import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
from collections import Counter

# 导入你的自定义模块
from datasets.PowerFlowData import PowerFlowData 
from networks.MPN import MaskEmbdMultiMPN_GPS
from utils.evaluation import load_model
from utils.argument_parser import argument_parser

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 物理辅助函数
# ==========================================

def rect_to_polar(e, f):
    """将直角坐标预测值转换为极坐标 (Vm, Va_degree)"""
    vm = torch.sqrt(e**2 + f**2 + 1e-12)
    va_rad = torch.atan2(f, e)
    va_deg = va_rad * (180.0 / torch.pi)
    return vm, va_deg

# ==========================================
# 2. 核心评估函数 (集成 Bias Correction + 混合策略数据收集)
# ==========================================

@torch.no_grad()
def evaluate_full_metrics(model, loader, device, xymean, xystd, edgemean, edgestd):
    model.eval()
    
    metrics = {
        'num_samples': 0,
        'mae_vm': 0., 
        'all_gt_labels': [],
        'all_pred_labels': [],
        'fp_details': [], # 误报样本详情
        'fn_details': [], # 漏报样本详情
        'all_samples_min_vm': [] # 存储所有样本的 (Pred_Min, True_Min)，用于混合策略分析
    }
    
    ef_mean = xymean[:, 2:4].to(device)
    ef_std  = xystd[:, 2:4].to(device)
    
    # 设定校准参数
    BIAS_CORRECTION = 0.002  # 强制压低预测电压
    SAFETY_LIMIT = 0.95      # 安全阈值

    for data in loader:
        data = data.to(device)
        out = model(data) # 预测 e, f
        
        # 1. 反归一化
        target_ef = data.y[:, 2:4]
        pred_real = out * (ef_std + 1e-7) + ef_mean
        target_real = target_ef * (ef_std + 1e-7) + ef_mean
        
        # 2. 计算节点 Vm
        pred_vm, _ = rect_to_polar(pred_real[:, 0], pred_real[:, 1])
        true_vm, _ = rect_to_polar(target_real[:, 0], target_real[:, 1])
        
        # ==========================================
        # 💉 【核心修改】 偏差校准 (Bias Correction)
        # ==========================================
        pred_vm = pred_vm - BIAS_CORRECTION
        # ==========================================
        
        # 3. 安全判定逻辑 (只看低压 < 0.95)
        # 注意：这里的高压判定被移除了，为了减少 FP
        v_unsafe_node = (pred_vm < SAFETY_LIMIT)
        
        # 4. 逐样本分析
        for i in range(data.num_graphs):
            node_mask = (data.batch == i)
            
            # A. Ground Truth (只要 Dataset label > 0 就算不安全)
            gt_status = 1 if data.label[i].item() > 0 else 0
            metrics['all_gt_labels'].append(gt_status)
            
            # B. AI Prediction
            ai_v_unsafe = v_unsafe_node[node_mask].any().item()
            pred_status = 1 if ai_v_unsafe else 0
            metrics['all_pred_labels'].append(pred_status)
            
            # C. 收集混合策略所需数据 (Min Vm)
            p_min = pred_vm[node_mask].min().item()
            t_min = true_vm[node_mask].min().item()
            metrics['all_samples_min_vm'].append((p_min, t_min))

            # D. 诊断误报 (FP)
            if gt_status == 0 and pred_status == 1:
                max_err = (true_vm[node_mask] - pred_vm[node_mask]).abs().max().item()
                metrics['fp_details'].append((t_min, p_min, max_err))
                
            # E. 诊断漏报 (FN)
            if gt_status == 1 and pred_status == 0:
                max_err = (true_vm[node_mask] - pred_vm[node_mask]).abs().max().item()
                metrics['fn_details'].append((t_min, p_min, max_err))

        # 5. 基础误差统计
        node_mask_all = data.pred_mask[:, 2] 
        m_sum = node_mask_all.sum().item() + 1e-6
        metrics['mae_vm'] += ((pred_vm - true_vm).abs() * node_mask_all).sum().item() / m_sum * data.num_graphs
        metrics['num_samples'] += data.num_graphs

    metrics['mae_vm'] /= metrics['num_samples']
    return metrics

# ==========================================
# 3. 混合策略模拟函数 (Hybrid Simulation)
# ==========================================
def simulate_hybrid_strategy(metrics, base_threshold=0.95):
    all_vm = np.array(metrics['all_samples_min_vm']) # Shape: [N, 2]
    pred_min = all_vm[:, 0]
    true_min = all_vm[:, 1]
    
    # 真实的“不安全”样本 (True Min < 0.95)
    actual_unsafe = (true_min < base_threshold)
    total_samples = len(pred_min)

    print("\n" + "="*60)
    print("🤖 AI + 物理混合策略模拟 (AI-Physics Hybrid Solver)")
    print("   策略: 预测值在 [0.95-Margin, 0.95+Margin] 之间的样本，")
    print("         交给物理求解器重算(耗时)，其余直接信AI(极速)。")
    print("="*60)
    
    # 测试不同的裕度 (Margin)
    margins = [0.005, 0.010, 0.015, 0.020, 0.025]
    
    print(f"{'Margin':<10} | {'区间 (重算区)':<20} | {'需重算比例(成本)':<15} | {'剩余漏报(风险)':<15} | {'评价'}")
    print("-" * 90)
    
    for margin in margins:
        lower = base_threshold - margin
        upper = base_threshold + margin
        
        # 1. 灰区 (重算)
        mask_gray = (pred_min >= lower) & (pred_min <= upper)
        n_recalc = np.sum(mask_gray)
        ratio_recalc = n_recalc / total_samples
        
        # 2. 绿区 (AI 放行)
        # AI 认为 > upper (非常安全)，直接放行
        mask_green = (pred_min > upper)
        
        # 3. 恶性漏报 (Critical FN)
        # AI 放行了，但其实是危险的
        n_critical_fn = np.sum(mask_green & actual_unsafe)
        
        status = "✅ 完美" if n_critical_fn == 0 else "⚠️ 有风险"
        
        print(f"+/- {margin:<5.3f} | [{lower:.3f}, {upper:.3f}]      | {ratio_recalc:<6.2%} ({n_recalc})   | {n_critical_fn:<13d}   | {status}")

    print("-" * 90)

# ==========================================
# 4. 绘图函数
# ==========================================
def plot_fn_analysis(fn_details, save_path='fn_analysis.png'):
    if len(fn_details) == 0: return
    fn_array = np.array(fn_details)
    true_min = fn_array[:, 0]
    pred_min = fn_array[:, 1]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 8))
    
    plt.scatter(true_min, pred_min, alpha=0.6, color='orange', label='FN Samples')
    
    # 画线
    plt.plot([0.85, 1.0], [0.85, 1.0], 'k--', label='Ideal x=y')
    plt.axhline(0.95, color='r', linestyle='--', label='AI Threshold (0.95)')
    plt.axvline(0.95, color='b', linestyle='--', label='True Threshold (0.95)')
    
    # 填充恶性漏报区域
    plt.fill_between([0.85, 0.95], 0.95, 1.0, color='red', alpha=0.1, label='Critical FN Zone')

    plt.title('False Negative Analysis\n(Why AI missed them?)')
    plt.xlabel('True Min Voltage (Unsafe)')
    plt.ylabel('Pred Min Voltage (Safe)')
    plt.legend()
    plt.savefig(save_path)
    print(f"[Plot] 漏报分布图已保存至: {save_path}")

# ==========================================
# 5. 主程序
# ==========================================
def main():
    # ---------------------------------------------
    # 请确保这里的 ID 和 case 名字是对的
    run_id = '20251223-6480' 
    case_name = '118v_n1_train' 
    # ---------------------------------------------
    
    args = argument_parser()
    # 强制覆盖 args 以便直接运行
    args.case = case_name 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载参数
    data_param_path = os.path.join(args.data_dir, 'params', f'data_params_{run_id}.pt')
    if not os.path.exists(data_param_path):
        print(f"Error: 找不到参数文件 {data_param_path}")
        return
    data_param = torch.load(data_param_path, map_location='cpu')
    
    # 2. 加载数据
    print("Loading Test Data...")
    testset = PowerFlowData(root=args.data_dir, case=args.case,
                            split=[.5, .2, .3], task='test',
                            xymean=data_param['xymean'], xystd=data_param['xystd'],
                            edgemean=data_param['edgemean'], edgestd=data_param['edgestd'])
    loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    # 3. 加载模型
    node_in, _, edge_dim = testset.get_data_dimensions()
    model = MaskEmbdMultiMPN_GPS(nfeature_dim=node_in, efeature_dim=edge_dim, output_dim=2, 
                                 hidden_dim=args.hidden_dim, n_gnn_layers=args.n_gnn_layers).to(device)
    model, _ = load_model(model, run_id, device)
    
    # 4. 运行评估
    print("Running Evaluation (Bias Correction = -0.002)...")
    res = evaluate_full_metrics(model, loader, device, 
                                data_param['xymean'], data_param['xystd'],
                                data_param['edgemean'], data_param['edgestd'])
    
    # 5. 输出常规报告
    gt = np.array(res['all_gt_labels'])
    pred = np.array(res['all_pred_labels'])
    cm = confusion_matrix(gt, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)

    print("\n" + "="*50)
    print(f"🛡️  N-1 安全判定评估 (校准后)")
    print(f"  准确识别安全 (TN): {tn}")
    print(f"  误报 (FP): {fp}  (Bias Correction helps here)")
    print(f"  漏报 (FN): {fn}")
    print(f"  正确识别故障 (TP): {tp}")
    print("-" * 30)
    print(f"  Recall (捕捉率): {tp/(tp+fn+1e-9):.2%} (Goal: >90%)")
    print(f"  FPR (误报率)   : {fp/(fp+tn+1e-9):.2%}")
    print("="*50)
    
    # 6. 运行混合策略模拟
    simulate_hybrid_strategy(res)

    # 7. 画漏报分析图
    plot_fn_analysis(res['fn_details'])

if __name__ == "__main__":
    main()
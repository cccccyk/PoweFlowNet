import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
from collections import Counter
import seaborn as sns

from datasets.PowerFlowData import PowerFlowData 
from networks.MPN import MaskEmbdMultiMPN_GPS
from utils.evaluation import load_model
from utils.argument_parser import argument_parser

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

@torch.no_grad()
def debug_max_voltage_source(model, loader, device, xymean, xystd):
    """
    专项体检：诊断全网最高电压(Max Vm)到底来自哪里？
    """
    model.eval()
    
    ef_mean = xymean[:, 2:4].to(device)
    ef_std  = xystd[:, 2:4].to(device)
    
    # 存储每一张图的诊断信息
    records = []
    
    print("🕵️ 开始 Max Voltage 溯源诊断...")
    
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        out = model(data)
        
        # 1. 反归一化
        pred_real = out * (ef_std + 1e-7) + ef_mean
        target_real = data.y[:, 2:4] * (ef_std + 1e-7) + ef_mean
        
        # 2. 计算 Vm
        pred_vm = torch.sqrt(pred_real[:, 0]**2 + pred_real[:, 1]**2 + 1e-12)
        true_vm = torch.sqrt(target_real[:, 0]**2 + target_real[:, 1]**2 + 1e-12)
        
        # =========================================================
        # 🛡️ 步骤 A: 强力验证 PV 替换逻辑是否生效
        # =========================================================
        is_controlled = (data.bus_type != 2) # 0=Slack, 1=PV
        
        # 替换前记录一下，看看有多少误差
        diff_before = (pred_vm[is_controlled] - true_vm[is_controlled]).abs().mean().item()
        
        # --- 执行替换 ---
        pred_vm[is_controlled] = true_vm[is_controlled]
        
        # 替换后检查：误差必须是 0.0
        diff_after = (pred_vm[is_controlled] - true_vm[is_controlled]).abs().max().item()
        if diff_after > 1e-5:
            print(f"⚠️ 警告！Batch {batch_idx} 替换失败！PV节点最大残差: {diff_after}")
        # =========================================================

        # 3. 逐样本溯源
        # 找出每个图的 Max 电压是由哪个节点贡献的
        num_graphs = data.num_graphs
        for i in range(num_graphs):
            # 获取属于这张图的所有节点索引
            node_indices = torch.where(data.batch == i)[0]
            
            # 取出这张图的电压
            local_pred_vm = pred_vm[node_indices]
            local_true_vm = true_vm[node_indices]
            local_types   = data.bus_type[node_indices]
            
            # --- 关键：寻找 Max 的“肇事者” ---
            # 找到预测电压最高的那个节点的局部索引
            max_val_pred, argmax_idx = torch.max(local_pred_vm, dim=0)

            max_val_true, true_argmax_idx = torch.max(local_true_vm, dim=0)
            max_node_type_true = local_types[true_argmax_idx].item()
            
            # 查户口：这个节点是什么类型？
            max_node_type = local_types[argmax_idx].item() # 0, 1, or 2
            
            # 对应的真实电压是多少？
            max_val_true_at_that_node = local_true_vm[argmax_idx].item()
            
            # 这张图真实的最高电压是多少（可能不是同一个点）
            global_true_max = local_true_vm.max().item()
            
            records.append({
                'pred_max': max_val_pred.item(),
                'true_max': global_true_max,
                'source_type': max_node_type, # 0=Slack, 1=PV, 2=PQ
                'is_aligned': abs(max_val_pred.item() - global_true_max) < 1e-4,
                'true_max_source_type': max_node_type_true # 记录真值的冠军类型
            })
            
    print(f"✅ 诊断完成，共分析 {len(records)} 个样本。正在绘图...")
    plot_diagnosis_results(records)

def plot_diagnosis_results(records):
    # 转换为 Numpy 方便切片
    pred_max = np.array([r['pred_max'] for r in records])
    true_max = np.array([r['true_max'] for r in records])
    source_type = np.array([r['source_type'] for r in records])
    
    # 分类
    mask_slack = (source_type == 0)
    mask_pv    = (source_type == 1)
    mask_pq    = (source_type == 2)
    
    plt.figure(figsize=(10, 10))
    
    # 1. 画 Slack 贡献的 Max (应该在线上)
    plt.scatter(true_max[mask_slack], pred_max[mask_slack], 
                c='green', s=20, alpha=0.6, label=f'Max from Slack (N={mask_slack.sum()})')
    
    # 2. 画 PV 贡献的 Max (应该在线上)
    plt.scatter(true_max[mask_pv], pred_max[mask_pv], 
                c='blue', s=20, alpha=0.6, label=f'Max from PV (N={mask_pv.sum()})')
    
    # 3. 画 PQ 贡献的 Max (可能是罪魁祸首)
    plt.scatter(true_max[mask_pq], pred_max[mask_pq], 
                c='red', s=20, alpha=0.5, label=f'Max from PQ (N={mask_pq.sum()})')
    
    # 画对角线和边界
    plt.plot([0.9, 1.2], [0.9, 1.2], 'k--', linewidth=1)
    plt.axhline(1.05, color='gray', linestyle='--')
    plt.axvline(1.05, color='gray', linestyle='--')
    
    plt.title("Diagnostics: Which Node Type Determines Pred Max Voltage?")
    plt.xlabel("True Max Voltage of Graph")
    plt.ylabel("Pred Max Voltage of Graph")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = 'max_voltage_source_debug.png'
    plt.savefig(save_path, dpi=150)
    print(f"🖼️ 诊断图已保存至: {save_path}")
    
    # --- 文字统计报告 ---
    print("\n" + "="*50)
    print("📊 Max Voltage 来源统计报告")
    print("="*50)
    total = len(records)
    print(f"总样本数: {total}")
    
    # 统计 PQ 导致的过冲
    # 过冲定义：Pred Max > True Max + 0.001 (允许一点浮点误差)
    overshoot = (pred_max > true_max + 0.001)
    
    n_pq = mask_pq.sum()
    n_pq_overshoot = (mask_pq & overshoot).sum()
    
    print(f"1. 由 PQ 节点决定最高电压的样本数: {n_pq} ({n_pq/total:.2%})")
    print(f"   -> 其中发生过冲(Pred > True)的数量: {n_pq_overshoot}")
    if n_pq > 0:
        print(f"   -> PQ过冲率: {n_pq_overshoot/n_pq:.2%}")
        
    n_pv_slack = mask_pv.sum() + mask_slack.sum()
    n_pv_error = ((mask_pv | mask_slack) & (np.abs(pred_max - true_max) > 1e-4)).sum()
    print(f"2. 由 PV/Slack 决定最高电压的样本数: {n_pv_slack} ({n_pv_slack/total:.2%})")
    print(f"   -> 其中偏离对角线的数量: {n_pv_error} (应该接近0)")
    
    if n_pv_error > 0:
        print("⚠️ 警告：检测到 PV/Slack 节点不在对角线上！这意味着替换逻辑代码没生效，或者统计逻辑有Bug。")
    else:
        print("✅ 确认：所有由 PV/Slack 主导的样本均完美落在对角线上。")

    true_pq_wins = np.array([r['true_max_source_type'] == 2 for r in records]).sum()
    print(f"🔎 真相核查：在真实标签(Ground Truth)中，PQ节点是最高电压的比例: {true_pq_wins/total:.2%}")

# ==========================================
# 2. 核心评估函数 (已修改：引入已知节点电压替换)
# ==========================================

@torch.no_grad()
def evaluate_full_metrics(model, loader, device, xymean, xystd, edgemean, edgestd):
    model.eval()
    
    metrics = {
        'num_samples': 0,
        'mae_vm': 0., 
        'all_gt_labels': [],
        'all_pred_labels': [],
        'fp_details': [], 
        'fn_details': [],
        # 这里存的将是 (Global_Min, PQ_Max, True_Global_Min, True_PQ_Max)
        'all_samples_extremes': [] 
    }
    
    ef_mean = xymean[:, 2:4].to(device)
    ef_std  = xystd[:, 2:4].to(device)
    
    BIAS_CORRECTION = 0.00  
    
    LIMIT_LOW = 0.95
    LIMIT_PQ_HIGH = 1.05

    for data in loader:
        data = data.to(device)
        out = model(data)
        
        # 1. 反归一化
        target_ef = data.y[:, 2:4]
        pred_real = out * (ef_std + 1e-7) + ef_mean
        target_real = target_ef * (ef_std + 1e-7) + ef_mean
        
        # 2. 计算节点 Vm
        pred_vm, _ = rect_to_polar(pred_real[:, 0], pred_real[:, 1])
        true_vm, _ = rect_to_polar(target_real[:, 0], target_real[:, 1])
        
        pred_vm = pred_vm - BIAS_CORRECTION
        
        # ---------------------------------------------------------
        # 【判定逻辑重构】
        # ---------------------------------------------------------
        if hasattr(data, 'bus_type'):
            is_pq = (data.bus_type == 2)
        else:
            # 防御性代码
            is_pq = torch.ones_like(pred_vm, dtype=torch.bool)
        
        # A. 低压判定：全网任何节点 < 0.95
        # (pred_vm < 0.95)
        v_unsafe_low_node = (pred_vm < LIMIT_LOW)
        
        # B. 高压判定：仅 PQ 节点 > 1.05
        # (pred_vm > 1.05) AND (is_pq)
        v_unsafe_high_node = (pred_vm > LIMIT_PQ_HIGH) & is_pq
        
        # C. 综合节点越限情况
        v_unsafe_node = v_unsafe_low_node | v_unsafe_high_node
        
        # ---------------------------------------------------------
        
        # 4. 逐样本分析
        for i in range(data.num_graphs):
            node_mask = (data.batch == i)
            
            # --- 关键：构建 PQ 掩码 ---
            # 当前图中，既属于该图，又是 PQ 类型的节点
            pq_mask_in_graph = node_mask & is_pq
            
            # A. Ground Truth
            gt_status = 1 if data.label[i].item() > 0 else 0
            metrics['all_gt_labels'].append(gt_status)
            
            # B. AI Prediction
            ai_v_unsafe = v_unsafe_node[node_mask].any().item()
            pred_status = 1 if ai_v_unsafe else 0
            metrics['all_pred_labels'].append(pred_status)
            
            # C. 收集极值数据 (核心修改)
            # ---------------------------------------------
            # Min: 取全网最低 (包括 PV，因为 PV 跌落也算故障)
            p_curr_all = pred_vm[node_mask]
            t_curr_all = true_vm[node_mask]
            p_min = p_curr_all.min().item()
            t_min = t_curr_all.min().item()
            
            # Max: 只取 PQ 节点的最高值！
            # 只有当该图有 PQ 节点时才计算 (正常都有)
            if pq_mask_in_graph.any():
                p_curr_pq = pred_vm[pq_mask_in_graph]
                t_curr_pq = true_vm[pq_mask_in_graph]
                p_max = p_curr_pq.max().item()
                t_max = t_curr_pq.max().item()
            else:
                # 极端情况 fallback
                p_max = -1.0 
                t_max = -1.0
            # ---------------------------------------------
            
            metrics['all_samples_extremes'].append((p_min, p_max, t_min, t_max))

            # D. 诊断误报 (FP)
            if gt_status == 0 and pred_status == 1:
                # 记录误差时，简单记录最大绝对误差即可
                max_err = (t_curr_all - p_curr_all).abs().max().item()
                metrics['fp_details'].append((t_min, t_max, p_min, p_max, max_err))
                
            # E. 诊断漏报 (FN)
            if gt_status == 1 and pred_status == 0:
                max_err = (t_curr_all - p_curr_all).abs().max().item()
                metrics['fn_details'].append((t_min, t_max, p_min, p_max, max_err))

        # 5. MAE 统计 (只统计 PQ，反映负载侧精度)
        m_sum = is_pq.sum().item() + 1e-6
        metrics['mae_vm'] += ((pred_vm - true_vm).abs() * is_pq).sum().item() / m_sum * data.num_graphs
        metrics['num_samples'] += data.num_graphs

    metrics['mae_vm'] /= metrics['num_samples']
    return metrics

# ==========================================
# 3. 混合策略模拟函数 (保持不变)
# ==========================================
def simulate_hybrid_strategy(metrics, limit_low=0.95, limit_high=1.05):
    data = np.array(metrics['all_samples_extremes']) 
    p_min, p_max = data[:, 0], data[:, 1]
    t_min, t_max = data[:, 2], data[:, 3]
    
    actual_unsafe = (t_min < limit_low) | (t_max > limit_high)
    actual_safe = ~actual_unsafe
    total_samples = len(p_min)

    print("\n" + "="*80)
    print("🤖 AI + 物理混合策略模拟 (双向检测)")
    print("="*80)
    
    margins = [0.005, 0.010, 0.015, 0.020, 0.025]
    print(f"{'Margin':<8} | {'需重算(成本)':<15} | {'剩余漏报(FN)':<12} | {'红区误报(FP)':<12}")
    print("-" * 80)
    
    for margin in margins:
        l_gray_start, l_gray_end = limit_low - margin, limit_low + margin
        h_gray_start, h_gray_end = limit_high - margin, limit_high + margin
        
        # 1. 灰区 (重算)
        mask_recalc = ((p_min >= l_gray_start) & (p_min <= l_gray_end)) | \
                      ((p_max >= h_gray_start) & (p_max <= h_gray_end))
        n_recalc = np.sum(mask_recalc)
        
        # 2. 绿区 (AI 放行)
        mask_green = (p_min > l_gray_end) & (p_max < h_gray_start)
        
        # 3. 红区 (AI 报警)
        mask_red = (p_min < l_gray_start) | (p_max > h_gray_end)
        
        n_crit_fn = np.sum(mask_green & actual_unsafe)
        n_red_fp = np.sum(mask_red & actual_safe)
        
        print(f"+/-{margin:<.3f} | {n_recalc/total_samples:<6.2%} ({n_recalc})   | {n_crit_fn:<12d} | {n_red_fp:<12d}")
    print("-" * 80)

# ==========================================
# 4. 分布诊断函数 (保持不变)
# ==========================================
def plot_distribution_debug(metrics, save_path='dist_debug.png'):
    # ... (代码保持不变，直接复制即可)
    # 略去以节省篇幅，请保留你原有的绘图代码
    print("\n" + "="*50)
    print("🔬 正在进行分布“尸检”诊断...")
    if len(metrics['all_samples_extremes']) == 0: return

    data = np.array(metrics['all_samples_extremes'])
    p_min, p_max = data[:, 0], data[:, 1]
    t_min, t_max = data[:, 2], data[:, 3]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Min Voltage 分布 (Sag)
    sns.kdeplot(t_min, ax=axes[0,0], color='blue', fill=True, label='True Min', clip=(0.8, 1.05))
    sns.kdeplot(p_min, ax=axes[0,0], color='red', fill=True, label='Pred Min', clip=(0.8, 1.05))
    axes[0,0].axvline(0.95, color='k', linestyle='--', label='Limit 0.95')
    axes[0,0].set_title('Distribution of MIN Voltage')
    axes[0,0].legend()

    # 2. Max Voltage 分布 (Swell)
    sns.kdeplot(t_max, ax=axes[0,1], color='blue', fill=True, label='True Max', clip=(0.95, 1.2))
    sns.kdeplot(p_max, ax=axes[0,1], color='red', fill=True, label='Pred Max', clip=(0.95, 1.2))
    axes[0,1].axvline(1.05, color='k', linestyle='--', label='Limit 1.05')
    axes[0,1].set_title('Distribution of MAX Voltage')
    axes[0,1].legend()

    # 3. 散点图 Min
    axes[1,0].scatter(t_min, p_min, alpha=0.3, s=5, color='purple')
    axes[1,0].plot([0.8, 1.1], [0.8, 1.1], 'k--')
    axes[1,0].axhline(0.95, color='r', linestyle='--')
    axes[1,0].axvline(0.95, color='b', linestyle='--')
    axes[1,0].set_title('Scatter: True vs Pred Min')
    axes[1,0].set_xlabel('True Min')
    axes[1,0].set_ylabel('Pred Min')

    # 4. 散点图 Max
    axes[1,1].scatter(t_max, p_max, alpha=0.3, s=5, color='green')
    axes[1,1].plot([0.9, 1.2], [0.9, 1.2], 'k--')
    axes[1,1].axhline(1.05, color='r', linestyle='--')
    axes[1,1].axvline(1.05, color='b', linestyle='--')
    axes[1,1].set_title('Scatter: True vs Pred Max')
    axes[1,1].set_xlabel('True Max')
    axes[1,1].set_ylabel('Pred Max')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot] 分布诊断图已保存至: {save_path}")

    # --- 关键统计 ---
    total = len(p_min)
    # 统计落在灰区 (0.945~0.955) 和 (1.045~1.055) 的比例
    in_low_gray = np.sum((p_min >= 0.945) & (p_min <= 0.955))
    in_high_gray = np.sum((p_max >= 1.045) & (p_max <= 1.055))

    print("\n[📊 灰区堆积分析]")
    print(f"Pred Min 落在 [0.945, 0.955] (低压边界) 的比例: {in_low_gray/total:.2%} ({in_low_gray})")
    print(f"Pred Max 落在 [1.045, 1.055] (高压边界) 的比例: {in_high_gray/total:.2%} ({in_high_gray})")
    print("--> 如果这两个比例很高，说明大量样本卡在边界上，导致混合策略失效(必须重算)。")

# ==========================================
# 5. 主程序
# ==========================================
def main():
    run_id = '20260108-2395'  # 记得改成你最新训练的 ID (门控+NodeID版)
    case_name = '118v2_30w_n1' 
    
    args = argument_parser()
    args.case = case_name 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    # 注意：确保这里实例化模型时参数与你训练时一致 (例如 num_nodes)
    node_in, _, edge_dim = testset.get_data_dimensions()
    model = MaskEmbdMultiMPN_GPS(nfeature_dim=node_in, efeature_dim=edge_dim, output_dim=2, 
                                 hidden_dim=args.hidden_dim, n_gnn_layers=args.n_gnn_layers,
                                 num_nodes=118).to(device) # 加上 num_nodes=118
    model, _ = load_model(model, run_id, device)
    
    # 4. 运行评估
    print("Running Full Evaluation...")
    res = evaluate_full_metrics(model, loader, device, 
                                data_param['xymean'], data_param['xystd'],
                                data_param['edgemean'], data_param['edgestd'])
    
    # debug_max_voltage_source(model, loader, device, data_param['xymean'], data_param['xystd'])
    
    # 5. 输出报告
    gt = np.array(res['all_gt_labels'])
    pred = np.array(res['all_pred_labels'])
    cm = confusion_matrix(gt, pred)
    # 处理可能的 shape 不匹配 (例如测试集全是安全样本)
    if cm.size == 1:
        tn = cm[0,0]
        fp, fn, tp = 0, 0, 0
    else:
        tn, fp, fn, tp = cm.ravel()

    print("\n" + "="*50)
    print(f"🛡️  N-1 安全判定评估 (PV修正后)")
    print(f"  准确识别安全 (TN): {tn}")
    print(f"  误报 (FP): {fp}")
    print(f"  漏报 (FN): {fn}")
    print(f"  正确识别故障 (TP): {tp}")
    print(f"  Recall: {tp/(tp+fn+1e-9):.2%}")
    print(f"  FPR   : {fp/(fp+tn+1e-9):.2%}")
    print("="*50)
    
    simulate_hybrid_strategy(res)
    plot_distribution_debug(res) # 需要补全上面的函数才能运行绘图

if __name__ == "__main__":
    main()
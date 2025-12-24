import os
import logging
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, classification_report # [新增] 用于安全评估统计

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

def compute_branch_flows(e, f, edge_index, edge_attr, baseMVA=100.0):
    """
    根据节点电压和 Ybus 边特征计算支路潮流 (P_ij, Q_ij)
    """
    e_i, f_i = e[edge_index[0]], f[edge_index[0]]
    e_j, f_j = e[edge_index[1]], f[edge_index[1]]

    g_line = -edge_attr[:, 0]
    b_line = -edge_attr[:, 1]

    de = e_i - e_j
    df = f_i - f_j

    i_real = g_line * de - b_line * df
    i_imag = g_line * df + b_line * de

    p_ij = (e_i * i_real + f_i * i_imag) * baseMVA
    q_ij = (f_i * i_real - e_i * i_imag) * baseMVA
    
    return p_ij, q_ij

# ==========================================
# 2. 核心评估函数 (集成安全评估)
# ==========================================

@torch.no_grad()
def evaluate_full_metrics(model, loader, device, xymean, xystd, edgemean, edgestd):
    model.eval()
    
    metrics = {
        'num_samples': 0,
        'mae_vm': 0., 'mae_va': 0.,
        'max_err_vm': 0., 'max_err_va': 0.,
        'mae_e': 0., 'mae_f': 0.,
        'branch_p_mae': 0., 'branch_p_max': 0.,
        # [新增] 安全评估相关
        'all_gt_labels': [],
        'all_pred_labels': []
    }
    
    ef_mean = xymean[:, 2:4].to(device)
    ef_std  = xystd[:, 2:4].to(device)
    edgemean = edgemean.to(device)
    edgestd = edgestd.to(device)

    for data in loader:
        data = data.to(device)
        out = model(data) # 预测 e, f
        
        # 1. 反归一化
        target_ef = data.y[:, 2:4]
        pred_real = out * (ef_std + 1e-7) + ef_mean
        target_real = target_ef * (ef_std + 1e-7) + ef_mean
        
        # 2. 计算节点 Vm, Va
        pred_vm, pred_va = rect_to_polar(pred_real[:, 0], pred_real[:, 1])
        pred_vm = pred_vm - 0.002
        true_vm, true_va = rect_to_polar(target_real[:, 0], target_real[:, 1])
        
        # 3. 支路潮流计算
        real_edge_attr = data.edge_attr * (edgestd + 1e-7) + edgemean
        p_pred, q_pred = compute_branch_flows(pred_real[:,0], pred_real[:,1], data.edge_index, real_edge_attr)
        p_true, q_true = compute_branch_flows(target_real[:,0], target_real[:,1], data.edge_index, real_edge_attr)
        
        # ==========================================
        # [核心新增] 安全评估判定逻辑 (Security Assessment)
        # ==========================================
        # 判定标准：电压 < 0.95 或 > 1.05 为不安全
        # 注意：此处只演示电压判定。如有线路限值，也可加入 s_pred > s_limit 的判定。
        v_unsafe_node = (pred_vm < 0.95) | (pred_vm > 1.05)
        
        fn_labels = []
        for i in range(data.num_graphs):
            # A. 提取该样本的 Ground Truth (二分类：安全 0 vs 不安全 1)
            gt_status = 1 if data.label[i].item() > 0 else 0
            metrics['all_gt_labels'].append(gt_status)
            
            # B. 提取 AI 的判定结果
            node_mask = (data.batch == i)
            # 如果该样本中任何一个节点电压越限，则判定为不安全
            ai_v_unsafe = v_unsafe_node[node_mask].any().item()
            
            # TODO: 如果你有支路限值，可以在此加入支路过载判定
            # ai_p_overload = (p_pred[edge_mask].abs() > limit).any().item()
            
            pred_status = 1 if ai_v_unsafe else 0
            metrics['all_pred_labels'].append(pred_status)

            if gt_status == 1 and pred_status == 0:
                # 记录漏报样本的原始 Label (1:V, 2:P, 3:Both)
                fn_labels.append(data.label[i].item())
        # 运行结束后打印
        from collections import Counter
        print(f"漏报样本原始类型分布: {Counter(fn_labels)}")

        # 4. 基础误差统计 (原有逻辑)
        node_mask_all = data.pred_mask[:, 2] # 预测掩码
        m_sum = node_mask_all.sum().item() + 1e-6
        batch_size = data.num_graphs
        
        metrics['mae_vm'] += ((pred_vm - true_vm).abs() * node_mask_all).sum().item() / m_sum * batch_size
        metrics['mae_va'] += ((pred_va - true_va).abs() * node_mask_all).sum().item() / m_sum * batch_size
        metrics['branch_p_mae'] += (p_pred - p_true).abs().mean().item() * batch_size
        metrics['branch_p_max'] = max(metrics['branch_p_max'], (p_pred - p_true).abs().max().item())
        metrics['num_samples'] += batch_size

    # 平均化基础指标
    n = metrics['num_samples']
    for k in ['mae_vm', 'mae_va', 'mae_e', 'mae_f', 'branch_p_mae']:
        metrics[k] /= n
            
    return metrics

# ==========================================
# 3. 主程序
# ==========================================

def main():
    run_id = '20251223-6480' # 修改为你的模型 ID
    args = argument_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载参数与数据
    data_param_path = os.path.join(args.data_dir, 'params', f'data_params_{run_id}.pt')
    data_param = torch.load(data_param_path, map_location='cpu')
    
    # 注意：此处 PowerFlowData 必须是修改过、支持 label 的版本
    testset = PowerFlowData(root=args.data_dir, case=args.case,
                            split=[.5, .2, .3], task='test',
                            xymean=data_param['xymean'], xystd=data_param['xystd'],
                            edgemean=data_param['edgemean'], edgestd=data_param['edgestd'])
    loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    # 2. 构建并加载模型
    node_in, _, edge_dim = testset.get_data_dimensions()
    model = MaskEmbdMultiMPN_GPS(nfeature_dim=node_in, efeature_dim=edge_dim, output_dim=2, 
                                 hidden_dim=args.hidden_dim, n_gnn_layers=args.n_gnn_layers).to(device)
    
    model, _ = load_model(model, run_id, device)
    
    # 3. 运行评估
    res = evaluate_full_metrics(model, loader, device, 
                                data_param['xymean'], data_param['xystd'],
                                data_param['edgemean'], data_param['edgestd'])
    
    # 4. 输出常规指标
    print("\n" + "="*50)
    print(f"📊 基础误差评估: {args.case}")
    print(f"  MAE Vm : {res['mae_vm']:.6f} p.u. | MAE Va : {res['mae_va']:.4f} deg")
    print(f"  P MAE  : {res['branch_p_mae']:.4f} MW   | P MAX Err: {res['branch_p_max']:.4f} MW")
    
    # ==========================================
    # 5. [新增] 安全评估报告 (Security Report)
    # ==========================================
    gt = np.array(res['all_gt_labels'])
    pred = np.array(res['all_pred_labels'])
    
    cm = confusion_matrix(gt, pred)
    # 处理全安全或全不安全的特殊情况，防止索引错误
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0)

    print("\n" + "="*50)
    print(f"🛡️  N-1 安全判定评估 (二分类)")
    print("="*50)
    print(f"  准确识别安全 (TN): {tn:4d} | 误报 (FP): {fp:4d}")
    print(f"  漏报故障 (FN)  : {fn:4d} | 正确识别故障 (TP): {tp:4d}  <-- 重点关注!")
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    recall = tp / (tp + fn + 1e-9)   # 召回率：所有真正的故障中有多少被AI发现了
    precision = tp / (tp + fp + 1e-9) # 精确率：AI报出的故障中有多少是真的
    fnr = fn / (tp + fn + 1e-9)      # 漏报率

    print(f"-"*50)
    print(f"  总准确率 (Accuracy) : {accuracy:.2%}")
    print(f"  故障捕捉率 (Recall)   : {recall:.2%}")
    print(f"  漏报率 (Miss Rate/FNR): {fnr:.2%}  (越低越安全)")
    print(f"  误报率 (Fall-out/FPR) : {fp/(fp+tn+1e-9):.2%} (越低越经济)")
    print("="*50)

if __name__ == "__main__":
    main()
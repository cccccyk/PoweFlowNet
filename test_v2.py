import os
import logging
import torch
import numpy as np
from torch_geometric.loader import DataLoader

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
    e, f: 节点电压实部/虚部 [N]
    edge_index: 边索引 [2, E]
    edge_attr: Ybus 非对角元 [Gij, Bij]
    """
    # 提取 from (i) 和 to (j) 节点的电压
    e_i, f_i = e[edge_index[0]], f[edge_index[0]]
    e_j, f_j = e[edge_index[1]], f[edge_index[1]]

    # Ybus 非对角元 Yij = -y_line_ij，所以线路导纳为:
    g_line = -edge_attr[:, 0]
    b_line = -edge_attr[:, 1]

    # 计算电压差
    de = e_i - e_j
    df = f_i - f_j

    # 计算支路电流 I_ij = y_line * (V_i - V_j)
    # 复数乘法: (g+jb)*(de+jdf) = (g*de - b*df) + j(g*df + b*de)
    i_real = g_line * de - b_line * df
    i_imag = g_line * df + b_line * de

    # 计算支路功率 S_ij = V_i * conj(I_ij)
    # P = e_i*Ir + f_i*Ii
    # Q = f_i*Ir - e_i*Ii
    p_ij = (e_i * i_real + f_i * i_imag) * baseMVA
    q_ij = (f_i * i_real - e_i * i_imag) * baseMVA
    
    return p_ij, q_ij

# ==========================================
# 2. 核心评估函数
# ==========================================

@torch.no_grad()
def evaluate_full_metrics(model, loader, device, xymean, xystd, edgemean, edgestd):
    model.eval()
    
    # 初始化统计字典
    metrics = {
        'num_samples': 0,
        # 节点电压指标
        'mae_vm': 0., 'mae_va': 0.,
        'max_err_vm': 0., 'max_err_va': 0.,
        # 节点直角指标
        'mae_e': 0., 'mae_f': 0.,
        'mape_f_reliable': 0.,
        # 支路潮流指标 (MW/MVAR)
        'branch_p_mae': 0., 'branch_p_rmse': 0., 'branch_p_max': 0.,
        'branch_q_mae': 0., 'branch_q_rmse': 0.
    }
    
    # 提取反归一化参数 [P, Q, e, f, Gii, Bii]
    ef_mean = xymean[:, 2:4].to(device)
    ef_std  = xystd[:, 2:4].to(device)
    edgemean = edgemean.to(device)
    edgestd = edgestd.to(device)

    MAPE_THRESHOLD = 1e-2 # 用于过滤 f 的 MAPE 计算

    for data in loader:
        data = data.to(device)
        out = model(data) # 预测 [N, 2] -> e, f
        
        # 1. 反归一化节点电压
        target_ef = data.y[:, 2:4]
        mask_ef = data.pred_mask[:, 2:4]
        
        pred_real = out * (ef_std + 1e-7) + ef_mean
        target_real = target_ef * (ef_std + 1e-7) + ef_mean
        
        # 2. 计算节点电压 Vm, Va 指标
        pred_vm, pred_va = rect_to_polar(pred_real[:, 0], pred_real[:, 1])
        true_vm, true_va = rect_to_polar(target_real[:, 0], target_real[:, 1])
        
        node_mask = mask_ef[:, 0] # 只取待预测节点
        diff_vm = (pred_vm - true_vm) * node_mask
        diff_va = (pred_va - true_va) * node_mask
        diff_va = (diff_va + 180) % 360 - 180 # 角度环路处理
        
        m_sum = node_mask.sum().item() + 1e-6
        batch_size = data.num_graphs
        
        metrics['mae_vm'] += (diff_vm.abs().sum().item() / m_sum) * batch_size
        metrics['mae_va'] += (diff_va.abs().sum().item() / m_sum) * batch_size
        metrics['max_err_vm'] = max(metrics['max_err_vm'], diff_vm.abs().max().item())
        metrics['max_err_va'] = max(metrics['max_err_va'], diff_va.abs().max().item())

        # 3. 计算节点 e, f 指标
        diff_ef = (pred_real - target_real) * mask_ef
        metrics['mae_e'] += (diff_ef[:, 0].abs().sum().item() / m_sum) * batch_size
        metrics['mae_f'] += (diff_ef[:, 1].abs().sum().item() / m_sum) * batch_size
        
        # 可靠的 MAPE f (分母大于阈值才计算)
        f_true_abs = target_real[:, 1].abs()
        f_mask_reliable = (f_true_abs > MAPE_THRESHOLD) * mask_ef[:, 1]
        if f_mask_reliable.sum() > 0:
            mape_f = (diff_ef[:, 1].abs() / (f_true_abs + 1e-8))[f_mask_reliable > 0].mean().item()
            metrics['mape_f_reliable'] += mape_f * batch_size

        # 4. 支路潮流计算与指标 (P, Q)
        real_edge_attr = data.edge_attr * (edgestd + 1e-7) + edgemean
        p_pred, q_pred = compute_branch_flows(pred_real[:,0], pred_real[:,1], data.edge_index, real_edge_attr)
        p_true, q_true = compute_branch_flows(target_real[:,0], target_real[:,1], data.edge_index, real_edge_attr)
        
        err_p = (p_pred - p_true)
        err_q = (q_pred - q_true)
        
        metrics['branch_p_mae'] += err_p.abs().mean().item() * batch_size
        metrics['branch_p_rmse'] += torch.sqrt((err_p**2).mean()).item() * batch_size
        metrics['branch_p_max'] = max(metrics['branch_p_max'], err_p.abs().max().item())
        metrics['branch_q_mae'] += err_q.abs().mean().item() * batch_size
        metrics['branch_q_rmse'] += torch.sqrt((err_q**2).mean()).item() * batch_size

        metrics['num_samples'] += batch_size

    # 平均化
    n = metrics['num_samples']
    for k in metrics:
        if 'max' not in k and k != 'num_samples':
            metrics[k] /= n
            
    return metrics

# ==========================================
# 3. 主程序
# ==========================================

def main():
    # 这里的 run_id 替换为你保存的模型 ID
    run_id = '20251222-9778' 
    
    args = argument_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载参数与数据
    data_param_path = os.path.join(args.data_dir, 'params', f'data_params_{run_id}.pt')
    data_param = torch.load(data_param_path, map_location='cpu')
    
    testset = PowerFlowData(root=args.data_dir, case=args.case,
                            split=[.5, .2, .3], task='test',
                            xymean=data_param['xymean'], xystd=data_param['xystd'],
                            edgemean=data_param['edgemean'], edgestd=data_param['edgestd'])
    loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    # 2. 构建并加载模型
    node_in, _, edge_dim = testset.get_data_dimensions()
    model = MaskEmbdMultiMPN_GPS(
        nfeature_dim=node_in,
        efeature_dim=edge_dim,
        output_dim=2, # 输出 e, f
        hidden_dim=args.hidden_dim,
        n_gnn_layers=args.n_gnn_layers,
        nhead=4,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    model, _ = load_model(model, run_id, device)
    print(f"✅ Loaded GPS Model: {run_id}")
    
    # 3. 运行评估
    res = evaluate_full_metrics(model, loader, device, 
                                data_param['xymean'], data_param['xystd'],
                                data_param['edgemean'], data_param['edgestd'])
    
    # 4. 格式化输出
    print("\n" + "="*50)
    print(f"📊 Full Evaluation Results: {args.case}")
    print("="*50)
    print(f"【Node Voltage (Physical)】")
    print(f"  MAE Vm : {res['mae_vm']:.6f} p.u. | Max Err: {res['max_err_vm']:.6f}")
    print(f"  MAE Va : {res['mae_va']:.4f} deg  | Max Err: {res['max_err_va']:.4f}")
    
    print(f"\n【Node Rectangular (Internal)】")
    print(f"  MAE e  : {res['mae_e']:.6f} | f  : {res['mae_f']:.6f}")
    print(f"  Reliable MAPE f: {res['mape_f_reliable']*100:.4f}%")
    
    print(f"\n【Branch Power Flow (MW/MVAR)】")
    print(f"  P MAE  : {res['branch_p_mae']:.4f} MW   | RMSE: {res['branch_p_rmse']:.4f}")
    print(f"  P MAX  : {res['branch_p_max']:.4f} MW   (Critical for N-1!)")
    print(f"  Q MAE  : {res['branch_q_mae']:.4f} MVAR | RMSE: {res['branch_q_rmse']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
import os
import torch
import numpy as np
import logging
from torch_geometric.loader import DataLoader

# 引入你的数据处理和参数解析
from datasets.PowerFlowData_copy import PowerFlowData
from utils.argument_parser import argument_parser
from utils.evaluation import load_model

# 引入你的模型定义
from networks.MPN import MaskEmbdMultiMPN,MaskEmbdMultiMPN_NNConv_v2

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def denormalize(val, mean, std):
    """反归一化：将模型输出还原为真实物理值"""
    return val * std + mean

@torch.no_grad()
def main():
    # 1. 获取参数
    args = argument_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ================= [配置区域] =================
    # 你想查看测试集里的第几个样本？
    SAMPLE_IDX = 0  
    # 你的模型运行ID
    RUN_ID = '20251201-1019' 
    # ============================================

    # 2. 加载数据参数 (均值/标准差)
    # 我们只需要 xymean/xystd 来还原 V, Theta, P, Q
    # 不再需要 edgemean/edgestd，因为不做物理计算了
    data_dir = args.data_dir
    data_param_path = os.path.join(data_dir, 'params', f'data_params_{RUN_ID}.pt')
    
    if not os.path.exists(data_param_path):
        logger.error(f"找不到数据参数文件: {data_param_path}")
        return

    data_param = torch.load(data_param_path, map_location='cpu')
    xymean = data_param['xymean'].to(device)
    xystd = data_param['xystd'].to(device)

    # 3. 加载数据集 (Test Set)
    logger.info(f"Loading Test Data from {args.case}...")
    testset = PowerFlowData(root=data_dir, case=args.case,
                            split=[.5, .2, .3], task='test',
                            xymean=xymean.cpu(), xystd=xystd.cpu())
    
    if SAMPLE_IDX >= len(testset):
        logger.error(f"索引越界！测试集只有 {len(testset)} 个样本。")
        return
    
    data = testset[SAMPLE_IDX].to(device)
    
    # 4. 初始化并加载模型
    node_in_dim, node_out_dim, edge_dim = testset.get_data_dimensions()
    
    model = MaskEmbdMultiMPN_NNConv_v2(
        nfeature_dim=node_in_dim,
        efeature_dim=edge_dim,
        output_dim=node_out_dim,
        hidden_dim=args.hidden_dim,
        n_gnn_layers=args.n_gnn_layers,
        K=args.K,
        dropout_rate=args.dropout_rate
    ).to(device)

    model, _ = load_model(model, RUN_ID, device)
    model.eval()

    # 5. 模型预测
    logger.info(f"Running inference on Sample #{SAMPLE_IDX}...")
    out = model(data)

    # 6. 数据还原 (反归一化)
    # pred_real: [N, 4] -> (Vm, Va, P, Q)
    pred_real = denormalize(out, xymean, xystd)
    truth_real = denormalize(data.y, xymean, xystd)
    
    # 7. 打印详细对比
    print_comparison(data, pred_real, truth_real)

def print_comparison(data, pred, truth):
    """
    格式化打印对比结果: Pred vs True
    """
    num_nodes = data.num_nodes
    bus_type = data.bus_type.cpu().numpy() # 0=Slack, 1=PV, 2=PQ
    
    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()
    
    # 计算各项绝对误差
    diff = np.abs(pred - truth)
    
    print("\n" + "="*120)
    print(f"| {'Node':<4} | {'Type':<5} | {'Vm (Pred / True)':<20} | {'Va_deg (Pred / True)':<22} | {'P (Pred / True)':<20} | {'Q (Pred / True)':<20} |")
    print("-" * 120)

    # 累加器用于计算平均误差
    err_vm_sum, err_va_sum = 0, 0
    err_p_slack, err_q_pv = 0, 0
    count_pv = 0
    
    for i in range(num_nodes):
        b_type = int(bus_type[i])
        type_str = "Slk" if b_type == 0 else ("PV " if b_type == 1 else "PQ ")
        
        # 数值格式化
        vm_str = f"{pred[i,0]:.4f} / {truth[i,0]:.4f}"
        va_str = f"{pred[i,1]:.2f} / {truth[i,1]:.2f}" # 相角保留2位小数
        p_str  = f"{pred[i,2]:.4f} / {truth[i,2]:.4f}"
        q_str  = f"{pred[i,3]:.4f} / {truth[i,3]:.4f}"
        
        # 统计误差
        err_vm_sum += diff[i, 0]
        err_va_sum += diff[i, 1]
        
        # 高亮标记
        prefix = " "
        if b_type == 0: # Slack
            prefix = ">>" 
            err_p_slack = diff[i, 2] # Slack 节点的 P 是预测重点
        if b_type == 1: # PV
            prefix = "*"  
            err_q_pv += diff[i, 3] # PV 节点的 Q 是预测重点
            count_pv += 1
            
        print(f"{prefix}{i:<4} | {type_str:<5} | {vm_str:<20} | {va_str:<22} | {p_str:<20} | {q_str:<20} |")

    print("="*120)
    
    # --- 最终评估报告 ---
    print("\n[Evaluation Summary]")
    print(f"1. Voltage Magnitude (Vm):")
    print(f"   Mean Absolute Error (All Nodes): {err_vm_sum / num_nodes:.5f} p.u.")
    
    print(f"\n2. Voltage Angle (Va):")
    print(f"   Mean Absolute Error (All Nodes): {err_va_sum / num_nodes:.4f} degrees")
    if (err_va_sum / num_nodes) < 1.0:
        print("   >>> Result: EXCELLENT (< 1.0 deg)")
    else:
        print("   >>> Result: Good, but can be improved")

    print(f"\n3. Active Power (P) at Slack Node:")
    print(f"   Absolute Error: {err_p_slack:.4f} p.u.")
    
    print(f"\n4. Reactive Power (Q) at PV Nodes (Count: {count_pv}):")
    if count_pv > 0:
        mean_q_err = err_q_pv / count_pv
        print(f"   Mean Absolute Error: {mean_q_err:.4f} p.u.")
        # 计算 MAPE (排除接近0的值防止除零)
        pv_indices = np.where(bus_type == 1)[0]
        q_true_pv = truth[pv_indices, 3]
        q_pred_pv = pred[pv_indices, 3]
        valid_mask = np.abs(q_true_pv) > 1e-3
        if np.sum(valid_mask) > 0:
            mape = np.mean(np.abs((q_pred_pv[valid_mask] - q_true_pv[valid_mask]) / q_true_pv[valid_mask])) * 100
            print(f"   MAPE (excluding near-zero): {mape:.2f}%")
    else:
        print("   No PV nodes found in this sample.")

if __name__ == "__main__":
    main()
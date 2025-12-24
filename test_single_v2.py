import os
import torch
import numpy as np
import logging

# 引入你的数据处理和参数解析
from datasets.PowerFlowData_2 import PowerFlowData
from utils.argument_parser import argument_parser
from utils.evaluation import load_model
from networks.MPN import MaskEmbdMultiMPN

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. 纯手写物理计算函数 (No External Libs)
# ==========================================
def calculate_physics_manually(pred_real, data, edgemean, edgestd, device):
    # --- Step A: 还原线路参数 ---
    # 根据你的数据生成，这里直接就是 G 和 B
    edge_attr_real = data.edge_attr * edgestd + edgemean
    g_ij = edge_attr_real[:, 0]
    b_ij = edge_attr_real[:, 1]
    
    # 构建串联导纳
    Y_series = torch.complex(g_ij, b_ij) # [E]

    # --- Step B: 构建复数电压 ---
    Vm = pred_real[:, 0]
    Va_deg = pred_real[:, 1]
    Va_rad = Va_deg * (torch.pi / 180.0)
    V_node = torch.complex(Vm * torch.cos(Va_rad), Vm * torch.sin(Va_rad))

    # --- Step C: 计算电流 ---
    src, dst = data.edge_index
    
    # 既然已知 Y，直接算 I = Y * dV
    # 流出 i 的电流 (忽略对地支路)
    I_series = (V_node[src] - V_node[dst]) * Y_series
    
    # --- Step D: 聚合 ---
    I_injected = torch.zeros(data.num_nodes, dtype=torch.complex64, device=device)
    I_injected.index_add_(0, src, I_series)
    I_injected.index_add_(0, dst, -I_series)
    
    # --- Step E: 功率 ---
    S_injected = V_node * torch.conj(I_injected)
    
    # 这里不需要取负号，因为 PandaPower 定义 P_gen - P_load
    # 注入网络为正。
    P_calc = S_injected.real
    Q_calc = S_injected.imag
    
    return -P_calc, -Q_calc

# ==========================================
# 2. 辅助函数
# ==========================================
def denormalize(val, mean, std):
    return val * std + mean

# ==========================================
# 3. 主程序
# ==========================================
@torch.no_grad()
def main():
    args = argument_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # === 配置 ===
    SAMPLE_IDX = 0  
    RUN_ID = '20251209-4802' 
    # ===========

    # 1. 加载参数
    data_dir = args.data_dir
    data_param_path = os.path.join(data_dir, 'params', f'data_params_{RUN_ID}.pt')
    if not os.path.exists(data_param_path):
        logger.error(f"找不到参数文件: {data_param_path}")
        return

    data_param = torch.load(data_param_path, map_location='cpu')
    xymean = data_param['xymean'].to(device)
    xystd = data_param['xystd'].to(device)
    
    # 尝试加载边参数
    if 'edgemean' in data_param:
        edgemean = data_param['edgemean'].to(device)
        edgestd = data_param['edgestd'].to(device)
    else:
        logger.warning("未找到 edgemean，将使用默认值 (可能导致计算不准)")
        edgemean = torch.zeros(2).to(device)
        edgestd = torch.ones(2).to(device)

    # 2. 加载数据集
    logger.info(f"Loading Test Data from {args.case}...")
    testset = PowerFlowData(root=data_dir, case=args.case,
                            split=[.5, .2, .3], task='test',
                            xymean=xymean.cpu(), xystd=xystd.cpu())
    
    if SAMPLE_IDX >= len(testset): return
    data = testset[SAMPLE_IDX].to(device)
    
    # 3. 加载模型 & 预测
    node_in, node_out, edge_dim = testset.get_data_dimensions()
    model = MaskEmbdMultiMPN(
        nfeature_dim=node_in, efeature_dim=edge_dim, output_dim=node_out,
        hidden_dim=args.hidden_dim, n_gnn_layers=args.n_gnn_layers, K=args.K, dropout_rate=args.dropout_rate
    ).to(device)

    model, _ = load_model(model, RUN_ID, device)
    model.eval()

    logger.info(f"Running inference on Sample #{SAMPLE_IDX}...")
    out = model(data)

    # 4. 数据还原 (反归一化)
    pred_real = denormalize(out, xymean, xystd)
    truth_real = denormalize(data.y, xymean, xystd)

    # 在这一步使用模型算出了模型的预测值，
    
    # 5. 【核心步骤】手写物理计算
    logger.info("Executing Manual Physics Calculation (Using Predicted V & Theta)...")
    p_calc, q_calc = calculate_physics_manually(truth_real, data, edgemean, edgestd, device)
    
    # 6. 打印对比表格
    print_comparison(data, pred_real, truth_real, p_calc, q_calc)


def print_comparison(data, pred, truth, p_calc, q_calc):
    num_nodes = data.num_nodes
    bus_type = data.bus_type.cpu().numpy()
    
    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()
    p_calc = p_calc.cpu().numpy()
    q_calc = q_calc.cpu().numpy()
    
    print("\n" + "="*165)
    print(f"| {'Node':<4} | {'Type':<5} | {'Vm Pred':<8} {'Vm True':<8} | {'Va Pred':<8} {'Va True':<8} {'Va_err':<8} | {'P Pred':<9} {'P cal':<9} {'P True':<9} | {'Q Pred':<9} {'Q cal':<9} {'Q True':<9} |")
    print("-" * 165)

    sum_p_err = 0
    sum_q_err = 0

    for i in range(num_nodes):
        b_type = int(bus_type[i])
        type_str = "Slk" if b_type == 0 else ("PV " if b_type == 1 else "PQ ")
        vm_t = f"{truth[i,0]:.4f}"
        vm_s = f"{pred[i,0]:.4f}"

        va_t = f"{truth[i,1]:.4f}"
        va_s = f"{pred[i,1]:.2f}"
        va_err = f"{pred[i,1]-truth[i,1]:.2f}"
        
        # P
        p_p = f"{pred[i,2]:.4f}"
        p_c = f"{p_calc[i]:.4f}"
        p_t = f"{truth[i,2]:.4f}"
        
        # Q
        q_p = f"{pred[i,3]:.4f}"
        q_c = f"{q_calc[i]:.4f}"
        q_t = f"{truth[i,3]:.4f}"
        
        # 统计物理计算与真实值的误差
        sum_p_err += abs(p_calc[i] - truth[i,2])
        sum_q_err += abs(q_calc[i] - truth[i,3])

        prefix = " "
        if b_type == 0: prefix = ">>"
        if b_type == 1: prefix = "*" 
        
        print(f"{prefix}{i:<4} | {type_str:<5} | {vm_s:<8}  {vm_t:<8} | {va_s:<8}  {va_t:<8} {va_err:<8} | {p_p:<9} {p_c:<9} {p_t:<9} | {q_p:<9} {q_c:<9} {q_t:<9} |")

    print("="*165)
    print(f"[Manual Physics Error Summary]")
    print(f"Mean P Error (Calc vs True): {sum_p_err/num_nodes:.4f} p.u.")
    print(f"Mean Q Error (Calc vs True): {sum_q_err/num_nodes:.4f} p.u.")
    print(f"Note: If these errors are large, it confirms missing physical parameters (Tap/B) in dataset.")

    

if __name__ == "__main__":
    main()
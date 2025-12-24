import os
import torch
import numpy as np
import logging
from torch_geometric.loader import DataLoader
from datasets.PowerFlowData import PowerFlowData

# 引入你的模型定义
from networks.MPN import (
    MaskEmbdMultiMPN_Transformer,
    MaskEmbdMultiMPN_GPS,

)
from utils.evaluation import load_model
from utils.argument_parser import argument_parser

# ==========================================
# 物理计算核心 (修改版：增加返回 e, f)
# ==========================================
def calculate_physics_from_prediction(pred_ef, data, xymean, xystd, edgemean, edgestd, device):
    """
    输入:
        pred_ef: 模型预测的 [e, f] (归一化态)
        data: Batch 数据对象
    输出:
        P_calc, Q_calc: 模型预测电压推导出的功率
        P_target, Q_target: 真实功率
        e_real, f_real: 模型预测的电压 (物理值)
        e_true, f_true: 真实的电压 (物理值)
    """
    
    # --- 1. 反归一化 (还原物理值) ---
    
    # A. 预测电压 (e, f) [模型输出]
    ef_mean = xymean[:, 2:4].to(device)
    ef_std = xystd[:, 2:4].to(device)
    
    # 反归一化预测值
    real_ef_pred = pred_ef * (ef_std + 1e-7) + ef_mean
    e_pred, f_pred = real_ef_pred[:, 0], real_ef_pred[:, 1]
    
    # B. 真实电压 (e, f) [Ground Truth]
    # data.y 的后两列是 e, f (归一化态)
    target_ef_norm = data.y[:, 2:]
    real_ef_true = target_ef_norm * (ef_std + 1e-7) + ef_mean
    e_true, f_true = real_ef_true[:, 0], real_ef_true[:, 1]
    
    # C. 节点自导纳 (Gii, Bii)
    gb_mean = xymean[:, 4:6].to(device)
    gb_std = xystd[:, 4:6].to(device)
    node_gb_norm = data.x[:, 4:6]
    real_node_gb = node_gb_norm * (gb_std + 1e-7) + gb_mean
    g_ii, b_ii = real_node_gb[:, 0], real_node_gb[:, 1]
    
    # D. 边互导纳 (Gij, Bij)
    real_edge = data.edge_attr * (edgestd.to(device) + 1e-7) + edgemean.to(device)
    g_ij, b_ij = real_edge[:, 0], real_edge[:, 1]
    
    # E. 真实功率 Target (P, Q)
    pq_mean = xymean[:, :2].to(device)
    pq_std = xystd[:, :2].to(device)
    target_pq_norm = data.y[:, :2]
    real_pq_target = target_pq_norm * (pq_std + 1e-7) + pq_mean
    
    # --- 2. 物理计算 (基于预测电压 e_pred, f_pred) ---
    
    # A. 自项电流 I_self
    i_self_real = g_ii * e_pred - b_ii * f_pred
    i_self_imag = g_ii * f_pred + b_ii * e_pred
    
    # B. 邻居电流 I_neigh
    src, dst = data.edge_index
    e_j, f_j = e_pred[dst], f_pred[dst]
    
    i_msg_real = g_ij * e_j - b_ij * f_j
    i_msg_imag = g_ij * f_j + b_ij * e_j
    
    i_neigh_real = torch.zeros_like(e_pred)
    i_neigh_imag = torch.zeros_like(e_pred)
    
    i_neigh_real.index_add_(0, src, i_msg_real)
    i_neigh_imag.index_add_(0, src, i_msg_imag)
    
    # C. 总注入电流
    i_tot_real = i_self_real + i_neigh_real
    i_tot_imag = i_self_imag + i_neigh_imag
    
    # D. 计算功率
    p_calc = e_pred * i_tot_real + f_pred * i_tot_imag
    q_calc = f_pred * i_tot_real - e_pred * i_tot_imag
    
    # 返回所有需要对比的值
    return (p_calc, q_calc, 
            real_pq_target[:, 0], real_pq_target[:, 1], 
            e_pred, f_pred, 
            e_true, f_true)


@torch.no_grad()
def main():
    args = argument_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ================= 配置 =================
    run_id = '20251221-4629'  # 替换成你想测试的模型ID
    # =======================================

    print(f"Testing Model: {run_id}")
    
    # 1. 加载参数
    data_dir = args.data_dir
    data_param_path = os.path.join(data_dir, 'params', f'data_params_{run_id}.pt')
    if not os.path.exists(data_param_path):
        print("Data params not found.")
        return
    data_param = torch.load(data_param_path, map_location='cpu')
    xymean, xystd = data_param['xymean'], data_param['xystd']
    edgemean, edgestd = data_param['edgemean'], data_param['edgestd']
    
    # 2. 加载数据
    testset = PowerFlowData(root=data_dir, case=args.case,
                            split=[.5, .2, .3], task='test',
                            xymean=xymean, xystd=xystd,
                            edgemean=edgemean, edgestd=edgestd)
    # 这里的 batch_size 设为 1，方便逐条打印
    loader = DataLoader(testset, batch_size=1, shuffle=False)
    
    # 3. 加载模型
    model_cls = MaskEmbdMultiMPN_GPS # 请根据实际情况修改类名
    
    node_in, _, edge_dim = testset.get_data_dimensions()
    model = model_cls(
        nfeature_dim=node_in,
        efeature_dim=edge_dim,
        output_dim=2, 
        hidden_dim=args.hidden_dim,
        n_gnn_layers=args.n_gnn_layers,
        K=args.K,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    model, _ = load_model(model, run_id, device)
    model.eval()
    
    # 4. 运行验证
    # 打印宽一点的表头
    header = (f"| {'Node':<4} | {'Type':<4} | "
              f"{'e_True':<7} {'e_Mod':<7} {'err_e':<7}| {'f_True':<7} {'f_Mod':<7} {'err_f':<7}| "
              f"{'P_True':<7} {'P_Mod':<7} {'err_P':<7}| {'Q_True':<7} {'Q_Mod':<7} {'err_Q':<7}| "
            )
    
    print("\n" + "="*len(header))
    print(header)
    print("-" * len(header))

    # 用于累加 MAPE 的列表 (使用列表方便最后处理)
    mape_e, mape_f, mape_p, mape_q = [], [], [], []
    
    total_p_err = 0.
    total_q_err = 0.
    total_e_err = 0.
    total_f_err = 0.
    
    count = 0
    
    # 只看第一个样本
    for data in loader:
        data = data.to(device)
        out = model(data)

        
        # 解包所有返回数据
        p_c, q_c, p_t, q_t, e_c, f_c, e_t, f_t = calculate_physics_from_prediction(
            out, data, xymean, xystd, edgemean, edgestd, device
        )
        
        num_nodes = data.num_nodes
        bus_type = data.bus_type
        
        for i in range(num_nodes):
            bt = bus_type[i].item()
            type_str = "Slk" if bt==0 else ("PV" if bt==1 else "PQ")

            # --- 定义计算 MAPE 的小函数，防止除以 0 ---
            def get_ape(true, pred):
                t_val = true[i].item()
                p_val = pred[i].item()
                if abs(t_val) < 1e-2: return None # 忽略极小值点，避免 MAPE 误导
                return abs((t_val - p_val) / t_val) * 100
            
            # 计算各项误差
            err_p = abs(p_t[i].item() - p_c[i].item())
            err_q = abs(q_t[i].item() - q_c[i].item())
            err_e = abs(e_t[i].item() - e_c[i].item())
            err_f = abs(f_t[i].item() - f_c[i].item())

            ape_e = get_ape(e_t, e_c)
            ape_f = get_ape(f_t, f_c)
            ape_p = get_ape(p_t, p_c)
            ape_q = get_ape(q_t, q_c)

            # 收集有效数据点
            if ape_e is not None: mape_e.append(ape_e)
            if ape_f is not None: mape_f.append(ape_f)
            if ape_p is not None: mape_p.append(ape_p)
            if ape_q is not None: mape_q.append(ape_q)
            
            # 标记大误差 (物理不平衡 > 0.05)
            if (err_p/(p_t[i]+1e-5) > 0.05) and (err_q/(q_t[i]+1e-5) < 0.05):
                flag = "🔴"
            elif (err_q/(q_t[i]+1e-5) > 0.05) and (err_p/(p_t[i]+1e-5) < 0.05):
                flag = "🟢"
            elif (err_q/(q_t[i]+1e-5) > 0.05) and (err_p/(p_t[i]+1e-5) > 0.05):
                flag = "🔵"
            else :
                flag = ""
            
            
            # 格式化打印
            print(f"| {i:<4} | {type_str:<4} | "
                  f"{e_t[i]:<7.4f} {e_c[i]:<7.4f} {err_e:7.4f}| {f_t[i]:<7.4f} {f_c[i]:<7.4f} {err_f:7.4f}| "
                  f"{p_t[i]:<7.4f} {p_c[i]:<7.4f} {err_p:7.4f}| {q_t[i]:<7.4f} {q_c[i]:<7.4f} {err_q:7.4f}| "
                  f'{flag}'
                )
            
            total_p_err += err_p
            total_q_err += err_q
            total_e_err += err_e
            total_f_err += err_f
            count += 1
            
        break # 只看第一张图
    # 计算最终平均值
    def safe_mean(lst): return np.mean(lst) if len(lst) > 0 else 0.0


    print("-" * len(header))
    print(f"Mean e Error (Direct):  {total_e_err/count:.6f} p.u.")
    print(f"Mean f Error (Direct):  {total_f_err/count:.6f} p.u.")
    print(f"Mean P Error (Physics): {total_p_err/count:.6f} p.u.")
    print(f"Mean Q Error (Physics): {total_q_err/count:.6f} p.u.")
    print(f"MAPE e: {safe_mean(mape_e):.4f} %")
    print(f"MAPE f: {safe_mean(mape_f):.4f} %")
    print(f"MAPE P: {safe_mean(mape_p):.4f} %")
    print(f"MAPE Q: {safe_mean(mape_q):.4f} %")
    print("=" * len(header))



if __name__ == "__main__":
    main()
import torch
import numpy as np
import os
from datasets.PowerFlowData import PowerFlowData
from torch_geometric.loader import DataLoader

# ================= 配置区域 =================
# 这里填你生成数据时用的 case 名字 (例如 '118v2' 或 '14v2')
CASE_NAME = '118v1_n1' 
DATA_ROOT = 'data'
# ===========================================

def verify_pt_physics():
    print(f">>> 1. 加载处理后的数据集 (PowerFlowData) - Case: {CASE_NAME}...")
    
    # 实例化 Dataset
    # 这会自动查找 processed 文件夹下的 .pt 文件
    # 如果没找到，它会自动运行 process()
    dataset = PowerFlowData(
        root=DATA_ROOT, 
        case=CASE_NAME, 
        split=[.5, .2, .3], # Split 比例不影响物理验证，随便填
        task='train'
    )
    
    # 获取归一化参数
    print(">>> 2. 获取归一化参数 (Mean/Std)...")
    # xymean: [1, 6] -> [P, Q, e, f, Gii, Bii]
    xymean, xystd, edgemean, edgestd = dataset.get_data_means_stds()
    
    print(f"   Node Mean shape: {xymean.shape}")
    print(f"   Node Std shape:  {xystd.shape}")
    
    # 检查 e, f 的归一化参数是否符合预期 (0, 1)
    e_mean, f_mean = xymean[0, 2].item(), xymean[0, 3].item()
    e_std, f_std = xystd[0, 2].item(), xystd[0, 3].item()
    print(f"   [检查] e/f 归一化参数: Mean=({e_mean:.1f}, {f_mean:.1f}), Std=({e_std:.1f}, {f_std:.1f})")
    if e_std != 1.0 or f_std != 1.0:
        print("   ⚠️ 警告：e/f 似乎被归一化了 (Std != 1)。请确认这是你的预期。")
    else:
        print("   ✅ e/f 未归一化 (符合预期)。")

    # 取第一条数据进行验证
    data = dataset[0]
    num_nodes = data.num_nodes
    print(f"\n>>> 3. 取出第 0 条样本进行反归一化与物理回推...")
    
    # =======================================================
    # A. 反归一化 (Denormalization)
    # =======================================================
    
    # 1. 还原真实电压 [e, f] (来自 data.y)
    # data.y: [P, Q, e, f]
    # mean/std 对应索引 2, 3
    ef_norm = data.y[:, 2:]
    ef_mean = xymean[:, 2:4]
    ef_std = xystd[:, 2:4]
    
    real_ef = ef_norm * (ef_std + 1e-7) + ef_mean
    e = real_ef[:, 0]
    f = real_ef[:, 1]
    
    # 2. 还原真实目标功率 [P, Q] (来自 data.y)
    # mean/std 对应索引 0, 1
    pq_norm = data.y[:, :2]
    pq_mean = xymean[:, :2]
    pq_std = xystd[:, :2]
    
    real_pq_target = pq_norm * (pq_std + 1e-7) + pq_mean
    
    # 3. 还原节点自导纳 [Gii, Bii] (来自 data.x)
    # data.x: [P_in, Q_in, e_in, f_in, Gii, Bii]
    # Gii, Bii 在最后两列 (索引 4, 5)
    gb_node_norm = data.x[:, 4:]
    gb_node_mean = xymean[:, 4:]
    gb_node_std = xystd[:, 4:]
    
    real_gb_node = gb_node_norm * (gb_node_std + 1e-7) + gb_node_mean
    g_ii = real_gb_node[:, 0]
    b_ii = real_gb_node[:, 1]
    
    # 4. 还原边互导纳 [Gij, Bij] (来自 data.edge_attr)
    edge_norm = data.edge_attr
    real_edge = edge_norm * (edgestd + 1e-7) + edgemean
    
    # =======================================================
    # B. 物理计算 (Physics Calculation)
    # 公式: I_i = (Gii + jBii)*Vi + Sum( (Gij + jBij)*Vj )
    # =======================================================
    
    # 准备计算容器
    p_calc = torch.zeros(num_nodes)
    q_calc = torch.zeros(num_nodes)
    
    # 转换图结构以便遍历
    src = data.edge_index[0]
    dst = data.edge_index[1]
    g_ij = real_edge[:, 0]
    b_ij = real_edge[:, 1]
    
    # --- 计算 ---
    # 1. 自项 (Self Term)
    # I_self = (Gii + jBii) * (e + jf)
    i_real_self = g_ii * e - b_ii * f
    i_imag_self = g_ii * f + b_ii * e
    
    # 2. 邻居项 (Neighbor Term)
    # 使用 scatter_add 模拟 GNN 聚合 (或者简单循环)
    # I_neigh_real = Sum( Gij*ej - Bij*fj ) for j in neighbors
    i_real_neigh_msg = g_ij * e[dst] - b_ij * f[dst]
    i_imag_neigh_msg = g_ij * f[dst] + b_ij * e[dst]
    
    i_real_neigh = torch.zeros(num_nodes)
    i_imag_neigh = torch.zeros(num_nodes)
    
    # 将消息聚合到源节点 src
    i_real_neigh.index_add_(0, src, i_real_neigh_msg)
    i_imag_neigh.index_add_(0, src, i_imag_neigh_msg)
    
    # 3. 总注入电流
    i_total_real = i_real_self + i_real_neigh
    i_total_imag = i_imag_self + i_imag_neigh
    
    # 4. 计算功率 S = V * conj(I)
    # P = e*Ir + f*Ii
    # Q = f*Ir - e*Ii
    p_calc = e * i_total_real + f * i_total_imag
    q_calc = f * i_total_real - e * i_total_imag
    
    # =======================================================
    # C. 误差验证
    # =======================================================
    diff_p = real_pq_target[:, 0] - p_calc
    diff_q = real_pq_target[:, 1] - q_calc
    
    print("\n" + "="*80)
    print(f"{'Node':<5} | {'P_Target':<12} {'P_Calc':<12} | {'Q_Target':<12} {'Q_Calc':<12} | {'Error':<10}")
    print("-" * 80)
    
    total_err = diff_p.abs() + diff_q.abs()
    top_indices = torch.argsort(total_err, descending=True)[:10] # 看误差最大的10个
    
    for i in top_indices:
        i = i.item()
        pt = real_pq_target[i, 0].item()
        pc = p_calc[i].item()
        qt = real_pq_target[i, 1].item()
        qc = q_calc[i].item()
        err = total_err[i].item()
        
        print(f"{i:<5} | {pt:<12.6f} {pc:<12.6f} | {qt:<12.6f} {qc:<12.6f} | {err:.2e}")
        
    print("-" * 80)
    max_p_err = diff_p.abs().max().item()
    max_q_err = diff_q.abs().max().item()
    print(f"Max P Error: {max_p_err:.4e}")
    print(f"Max Q Error: {max_q_err:.4e}")
    
    if max_p_err < 1e-4 and max_q_err < 1e-4:
        print("\n✅ [完美通过] .pt 文件数据经过反归一化后，物理完全守恒！")
        print("   结论：Dataset 的处理逻辑和归一化参数完全正确。")
    else:
        print("\n❌ [验证失败] 物理不守恒。")
        print("   可能原因：")
        print("   1. PowerFlowData 中的归一化参数计算有误 (检查是否混入了 data.x)")
        print("   2. data.y 中的 P, Q 与 Ybus 定义的注入方向不一致")

if __name__ == "__main__":
    verify_pt_physics()
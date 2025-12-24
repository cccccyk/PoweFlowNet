import numpy as np
import os

# ================= 配置区域 =================
# 必须和你刚才生成的文件名一致
CASE_NAME = 'case118v_n1_train' 
DATA_DIR = 'data/raw'
SAMPLE_IDX = 1  # 检查第 0 个样本
# ===========================================

def verify_physics_from_npy():
    node_path = os.path.join(DATA_DIR, f"{CASE_NAME}_node_features.npy")
    edge_path = os.path.join(DATA_DIR, f"{CASE_NAME}_edge_features.npy")
    
    if not os.path.exists(node_path):
        print(f"❌ 找不到文件: {node_path}")
        return

    # 1. 加载数据
    print(f">>> 正在读取 {CASE_NAME} ...")
    
    # [关键修改] N-1 数据是变长的，不能用 mmap，必须 allow_pickle=True
    try:
        nodes_all = np.load(node_path, allow_pickle=True)
        edges_all = np.load(edge_path, allow_pickle=True)
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    # 获取单条样本
    nodes = nodes_all[SAMPLE_IDX] 
    edges = edges_all[SAMPLE_IDX]

    nodes = nodes.astype(float)
    edges = edges.astype(float)
    
    print(f"Sample {SAMPLE_IDX} | Nodes: {nodes.shape[0]} | Edges: {edges.shape[0]}")
    
    # -------------------------------------------------------
    # 2. 解析数据 (对应 extract_ybus_data 的存储格式)
    # -------------------------------------------------------
    # node_features: [Idx, Type, Vm, Va, P, Q, Gii, Bii]
    # 索引:            0     1    2   3   4  5   6    7
    
    vm = nodes[:, 2]
    va_deg = nodes[:, 3]
    p_target = nodes[:, 4]  # 这里的 P, Q 是 Pandapower 计算出的真值
    q_target = nodes[:, 5]
    
    # 提取自导纳 (Y_ii)
    g_ii = nodes[:, 6]
    b_ii = nodes[:, 7]
    
    # 转复数电压 V
    va_rad = np.deg2rad(va_deg)
    V = vm * np.exp(1j * va_rad)
    
    # -------------------------------------------------------
    # 3. 解析边数据
    # -------------------------------------------------------
    # edge_features: [From, To, Gij, Bij]
    src = edges[:, 0].astype(int)
    dst = edges[:, 1].astype(int)
    g_ij = edges[:, 2]
    b_ij = edges[:, 3]
    
    # -------------------------------------------------------
    # 4. 模拟 GNN 物理计算 (I = Y * V)
    # -------------------------------------------------------
    # 公式: I_i = Y_ii * V_i + Sum(Y_ij * V_j)
    
    # A. 自项电流 (Self Current)
    Y_ii = g_ii + 1j * b_ii
    I_self = Y_ii * V
    
    # B. 邻居电流 (Neighbor Current)
    I_neigh = np.zeros_like(I_self)
    
    # 模拟消息传递
    Y_ij = g_ij + 1j * b_ij
    
    # 遍历每条边，计算 j 对 i 的贡献
    for k in range(len(src)):
        s, d = src[k], dst[k]
        y = Y_ij[k]
        
        # 你的 extract_ybus_data 生成的是 COO 格式
        # 如果 Ybus 是对称的，通常会有 (i, j) 和 (j, i) 两条边
        # 所以我们只需要处理 s <- d 的贡献 (如果 edges 里包含了双向)
        # 或者处理 s <-> d (如果 edges 里只有单向)
        
        # Pandapower 的 tocoo() 会包含双向边 (除非它是上三角矩阵)
        # 我们假设 edges 里已经包含了所有非零元素
        
        I_neigh[s] += y * V[d]

    # C. 总注入电流
    I_inj = I_self + I_neigh
    
    # D. 计算复功率 S = V * conj(I)
    S_calc = V * np.conj(I_inj)
    P_calc = S_calc.real
    Q_calc = S_calc.imag
    
    # -------------------------------------------------------
    # 5. 误差对比
    # -------------------------------------------------------
    # Pandapower 的 P_target 定义可能与注入方向相反
    # 我们检查 sum 和 diff，看哪个接近 0
    
    diff_p = p_target - P_calc
    sum_p = p_target + P_calc
    
    # 自动检测符号方向
    if np.mean(np.abs(diff_p)) < np.mean(np.abs(sum_p)):
        final_p_err = diff_p
        p_sign = "Target - Calc"
    else:
        final_p_err = sum_p
        p_sign = "Target + Calc"

    diff_q = q_target - Q_calc
    sum_q = q_target + Q_calc
    
    if np.mean(np.abs(diff_q)) < np.mean(np.abs(sum_q)):
        final_q_err = diff_q
    else:
        final_q_err = sum_q

    # 打印结果
    print("\n" + "="*80)
    print(f"{'Node':<5} | {'P_Target':<12} {'P_Calc':<12} | {'Q_Target':<12} {'Q_Calc':<12} | {'Err_P':<10}")
    print("-" * 80)
    
    # 打印误差最大的几个点
    err_abs = np.abs(final_p_err) + np.abs(final_q_err)
    top_indices = np.argsort(err_abs)[-10:] # 看误差最大的10个
    
    for i in sorted(top_indices):
        print(f"{i:<5} | {p_target[i]:<12.6f} {P_calc[i]:<12.6f} | {q_target[i]:<12.6f} {Q_calc[i]:<12.6f} | {abs(final_p_err[i]):.2e}")

    print("-" * 80)
    print(f"检测到的符号关系: {p_sign}")
    print(f"Max P Error: {np.max(np.abs(final_p_err)):.4e}")
    print(f"Max Q Error: {np.max(np.abs(final_q_err)):.4e}")
    
    if np.max(np.abs(final_p_err)) < 1e-5:
        print("\n✅ 完美匹配！数据中的 Ybus 信息足以精确反推 PQ。")
    else:
        print("\n❌ 匹配失败。可能原因：Ybus提取不完整 或 边方向覆盖不全。")

if __name__ == "__main__":
    verify_physics_from_npy()
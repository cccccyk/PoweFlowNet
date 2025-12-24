import numpy as np
import os
import time

# ================= 配置区域 =================
CASE_NAME = 'case118v_n1_train' 
DATA_DIR = 'data/raw'
CHECK_COUNT = -1  # 检查的样本数量（设为 -1 则检查所有数据）
PHYSICS_TOLERANCE = 1e-4  # 物理误差容忍度 (通常 1e-5 ~ 1e-3 都是合理的)
# ===========================================

def validate_dataset_full():
    node_path = os.path.join(DATA_DIR, f"{CASE_NAME}_node_features.npy")
    edge_path = os.path.join(DATA_DIR, f"{CASE_NAME}_edge_features.npy")
    label_path = os.path.join(DATA_DIR, f"{CASE_NAME}_labels.npy")
    
    print(f"Loading data from {DATA_DIR} ...")
    if not os.path.exists(node_path):
        print("❌ 文件不存在！请检查路径。")
        return

    # 加载数据
    try:
        nodes_all = np.load(node_path, allow_pickle=True)
        edges_all = np.load(edge_path, allow_pickle=True)
        labels_all = np.load(label_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    total_samples = len(nodes_all)
    print(f"✅ 数据加载成功! 总样本数: {total_samples}")
    
    # 确定检查范围
    num_to_check = total_samples if CHECK_COUNT == -1 else min(total_samples, CHECK_COUNT)
    indices = np.arange(total_samples)
    if num_to_check < total_samples:
        np.random.seed(42)
        indices = np.random.choice(indices, num_to_check, replace=False)
    
    print(f"🚀 开始执行深度检查 (抽样: {num_to_check} 条)...")
    
    # 统计变量
    stats = {
        'nan_found': 0,
        'inf_found': 0,
        'physics_fail': 0,
        'max_p_error': 0.0,
        'max_q_error': 0.0,
        'edge_counts': [],
        'label_dist': {}
    }

    start_time = time.time()

    for idx in indices:
        # 1. 提取单条数据
        nodes = nodes_all[idx].astype(float)
        edges = edges_all[idx].astype(float)
        label = labels_all[idx]

        # =========================================
        # check A: 数值稳定性 (NaN / Inf)
        # =========================================
        if np.isnan(nodes).any() or np.isnan(edges).any():
            stats['nan_found'] += 1
            continue # 如果有 NaN，后续计算无意义，跳过
            
        if np.isinf(nodes).any() or np.isinf(edges).any():
            stats['inf_found'] += 1
            continue

        # =========================================
        # Check B: 物理一致性 (P_calc vs P_target)
        # =========================================
        # node_features: [Idx, Type, Vm, Va, P, Q, Gii, Bii]
        vm = nodes[:, 2]
        va_rad = np.deg2rad(nodes[:, 3])
        p_target = nodes[:, 4]
        q_target = nodes[:, 5]
        g_ii = nodes[:, 6]
        b_ii = nodes[:, 7]
        
        # 复电压
        V = vm * np.exp(1j * va_rad)
        
        # 自导纳电流
        I_inj = (g_ii + 1j * b_ii) * V
        
        # 邻居电流注入 (利用 numpy 高级索引加速，不写循环)
        # edges: [Src, Dst, Gij, Bij]
        src = edges[:, 0].astype(int)
        dst = edges[:, 1].astype(int)
        y_ij = edges[:, 2] + 1j * edges[:, 3]
        
        # 相当于 for k: I[src[k]] += y_ij[k] * V[dst[k]]
        # np.add.at 是处理这种稀疏累加的神器
        np.add.at(I_inj, src, y_ij * V[dst])
        
        # 计算复功率 S = V * conj(I)
        S_calc = V * np.conj(I_inj)
        P_calc = S_calc.real
        Q_calc = S_calc.imag
        
        # 误差计算 (取绝对误差)
        # 注意：这里假设生成器里 P_target 的符号和计算的一致
        # 如果生成器里 P 是注入为正，这里计算也是注入为正
        err_p = np.abs(p_target - P_calc)
        err_q = np.abs(q_target - Q_calc)
        
        # 更新最大误差统计
        current_max_p = np.max(err_p)
        current_max_q = np.max(err_q)
        
        stats['max_p_error'] = max(stats['max_p_error'], current_max_p)
        stats['max_q_error'] = max(stats['max_q_error'], current_max_q)
        
        if current_max_p > PHYSICS_TOLERANCE or current_max_q > PHYSICS_TOLERANCE:
            stats['physics_fail'] += 1

        # =========================================
        # Check C: 拓扑统计 (N-0 vs N-1)
        # =========================================
        num_edges = edges.shape[0]
        stats['edge_counts'].append(num_edges)
        
        # 统计 Label
        l_val = int(label)
        stats['label_dist'][l_val] = stats['label_dist'].get(l_val, 0) + 1

    end_time = time.time()
    
    # =========================================
    # 打印最终报告
    # =========================================
    print("\n" + "="*40)
    print("📊 数据集体检报告")
    print("="*40)
    print(f"检查样本数: {num_to_check}")
    print(f"耗时: {end_time - start_time:.2f} 秒")
    print("-" * 40)
    
    print(f"1. [数值完整性]")
    if stats['nan_found'] == 0 and stats['inf_found'] == 0:
        print(f"   ✅ 通过: 未发现 NaN 或 Inf。")
    else:
        print(f"   ❌ 失败: 发现 {stats['nan_found']} 个样本含 NaN，{stats['inf_found']} 个样本含 Inf。")
        
    print("-" * 40)
    print(f"2. [物理一致性] (容差 Threshold={PHYSICS_TOLERANCE})")
    print(f"   最大 P 误差: {stats['max_p_error']:.6f}")
    print(f"   最大 Q 误差: {stats['max_q_error']:.6f}")
    if stats['physics_fail'] == 0:
        print(f"   ✅ 完美: 所有检查样本均满足物理约束。")
    else:
        fail_rate = stats['physics_fail'] / num_to_check * 100
        print(f"   ⚠️ 警告: 有 {stats['physics_fail']} 个样本 ({fail_rate:.2f}%) 超出物理误差容差。")
        print("      (如果误差很小如 1e-3 级别，通常可接受，可能是潮流计算精度问题)")

    print("-" * 40)
    print(f"3. [拓扑多样性 N-0/N-1]")
    unique_counts = np.unique(stats['edge_counts'])
    print(f"   边数量分布 (Edge Counts): {unique_counts}")
    if len(unique_counts) > 1:
        print(f"   ✅ 通过: 检测到多种拓扑结构 (N-0 和 N-1 混合)。")
        print(f"      最少边数: {min(unique_counts)}, 最多边数: {max(unique_counts)}")
    else:
        print(f"   ⚠️ 警告: 所有样本边数相同，可能未成功生成 N-1 样本。")
        
    print("-" * 40)
    print(f"4. [标签分布]")
    for k, v in stats['label_dist'].items():
        print(f"   Label {k}: {v} 条 ({v/num_to_check*100:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    validate_dataset_full()
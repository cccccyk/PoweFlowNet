"""
N-1 Training Data Generator (Style Matched with dataset_generator_GB.py)
包含 N-1 故障及越限（不安全）场景的训练数据生成。
"""
import argparse
import pandas as pd
import pandapower as pp
import numpy as np
import multiprocessing as mp
import os
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makeYbus import makeYbus

# =========================================================
# 核心函数复用
# =========================================================
def extract_ybus_data(net):
    """
    提取 Ybus 特征 (Gij, Bij, Gii, Bii)
    return:
    edge_features:[From,To,G_ij,B_ij]
    node_g 节点自身的导纳实部
    node_b 节点自身的导纳虚部
    """
    # 1. 转换为 PYPOWER 格式
    ppc, ppci = _pd2ppc(net)
    baseMVA, bus, branch = ppc["baseMVA"], ppc["bus"], ppc["branch"]
    
    # 2. 生成 Ybus
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    
    # 3. 转换为 COO 格式
    y_coo = Ybus.tocoo()
    row = y_coo.row
    col = y_coo.col
    data = y_coo.data
    
    # 提取边特征 (非对角)
    mask_edge = (row != col)
    src = row[mask_edge]
    dst = col[mask_edge]
    y_edge = data[mask_edge]
    
    edge_features = np.zeros((len(src), 4))
    edge_features[:, 0] = src
    edge_features[:, 1] = dst
    edge_features[:, 2] = y_edge.real
    edge_features[:, 3] = y_edge.imag
    
    # 提取节点自导纳 (对角)
    n_nodes = len(net.bus)
    y_diag_g = np.zeros(n_nodes)
    y_diag_b = np.zeros(n_nodes)
    
    mask_diag = (row == col)
    if np.any(mask_diag):
        y_diag_g[row[mask_diag]] = data[mask_diag].real
        y_diag_b[row[mask_diag]] = data[mask_diag].imag
    
    return edge_features, y_diag_g, y_diag_b

def calculate_net_injection(net):
    """
    计算净注入功率 P, Q (Gen + Ext - Load)
    """
    n_nodes = len(net.bus)
    p_inj = np.zeros(n_nodes)
    q_inj = np.zeros(n_nodes)
    
    # 1. 减去负载
    if not net.load.empty:
        load_bus = net.load['bus'].values.astype(int)
        np.add.at(p_inj, load_bus, -net.load['p_mw'].values)
        np.add.at(q_inj, load_bus, -net.load['q_mvar'].values)
        
    # 2. 加上发电机
    if not net.res_gen.empty:
        gen_bus = net.gen['bus'].values.astype(int)
        np.add.at(p_inj, gen_bus, net.res_gen['p_mw'].values)
        np.add.at(q_inj, gen_bus, net.res_gen['q_mvar'].values)
        
    # 3. 加上外部电网
    if not net.res_ext_grid.empty:
        ext_bus = net.ext_grid['bus'].values.astype(int)
        np.add.at(p_inj, ext_bus, net.res_ext_grid['p_mw'].values)
        np.add.at(q_inj, ext_bus, net.res_ext_grid['q_mvar'].values)
        
    # 4. 转标幺值
    return p_inj / net.sn_mva, q_inj / net.sn_mva

# =========================================================
# N-1 生成逻辑 (主函数) - 修改版
# =========================================================
def generate_n1_data(sublist_size, rng, base_net_create):
    edge_features_list = []
    node_features_list = []
    labels_list = [] 
    
    # 统计计数器 (新增 n0_count, n1_count)
    stats = {'safe': 0, 'unsafe': 0, 'failed': 0, 'nan_skip': 0,'n0_count': 0, 'n1_count': 0}
    N1_RATIO = 0.4

    while len(edge_features_list) < sublist_size:
        net = base_net_create()
        net.bus['name'] = net.bus.index


        if 'va_degree' in net.ext_grid.columns: 
            net.ext_grid['va_degree'] = 0.0
            net.ext_grid['vm_pu'] = 1.00

        # 1. 负荷缩放: 0.2 ～ 1.0 (鼓励轻载以激发过电压)
        load_scale = rng.uniform(0.3, 1.0, size=len(net.load))
        net.load['p_mw'] *= load_scale
        net.load['q_mvar'] *= load_scale

        # 2. 发电机有功缩放: 0.8 ～ 1.3 (支持重载，减少低压崩溃)
        gen_p_scale = rng.uniform(0.75, 1.25, size=len(net.gen))
        net.gen['p_mw'] *= gen_p_scale

        # 3. 发电机 PV 电压设定: 0.95 ～ 1.06 p.u. (直接赋值，非缩放!)
        net.gen['vm_pu'] = rng.uniform(0.95, 1.05, size=len(net.gen))

        # 4. 外部电网 (Slack) 保持固定
        net.ext_grid['vm_pu'] = 1.0
        net.ext_grid['va_degree'] = 0.0

        is_n1_scenario = rng.random() < N1_RATIO
        
        if is_n1_scenario:
            # === 执行 N-1 ===
            line_indices = net.line.index.tolist()
            if len(line_indices) > 0:
                drop_line = rng.choice(line_indices)
                net.line.at[drop_line, 'in_service'] = False
        else:
            # === 保持 N-0 ===
            # 不做任何断线操作，保留完整拓扑
            pass

        # ---------------------------------------------------
        # 3. 运行潮流
        # ---------------------------------------------------
        try:
            pp.runpp(net, algorithm='nr', init="flat", numba=False)
        except:
            stats['failed'] += 1
            continue

        if not net.converged:
            stats['failed'] += 1
            continue

        # ---------------------------------------------------
        # 4. 统计安全性 (生成 Label) —— 仅 PQ 节点参与高压判断
        # ---------------------------------------------------
        max_load = net.res_line['loading_percent'].max()
        min_vm = net.res_bus['vm_pu'].min()  # 低压：全网最小（合理）

        # ✅ 高压：仅 PQ 节点
        pv_buses = set(net.gen['bus'].tolist()) | set(net.ext_grid['bus'].tolist())
        pq_mask = ~net.res_bus.index.isin(pv_buses)
        pq_vms = net.res_bus.loc[pq_mask, 'vm_pu']
        max_pq_vm = pq_vms.max() if len(pq_vms) > 0 else -np.inf

        # 判断越限
        is_v_bad = (min_vm < 0.95) or (max_pq_vm > 1.05)
        is_l_bad = (max_load > 80.0)
        is_unsafe = is_v_bad or (max_load > 100.0)

        # 生成标签
        status = 0
        if is_v_bad:
            status = 1
        if is_l_bad:
            status = 3 if status == 1 else 2

        # ---------------------------------------------------
        # 5. 提取特征 & NaN 检查
        # ---------------------------------------------------
        try:
            edge_features, node_g, node_b = extract_ybus_data(net)
            p_true, q_true = calculate_net_injection(net)

            # 构建节点特征
            n = len(net.bus)
            types = np.ones(n) * 2
            for j in range(net.gen.shape[0]): 
                idx = np.where(net.gen['bus'].values[j] == net.bus['name'])[0][0]
                types[idx] = 1 
            for j in range(net.ext_grid.shape[0]):
                idx = np.where(net.ext_grid['bus'].values[j] == net.bus['name'])[0][0]
                types[idx] = 0

            base_node_features = np.zeros((n, 6))
            base_node_features[:, 0] = net.bus.index
            base_node_features[:, 1] = types
            base_node_features[:, 2] = net.res_bus['vm_pu']
            base_node_features[:, 3] = net.res_bus['va_degree']
            base_node_features[:, 4] = p_true
            base_node_features[:, 5] = q_true
            
            final_node_features = np.concatenate([
                base_node_features, 
                node_g.reshape(-1, 1), 
                node_b.reshape(-1, 1)
            ], axis=1)

            # === NaN / Inf 检查 ===
            if np.isnan(final_node_features).any() or np.isinf(final_node_features).any():
                stats['nan_skip'] += 1
                continue
            if np.isnan(edge_features).any() or np.isinf(edge_features).any():
                stats['nan_skip'] += 1
                continue

            # === 数据有效，进行保存统计 ===
            if is_n1_scenario:
                stats['n1_count'] += 1
            else:
                stats['n0_count'] += 1

            if is_unsafe: stats['unsafe'] += 1
            else: stats['safe'] += 1

            edge_features_list.append(edge_features)
            node_features_list.append(final_node_features)
            labels_list.append(status)

        except Exception as e:
            stats['failed'] += 1
            continue

        # 打印日志
        if len(edge_features_list) % 500 == 0:
            total = stats['safe'] + stats['unsafe'] + 1e-6
            unsafe_ratio = stats['unsafe'] / total
            n1_ratio_actual = stats['n1_count'] / total
            print(f'[PID {os.getpid()}] Gen {len(edge_features_list)} | '
                  f'Unsafe: {unsafe_ratio:.1%} | '
                  f'N-1 Ratio: {n1_ratio_actual:.1%} | ' # 显示实际生成的 N-1 比例
                  f'NaN Skip: {stats["nan_skip"]}')
            
    return edge_features_list, node_features_list, labels_list

def generate_parallel(num_samples, num_proc, base_net_create):
    sublist_size = num_samples // num_proc
    parent_rng = np.random.default_rng(42)
    streams = parent_rng.spawn(num_proc)
    pool = mp.Pool(processes=num_proc)
    args = [[sublist_size, st, base_net_create] for st in streams]
    
    results = pool.starmap(generate_n1_data, args)
    pool.close()
    pool.join()
    
    ef_list, nf_list, lbl_list = [], [], []
    for res in results:
        ef_list += res[0]
        nf_list += res[1]
        lbl_list += res[2]
    return ef_list, nf_list, lbl_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='118')
    parser.add_argument('--samples', type=int, default=300000)
    args = parser.parse_args()

    if args.case == '14': base_net = pp.networks.case14
    elif args.case == '118': base_net = pp.networks.case118
    elif args.case == '300': base_net = pp.networks.case300
    else: raise ValueError("Unknown case")
        
    # 定义文件名
    case_name = f'case{args.case}v2_30w_n1'
    print(f"Generating N-1 Training Data for {case_name}...")
    
    edges, nodes, labels = generate_parallel(args.samples, 30, base_net)
    
    os.makedirs("./data/raw", exist_ok=True)
    # 使用 object 类型保存 (因为N-1导致边数不固定)
    np.save(f"./data/raw/{case_name}_edge_features.npy", np.array(edges, dtype=object))
    np.save(f"./data/raw/{case_name}_node_features.npy", np.array(nodes, dtype=object))
    
    # [新增] 保存 Labels，用于后续统计或分类任务
    np.save(f"./data/raw/{case_name}_labels.npy", np.array(labels))
    
    print("Done!")
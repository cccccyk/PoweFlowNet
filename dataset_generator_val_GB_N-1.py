# 这是N-1的数据生成的代码，有包含不安全的数据，还没有做相应的检测

import pandapower as pp
import pandapower.networks as pn
import numpy as np
import os
import pandas as pd
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makeYbus import makeYbus

# 提取Ybus的函数
def extract_ybus_data(net):
    ppc, ppci = _pd2ppc(net)
    baseMVA, bus, branch = ppc["baseMVA"], ppc["bus"], ppc["branch"]
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    y_coo = Ybus.tocoo()
    row, col, data = y_coo.row, y_coo.col, y_coo.data
    
    # 边特征
    mask_edge = (row != col)
    src, dst = row[mask_edge], col[mask_edge]
    y_edge = data[mask_edge]
    edge_features = np.stack([src, dst, y_edge.real, y_edge.imag], axis=1)
    
    # 节点自导纳
    n_nodes = len(net.bus)
    y_diag_g = np.zeros(n_nodes)
    y_diag_b = np.zeros(n_nodes)
    mask_diag = (row == col)
    y_diag_g[row[mask_diag]] = data[mask_diag].real
    y_diag_b[row[mask_diag]] = data[mask_diag].imag
    
    return edge_features, y_diag_g, y_diag_b

def calculate_net_injection(net):
    n_nodes = len(net.bus)
    p_inj = np.zeros(n_nodes)
    q_inj = np.zeros(n_nodes)
    
    # 负载 (减)
    if not net.load.empty:
        np.add.at(p_inj, net.load['bus'].values.astype(int), -net.load['p_mw'].values)
        np.add.at(q_inj, net.load['bus'].values.astype(int), -net.load['q_mvar'].values)
    # 发电 (加)
    if not net.res_gen.empty:
        np.add.at(p_inj, net.gen['bus'].values.astype(int), net.res_gen['p_mw'].values)
        np.add.at(q_inj, net.gen['bus'].values.astype(int), net.res_gen['q_mvar'].values)
    # 外网 (加)
    if not net.res_ext_grid.empty:
        np.add.at(p_inj, net.ext_grid['bus'].values.astype(int), net.res_ext_grid['p_mw'].values)
        np.add.at(q_inj, net.ext_grid['bus'].values.astype(int), net.res_ext_grid['q_mvar'].values)
        
    return p_inj / net.sn_mva, q_inj / net.sn_mva

# =========================================================
# 主生成逻辑
# =========================================================
def generate_n1_dataset():
    OUTPUT_DIR = "data/n1_eval"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载基准网络
    net_base = pn.case118()
    net_base.bus['name'] = net_base.bus.index
    
    # [重要] 保存线路电流限值 (kA) 到静态文件
    # 用于后续 GNN 预测出电流后计算负载率
    # 注意：如果数据是 p.u.，我们需要知道 Base Current。但 Pandapower 通常直接给 kA 限值。
    # 这里我们存下 max_i_ka
    line_limits = net_base.line['max_i_ka'].values
    np.save(f"{OUTPUT_DIR}/static_line_limits.npy", line_limits)
    print(f"已保存线路限值，共 {len(line_limits)} 条线路。")

    # 2. 配置生成场景
    # 我们生成 5 组基准工况 (Base Cases)，每组都跑完所有线路的 N-1
    # 场景设计：[正常, 重载, 重载+低压, 轻载+高压, 随机]
    SCENARIOS = [
        {'load_scale': 1.0, 'gen_v_range': (0.98, 1.02)}, # 正常
        {'load_scale': 1.3, 'gen_v_range': (0.98, 1.02)}, # 重载 (易过载)
        {'load_scale': 1.4, 'gen_v_range': (0.94, 0.98)}, # 重载+低压 (易电压越下限)
        {'load_scale': 0.6, 'gen_v_range': (1.04, 1.08)}, # 轻载+高压 (易电压越上限)
        {'load_scale': 1.1, 'gen_v_range': (0.95, 1.05)}, # 随机
    ]
    
    line_indices = net_base.line.index.tolist()
    
    edge_list = []
    node_list = []
    label_list = [] # [Scenario_ID, Line_ID, Status, Max_Load%, Min_Vm, Max_Vm]
    
    total_count = 0
    
    print(f"开始生成 N-1 数据，共 {len(SCENARIOS)} 组基准工况 x {len(line_indices)} 条线路...")

    for scen_idx, scen in enumerate(SCENARIOS):
        # 初始化基准工况
        net_scen = net_base.deepcopy()
        
        # A. 调整负荷
        net_scen.load['p_mw'] *= scen['load_scale']
        net_scen.load['q_mvar'] *= scen['load_scale']
        
        # B. 调整发电机电压设定值 (PV)
        v_low, v_high = scen['gen_v_range']
        # 随机设定每个发电机的电压
        net_scen.gen['vm_pu'] = np.random.uniform(v_low, v_high, size=len(net_scen.gen))
        # Slack 节点也调整
        net_scen.ext_grid['vm_pu'] = np.random.uniform(v_low, v_high, size=len(net_scen.ext_grid))
        
        # C. 遍历 N-1
        for line_idx in line_indices:
            net = net_scen.deepcopy()
            net.line.at[line_idx, 'in_service'] = False
            
            # D. 运行潮流
            try:
                pp.runpp(net, algorithm='nr', init="flat")
                converged = True
            except:
                converged = False
            
            if not net.converged:
                converged = False
            
            # E. 提取标签 (Labels)
            status = 0 # 0:Safe
            max_load_pct = 0.0
            min_vm = 0.0
            max_vm = 0.0
            
            if not converged:
                status = 4 # 4:Collapsed (最危险)
                # 不收敛时无法提取特征，跳过 feature append，只记录 label 或者直接丢弃
                # 为了 GNN 测试方便，通常直接丢弃不收敛样本，或者用上一时刻数据填充
                # 这里我们选择【跳过不保存】，只打印日志
                # print(f"Scen {scen_idx} Line {line_idx}: Not Converged")
                continue 
            else:
                # 提取结果
                max_load_pct = net.res_line['loading_percent'].max()
                min_vm = net.res_bus['vm_pu'].min()
                max_vm = net.res_bus['vm_pu'].max()
                
                # 判定状态
                is_overload = max_load_pct > 100.0
                is_voltage_bad = (min_vm < 0.95) or (max_vm > 1.05)
                
                if is_overload and is_voltage_bad: status = 3
                elif is_overload: status = 2
                elif is_voltage_bad: status = 1
                else: status = 0 # Safe
                
                # F. 提取特征 (仅当收敛时)
                edge_feat, node_g, node_b = extract_ybus_data(net)
                p_true, q_true = calculate_net_injection(net)
                
                # 构建 Node Features [Index, Type, Vm, Va, P, Q] + [Gii, Bii]
                n = len(net.bus)
                # ... (此处省略 types 的构建代码，复用之前的) ...
                types = np.ones(n) * 2
                for j in range(net.gen.shape[0]): types[np.where(net.gen['bus'].values[j] == net.bus['name'])[0][0]] = 1
                for j in range(net.ext_grid.shape[0]): types[np.where(net.ext_grid['bus'].values[j] == net.bus['name'])[0][0]] = 0

                base_node = np.zeros((n, 6))
                base_node[:, 0] = net.bus.index
                base_node[:, 1] = types
                base_node[:, 2] = net.res_bus['vm_pu']
                base_node[:, 3] = net.res_bus['va_degree']
                base_node[:, 4] = p_true
                base_node[:, 5] = q_true
                
                final_node = np.concatenate([base_node, node_g.reshape(-1,1), node_b.reshape(-1,1)], axis=1)
                
                edge_list.append(edge_feat)
                node_list.append(final_node)
                # 记录标签: [Line_ID_Broken, Status, Max_Load, Min_Vm, Max_Vm]
                label_list.append([line_idx, status, max_load_pct, min_vm, max_vm])
                
                total_count += 1

    # G. 保存
    # 使用 object 数组，因为 N-1 导致边数变化
    np.save(f"{OUTPUT_DIR}/n1_edge_features.npy", np.array(edge_list, dtype=object))
    np.save(f"{OUTPUT_DIR}/n1_node_features.npy", np.array(node_list, dtype=object))
    np.save(f"{OUTPUT_DIR}/n1_labels.npy", np.array(label_list))
    
    print(f"\n>>> 生成完毕！共生成 {total_count} 个有效收敛样本。")
    print(f"    不安全样本分布: 需后续统计 (Status > 0)")

if __name__ == "__main__":
    generate_n1_dataset()
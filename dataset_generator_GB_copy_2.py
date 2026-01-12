"""
生成使用Ybus为信息的数据。这份代码有Yii,1220版本
在此基础上，对比N-0情况下，安全和不安全的数据对于结果的影响
"""
import time
import argparse
import pandas as pd
import pandapower as pp
import numpy as np
import networkx as nx
import multiprocessing as mp
import os

# 导入需要的包
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makeYbus import makeYbus
from scipy.sparse import coo_matrix

from utils.data_utils import perturb_topology

# 生成30000个数据，开10个进程
number_of_samples = 30000
number_of_processes = 10
ENFORCE_Q_LIMS = False

def create_case3():
    net = pp.create_empty_network()
    net.sn_mva = 100
    b0 = pp.create_bus(net, vn_kv=345., name='bus 0')
    b1 = pp.create_bus(net, vn_kv=345., name='bus 1')
    b2 = pp.create_bus(net, vn_kv=345., name='bus 2')
    pp.create_ext_grid(net, bus=b0, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=b2, p_mw=10.3, q_mvar=3, name="Load")
    # pp.create_gen(net, bus=b1, p_mw=0.5, vm_pu=1.03, name="Gen", max_p_mw=1)
    pp.create_line(net, from_bus=b0, to_bus=b1, length_km=10, name='line 01', std_type='NAYY 4x50 SE')
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=5, name='line 01', std_type='NAYY 4x50 SE')
    pp.create_line(net, from_bus=b2, to_bus=b0, length_km=20, name='line 01', std_type='NAYY 4x50 SE')
    
    net.line['c_nf_per_km'] = pd.Series(0., index=net.line['c_nf_per_km'].index, name=net.line['c_nf_per_km'].name)
    
    return net

def calculate_net_injection(net):
    """
    计算每个节点的净注入功率 (Gen + Ext_Grid - Load)
    注意：这是为了匹配 Ybus 的计算结果 (流出功率)
    """
    n_nodes = len(net.bus)
    p_inj = np.zeros(n_nodes)
    q_inj = np.zeros(n_nodes)
    
    # 1. 减去负载 (Load)
    # Pandapower 中 load['p_mw'] > 0 表示消耗
    if not net.load.empty:
        # 这种写法比 iterrows 快很多
        load_bus = net.load['bus'].values.astype(int)
        p_load = net.load['p_mw'].values
        q_load = net.load['q_mvar'].values
        # 累加 (处理多个负载接在同一个节点的情况)
        np.add.at(p_inj, load_bus, -p_load)
        np.add.at(q_inj, load_bus, -q_load)
        
    # 2. 加上发电机 (Gen) - 使用 res_gen (计算结果)
    # 因为 PV 节点的 Q 是算出来的，必须用 res_gen
    if not net.res_gen.empty:
        gen_bus = net.gen['bus'].values.astype(int)
        p_gen = net.res_gen['p_mw'].values
        q_gen = net.res_gen['q_mvar'].values
        np.add.at(p_inj, gen_bus, p_gen)
        np.add.at(q_inj, gen_bus, q_gen)
        
    # 3. 加上外部电网 (Ext_Grid) - 使用 res_ext_grid
    if not net.res_ext_grid.empty:
        ext_bus = net.ext_grid['bus'].values.astype(int)
        p_ext = net.res_ext_grid['p_mw'].values
        q_ext = net.res_ext_grid['q_mvar'].values
        np.add.at(p_inj, ext_bus, p_ext)
        np.add.at(q_inj, ext_bus, q_ext)
        
    # 4. 转换为标幺值 (p.u.)
    p_pu = p_inj / net.sn_mva
    q_pu = q_inj / net.sn_mva
    
    return p_pu, q_pu
    
# 这个函数用于从net中提取相应的Ybus的信息
def extract_ybus_data(net):
    '''
    return:
    edge_features:[From,To,G_ij,B_ij]
    node_g 节点自身的导纳实部
    node_b 节点自身的导纳虚部
    '''

    # 1. 转换为 PYPOWER 格式 (这是 Pandapower 计算的核心结构)
    ppc, ppci = _pd2ppc(net)
    baseMVA, bus, branch = ppc["baseMVA"], ppc["bus"], ppc["branch"]

    # 2. 生成 Ybus (稀疏复数矩阵)
    # makeYbus 会自动处理变压器变比、线路充电电容、并联补偿等所有物理细节
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    
    # print(f"Ybus = {Ybus}") # 这个可以作为相应的检查的点

    # 3. 转换为 COO 格式以便提取数据
    y_coo = Ybus.tocoo()
    row = y_coo.row
    col = y_coo.col
    data = y_coo.data

    # 提取边特征，非对角元素，Yij
    mask_edge = (row != col)

    src = row[mask_edge]    # 即from
    dst = col[mask_edge]    # 即to
    y_edge = data[mask_edge]

    # 用边特征构建edge_feature
    edge_features = np.zeros((len(src),4))
    edge_features[:,0] = src
    edge_features[:,1] = dst
    edge_features[:,2] = y_edge.real
    edge_features[:,3] = y_edge.imag

    # 提取节点自身导纳，Yii
    n_nodes = len(net.bus)
    y_diag_g = np.zeros(n_nodes)
    y_diag_b = np.zeros(n_nodes)

    mask_diag = (row==col)
    diag_nodes = row[mask_diag]
    diag_vals = data[mask_diag]

    y_diag_g[diag_nodes] = diag_vals.real
    y_diag_b[diag_nodes] = diag_vals.imag

    # 最后返回边特征（构建好），节点导纳，分开
    return edge_features , y_diag_g , y_diag_b
    

# 这一段是生成数据的代码
def generate_data(sublist_size, rng, base_net_create, num_lines_to_remove=0, num_lines_to_add=0):
    # 先准备好空的列表
    edge_features_list = []
    node_features_list = []
    # graph_feature_list = []

    while len(edge_features_list) < sublist_size:

        net = base_net_create()
        
        # 这一段是有关拓扑扰动的，由于我们的num_lines_to_remove=0, num_lines_to_add=0，所以并无影响
        success_flag, net = perturb_topology(net, num_lines_to_remove=num_lines_to_remove, num_lines_to_add=num_lines_to_add) # TODO 
        if success_flag == 1:
            exit()
        
        n = net.bus.values.shape[0]
        
        net.bus['name'] = net.bus.index
        scenario = rng.choice(['normal','heavy_overload', 'high_voltage', 'low_voltage'], p=[0.3, 0.25, 0.25, 0.2])

        # 强制 Slack 相角为 0 方便模型学习
        if 'va_degree' in net.ext_grid.columns:
            net.ext_grid['va_degree'] = 0.0

        # 获取参数
        Pg = net.gen['p_mw'].values
        Pd = net.load['p_mw'].values
        Qd = net.load['q_mvar'].values
        

        if scenario == 'normal':
            # 增加扰动，其中电压和节点的参数都在原本的数据的基础上进行一定波动
            Vg = rng.uniform(1.00, 1.04, net.gen['vm_pu'].shape[0])
            Pg = rng.normal(Pg, 0.2*np.abs(Pg), net.gen['p_mw'].shape[0])
            Pd = rng.normal(Pd, 0.2*np.abs(Pd), net.load['p_mw'].shape[0])
            Qd = rng.normal(Qd, 0.2*np.abs(Qd), net.load['q_mvar'].shape[0])
        
            # 赋值
            net.gen['vm_pu'] = Vg
            net.gen['p_mw'] = Pg
            net.load['p_mw'] = Pd
            net.load['q_mvar'] = Qd
        elif scenario == 'heavy_overload':
            # 增加扰动，其中电压和节点的参数都在原本的数据的基础上进行一定波动
            scale = rng.uniform(1.3, 1.6)
            Vg = rng.uniform(1.00, 1.04, net.gen['vm_pu'].shape[0])
            Pg = rng.normal(Pg, 0.2*np.abs(Pg), net.gen['p_mw'].shape[0])
            Pd = rng.normal(Pd*scale, 0.05*np.abs(Pd), net.load['p_mw'].shape[0])
            Qd = rng.normal(Qd*scale, 0.05*np.abs(Qd), net.load['q_mvar'].shape[0])
        
            # 赋值
            net.gen['vm_pu'] = Vg
            net.gen['p_mw'] = Pg
            net.load['p_mw'] = Pd
            net.load['q_mvar'] = Qd

        elif scenario == 'high_voltage':
            # 增加扰动，其中电压和节点的参数都在原本的数据的基础上进行一定波动
            base_v = rng.uniform(1.02, 1.05)
            half_width = 0.015
            Vg = rng.uniform(base_v - half_width, base_v+half_width, net.gen['vm_pu'].shape[0])
            # Pg = rng.normal(Pg, 0.2*np.abs(Pg), net.gen['p_mw'].shape[0])

            scale = rng.uniform(0.5, 0.9)
            Pd = rng.normal(Pd*scale, 0.05*np.abs(Pd), net.load['p_mw'].shape[0])
            Qd = rng.normal(Qd*scale, 0.05*np.abs(Qd), net.load['q_mvar'].shape[0])
        
            # 赋值
            net.gen['vm_pu'] = Vg
            net.gen['p_mw'] = Pg
            net.load['p_mw'] = Pd
            net.load['q_mvar'] = Qd

        elif scenario == 'low_voltage':
            # 增加扰动，其中电压和节点的参数都在原本的数据的基础上进行一定波动
            scale = rng.uniform(1.2, 1.5)
            Vg = rng.uniform(0.96, 1.00, net.gen['vm_pu'].shape[0])
            # Pg = rng.normal(Pg, 0.2*np.abs(Pg), net.gen['p_mw'].shape[0])
            Pd = rng.normal(Pd*scale, 0.05*np.abs(Pd), net.load['p_mw'].shape[0])
            Qd = rng.normal(Qd*scale, 0.05*np.abs(Qd), net.load['q_mvar'].shape[0])
        
            # 赋值
            net.gen['vm_pu'] = Vg
            net.gen['p_mw'] = Pg
            net.load['p_mw'] = Pd
            net.load['q_mvar'] = Qd


        # 其他的改变不动，获取原始的net的数据
        try:
            net['converged'] = False
            pp.runpp(net, algorithm='nr', init="results", numba=False, enforce_q_lims=ENFORCE_Q_LIMS)
        except:
            if not net['converged']:
                continue      
        # 在这一步，潮流已经跑完了，要开始提取特征了，我们使用Ybus进行特征提取
        edge_features , node_g , node_b = extract_ybus_data(net)

        p_true, q_true = calculate_net_injection(net)

        # 构建node_features
        # [Index,Type,Vm,Va,P,Q,Gii,Bii]
        # 这一句不理解
        n = net.bus.values.shape[0]
        types = np.ones(n)*2 # 默认全部设为2，如果是其他节点类型就更改相应的节点

        # 确定节点类型，这一步不理解
        for j in range(net.gen.shape[0]):    
            index = np.where(net.gen['bus'].values[j] == net.bus['name'])[0][0] 
            types[index] = 1 
        for j in range(net.ext_grid.shape[0]):
            index = np.where(net.ext_grid['bus'].values[j] == net.bus['name'])[0][0]
            types[index] = 0

        base_node_features = np.zeros((n,6))
        base_node_features[:,0] = net.bus['name'].values
        base_node_features[:,1] = types
        base_node_features[:,2] = net.res_bus['vm_pu']
        base_node_features[:,3] = net.res_bus['va_degree']
        base_node_features[:,4] = p_true
        base_node_features[:,5] = q_true

        # 拼接成完整的节点特征
        node_g = node_g.reshape(-1, 1)
        node_b = node_b.reshape(-1, 1)
        final_node_features = np.concatenate([base_node_features, node_g, node_b], axis=1)

        edge_features_list.append(edge_features)
        node_features_list.append(final_node_features)

        if len(edge_features_list) % 100 == 0:
            print(f'[Process {os.getpid()}] Sample {len(edge_features_list)}')

    return edge_features_list, node_features_list


def generate_data_parallel(num_samples, num_processes, base_net_create, num_lines_to_remove=0, num_lines_to_add=0):
    sublist_size = num_samples // num_processes
    parent_rng = np.random.default_rng(123456)
    streams = parent_rng.spawn(num_processes)
    pool = mp.Pool(processes=num_processes)
    args = [[sublist_size, st, base_net_create, num_lines_to_remove, num_lines_to_add] for st in streams]
    results = pool.starmap(generate_data, args)
    # results = generate_data(*args[0]) # DEBUG LINE
    pool.close()
    pool.join()
    
    edge_features_list = []
    node_features_list = []
    for sub_res in results:
        edge_features_list += sub_res[0]
        node_features_list += sub_res[1]
        
    return edge_features_list, node_features_list

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(prog='Power Flow Data Generator', description='')
    parser.add_argument('--case', type=str, default='118', help='e.g. 118, 14, 6470rte')
    parser.add_argument('--num_lines_to_remove', '-r', type=int, default=0, help='Number of lines to remove')
    parser.add_argument('--num_lines_to_add', '-a', type=int, default=0, help='Number of lines to add')
    args = parser.parse_args()

    num_lines_to_remove = args.num_lines_to_remove
    num_lines_to_add = args.num_lines_to_add
    case = args.case

    if case == '3':
        base_net_create = create_case3
    elif case == '14':
        base_net_create = pp.networks.case14
    elif case == '118':
        base_net_create = pp.networks.case118
    elif case == '6470rte':
        base_net_create = pp.networks.case6470rte
    else:
        print('Invalid test case.')
        exit()
    if num_lines_to_remove > 0 or num_lines_to_add > 0:
        complete_case_name = 'case' + case + 'perturbed' + f'{num_lines_to_remove:1d}' + 'r' + f'{num_lines_to_add:1d}' + 'a'
    else:
        complete_case_name = 'case' + case + '_n0_safe'
    base_net = base_net_create()
    base_net.bus['name'] = base_net.bus.index
    print(base_net.bus)
    print(base_net.line)
    
    # Generate data
    edge_features_list, node_features_list = generate_data_parallel(number_of_samples, number_of_processes, base_net_create,
                                                                    num_lines_to_remove=num_lines_to_remove, num_lines_to_add=num_lines_to_add)
    
    # Turn the lists into numpy arrays
    edge_features = np.array(edge_features_list)
    node_features = np.array(node_features_list)
    # graph_features = np.array(graph_feature_list)

    # Print the shapes
    print(f'edge_features shape: {edge_features.shape}')
    print(f'node_features_x shape: {node_features.shape}')

    print(f'range of edge_features "from": {np.min(edge_features[:,:,0])} - {np.max(edge_features[:,:,0])}')
    print(f'range of edge_features "to": {np.min(edge_features[:,:,1])} - {np.max(edge_features[:,:,1])}')

    print(f'range of node_features "index": {np.min(node_features[:,:,0])} - {np.max(node_features[:,:,0])}')

    # print(f"A. {A}")
    # print(f"edge_features. {edge_features}")
    # print(f"node_features_x. {node_features_x}")
    # print(f"node_features_y. {node_features_y}")

    # save the features
    os.makedirs("./data/raw", exist_ok=True)
    with open("./data/raw/"+complete_case_name+"_edge_features.npy", 'wb') as f:
        np.save(f, edge_features)

    with open("./data/raw/"+complete_case_name+"_node_features.npy", 'wb') as f:
        np.save(f, node_features)

    # with open("./data/"+test_case+"_graph_features.npy", 'wb') as f:
    #     np.save(f, graph_features)

    # with open("./data/raw/"+test_case+"_adjacency_matrix.npy", 'wb') as f:
    #     np.save(f, A)
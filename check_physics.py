import torch
import numpy as np
from datasets.PowerFlowData import PowerFlowData
from utils.custom_loss_functions import RectangularPowerImbalance

def check_physics_detail(sample_idx=5):
    # 1. 加载数据 (测试集)
    # 确保 PowerFlowData 已经重新生成过，且 e/f 是 std=1 的
    dataset = PowerFlowData(root='data', case='14v11', split=[.5, .2, .3], task='test')
    
    if sample_idx >= len(dataset):
        print(f"Sample index {sample_idx} out of range.")
        return

    data = dataset[sample_idx]
    
    # 2. 获取归一化参数
    xymean, xystd, edgemean, edgestd = dataset.get_data_means_stds()
    
    # 3. 初始化物理引擎
    phys_engine = RectangularPowerImbalance(xymean, xystd, edgemean, edgestd)
    
    print(f"\n>>> Analyzing Sample {sample_idx} (Case {dataset.case}) <<<")
    
    # 4. 准备输入 (全部使用 Ground Truth)
    # pred_ef: 物理值 e, f (因为 PowerFlowData 里 std=1)
    target_ef = data.y[:, 2:] 
    # input_pq: 归一化值 P, Q
    input_pq = data.y[:, :2] 
    
    # 5. 反归一化 P, Q 以便对比 (手动做一次，为了打印)
    pq_mean = xymean[:, :2]
    pq_std = xystd[:, :2]

    # 真实的 P 值
    real_pq_target = input_pq * (pq_std + 1e-7) + pq_mean
    
    # 6. 计算物理流 (Physics Calculation)
    # 我们需要 hack 一下 phys_engine，让它返回每个节点的 P_calc, Q_calc 而不是 mean loss
    
    # --- 手动执行 forward 的逻辑 ---
    # 反归一化边
    real_edge = data.edge_attr * (edgestd + 1e-7) + edgemean
    
    # 处理无向图
    edge_index = data.edge_index
    if edge_index.shape[1] > 0:
        if edge_index[0,0] not in edge_index[1,edge_index[0,:]==edge_index[1,0]]:
            edge_index_dup = torch.stack([edge_index[1], edge_index[0]], dim=0)
            edge_index = torch.cat([edge_index, edge_index_dup], dim=1)
            real_edge = torch.cat([real_edge, real_edge], dim=0)
    
    # 消息传递 (计算计算出的注入功率)
    calc_pq_flow = phys_engine.propagate(edge_index, x=target_ef, edge_attr=real_edge)
    # -------------------------------
    
    # 7. 打印对比表
    print(f"{'Node':<5} | {'Type':<5} | {'P_Target':<10} {'P_Calc':<10} {'P_Diff':<10} | {'Q_Target':<10} {'Q_Calc':<10} {'Q_Diff':<10}")
    print("-" * 85)
    
    total_p_err = 0
    total_q_err = 0
    
    for i in range(data.num_nodes):
        p_t = real_pq_target[i, 0].item()
        p_c = calc_pq_flow[i, 0].item()
        p_diff = p_t + p_c
        
        q_t = real_pq_target[i, 1].item()
        q_c = calc_pq_flow[i, 1].item()
        q_diff = q_t + q_c
        
        b_type = data.bus_type[i].item()
        type_str = "Slk" if b_type==0 else ("PV" if b_type==1 else "PQ")
        
        total_p_err += abs(p_diff)
        total_q_err += abs(q_diff)
        
        print(f"{i:<5} | {type_str:<5} | {p_t:<10.4f} {p_c:<10.4f} {p_diff:<10.4f} | {q_t:<10.4f} {q_c:<10.4f} {q_diff:<10.4f}")
        
    print("-" * 85)
    print(f"Mean P Error: {total_p_err / data.num_nodes:.4f}")
    print(f"Mean Q Error: {total_q_err / data.num_nodes:.4f}")
    
    if (total_p_err / data.num_nodes) > 0.01:
        print("\n❌ 结论：物理公式与真实数据不匹配！")
        print("原因可能是：")
        print("1. 忽略了变压器变比 (Tap Ratio)")
        print("2. 忽略了线路对地电容 (Shunt)")
        print("3. 忽略了并联补偿装置 (Shunt Capacitor/Reactor)")
    else:
        print("\n✅ 结论：物理公式与数据基本吻合。")

if __name__ == '__main__':
    check_physics_detail()
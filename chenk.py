import torch
import numpy as np
from torch_geometric.loader import DataLoader
from datasets.PowerFlowData import PowerFlowData
from utils.argument_parser import argument_parser
import os

# 引入你的物理计算模块 (直接用 Loss 里的类，确保逻辑一致)
from utils.custom_loss_functions import RectangularPowerImbalance 

def verify_pipeline_correctness():
    print("🕵️‍♂️ 正在验证数据管道 (Data Pipeline Sanity Check)...")
    
    # 1. 配置 (直接读取新清理的数据)
    CASE_NAME = '118v_n1_train' 
    DATA_DIR = 'data' # 根据你的实际路径调整
    BATCH_SIZE = 32
    
    # 2. 初始化 Dataset
    # 这会自动触发 process，重新生成 .pt 文件
    print("正在加载/处理数据集...")
    dataset = PowerFlowData(
        root=DATA_DIR, 
        case=CASE_NAME,
        split=[0.8, 0.1, 0.1], 
        task='train'
    )
    
    # 获取归一化参数
    xymean, xystd, edgemean, edgestd = dataset.get_data_means_stds()
    print("✅ 归一化参数获取成功")
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. 初始化物理计算引擎
    # 我们用这个引擎来检查 Ground Truth 是否自洽
    physics_engine = RectangularPowerImbalance(xymean, xystd, edgemean, edgestd)
    
    print("🔄 开始验证 Batch...")
    
    # 只检查第一个 Batch 即可，如果有问题，第一个 Batch 就会炸
    for i, data in enumerate(loader):
        # data.y: [P, Q, e, f] (归一化后的真值)
        # data.x: [P, Q, e, f, Gii, Bii, PE...] (归一化后的输入)
        
        # 我们用真值 e, f (data.y[:, 2:4]) 
        # 和 真值 P, Q (data.y[:, 0:2])
        # 和 真值 G, B (data.edge_attr, data.x[:, 4:6])
        # 来计算物理 Loss
        
        target_ef = data.y[:, 2:4] # 真值电压
        target_pq = data.y[:, 0:2] # 真值功率
        node_gb = data.x[:, 4:6]   # 真值节点导纳
        
        # 计算物理误差
        # 理论上，如果是真值代入，这个 Loss 应该几乎为 0
        loss = physics_engine(target_ef, target_pq, data.edge_index, data.edge_attr, node_gb)
        
        print(f"\n[Batch {i}] 物理自洽性检查:")
        print(f"  Power Imbalance Loss (Ground Truth): {loss.item():.8f}")
        
        # 4. 额外检查：N-1 拓扑是否生效？
        # 统计 batch 里每个图的边数
        batch_idx = data.batch
        edges_src = data.edge_index[0]
        # 计算每个图有多少条边
        edges_per_graph = torch.bincount(batch_idx[edges_src])
        
        min_edges = edges_per_graph.min().item()
        max_edges = edges_per_graph.max().item()
        
        print(f"  拓扑检查: Min Edges={min_edges}, Max Edges={max_edges}")
        
        if min_edges < max_edges:
            print("  ✅ 确认：检测到变拓扑 (N-1 生效)")
        elif min_edges == max_edges:
            print("  ⚠️ 警告：所有图边数相同，可能是 N-0 数据？或者正好抽到了相同的拓扑")

        # 阈值判断
        if loss.item() < 1e-4:
            print("\n✅ 结论：数据管道完美！代码逻辑正确。")
            print("   (Ground Truth 满足物理方程，说明归一化/构图/加载全都没问题)")
        else:
            print("\n❌ 结论：数据管道存在 BUG！")
            print("   (真值代入方程都有误差，请检查：1. 归一化参数是否对齐? 2. 边方向是否搞反? 3. 物理公式单位?)")
            
        break # 只看一个 Batch 就够了

if __name__ == "__main__":
    verify_pipeline_correctness()
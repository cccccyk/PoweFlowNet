import torch
import os
import numpy as np
from datasets.PowerFlowData import PowerFlowData

# === 请填入那个结果异常的 NNConv 的 Run ID ===
RUN_ID = '20251213-2019'  # 替换为你出问题的那个 ID
DATA_DIR = 'data'
CASE = '14' # 或者 118
# ============================================

def check_params():
    print(f"🔍 正在诊断 Run ID: {RUN_ID} ...")
    
    path = os.path.join(DATA_DIR, 'params', f'data_params_{RUN_ID}.pt')
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return

    params = torch.load(path, map_location='cpu')
    
    print("\n[1] 检查归一化参数 (xymean/std)")
    print(f"   Shape: {params['xymean'].shape}") # 应该是 [1, 4]
    
    # 打印 e (index 2) 和 f (index 3) 的参数
    e_mean = params['xymean'][0, 2].item()
    e_std = params['xystd'][0, 2].item()
    print(f"   e (实部) -> Mean: {e_mean:.4f}, Std: {e_std:.4f}")
    
    if abs(e_std - 1.0) < 1e-3 and abs(e_mean) < 1e-3:
        print("   ✅ e 使用了 Flat Start 归一化 (不归一化)")
    else:
        print("   ⚠️ e 使用了统计归一化 (Mean!=0 或 Std!=1)")

    print("\n[2] 检查边归一化参数 (edgemean/std)")
    edge_mean = params['edgemean']
    edge_std = params['edgestd']
    print(f"   Edge Mean: {edge_mean.numpy().flatten()}")
    print(f"   Edge Std : {edge_std.numpy().flatten()}")
    
    # 检查边参数是否过小 (导致归一化后数值爆炸)
    if (edge_std < 1e-4).any():
        print("   ❌ 警告：边的标准差极小！这会导致归一化后的边特征巨大，引爆 NNConv！")

def check_data():
    print("\n[3] 检查处理后的数据分布 (Processed Data)")
    try:
        # 加载测试集
        dataset = PowerFlowData(root=DATA_DIR, case=CASE, split=[.5, .2, .3], task='test')
        data = dataset[0]
        
        print(f"   Sample Edge Attr (前5行):\n{data.edge_attr[:5]}")
        max_edge = data.edge_attr.abs().max().item()
        print(f"   Max Edge Value: {max_edge:.4f}")
        
        if max_edge > 10.0:
            print("   ❌ 严重警告：边特征数值过大！NNConv 会生成巨大的权重矩阵！")
        else:
            print("   ✅ 边特征数值范围正常。")
            
    except Exception as e:
        print(f"   无法加载数据: {e}")

if __name__ == '__main__':
    check_params()
    check_data()
import torch
import numpy as np
import os
from datasets.PowerFlowData import PowerFlowData

# ================= 配置 =================
CASE_NAME = '118v_n1_train' # 你的 N-1 数据集名字
DATA_DIR = 'data'
# =======================================

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"❌ {name} 中发现 NaN!")
        # 打印出具体是哪一列 (Feature) 出现了 NaN
        nan_cols = torch.where(torch.isnan(tensor).any(dim=0))[0]
        print(f"   出错的列索引: {nan_cols.tolist()}")
        return True
    return False

def check_inf(tensor, name):
    if torch.isinf(tensor).any():
        print(f"❌ {name} 中发现 Inf (无穷大)!")
        return True
    return False

def main():
    print(f">>> 正在排查数据集: {CASE_NAME}")
    
    # -------------------------------------------------------------
    # 步骤 1: 检查归一化参数 (Data Params)
    # -------------------------------------------------------------
    # 这里我们假设你用 finetune.py 里的逻辑加载了某个旧的 params
    # 或者我们直接看 PowerFlowData 刚刚生成的 params
    # 为了简单，我们直接实例化 Dataset，让它自己加载
    
    try:
        dataset = PowerFlowData(
            root=DATA_DIR, case=CASE_NAME, split=[.9, .05, .05], task='train'
        )
    except Exception as e:
        print(f"Dataset 加载失败: {e}")
        return

    print("\n[1] 检查归一化统计量 (Mean/Std)")
    xymean, xystd, _, _ = dataset.get_data_means_stds()
    
    if check_nan(xymean, "xymean (均值)") or check_nan(xystd, "xystd (方差)"):
        print("   👉 结论：原始数据中可能有 NaN，导致算出来的均值方差也是 NaN。")
        print("   👉 请跳到步骤 2 检查原始 .npy 文件。")
    else:
        print("   ✅ 归一化参数正常。")
        # 检查是否有 0 方差
        zero_std = torch.where(xystd == 0)[1]
        if len(zero_std) > 0:
            print(f"   ⚠️ 警告：以下列的方差为 0: {zero_std.tolist()}")
            print("   这可能导致归一化时除以 1e-7，产生巨大的数值。")

    # -------------------------------------------------------------
    # 步骤 2: 检查处理后的数据 (Processed .pt)
    # -------------------------------------------------------------
    print("\n[2] 检查 PyG Data 对象 (Processed)")
    has_error = False
    for i in range(len(dataset)):
        data = dataset[i]
        
        # 检查 x (Input)
        if check_nan(data.x, f"Sample {i} - data.x"):
            has_error = True
        
        # 检查 y (Target)
        if check_nan(data.y, f"Sample {i} - data.y"):
            has_error = True
            
        if has_error:
            print(f"   样本 {i} 数据异常！停止检查。")
            break
            
    if not has_error:
        print("   ✅ Processed 数据似乎没有 NaN。")
    
    # -------------------------------------------------------------
    # 步骤 3: 检查原始 .npy 文件 (Raw)
    # -------------------------------------------------------------
    print("\n[3] 检查原始 .npy 文件 (Raw Source)")
    raw_node_path = os.path.join(DATA_DIR, f"raw/case{CASE_NAME}_node_features.npy")
    
    if os.path.exists(raw_node_path):
        # Allow pickle for object arrays
        raw_data = np.load(raw_node_path, allow_pickle=True)
        
        print(f"   加载了 {len(raw_data)} 条原始数据")
        for i in range(len(raw_data)):
            sample = raw_data[i]
            # sample shape: [Nodes, 8] -> [Idx, Type, Vm, Va, P, Q, Gii, Bii]
            
            # 转 float 检查
            try:
                sample_float = sample.astype(float)
                if np.isnan(sample_float).any():
                    print(f"❌ 原始样本 {i} 包含 NaN!")
                    # 找出是哪一列
                    nan_mask = np.isnan(sample_float)
                    rows, cols = np.where(nan_mask)
                    print(f"   出错列: {np.unique(cols)}")
                    print(f"   出错行(节点): {np.unique(rows)}")
                    break
            except:
                print(f"   ⚠️ 样本 {i} 无法转换为 float，可能包含非数值类型。")
    else:
        print("   ⚠️ 找不到原始 .npy 文件。")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def analyze_dataset_distribution(data_dir, case_name):
    print(f"🔍 正在全方位审计数据集: {case_name}")
    
    # 构建路径
    node_path = os.path.join(data_dir, f"{case_name}_node_features.npy")
    # label_path = os.path.join(data_dir, f"{case_name}_labels.npy")
    
    if not os.path.exists(node_path):
        print(f"❌ 错误：找不到文件 {node_path}")
        return

    # 1. 加载数据
    print("   正在加载 .npy 文件 (可能需要几秒钟)...")
    nodes_raw = np.load(node_path, allow_pickle=True)
    # labels_raw = np.load(label_path, allow_pickle=True)
    
    total_samples = len(nodes_raw)
    print(f"   样本总数: {total_samples}")
    
    # 2. 提取关键指标
    min_vms = []
    max_vms = []
    all_vms = [] # 采样存一些看全局
    
    print("   正在统计电压分布...")
    for i in range(total_samples):
        # 确保转为 float
        sample_nodes = nodes_raw[i].astype(np.float32)
        # 第2列是 Vm
        vms = sample_nodes[:, 2]
        
        min_vms.append(vms.min())
        max_vms.append(vms.max())
        
        # 降采样存全局分布，防止内存爆
        if i % 20 == 0: 
            all_vms.extend(vms)
            
    min_vms = np.array(min_vms)
    max_vms = np.array(max_vms)
    all_vms = np.array(all_vms)
    # labels = np.array(labels_raw)

    # ==========================================
    # 3. 核心统计分析
    # ==========================================
    
    # 定义阈值
    LOW_THRESHOLDS = [0.95, 0.94, 0.92, 0.90, 0.85]
    HIGH_THRESHOLDS = [1.05, 1.06, 1.08, 1.10, 1.15]
    
    print("\n" + "="*70)
    print("📊 数据集电压极值审计报告")
    print("="*70)
    
    # --- 低电压分析 ---
    print("\n[▼] 低电压 (Low Voltage) 稀缺性分析:")
    print("-" * 70)
    print(f"{'阈值 (Min V < X)':<20} | {'样本数':<10} | {'占比 (%)':<10} | {'评价'}")
    print("-" * 70)
    for thresh in LOW_THRESHOLDS:
        count = np.sum(min_vms < thresh)
        ratio = count / total_samples * 100
        
        severity = ""
        if ratio == 0: severity = "💀 绝迹 (没救了)"
        elif ratio < 0.1: severity = "❌ 极度稀缺"
        elif ratio < 1.0: severity = "⚠️ 稀缺 (长尾)"
        elif ratio < 5.0: severity = "👌 较少"
        else: severity = "✅ 充足"
        print(f"{thresh:<20} | {count:<10} | {ratio:<10.2f} | {severity}")

    # --- 高电压分析 ---
    print("\n[▲] 高电压 (High Voltage) 分布分析:")
    print("-" * 70)
    print(f"{'阈值 (Max V > X)':<20} | {'样本数':<10} | {'占比 (%)':<10} | {'评价'}")
    print("-" * 70)
    for thresh in HIGH_THRESHOLDS:
        count = np.sum(max_vms > thresh)
        ratio = count / total_samples * 100
        
        severity = ""
        if ratio == 0: severity = "⚪ 无"
        elif ratio < 1.0: severity = "⚠️ 稀缺"
        elif ratio > 50.0: severity = "🔥 泛滥 (模型偏向高压)"
        else: severity = "✅ 存在"
        print(f"{thresh:<20} | {count:<10} | {ratio:<10.2f} | {severity}")

    # --- Label 分析 ---
    print("\n[🏷️] Label 分布 (0=Safe, 1=V_Err, 2=L_Err, 3=Both):")
    # unique, counts = np.unique(labels, return_counts=True)
    # for u, c in zip(unique, counts):
    #     print(f"  Class {u}: {c:<6} ({c/total_samples:.2%})")

    # ==========================================
    # 4. 可视化
    # ==========================================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 图1：最小电压分布 (关注左尾)
    sns.histplot(min_vms, bins=100, kde=False, color='blue', ax=axes[0])
    axes[0].axvline(0.95, color='r', linestyle='--', label='Limit 0.95')
    axes[0].set_title('Distribution of Min Voltage (Sag)')
    axes[0].set_xlabel('Min Voltage (p.u.)')
    axes[0].set_yscale('log') # 对数坐标看稀有样本
    axes[0].legend()
    
    # 图2：最大电压分布 (关注右尾)
    sns.histplot(max_vms, bins=100, kde=False, color='red', ax=axes[1])
    axes[1].axvline(1.05, color='r', linestyle='--', label='Limit 1.05')
    axes[1].set_title('Distribution of Max Voltage (Swell)')
    axes[1].set_xlabel('Max Voltage (p.u.)')
    axes[1].set_yscale('log')
    axes[1].legend()

    # 图3：全局电压分布
    sns.histplot(all_vms, bins=100, kde=True, color='purple', ax=axes[2])
    axes[2].axvline(0.95, color='k', linestyle='--')
    axes[2].axvline(1.05, color='k', linestyle='--')
    axes[2].set_title(f'Global Voltage Dist (Mean: {np.mean(all_vms):.3f})')
    
    plt.tight_layout()
    plt.savefig('data_distribution_full_audit.png')
    print(f"\n[Plot] 全面分布图已保存至: data_distribution_full_audit.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 确保这里的路径指向你生成 .npy 的地方
    parser.add_argument('--data_dir', type=str, default='./data/raw')
    # 你的 case 名字 (如 case118v_n1_train)
    parser.add_argument('--case', type=str, default='case118_test') 
    args = parser.parse_args()
    
    analyze_dataset_distribution(args.data_dir, args.case)
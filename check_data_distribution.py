import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def analyze_new_data(data_dir, case_name):
    print(f"🔍 正在审计数据集: {case_name}")
    
    # 1. 构建文件路径
    # 假设文件名格式为: case{name}_node_features.npy
    # 如果你的文件名没有 'case' 前缀，请修改这里
    filename_prefix = f"case{case_name}" if not case_name.startswith("case") else case_name
    
    node_path = os.path.join(data_dir, f"{filename_prefix}_node_features.npy")
    # label_path = os.path.join(data_dir, f"{filename_prefix}_labels.npy")
    
    if not os.path.exists(node_path):
        print(f"❌ 错误：找不到文件 {node_path}")
        print(f"   请检查路径或文件名是否正确。")
        return

    # 2. 加载数据
    print("⏳ 正在加载 .npy 文件 (请稍候)...")
    try:
        nodes_raw = np.load(node_path, allow_pickle=True)
        # labels_raw = np.load(label_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    total_samples = len(nodes_raw)
    print(f"✅ 加载成功！样本总数: {total_samples}")

    # 3. 提取 V_min 和 V_max
    print("📊 正在统计电压极值...")
    min_vms = []
    max_vms = []
    
    for i in range(total_samples):
        sample_nodes = nodes_raw[i].astype(np.float32)
        vms = sample_nodes[:, 2]          # 电压幅值 (p.u.)
        types = sample_nodes[:, 1]        # 节点类型: 0=Slack, 1=PV, 2=PQ
        
        # 全网最小电压（低压可能出现在任何节点）
        min_vms.append(vms.min())
        
        # 仅 PQ 节点的最大电压（用于判断真实过电压风险）
        pq_mask = (types == 2)
        if np.any(pq_mask):
            max_pq_vm = vms[pq_mask].max()
        else:
            max_pq_vm = vms.max()  # fallback（理论上不会发生）
        max_vms.append(max_pq_vm)

    # ✅ 关键：必须转为 NumPy 数组！
    min_vms = np.array(min_vms, dtype=np.float32)
    max_vms = np.array(max_vms, dtype=np.float32)

    # ==========================================
    # 4. 打印统计报告
    # ==========================================
    
    # 阈值设置
    LOW_THRESHOLDS = [0.95, 0.94, 0.92, 0.90, 0.85, 0.80]
    HIGH_THRESHOLDS = [1.05, 1.055, 1.06, 1.08]
    
    print("\n" + "="*60)
    print(f"📄 数据集体检报告: {case_name}")
    print("="*60)
    
    # --- A. 低压分布 ---
    print(f"\n[▼] 最小电压 (Min Voltage) 分布:")
    print("-" * 60)
    print(f"{'阈值 (< X)':<15} | {'样本数':<10} | {'占比 (%)':<10} | {'评价'}")
    print("-" * 60)
    for thresh in LOW_THRESHOLDS:
        count = np.sum(min_vms < thresh)
        ratio = count / total_samples * 100
        
        severity = ""
        if ratio == 0: severity = "💀 无 (模型学不会严重故障)"
        elif ratio < 0.5: severity = "⚠️ 极度稀缺"
        elif ratio < 5.0: severity = "👌 较少 (需关注)"
        else: severity = "✅ 充足"
        print(f"{thresh:<15} | {count:<10} | {ratio:<10.2f} | {severity}")

    # --- B. 高压分布 ---
    print(f"\n[▲] 最大电压 (Max Voltage) 分布:")
    print("-" * 60)
    print(f"{'阈值 (> X)':<15} | {'样本数':<10} | {'占比 (%)':<10} | {'评价'}")
    print("-" * 60)
    for thresh in HIGH_THRESHOLDS:
        count = np.sum(max_vms > thresh)
        ratio = count / total_samples * 100
        print(f"{thresh:<15} | {count:<10} | {ratio:<10.2f} | {'⚠️ 注意堆积' if ratio > 10 else ''}")

    # --- C. Label 分布 ---
    print(f"\n[🏷️] Label 类别分布:")
    print("(0:安全, 1:电压越限, 2:线路过载, 3:两者都有)")
    print("-" * 60)
    # unique, counts = np.unique(labels, return_counts=True)
    # for u, c in zip(unique, counts):
    #     print(f"  Class {u}: {c:<8} ({c/total_samples:.2%})")

    # ==========================================
    # 5. 画图可视化
    # ==========================================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 图1：最小电压直方图
    sns.histplot(min_vms, bins=80, kde=False, color='#e74c3c', ax=axes[0])
    axes[0].axvline(0.95, color='k', linestyle='--', label='Limit 0.95')
    axes[0].set_title('Distribution of Sample MIN Voltage')
    axes[0].set_xlabel('Voltage (p.u.)')
    axes[0].set_ylabel('Count (Log Scale)')
    axes[0].set_yscale('log') # 关键：用对数坐标看长尾
    axes[0].legend()
    
    # 图2：最大电压直方图
    sns.histplot(max_vms, bins=80, kde=False, color='#3498db', ax=axes[1])
    axes[1].axvline(1.05, color='k', linestyle='--', label='Limit 1.05')
    axes[1].set_title('Distribution of Sample MAX Voltage')
    axes[1].set_xlabel('Voltage (p.u.)')
    axes[1].set_yscale('log')
    axes[1].legend()

    # # 图3：Label 饼图
    # axes[2].pie(counts, labels=[f'Class {u}' for u in unique], autopct='%1.1f%%', 
    #             colors=sns.color_palette('pastel'), startangle=140)
    # axes[2].set_title('Label Distribution')

    plt.tight_layout()
    save_name = f'check_distribution_{case_name}.png'
    plt.savefig(save_name)
    print(f"\n[Plot] 分布可视化图已保存至: {save_name}")
    print("✅ 检查完毕。请根据报告决定是否开始训练。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 你的数据目录
    parser.add_argument('--data_dir', type=str, default='./data/raw')
    # 你的 case 名字 (输入 118v_n1 即可，脚本会自动处理 case 前缀)
    parser.add_argument('--case', type=str, default='118v2_30w_n1') 
    args = parser.parse_args()
    
    analyze_new_data(args.data_dir, args.case)
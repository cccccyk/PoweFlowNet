import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader

# 导入你的自定义模块
from datasets.PowerFlowData import PowerFlowData 
from networks.MPN import MaskEmbdMultiMPN_GPS
from utils.evaluation import load_model

# ==========================================
# 1. 核心诊断逻辑
# ==========================================

@torch.no_grad()
def run_diagnostic(run_id, data_dir, case_name, hidden_dim, n_layers, batch_size, device):
    # --- A. 加载归一化参数 ---
    param_path = os.path.join(data_dir, 'params', f'data_params_{run_id}.pt')
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"找不到参数文件: {param_path}")
    
    params = torch.load(param_path, map_location='cpu')
    xymean = params['xymean'] # [1, 6] -> [P, Q, e, f, Gii, Bii]
    xystd = params['xystd']
    
    # 提取 e, f 的反归一化参数
    ef_mean = xymean[:, 2:4].to(device)
    ef_std = xystd[:, 2:4].to(device)

    # --- B. 加载数据 ---
    print(f"正在加载测试集: {case_name}...")
    testset = PowerFlowData(
        root=data_dir, case=case_name, split=[.5, .2, .3], task='test',
        xymean=xymean, xystd=xystd,
        edgemean=params['edgemean'], edgestd=params['edgestd']
    )
    loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # --- C. 构建并加载模型 ---
    node_in, _, edge_dim = testset.get_data_dimensions()
    model = MaskEmbdMultiMPN_GPS(
        nfeature_dim=node_in,
        efeature_dim=edge_dim,
        output_dim=2,
        hidden_dim=hidden_dim,
        n_gnn_layers=n_layers
    ).to(device)
    
    model, _ = load_model(model, run_id, device)
    model.eval()
    print(f"✅ 模型 {run_id} 加载成功，开始分析...")

    # --- D. 提取电压数据 ---
    all_true_vm = []
    all_pred_vm = []
    all_min_vm_true = []
    all_min_vm_pred = []
    
    for data in loader:
        data = data.to(device)
        out = model(data) # [N, 2] -> e, f
        
        # 反归一化还原物理值
        pred_real = out * (ef_std + 1e-7) + ef_mean
        target_real = data.y[:, 2:4] * (ef_std + 1e-7) + ef_mean
        
        # 计算 Vm
        pred_vm = torch.sqrt(pred_real[:, 0]**2 + pred_real[:, 1]**2)
        true_vm = torch.sqrt(target_real[:, 0]**2 + target_real[:, 1]**2)
        
        # 只统计需要预测的节点 (mask)
        mask = data.pred_mask[:, 2] > 0
        all_true_vm.extend(true_vm[mask].cpu().numpy())
        all_pred_vm.extend(pred_vm[mask].cpu().numpy())
        
        # 记录每个样本（图）的最小电压，用于判定安全性
        for i in range(data.num_graphs):
            m = (data.batch == i) & mask
            if m.any():
                all_min_vm_true.append(true_vm[m].min().item())
                all_min_vm_pred.append(pred_vm[m].min().item())

    all_true_vm = np.array(all_true_vm)
    all_pred_vm = np.array(all_pred_vm)
    all_min_vm_true = np.array(all_min_vm_true)
    all_min_vm_pred = np.array(all_min_vm_pred)

    # --- E. 计算定量统计 ---
    errors = all_pred_vm - all_true_vm
    me = np.mean(errors) # 系统性偏置
    mae = np.mean(np.abs(errors))
    
    print("\n" + "="*50)
    print("📊 数据分布诊断报告")
    print("="*50)
    print(f"Mean Error (系统性偏置): {me:.6f} p.u.")
    print(f"MAE (平均绝对误差)   : {mae:.6f} p.u.")
    print(f"物理安全阈值         : 0.95 p.u.")
    
    if me < -0.001:
        print(f"💡 结论：模型倾向于【低估】电压，这解释了为什么误报率(FP)那么高。")
    elif me > 0.001:
        print(f"💡 结论：模型倾向于【高估】电压，这可能导致漏报(FN)。")
    else:
        print(f"💡 结论：模型无明显系统性偏置，误差主要来自随机扰动。")
    
    # 计算临界区样本比例
    border_samples = np.sum((all_min_vm_true > 0.94) & (all_min_vm_true < 0.96))
    print(f"边界区样本数 (0.94~0.96): {border_samples} ({border_samples/len(all_min_vm_true):.1%} of total)")
    print("="*50)

    # --- F. 绘图可视化 ---
    plt.figure(figsize=(20, 6))
    sns.set_theme(style="whitegrid")

    # 1. 全网节点电压分布图
    plt.subplot(1, 3, 1)
    sns.kdeplot(all_true_vm, color="blue", label="True Vm", fill=True, alpha=0.3)
    sns.kdeplot(all_pred_vm, color="red", label="Pred Vm", fill=True, alpha=0.3)
    plt.axvline(0.95, color='green', linestyle='--', label='Safety Limit (0.95)')
    plt.title("All Nodes Voltage Density", fontsize=14)
    plt.xlabel("Voltage (p.u.)")
    plt.legend()

    # 2. 每个样本最小电压分布 (决定安全判定的关键)
    plt.subplot(1, 3, 2)
    sns.histplot(all_min_vm_true, color="blue", label="True Min Vm", alpha=0.5, bins=50)
    sns.histplot(all_min_vm_pred, color="red", label="Pred Min Vm", alpha=0.5, bins=50)
    plt.axvline(0.95, color='black', linestyle='--', label='0.95 Limit')
    plt.title("Sample Minimum Voltage (Decision Critical)", fontsize=14)
    plt.xlabel("Min Voltage in Graph (p.u.)")
    plt.legend()

    # 3. 预测残差分布 (Pred - True)
    plt.subplot(1, 3, 3)
    sns.histplot(errors, color="purple", kde=True, bins=100)
    plt.axvline(0, color='black', linestyle='-')
    plt.axvline(me, color='red', linestyle='--', label=f'Mean Bias: {me:.4f}')
    plt.title("Prediction Residuals (Pred - True)", fontsize=14)
    plt.xlabel("Error (p.u.)")
    plt.legend()

    plot_name = f"diagnostic_{run_id}.png"
    plt.tight_layout()
    plt.savefig(plot_name, dpi=150)
    print(f"\n✅ 诊断图表已保存至: {plot_name}")

# ==========================================
# 2. 参数解析与入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default='20251223-6480', help='训练时的ID')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据根目录')
    parser.add_argument('--case', type=str, default='118v_n1_train', help='Case名称')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    run_diagnostic(
        args.run_id, args.data_dir, args.case, 
        args.hidden_dim, args.n_layers, args.batch_size, device
    )
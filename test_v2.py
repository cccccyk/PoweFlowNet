import os
import logging
import torch
import numpy as np
from functools import partial
import torch.nn.functional as F

# 引入你的数据和模型定义
# 注意：确保 PowerFlowData 是修改后的版本 (输出 P, Q, e, f)
from datasets.PowerFlowData import PowerFlowData 
from networks.MPN import (
    MaskEmbdMultiMPN, 
    MaskEmbdMultiMPN_NNConv,
    MaskEmbdMultiMPN_NNConv_v2,
    MaskEmbdMultiMPN_NNConv_v3,
    MaskEmbdMultiMPN_Transformer,
    MaskEmbdMultiMPN_Transformer_Large
    # ... 其他你用到的模型类
)
from utils.evaluation import load_model
from utils.argument_parser import argument_parser
from torch_geometric.loader import DataLoader
from utils.custom_loss_functions import PowerImbalance

logger = logging.getLogger(__name__)

# ==========================================
# 辅助函数：坐标转换
# ==========================================
def rect_to_polar(e, f):
    """
    将直角坐标转换为极坐标 (Vm, Va_degree)
    """
    vm = torch.sqrt(e**2 + f**2 + 1e-12)
    va_rad = torch.atan2(f, e)
    va_deg = va_rad * (180.0 / torch.pi)
    return vm, va_deg

# ==========================================
# 核心测试函数
# ==========================================
@torch.no_grad()
def evaluate_metrics(model, loader, device, xymean, xystd):
    model.eval()
    
    # 累积器
    metrics = {
        'mse_e': 0., 'mse_f': 0.,
        'mse_vm': 0., 'mse_va': 0.,
        'mae_e': 0., 'mae_f': 0.,
        'mae_vm': 0., 'mae_va': 0.,
        'phys_imbalance': 0.
    }
    num_samples = 0
    
    # 初始化物理计算器 (用于计算 P/Q Imbalance)
    # 注意：这里的 mean/std 必须和 training 时一致
    # 假设 PowerImbalance 已经改成了直角坐标版
    # phys_calc = PowerImbalance(xymean, xystd, None, None).to(device) # edgemean/std 在 forward 里传
    # 上面这行有点问题，因为 PowerImbalance 需要 edge 的 mean/std
    # 我们在 main 里初始化它比较好，这里先不初始化
    ef_mean = xymean[:,2:].to(device)
    ef_std  = xystd[:,2:].to(device)
    
    for data in loader:
        data = data.to(device)

        # 输出的是归一化的结果
        out = model(data) # [N, 2] -> (e, f)

        # label的标签也是归一化的结果
        target_ef = data.y[:, 2:]

        mask_ef = data.pred_mask[:, 2:] # [N, 2]

        # 指标1，归一化空间下的MSE，可以用于和val_loss和train_loss做比较
        loss_norm = ((out - target_ef)**2 * mask_ef).sum() / mask_ef.sum()
        print(f"归一化下的mse={loss_norm}")

        # 指标2：物理空间下的误差
        # 反归一化
        pred_real = out * (ef_std + 1e-7) + ef_mean
        target_real = target_ef * (ef_std + 1e-7) + ef_mean

        # 计算物理误差
        diff_ef = (pred_real - target_real) * mask_ef
        batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
        
        # 2. 计算基础指标 (e, f) - 纯数值
        # 只计算 mask 部分
        mse_ef = (diff_ef**2).sum(dim=0) / (mask_ef.sum(dim=0) + 1e-6)
        mae_ef = diff_ef.abs().sum(dim=0) / (mask_ef.sum(dim=0) + 1e-6)
        
        metrics['mse_e'] += mse_ef[0].item() * batch_size
        metrics['mse_f'] += mse_ef[1].item() * batch_size
        metrics['mae_e'] += mae_ef[0].item() * batch_size
        metrics['mae_f'] += mae_ef[1].item() * batch_size
        
        # 3. 计算衍生指标 (Vm, Va) - 物理意义
        # 先还原成物理值 (假设 e, f 未归一化或已处理，这里根据你的 PowerFlowData 逻辑)
        # 如果 PowerFlowData 里 e,f 没归一化(mean=0, std=1)，直接用
        # 如果归一化了，需要反归一化。
        # 假设：你的 PowerFlowData 现在的 xystd 对于 e,f 是 1.0 (不归一化)
        
        pred_e, pred_f = out[:, 0], out[:, 1]
        true_e, true_f = target_ef[:, 0], target_ef[:, 1]
        
        pred_vm, pred_va = rect_to_polar(pred_e, pred_f)
        true_vm, true_va = rect_to_polar(true_e, true_f)
        
        # 计算 Vm, Va 的误差 (只看 PQ 和 PV 节点，Slack 不看)
        # 简单起见，用 mask_ef[:, 0] 作为节点是否需要预测的标志
        node_mask = mask_ef[:, 0] 
        
        diff_vm = (pred_vm - true_vm) * node_mask
        # 相角差处理 (处理 180/-180 跳变)
        diff_va = (pred_va - true_va) * node_mask
        # 简单的去周期化: 使得误差在 -180 到 180 之间
        diff_va = (diff_va + 180) % 360 - 180
        
        metrics['mse_vm'] += (diff_vm**2).sum().item() / (node_mask.sum() + 1e-6) * batch_size
        metrics['mse_va'] += (diff_va**2).sum().item() / (node_mask.sum() + 1e-6) * batch_size
        
        metrics['mae_vm'] += diff_vm.abs().sum().item() / (node_mask.sum() + 1e-6) * batch_size
        metrics['mae_va'] += diff_va.abs().sum().item() / (node_mask.sum() + 1e-6) * batch_size
        
        num_samples += batch_size

    # 平均
    for k in metrics:
        metrics[k] /= num_samples
        
    return metrics

# ==========================================
# Main
# ==========================================
@torch.no_grad()
def main():
    # === 配置 ===
    run_id = '20251216-4323' # <--- 填入你的新 Run ID
    # ===========
    
    args = argument_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载参数
    data_dir = args.data_dir
    data_param_path = os.path.join(data_dir, 'params', f'data_params_{run_id}.pt')
    if not os.path.exists(data_param_path):
        print(f"Error: Data params not found: {data_param_path}")
        return
        
    data_param = torch.load(data_param_path, map_location='cpu')
    # 注意：这里的 mean/std 对应 [P, Q, e, f]
    xymean, xystd = data_param['xymean'], data_param['xystd']
    edgemean, edgestd = data_param['edgemean'], data_param['edgestd']
    
    # 2. 加载数据
    testset = PowerFlowData(root=data_dir, case=args.case,
                            split=[.5, .2, .3], task='test',
                            xymean=xymean, xystd=xystd,
                            edgemean=edgemean, edgestd=edgestd)
    
    loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    # 3. 加载模型
    # [关键] 强制输出维度为 2
    model_cls = MaskEmbdMultiMPN_Transformer_Large # 或者你在 args 里指定的模型
    
    node_in, _, edge_dim = testset.get_data_dimensions()
    model = model_cls(
        nfeature_dim=node_in,
        efeature_dim=edge_dim,
        output_dim=2, # <--- 2
        hidden_dim=args.hidden_dim,
        n_gnn_layers=args.n_gnn_layers,
        K=args.K,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    model, _ = load_model(model, run_id, device)
    print(f"Loaded model {run_id}")
    
    # 4. 执行测试
    metrics = evaluate_metrics(model, loader, device, xymean, xystd)
    
    # 5. 打印结果
    print("\n" + "="*40)
    print(f"Test Results for {args.case} (Rectangular)")
    print("="*40)
    print(f"[Direct Output]")
    print(f"  MSE e : {metrics['mse_e']:.6f}")
    print(f"  MSE f : {metrics['mse_f']:.6f}")
    print(f"  MAE e : {metrics['mae_e']:.6f}")
    print(f"  MAE f : {metrics['mae_f']:.6f}")
    
    print(f"\n[Derived Physics]")
    print(f"  MSE Vm: {metrics['mse_vm']:.6f}")
    print(f"  MAE Vm: {metrics['mae_vm']:.6f}")
    print(f"  MAE Va: {metrics['mae_va']:.4f} degrees") # 关注这个！
    print("="*40)
    
    # 6. (可选) 计算物理不平衡
    # 需要实例化 Rectangular PowerImbalance 并运行一遍
    # 这部分逻辑可以复用 evaluate_epoch_v2 或者再写一个循环
    # 关键看你想不想看 P/Q 的误差

if __name__ == "__main__":
    main()
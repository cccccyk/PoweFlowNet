import os
import logging
import torch
import torch_geometric
from functools import partial

# 引入你的数据和模型定义
from datasets.PowerFlowData_copy import PowerFlowData, denormalize
from networks.MPN import (
    MPN, MPN_simplenet, SkipMPN, MaskEmbdMPN, MultiConvNet, MultiMPN, 
    MaskEmbdMultiMPN, MaskEmbdMultiMPN_NNConv, MaskEmbdMultiMPN_NNConv_v2, 
    MaskEmbdMultiMPN_PhysicsAttn, MaskEmbdMultiMPN_TAG_NNConv,ImprovedPowerFlowGNN
)
from utils.evaluation import load_model, evaluate_epoch_v2
from utils.argument_parser import argument_parser
from torch_geometric.loader import DataLoader

# 引入 Loss 定义
from utils.custom_loss_functions import (
    Masked_L2_loss, PowerImbalance, MixedMSEPoweImbalance, MaskedL2V2, MaskedL1
)

logger = logging.getLogger(__name__)

# --- [辅助函数] 手动计算 MAPE ---
# 这个函数独立于 evaluate_epoch_v2，避免参数传递报错
def compute_mape_metrics(model, loader, xymean, xystd, device, eps=1e-2):
    """
    手动计算反归一化后的 MAPE。
    eps: 防止分母为0的阈值 (默认0.01 p.u.)
    """
    model.eval()
    
    # [Vm, Va, P, Q]
    total_mape = torch.zeros(4).to(device) 
    total_counts = torch.zeros(4).to(device)
    
    xymean = xymean.to(device)
    xystd = xystd.to(device)

    print(f"Start calculating MAPE with eps={eps}...")
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            
            # 1. 反归一化 (Pred 和 True)
            pred_real = denormalize(out, xymean, xystd)
            true_real = denormalize(data.y, xymean, xystd)
            
            # 2. 获取掩码 (只计算未知量)
            mask = data.pred_mask
            
            # 3. 计算绝对误差
            diff = torch.abs(pred_real - true_real)
            abs_true = torch.abs(true_real)
            
            # 4. 安全除法 (防止除以0或极小值导致 MAPE 爆炸)
            denom = torch.clamp(abs_true, min=eps)
            
            # 5. 计算 MAPE (element-wise)
            mape_matrix = (diff / denom) * mask
            
            # 6. 按通道累加
            total_mape += mape_matrix.sum(dim=0)
            total_counts += mask.sum(dim=0)

    # 计算平均值
    avg_mape = total_mape / (total_counts + 1e-6)
    
    return avg_mape.cpu().numpy()


@torch.no_grad()
def main():
    # ================= [配置区域] =================
    # 请在这里填入你想要测试的 Run ID
    run_id = '20251211-9602' 
    # ============================================

    args = argument_parser()
    
    # 定义模型字典
    models = {
        'MPN': MPN,
        'MPN_simplenet': MPN_simplenet,
        'SkipMPN': SkipMPN,
        'MaskEmbdMPN': MaskEmbdMPN,
        'MultiConvNet': MultiConvNet,
        'MultiMPN': MultiMPN,
        'MaskEmbdMultiMPN': MaskEmbdMultiMPN,
        'MaskEmbdMultiMPN_NNConv': MaskEmbdMultiMPN_NNConv,
        'MaskEmbdMultiMPN_NNConv_v2': MaskEmbdMultiMPN_NNConv_v2, # 你的新模型
        'MaskEmbdMultiMPN_PhysicsAttn': MaskEmbdMultiMPN_PhysicsAttn,
        'MaskEmbdMultiMPN_TAG_NNConv': MaskEmbdMultiMPN_TAG_NNConv,
        'ImprovedPowerFlowGNN':ImprovedPowerFlowGNN
    }

    batch_size = args.batch_size
    grid_case = args.case
    data_dir = args.data_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载数据参数 (Mean/Std)
    data_param_path = os.path.join(data_dir, 'params', f'data_params_{run_id}.pt')
    if not os.path.exists(data_param_path):
        print(f"Error: Data params not found at {data_param_path}")
        return

    data_param = torch.load(data_param_path, map_location='cpu')
    xymean, xystd = data_param['xymean'], data_param['xystd']
    edgemean, edgestd = data_param['edgemean'], data_param['edgestd']
        
    # 2. 加载测试集
    testset = PowerFlowData(root=data_dir, case=grid_case,
                            split=[.5, .2, .3], task='test',
                            xymean=xymean, xystd=xystd, 
                            edgemean=edgemean, edgestd=edgestd)
    
    # 打印数据统计信息
    print("-" * 30)
    print(f'Test Data Shape: {testset.data.y.shape}')
    
    # 简单的统计打印
    _y = testset.data.y * xystd + xymean
    is_slack = testset.data.bus_type == 0
    is_pv = testset.data.bus_type == 1
    is_pq = testset.data.bus_type == 2
    
    _std = lambda x: ((x-x.mean()).square().sum()/x.numel()).sqrt().item()
    _l1 = lambda x: ((x-x.mean()).abs().sum()/x.numel()).item()
    
    _v = _y[is_pq,0]
    _p = _y[is_slack,2]
    
    print(f'Global Stats (Test Set):')
    print(f'  PQ Vm: Min={_v.min():.4f}, Max={_v.max():.4f}, Mean={_v.mean():.4f}')
    print(f'  Slack P: Min={_p.min():.4f}, Max={_p.max():.4f}, Mean={_p.mean():.4f}')
    print("-" * 30)

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # 3. 初始化模型
    nfeature_dim = args.nfeature_dim
    hidden_dim = args.hidden_dim
    node_in_dim, node_out_dim, edge_dim = testset.get_data_dimensions()
    
    model_class = models[args.model]
    
    # 注意：确保这里传入的参数与你训练 MaskEmbdMultiMPN_NNConv_v2 时一致
    model = model_class(
        nfeature_dim=node_in_dim,
        efeature_dim=edge_dim,
        output_dim=node_out_dim,
        hidden_dim=hidden_dim,
        n_gnn_layers=args.n_gnn_layers,
        K=args.K,
        dropout_rate=args.dropout_rate,
    ).to(device)
    
    # 4. 加载权重
    model, _ = load_model(model, run_id, device)
    model.eval()
    
    print(f"Model Loaded: {args.model}")
    print(f"Run ID: {run_id}")
    print(f"Case: {grid_case}")
    print("=" * 50)

    # -----------------------------------------------------------
    # 5. 评估指标计算
    # -----------------------------------------------------------

    # (A) Masked L2 (Normalized) - 原始 Loss
    print("\n[1] Masked L2 Loss (Normalized):")
    _loss = MaskedL2V2()
    masked_l2_terms = evaluate_epoch_v2(model, test_loader, _loss, device)
    for key, value in masked_l2_terms.items():
        print(f"  {key:<10}: {value:.6f}")

    # (B) Masked L2 (Denormalized) - 相当于 RMSE/MSE 物理值
    print("\n[2] Masked L2 (Denormalized - Physical Value MSE):")
    masked_l2_terms_de = evaluate_epoch_v2(
        model, test_loader, _loss, 
        pre_loss_fn=partial(denormalize, mean=xymean, std=xystd), 
        device=device
    )
    for key, value in masked_l2_terms_de.items():
        print(f"  {key:<10}: {value:.6f}")

    # (C) Masked L1 (Denormalized) - 相当于 MAE 物理值
    print("\n[3] Masked L1 (Denormalized - Physical Value MAE):")
    masked_l1_terms_de = evaluate_epoch_v2(
        model, test_loader, MaskedL1(), 
        pre_loss_fn=partial(denormalize, mean=xymean, std=xystd), 
        device=device
    )
    for key, value in masked_l1_terms_de.items():
        print(f"  {key:<10}: {value:.6f}")

    # (D) Masked MAPE (Denormalized) - 百分比误差
    print("\n[4] Masked MAPE (Denormalized - Percentage Error):")
    # eps=0.1 意味着如果真实值小于 0.1，分母按 0.1 算，防止 MAPE 爆炸
    mape_results = compute_mape_metrics(model, test_loader, xymean, xystd, device, eps=0.1)
    
    print(f"  Vm (MAPE): {mape_results[0]*100:.4f}%")
    print(f"  Va (MAPE): {mape_results[1]*100:.4f}%")
    print(f"  P  (MAPE): {mape_results[2]*100:.4f}%")
    print(f"  Q  (MAPE): {mape_results[3]*100:.4f}%")
    print(f"  AVG MAPE : {mape_results.mean()*100:.4f}%")

    # (E) 物理约束 Loss (Power Imbalance)
    print("\n[5] Physical Consistency (Power Imbalance):")
    pwr_imb_loss = PowerImbalance(*testset.get_data_means_stds()).to(device)
    # evaluate_epoch_v2 会返回 total, ref 等
    phy_results = evaluate_epoch_v2(model, test_loader, pwr_imb_loss, device)
    print(f"  Total Imbalance: {phy_results['total']:.6f}")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
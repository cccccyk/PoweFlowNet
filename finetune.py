import os
import torch
import numpy as np
import wandb
import argparse
import shutil
from datetime import datetime
from torch_geometric.loader import DataLoader

# --- 项目模块引入 ---
from datasets.PowerFlowData import PowerFlowData
# 引入所有可能用到的模型，防止报错
from networks.MPN import (
    MaskEmbdMultiMPN_GPS, 
)
from utils.training import train_epoch
from utils.evaluation import evaluate_epoch
from utils.custom_loss_functions import RectangularMixedLoss

# ==============================================================================
# 默认配置
# ==============================================================================
DEFAULT_PRETRAINED_ID = "20251220-8946"
DEFAULT_NEW_CASE      = "118v_n1_train" # 确保这里是你生成的 N-1 数据集名字
DEFAULT_GPU           = "cuda:1"
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune PowerFlowNet on N-1 Data")
    
    parser.add_argument('--pretrained-id', type=str, default=DEFAULT_PRETRAINED_ID)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--models-dir', type=str, default='models')
    parser.add_argument('--params-dir', type=str, default='data/params')
    
    parser.add_argument('--case', type=str, default=DEFAULT_NEW_CASE)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    
    # [关键修改] 默认 LR 降为 1e-5，防止梯度爆炸
    parser.add_argument('--lr', type=float, default=1e-5)
    
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=5e-3)
    parser.add_argument('--anchor', type=float, default=0.1)
    
    parser.add_argument('--wandb', action='store_true')
    # 填入你自己的用户名，或者设为 None
    parser.add_argument('--wandb-entity', type=str, default=None) 
    
    return parser.parse_args()

def load_normalization_stats(params_path):
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"❌ 找不到归一化参数: {params_path}")
    print(f">>> ♻️ 继承归一化参数: {params_path}")
    return torch.load(params_path, map_location='cpu')

def check_data_health(loader, device):
    """检查数据集中是否有 NaN"""
    print(">>> 正在检查数据健康状况 (NaN Check)...")
    for batch_idx, data in enumerate(loader):
        if torch.isnan(data.x).any() or torch.isnan(data.y).any():
            raise ValueError(f"❌ 数据中发现 NaN！Batch Index: {batch_idx}")
        if torch.isinf(data.x).any() or torch.isinf(data.y).any():
            raise ValueError(f"❌ 数据中发现 Inf！Batch Index: {batch_idx}")
    print("✅ 数据健康检查通过，无 NaN/Inf。")

def main():
    args = parse_args()
    device = torch.device(DEFAULT_GPU if torch.cuda.is_available() else 'cpu')
    
    # 1. 路径与ID
    pretrained_model_path = os.path.join(args.models_dir, f'model_{args.pretrained_id}.pt')
    pretrained_params_path = os.path.join(args.params_dir, f'data_params_{args.pretrained_id}.pt')
    
    current_time = datetime.now().strftime("%m%d-%H%M")
    new_run_id = f"ft-{current_time}-from-{args.pretrained_id.split('-')[-1]}"
    save_path = os.path.join(args.models_dir, f'model_{new_run_id}.pt')
    
    print(f"{'='*60}")
    print(f"🚀 启动微调 | 基础模型: {args.pretrained_id} | 数据: {args.case}")
    print(f"   新 Run ID: {new_run_id}")
    print(f"{'='*60}\n")

    # 2. 加载预训练 Checkpoint
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f"找不到模型: {pretrained_model_path}")
    
    checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
    train_args = checkpoint['args']
    
    # 3. 数据集准备 (必须继承参数!)
    stats = load_normalization_stats(pretrained_params_path)
    
    print(f">>> 加载数据集...")
    # 注意：一定要确保 data/processed 下的旧缓存已被删除，否则 inherit stats 无效
    trainset = PowerFlowData(
        root=args.data_dir, case=args.case, split=[.9, .05, .05], task='train',
        xymean=stats['xymean'], xystd=stats['xystd'], 
        edgemean=stats['edgemean'], edgestd=stats['edgestd']
    )
    valset = PowerFlowData(
        root=args.data_dir, case=args.case, split=[.9, .05, .05], task='val',
        xymean=stats['xymean'], xystd=stats['xystd'], 
        edgemean=stats['edgemean'], edgestd=stats['edgestd']
    )
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    # [新增] 训练前自检
    check_data_health(train_loader, device)

    # 4. 模型重建
    node_in_dim, _, edge_dim = trainset.get_data_dimensions()
    
    # 动态匹配模型类
    model_name = getattr(train_args, 'model', 'MaskEmbdMultiMPN_GPS')
    print(f">>> 重建模型架构: {model_name}")
    
    # 简单的工厂模式
    if model_name == 'MaskEmbdMultiMPN_GPS': ModelClass = MaskEmbdMultiMPN_GPS
    else: raise ValueError(f"未知的模型架构: {model_name}")

    model = ModelClass(
        nfeature_dim=node_in_dim,
        efeature_dim=edge_dim,
        output_dim=2, 
        hidden_dim=train_args.hidden_dim,
        n_gnn_layers=train_args.n_gnn_layers,
        # 兼容旧参数可能没有 nhead 的情况
        nhead=getattr(train_args, 'nhead', 4), 
        K=getattr(train_args, 'K', 3),
        dropout_rate=train_args.dropout_rate
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print(">>> ✅ 权重加载完毕")

    # 5. Loss & Optimizer
    # 注意：这里直接使用物理 Loss
    loss_fn = RectangularMixedLoss(
        stats['xymean'], stats['xystd'], stats['edgemean'], stats['edgestd'],
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, lambda_anchor=args.anchor
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 使用 Cosine 退火可能比 ReduceLROnPlateau 更稳
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    # 6. WandB
    if args.wandb:
        try:
            wandb.init(project="PowerFlowNet", entity=args.wandb_entity, name=new_run_id, config=train_args)
        except Exception as e:
            print(f"⚠️ WandB 初始化失败 (可能是网络或权限问题)，继续训练... \n{e}")

    # 7. 训练循环
    best_val_loss = 1e9 
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate_epoch(model, val_loader, loss_fn, device)
        
        val_loss = val_metrics['total']
        scheduler.step() # Cosine 不需要传 loss
        lr_now = optimizer.param_groups[0]['lr']

        # 日志
        if args.wandb and wandb.run is not None:
            wandb.log({
                'epoch': epoch, 'lr': lr_now,
                'train_loss': train_metrics['total'], 'train_phys': train_metrics.get('phys', 0),
                'val_loss': val_loss, 'val_phys': val_metrics.get('phys', 0),
            })

        print(f"Ep {epoch+1}/{args.epochs} | LR={lr_now:.2e} | "
              f"Tr_Loss={train_metrics['total']:.5f} (Phys={train_metrics.get('phys',0):.3f}) | "
              f"Val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'args': train_args, 
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss
            }, save_path)

    print(f"\n>>> ✅ 微调结束！模型已保存至: {save_path}")

if __name__ == '__main__':
    main()
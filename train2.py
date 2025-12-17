from datetime import datetime
import os
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from datasets.PowerFlowData_copy import PowerFlowData, random_bus_type
from networks.MPN import MPN, MPN_simplenet, SkipMPN, MaskEmbdMPN, MultiConvNet, MultiMPN, MaskEmbdMultiMPN , MaskEmbdMultiMPN_NNConv , MaskEmbdMultiMPN_GATv2,MaskEmbdMultiMPN_NNConv_v2,MaskEmbdMultiMPN_PhysicsAttn,MaskEmbdMultiMPN_TAG_NNConv,ImprovedPowerFlowGNN ,MaskEmbdMultiMPN_General
from utils.argument_parser import argument_parser
from utils.training import train_epoch, append_to_json
from utils.evaluation import evaluate_epoch
from utils.custom_loss_functions import Masked_L2_loss, PowerImbalance, MixedMSEPoweImbalance , Weighted_Masked_L2_loss

import wandb


def main():
    # Step 0: Parse Arguments and Setup
    args = argument_parser()    # 在这里读取传入的参数，比如学习率之类的
    run_id = datetime.now().strftime("%Y%m%d") + '-' + str(random.randint(0, 9999)) # 为运行创建一个唯一的ID，用于命名日志文件和模型文件
    LOG_DIR = 'logs'    # 日志保存的根目录
    SAVE_DIR = 'models' # 模型保存的根目录
    TRAIN_LOG_PATH = os.path.join(LOG_DIR, 'train_log/train_log_'+run_id+'.pt')
    SAVE_LOG_PATH = os.path.join(LOG_DIR, 'save_logs.json')
    SAVE_MODEL_PATH = os.path.join(SAVE_DIR, 'model_'+run_id+'.pt')
    # 这一步是选择MPN的模型
    models = {
        'MPN': MPN,
        'MPN_simplenet': MPN_simplenet,
        'SkipMPN': SkipMPN,
        'MaskEmbdMPN': MaskEmbdMPN,
        'MultiConvNet': MultiConvNet,
        'MultiMPN': MultiMPN,
        'MaskEmbdMultiMPN': MaskEmbdMultiMPN,
        'MaskEmbdMultiMPN_NNConv': MaskEmbdMultiMPN_NNConv,
        'MaskEmbdMultiMPN_GATv2':MaskEmbdMultiMPN_GATv2,
        'MaskEmbdMultiMPN_NNConv_v2':MaskEmbdMultiMPN_NNConv_v2,
        'MaskEmbdMultiMPN_PhysicsAttn':MaskEmbdMultiMPN_PhysicsAttn,
        'MaskEmbdMultiMPN_TAG_NNConv':MaskEmbdMultiMPN_TAG_NNConv,
        'ImprovedPowerFlowGNN':ImprovedPowerFlowGNN,
        'MaskEmbdMultiMPN_General':MaskEmbdMultiMPN_General
    }
    mixed_cases = ['118v2', '14v2']

    # Training parameters 和训练有关的参数，主要由args文件进行传导
    data_dir = args.data_dir    # 数据的路径
    nomalize_data = not args.disable_normalize  # 室友归一化的布尔值
    num_epochs = args.num_epochs
    loss_fn = Masked_L2_loss(regularize=args.regularize, regcoeff=args.regularization_coeff)
    
    eval_loss_fn = Masked_L2_loss(regularize=False)
    lr = args.lr
    batch_size = args.batch_size
    grid_case = args.case   # 所选取的网络
    
    # Network parameters 网络相关的参数，主要由args进行传导
    nfeature_dim = args.nfeature_dim
    efeature_dim = args.efeature_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    n_gnn_layers = args.n_gnn_layers
    conv_K = args.K
    dropout_rate = args.dropout_rate
    # 这里去加载对应的模型
    model = models[args.model]

    log_to_wandb = args.wandb
    wandb_entity = args.wandb_entity
    if log_to_wandb:
        wandb.init(project="PowerFlowNet",
                   entity=wandb_entity,
                   name=run_id,
                   config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备是: {device}")
    torch.manual_seed(1234)
    np.random.seed(1234)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Step 1: Load data 加载数据，分别为训练集，验证集，测试集
    trainset = PowerFlowData(root=data_dir, case=grid_case, split=[.5, .2, .3], task='train', normalize=nomalize_data,
                             transform=random_bus_type) # 其中这个random_bus_type的意思是数据增强，对母线的结构进行相应的修改，该功能仅在训练时使用
    valset = PowerFlowData(root=data_dir, case=grid_case, split=[.5, .2, .3], task='val', normalize=nomalize_data)
    testset = PowerFlowData(root=data_dir, case=grid_case, split=[.5, .2, .3], task='test', normalize=nomalize_data)
    
    # save normalizing params
    os.makedirs(os.path.join(data_dir, 'params'), exist_ok=True)
    # 这一步是把数据预处理的参数保存下来，因为后面要验证/测试的时候要保持归一化的一致
    torch.save({
            'xymean': trainset.xymean,
            'xystd': trainset.xystd,
            'edgemean': trainset.edgemean,
            'edgestd': trainset.edgestd,
        }, os.path.join(data_dir, 'params', f'data_params_{run_id}.pt'))
    
    # 把相应的数据转换成dataset的形式，其中shuffle保证打乱数据训练的顺序，防止模型学习到数据的排列顺序
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # 但是在测试和验证的时候需要保持一致，这样才能进行横向的比较
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    ## [Optional] physics-informed loss function 物理-数据驱动的损失函数
    if args.train_loss_fn == 'power_imbalance': # 功率不平衡损失
        # overwrite the loss function
        loss_fn = PowerImbalance(*trainset.get_data_means_stds()).to(device)
    elif args.train_loss_fn == 'masked_l2':     # 掩码L2损失
        print("使用masked,并且保持 train 和 eval 一致")
        loss_fn = Masked_L2_loss(regularize=args.regularize, regcoeff=args.regularization_coeff)
        eval_loss_fn = Masked_L2_loss(regularize=args.regularize, regcoeff=args.regularization_coeff)
    elif args.train_loss_fn == 'mixed_mse_power_imbalance': # 混合损失函数，比较重要的
        print("使用混合损失函数")
        loss_fn = MixedMSEPoweImbalance(*trainset.get_data_means_stds(), alpha=0.9).to(device)

    elif args.train_loss_fn == 'Weighted_Masked_L2_loss':
        loss_fn = Weighted_Masked_L2_loss(weights=[1.0, 20.0, 0.001, 0.001]).to(device)
        eval_loss_fn = Weighted_Masked_L2_loss(weights=[1.0, 20.0, 0.001, 0.001]).to(device)
    else:
        loss_fn = torch.nn.MSELoss()    # 纯MSE损失
    
    # Step 2: Create model and optimizer (and scheduler)
    node_in_dim, node_out_dim, edge_dim = trainset.get_data_dimensions()    #  获取数据的维度，分别是几点输入维度，节点输出维度，边特征
    # assert node_in_dim == 16
    assert node_in_dim == 4 # 这里是做一个检查，如果输入的数据的维度不对，就会自动的停止
    # 这里对模型进行初始化
    model = model(
        nfeature_dim=node_in_dim,
        efeature_dim=edge_dim,
        output_dim=node_out_dim,
        hidden_dim=hidden_dim,
        n_gnn_layers=n_gnn_layers,
        K=conv_K,
        dropout_rate=dropout_rate
    ).to(device) 

    #calculate model size 这一步用于计算模型到底有多少大
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", pytorch_total_params)
    
    # 创建优化器 意味着他要管理模型的所有的参数，并设计学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        mode='min',
    #                                                        factor=0.5,
    #                                                        patience=5,
    #                                                        verbose=True)
    # 学习率调整器，scheduler，可以动态的调整学习率
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=num_epochs)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # Step 3: Train model
    # 先把初始的loss设置成一个很大的值，这样后面的都会比他现在的小
    best_train_loss = 10000.
    best_val_loss = 10000.
    # 创建一个字典
    train_log = {
        'train': {
            'loss': []},
        'val': {
            'loss': []},
    }
    # pbar = tqdm(range(num_epochs), total=num_epochs, position=0, leave=True)
    for epoch in range(num_epochs):
        # 训练，这里把训练的过程放到了training文件中，这一步只是简单的调用一下训练的函数
        train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device)
        # 验证
        val_loss = evaluate_epoch(model, val_loader, eval_loss_fn, device)
        # 调整学习率
        scheduler.step()
        # 将训练情况保存到日志中
        train_log['train']['loss'].append(train_loss)
        train_log['val']['loss'].append(val_loss)

        # 发送到wandb的云端
        if log_to_wandb:
            wandb.log({'train_loss': train_loss,
                      'val_loss': val_loss})
            
        # 模型检查点
        # 检查当前的验证损损失是否低于历史最佳验证损失，如果是好的模型，就进行相应的打包和保存
        if train_loss < best_train_loss:
            best_train_loss = train_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.save:
                _to_save = {
                    'epoch': epoch,
                    'args': args,
                    'val_loss': best_val_loss,
                    'model_state_dict': model.state_dict(),
                }
                os.makedirs('models', exist_ok=True)
                torch.save(_to_save, SAVE_MODEL_PATH)
                append_to_json(
                    SAVE_LOG_PATH,
                    run_id,
                    {
                        'val_loss': f"{best_val_loss: .4f}",
                        # 'test_loss': f"{test_loss: .4f}",
                        'train_log': TRAIN_LOG_PATH,
                        'saved_file': SAVE_MODEL_PATH,
                        'epoch': epoch,
                        'model': args.model,
                        'train_case': args.case,
                        'train_loss_fn': args.train_loss_fn,
                        'args': vars(args)
                    }
                )
                os.makedirs(os.path.join(LOG_DIR, 'train_log'), exist_ok=True)

                torch.save(train_log, TRAIN_LOG_PATH)

        print(f"Epoch {epoch+1} / {num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best_val_loss={best_val_loss:.4f}")

    
    
    print(f"Training Complete. Best validation loss: {best_val_loss:.4f}")
    
    # Step 4: Evaluate model 测试
    if args.save:
        _to_load = torch.load(SAVE_MODEL_PATH, weights_only=False) 
        model.load_state_dict(_to_load['model_state_dict'])
        test_loss = evaluate_epoch(model, test_loader, eval_loss_fn, device)
        print(f"Test loss: {best_val_loss:.4f}")
        # if log_to_wandb:
        #     wandb.log({'test_loss', test_loss})

    # Step 5: Save results
    os.makedirs(os.path.join(LOG_DIR, 'train_log'), exist_ok=True)
    if args.save:
        torch.save(train_log, TRAIN_LOG_PATH)


if __name__ == '__main__':
    main()
from typing import Callable, Optional, List, Tuple, Union
import os
import json

import torch
from torch_geometric.loader import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
from tqdm import tqdm

from utils.custom_loss_functions import Masked_L2_loss, PowerImbalance, MixedMSEPoweImbalance ,Weighted_Masked_L2_loss ,RectangularPureMSELoss ,RectangularMixedLoss


def append_to_json(log_path, run_id, result):
    log_entry = {str(run_id): result}

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)
    try:
        with open(log_path, "r") as json_file:
            exist_log = json.load(json_file)
    except FileNotFoundError:
        exist_log = {}
    with open(log_path, "w") as json_file:
        exist_log.update(log_entry)
        json.dump(exist_log, json_file, indent=4)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    optimizer: Optimizer,
    device: torch.device
) -> dict: # 将返回值从float修改成dict，因为我们需要返回多个值，用于相应的画图操作
    """
    Trains a neural network model for one epoch using the specified data loader and optimizer.

    Args:
        model (nn.Module): The neural network model to be trained.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the training data.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer used for training the model.
        device (str): The device used for training the model (default: 'cpu').

    Returns:
        float: The mean loss value over all the batches in the DataLoader.

    """
    # 加载model
    model = model.to(device)
    # 初始loss是0
    acc = {'total': 0., 'mse_e': 0., 'mse_f': 0., 'phys': 0., 'pv': 0. , 'loss_anchor':0., "loss_angle":0.}

    num_samples = 0
    # 训练模式
    model.train()
    # 进度条
    pbar = tqdm(loader, total=len(loader), desc='Training')
    # 开始一批一批的把数据送进去
    for data in pbar:
        # 把数据放到GPU上面
        data = data.to(device)
        # 梯度清零 
        optimizer.zero_grad()
        # 前向传播
        out = model(data)   # 输出是一个2维的输出 

        batch_size = len(data.y)
        num_samples += batch_size
                            
        # 这里是计算loss
        if isinstance(loss_fn, Masked_L2_loss):
            loss = loss_fn(out, data.y, data.pred_mask)
        elif isinstance(loss_fn, Weighted_Masked_L2_loss):
            loss = loss_fn(out, data.y, data.pred_mask)
        elif isinstance(loss_fn, PowerImbalance):
            # have to mask out the non-predicted values, otherwise
            #   the network can learn to predict full-zeros
            masked_out = out*data.pred_mask \
                        + data.x*(1-data.pred_mask)
            loss = loss_fn(masked_out, data.edge_index, data.edge_attr)
        elif isinstance(loss_fn, MixedMSEPoweImbalance):
            loss = loss_fn(out, data.edge_index, data.edge_attr, data.y)
        elif isinstance(loss_fn,RectangularPureMSELoss):
            loss = loss_fn(out,data.y,data.pred_mask[:,:4])
        elif isinstance(loss_fn,RectangularMixedLoss):
            loss, l_mse, l_phys, l_pv, l_anchor , l_angle = loss_fn(
                pred_ef=out, 
                target_y=data.y, 
                input_x=data.x,
                mask=data.pred_mask, 
                edge_index=data.edge_index, 
                edge_attr=data.edge_attr, 
                bus_type=data.bus_type, 
                target_vm=data.target_vm
            )
            acc['phys'] += l_phys.item() * batch_size
            acc['pv'] += l_pv.item() * batch_size
            acc['loss_anchor'] += l_anchor.item() * batch_size
            acc['loss_angle'] += l_angle.item() * batch_size

        else:
            loss = loss_fn(out, data.y)
            

        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        # 更新训练的进度，以及一个epoch的平均loss
        
        
        acc['total'] += loss.item() * batch_size
        

        with torch.no_grad():
            target_ef = data.y[:,2:]
            mask_ef = data.pred_mask[:,2:4]

            squared_diff = (out - target_ef) ** 2
            mask_diff = squared_diff * mask_ef

            # 输出的地方有两列，因为选用了dim=0，所以有两个值
            batch_mse = mask_diff.sum(dim=0) / (mask_ef.sum(dim=0)+1e-6)

            acc['mse_e'] += batch_mse[0].item() * batch_size
            acc['mse_f'] += batch_mse[1].item() * batch_size

   # 计算平均值
    for k in acc:
        acc[k] /= num_samples
    
    return acc

def main():
    log_path = 'logs/save_logs.json'
    run_id = 'arb_id_01'
    result = {
        'train_loss': 0.3,
        'val_loss': 0.2,
    }
    append_to_json(log_path, run_id, result)


if __name__ == '__main__':
    main()

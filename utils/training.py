from typing import Callable, Optional, List, Tuple, Union
import os
import json

import torch
from torch_geometric.loader import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
from tqdm import tqdm

from utils.custom_loss_functions import Masked_L2_loss, PowerImbalance, MixedMSEPoweImbalance


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
) -> float:
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
    total_loss = 0.
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
        out = model(data)   # (N, 6), care about the first four. 
                            # data.y.shape == (N, 6)
        # 这里是计算loss
        if isinstance(loss_fn, Masked_L2_loss):
            loss = loss_fn(out, data.y, data.pred_mask)
        elif isinstance(loss_fn, PowerImbalance):
            # have to mask out the non-predicted values, otherwise
            #   the network can learn to predict full-zeros
            masked_out = out*data.pred_mask \
                        + data.x*(1-data.pred_mask)
            loss = loss_fn(masked_out, data.edge_index, data.edge_attr)
        elif isinstance(loss_fn, MixedMSEPoweImbalance):
            loss = loss_fn(out, data.edge_index, data.edge_attr, data.y)
        else:
            loss = loss_fn(out, data.y)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        # 更新训练的进度，以及一个epoch的平均loss
        num_samples += len(data)
        total_loss += loss.item() * len(data)

    mean_loss = total_loss / num_samples
    return mean_loss


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

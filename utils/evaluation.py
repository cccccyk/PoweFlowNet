"""This module provides functions for 
    - evaluation_epoch - evaluate performance over a whole epoch
    - other evaluation metrics function [NotImplemented]
"""
from typing import Callable, Optional, Union, Tuple
import os

import torch
from torch_geometric.loader import DataLoader
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from tqdm import tqdm

from utils.custom_loss_functions import Masked_L2_loss, PowerImbalance, MixedMSEPoweImbalance, MaskedL2V2, MaskedL1,Weighted_Masked_L2_loss,RectangularPureMSELoss,RectangularMixedLoss

LOG_DIR = 'logs'
SAVE_DIR = 'models'


def load_model(
    model: nn.Module,
    run_id: str,
    device: Union[str, torch.device]
) -> Tuple[nn.Module, dict]:
    SAVE_MODEL_PATH = os.path.join(SAVE_DIR, 'model_'+run_id+'.pt')
    if type(device) == str:
        device = torch.device(device)

    try:
        saved = torch.load(SAVE_MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(saved['model_state_dict'])
    except FileNotFoundError:
        print("File not found. Could not load saved model.")
        return -1

    return model, saved


def num_params(model: nn.Module) -> int:
    """
    Returns the number of trainable parameters in a neural network model.

    Args:
        model (nn.Module): The neural network model.

    Returns:
        int: The number of trainable parameters in the model.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, device, pre_loss_fn=None) -> dict:
    model.eval()
    
    # 累积器
    acc = {'total': 0., 'mse_e': 0., 'mse_f': 0., 'phys': 0., 'pv': 0. ,'loss_anchor':0. ,"loss_angle":0.}
    num_samples = 0
    
    pbar = tqdm(loader, total=len(loader), desc='Evaluating')
    
    for data in pbar:
        data = data.to(device)
        out = model(data)
        
        batch_size = len(data.y)
        num_samples += batch_size

        # ----------------------------------------------------
        # [修改] 适配逻辑同 train_epoch
        # ----------------------------------------------------
        if isinstance(loss_fn, RectangularMixedLoss):
            loss, l_mse, l_phys, l_pv,l_anchor,l_angle = loss_fn(
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
            
        elif isinstance(loss_fn, RectangularPureMSELoss):
            loss = loss_fn(out, data.y, data.pred_mask[:,:4])
            
        else:
            loss = loss_fn(out, data.y)

        acc['total'] += loss.item() * batch_size
        
        # --- 监控 e, f ---
        target_ef = data.y[:, 2:]
        mask_ef = data.pred_mask[:, 2:4]
        squared = (out - target_ef)**2 * mask_ef
        batch_mse = squared.sum(dim=0) / (mask_ef.sum(dim=0) + 1e-6)
        
        acc['mse_e'] += batch_mse[0].item() * batch_size
        acc['mse_f'] += batch_mse[1].item() * batch_size

    for k in acc:
        acc[k] /= num_samples
        
    return acc

@torch.no_grad()
def evaluate_epoch_v2(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: Callable,
        device: str = 'cpu',
        pre_loss_fn: Callable|None=None,) -> float:
    """
    Evaluates the performance of a trained neural network model on a dataset using the specified data loader.

    Args:
        model (nn.Module): The trained neural network model to be evaluated.
        loader (DataLoader): The PyTorch Geometric DataLoader containing the evaluation data.
        device (str): The device used for evaluating the model (default: 'cpu').

    Returns:
        float: The mean loss value over all the batches in the DataLoader.

    """
    pre_loss_fn = pre_loss_fn or (lambda x: x)
    model.eval()
    total_loss_terms = None
    num_samples = 0
    pbar = tqdm(loader, total=len(loader), desc='Evaluating:')
    for data in pbar:
        loss_terms = {}
        data = data.to(device)
        out = model(data)

        if isinstance(loss_fn, Masked_L2_loss):
            out, target = pre_loss_fn(out), pre_loss_fn(data.y)
            loss = loss_fn(out, target, data.pred_mask)
            loss_terms['total'] = loss
        elif isinstance(loss_fn, MaskedL2V2) or isinstance(loss_fn, MaskedL1):
            out, target = pre_loss_fn(out), pre_loss_fn(data.y)
            loss_terms = loss_fn(out, target, data.pred_mask)
        elif isinstance(loss_fn, PowerImbalance):
            # have to mask out the non-predicted values, otherwise
            #   the network can learn to predict full-zeros
            masked_out = out*data.pred_mask \
                        + data.x*(1-data.pred_mask)
            masked_out = pre_loss_fn(masked_out)
            loss = loss_fn(masked_out, data.edge_index, data.edge_attr)
            loss_terms['total'] = loss
            loss_terms['ref'] = loss_fn(data.y, data.edge_index, data.edge_attr)
            # loss = loss_fn(data.y, data.edge_index, data.edge_attr)
        elif isinstance(loss_fn, MixedMSEPoweImbalance):
            out = pre_loss_fn(out)
            loss = loss_fn(out, data.edge_index, data.edge_attr, data.y)
            loss_terms['total'] = loss
        else:
            out, target = pre_loss_fn(out), pre_loss_fn(data.y)
            loss = loss_fn(out, target)
            loss_terms['total'] = loss

        num_samples += len(data)
        if total_loss_terms is None:
            total_loss_terms = {key: value.item() for key, value in loss_terms.items()}
        else:
            for key, value in total_loss_terms.items():
                total_loss_terms[key] = value + loss_terms[key].item() * len(data)

    mean_loss_terms = {key: value/num_samples for key, value in total_loss_terms.items()}
    return mean_loss_terms
"""
这个版本处理的是GB，同时还增加了transformer需要的位置坐标，1220版本
"""
import os
from typing import Callable, Optional, List, Tuple, Union

import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy.sparse.linalg as lg # 用于特征分解

# ==========================================
# 辅助函数
# ==========================================

def polar_to_rect(vm, va_degree):
    """辅助函数：极坐标转直角坐标"""
    va_rad = va_degree * np.pi / 180.0
    e = vm * np.cos(va_rad)
    f = vm * np.sin(va_rad)
    return e, f

def denormalize(input, mean, std):
    return input * (std.to(input.device) + 1e-7) + mean.to(input.device)

def random_bus_type(data: Data) -> Data:
    data.bus_type = torch.randint_like(data.bus_type, low=0, high=2)
    return data

def compute_laplacian_pe(edge_index, num_nodes, k=8):
    """
    计算拉普拉斯位置编码 (Laplacian Positional Encoding)
    原理: 利用图拉普拉斯矩阵的最小 k 个非平凡特征向量作为节点的"几何坐标"
    """
    # 1. 获取对称归一化拉普拉斯矩阵 L = I - D^-0.5 A D^-0.5
    edge_index_lap, edge_weight_lap = get_laplacian(
        edge_index, normalization='sym', num_nodes=num_nodes
    )
    
    # 2. 转为 Scipy 稀疏矩阵 (ARPACK 算法求解特征值更快)
    L = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes)
    
    try:
        # 3. 特征分解
        # k+1: 因为最小的特征值对应全1向量(连通分量)，通常包含的信息量少，我们取之后 k 个
        # which='SM': Smallest Magnitude
        k_eval = min(k + 1, num_nodes - 1)
        vals, vecs = lg.eigsh(L, k=k_eval, which='SM')
        
        # 对特征值排序 (eigsh 返回的不一定有序)
        idx = vals.argsort()
        vecs = vecs[:, idx]
        
        # 4. 取出特征向量 [Num_Nodes, k]
        # 丢弃第一个 (index 0)，取 1 到 k+1
        pe = torch.from_numpy(vecs[:, 1:k+1]).float()
        
        # 如果节点数太少不足 k 个，补 0
        if pe.shape[1] < k:
            pad = torch.zeros(num_nodes, k - pe.shape[1])
            pe = torch.cat([pe, pad], dim=1)
            
    except Exception as e:
        # 容错：如果计算失败 (极少见)，返回全 0
        print(f"PE Warning: {e}")
        pe = torch.zeros(num_nodes, k)
        
    return pe

# ==========================================
# Dataset 类定义
# ==========================================
class PowerFlowData(InMemoryDataset):   
    """
    加载并处理基于 Ybus 的潮流数据 (带位置编码 PE)

    x(input): [P, Q, e, f, Gii, Bii, PE_0, ..., PE_k-1] (维度 = 6 + k)
    y(label): [P, Q, e, f]
    edge_attr:[Gij, Bij]
    
    归一化策略:
    - P, Q: 统计归一化
    - e, f: 不归一化 (Mean=0, Std=1)
    - Gii, Bii, PE: 统计归一化 (自动处理)
    """
    partial_file_names = [
        "edge_features.npy",
        "node_features.npy",
    ]
    split_order = {"train": 0, "val": 1, "test": 2}
    mixed_cases = ['118v2', '14v2']
    
    # 定义位置编码的维度
    PE_DIM = 8 

    def __init__(self, 
                root: str, 
                case: str = '14', 
                split: Optional[List[float]] = None, 
                task: str = "train", 
                transform: Optional[Callable] = None, 
                pre_transform: Optional[Callable] = None, 
                pre_filter: Optional[Callable] = None,
                normalize=True,
                xymean=None, xystd=None, edgemean=None, edgestd=None):

        assert len(split) == 3
        assert task in ["train", "val", "test"]

        self.normalize = normalize
        self.case = case  
        self.split = split
        self.task = task
        
        super().__init__(root, transform, pre_transform, pre_filter) 
        self.mask = torch.tensor([])

        # 赋值归一化参数
        if xymean is not None and xystd is not None:
            self.xymean, self.xystd = xymean, xystd
        else:
            self.xymean, self.xystd = None, None

        if edgemean is not None and edgestd is not None:
            self.edgemean, self.edgestd = edgemean, edgestd
        else:
            self.edgemean, self.edgestd = None, None

        path = self.processed_paths[self.split_order[self.task]]
        print(f"Loading processed data from: {path}")
        loaded_data = torch.load(path, weights_only=False)
        self.data, self.slices = self._normalize_dataset(*loaded_data)

    def get_data_dimensions(self):
        return self[0].x.shape[1], self[0].y.shape[1], self[0].edge_attr.shape[1]

    def get_data_means_stds(self):
        assert self.normalize == True   
        return self.xymean[:1, :], self.xystd[:1, :], self.edgemean[:1, :], self.edgestd[:1, :]

    def _normalize_dataset(self, data, slices):
        if not self.normalize:
            return data, slices

        if self.xymean is None or self.xystd is None:
            # 1. 计算 P, Q, e, f 的统计量 (基于 y)
            y_stats = data.y
            mean_y = torch.mean(y_stats, dim=0, keepdim=True) # [1, 4]
            std_y = torch.std(y_stats, dim=0, keepdim=True)   # [1, 4]

            # [策略] 强制 e(2), f(3) 不归一化
            mean_y[:, 2] = 0.0
            mean_y[:, 3] = 0.0
            std_y[:, 2] = 1.0
            std_y[:, 3] = 1.0

            # 2. 计算 Gii, Bii 和 PE 的统计量 (基于 x 的后续列)
            # x 的结构: [P, Q, e, f, Gii, Bii, PE...]
            # 我们取 index 4 之后的所有列
            other_stats = data.x[:, 4:]
            mean_other = torch.mean(other_stats, dim=0, keepdim=True) 
            std_other = torch.std(other_stats, dim=0, keepdim=True)   

            # 3. 拼接
            self.xymean = torch.cat([mean_y, mean_other], dim=1)
            self.xystd = torch.cat([std_y, std_other], dim=1)

            print(f"✅ 归一化参数已生成 (Dim={self.xymean.shape[1]}):")
            print(f"   P/Q/e/f: {self.xymean[0,:4].numpy()}")
            print(f"   Gii/Bii/PE...: {self.xymean[0,4:].numpy()}")

        # 应用归一化
        # x 使用全量参数
        data.x = (data.x - self.xymean) / (self.xystd + 1e-7)
        # y 使用前4维参数
        data.y = (data.y - self.xymean[:, :4]) / (self.xystd[:, :4] + 1e-7)

        # 边归一化
        if self.edgemean is None:
            mean = torch.mean(data.edge_attr, dim=0, keepdim=True)
            std = torch.std(data.edge_attr, dim=0, keepdim=True)
            self.edgemean, self.edgestd = mean, std
            
        data.edge_attr = (data.edge_attr - self.edgemean) / (self.edgestd + 1e-7)

        return data, slices
    
    @property
    def raw_file_names(self) -> List[str]:
        return ["case"+f"{self.case}"+"_"+name for name in self.partial_file_names]

    @property
    def processed_file_names(self) -> List[str]:
        # [修改] 改名以触发重新生成，加入 _pe 后缀
        return [
            "case"+f"{self.case}"+"_processed_train_ybus_pe.pt",
            "case"+f"{self.case}"+"_processed_val_ybus_pe.pt",
            "case"+f"{self.case}"+"_processed_test_ybus_pe.pt",
        ]
    
    def len(self):
        return self.slices['x'].shape[0]-1

    def process(self):
        assert len(self.raw_paths) % 2 == 0
        raw_paths_per_case = [[self.raw_paths[i], self.raw_paths[i+1],] for i in range(0, len(self.raw_paths), 2)]
        all_case_data = [[],[],[]]

        for case, raw_paths in enumerate(raw_paths_per_case):
            print(f"Processing raw files: {raw_paths}")
            edge_features = torch.from_numpy(np.load(raw_paths[0])).float()
            node_features = torch.from_numpy(np.load(raw_paths[1])).float()

            if node_features.shape[-1] != 8:
                raise ValueError(f"Expected 8 features (P,Q,Vm,Va,P_net,Q_net,Gii,Bii), got {node_features.shape[-1]}")

            if self.split is not None:
                split_len = [int(len(node_features) * i) for i in self.split]

            split_edge_features = torch.split(edge_features, split_len, dim=0)
            split_node_features = torch.split(node_features, split_len, dim=0)

            for idx in range(len(split_edge_features)):
                raw_data = split_node_features[idx] 
                bus_type = raw_data[:,:,1].long()
                
                # A. Target y [P, Q, e, f]
                vm_true = raw_data[:, :, 2]
                va_true = raw_data[:, :, 3]
                p_true = raw_data[:, :, 4]
                q_true = raw_data[:, :, 5]
                g_ii = raw_data[:, :, 6]
                b_ii = raw_data[:, :, 7]
                
                va_rad = va_true * (torch.pi / 180.0)
                e_true = vm_true * torch.cos(va_rad)
                f_true = vm_true * torch.sin(va_rad)
                y = torch.stack([p_true, q_true, e_true, f_true], dim=-1)

                # B. Input x (Base) [P, Q, e, f, Gii, Bii]
                x_base = torch.zeros(y.shape[0], y.shape[1], 6)
                x_base[:, :, 4] = g_ii
                x_base[:, :, 5] = b_ii
                
                # C. Mask & Init
                mask = torch.zeros(y.shape[0], y.shape[1], 6) 
                
                is_slack = (bus_type == 0)
                is_pv = (bus_type == 1)
                is_pq = (bus_type == 2)

                # Slack
                x_base[is_slack, 0:2] = 0.0
                x_base[is_slack, 2] = e_true[is_slack]
                x_base[is_slack, 3] = f_true[is_slack]
                mask[is_slack, 0:2] = 1.0
                
                # PV
                x_base[is_pv, 0] = p_true[is_pv]
                x_base[is_pv, 1] = 0.0
                x_base[is_pv, 2] = vm_true[is_pv]
                x_base[is_pv, 3] = 0.0
                mask[is_pv, 1:4] = 1.0

                # PQ
                x_base[is_pq, 0] = p_true[is_pq]
                x_base[is_pq, 1] = q_true[is_pq]
                x_base[is_pq, 2] = 1.0
                x_base[is_pq, 3] = 0.0
                mask[is_pq, 2:4] = 1.0

                # D. Edge Info
                e_feat_raw = split_edge_features[idx]
                
                data_list = []
                for i in range(len(x_base)):
                    # [关键新增] 计算位置编码 PE
                    curr_edge_index = e_feat_raw[i, :, 0:2].T.long()
                    curr_edge_attr = e_feat_raw[i, :, 2:]
                    num_nodes = x_base.shape[1]
                    
                    # 计算 PE (8维)
                    pe = compute_laplacian_pe(curr_edge_index, num_nodes, k=self.PE_DIM)
                    
                    # 拼接到 x: [P, Q, e, f, Gii, Bii] + [PE...]
                    x_final = torch.cat([x_base[i], pe], dim=-1)
                    
                    # 拼接到 mask: PE 是已知量，mask=0
                    mask_pe = torch.zeros(num_nodes, self.PE_DIM)
                    mask_final = torch.cat([mask[i], mask_pe], dim=-1)

                    data = Data(
                        x=x_final,
                        y=y[i],
                        bus_type=bus_type[i],
                        pred_mask=mask_final,
                        target_vm=vm_true[i],
                        edge_index=curr_edge_index,
                        edge_attr=curr_edge_attr
                    )
                    data_list.append(data)

                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]
                    
                all_case_data[idx].extend(data_list)

        for idx, case_data in enumerate(all_case_data):
            data, slices = self.collate(case_data)
            print(f"Saving to {self.processed_paths[idx]}...")
            torch.save((data, slices), self.processed_paths[idx])
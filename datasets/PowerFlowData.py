"""
这个版本处理的是GB，适配 N-1 变拓扑数据 (Variable Topology)，
同时增加了 Transformer 需要的 Laplacian Positional Encoding (PE)
"""
import os
from typing import Callable, Optional, List, Tuple, Union

import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy.sparse.linalg as lg 

# ==========================================
# 辅助函数 (保持不变)
# ==========================================
def random_bus_type(data: Data) -> Data:
    data.bus_type = torch.randint_like(data.bus_type, low=0, high=2)
    return data

def polar_to_rect(vm, va_degree):
    va_rad = va_degree * np.pi / 180.0
    e = vm * np.cos(va_rad)
    f = vm * np.sin(va_rad)
    return e, f

def compute_laplacian_pe(edge_index, num_nodes, k=8):
    """
    计算拉普拉斯位置编码
    """
    # 如果边数为0 (极端的孤岛情况)，直接返回0
    if edge_index.numel() == 0:
         return torch.zeros(num_nodes, k)

    edge_index_lap, edge_weight_lap = get_laplacian(
        edge_index, normalization='sym', num_nodes=num_nodes
    )
    L = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes)
    
    try:
        k_eval = min(k + 1, num_nodes - 1)
        if k_eval <= 0: return torch.zeros(num_nodes, k) # 节点太少

        vals, vecs = lg.eigsh(L, k=k_eval, which='SM')
        idx = vals.argsort()
        vecs = vecs[:, idx]
        # ==========================================
        # [核心修改] 符号标准化 (Sign Canonicalization)
        # ==========================================
        # 遍历每一个特征向量 (列)
        for i in range(vecs.shape[1]):
            # 找到绝对值最大的元素的索引
            max_idx = np.argmax(np.abs(vecs[:, i]))
            # 如果该元素为负，则翻转整个向量
            if vecs[max_idx, i] < 0:
                vecs[:, i] *= -1 
        # ==========================================
        
        # 取出特征向量 [Num_Nodes, k]
        pe = torch.from_numpy(vecs[:, 1:k+1]).float()
        
        if pe.shape[1] < k:
            pad = torch.zeros(num_nodes, k - pe.shape[1])
            pe = torch.cat([pe, pad], dim=1)
            
    except Exception as e:
        # print(f"PE Warning: {e}")
        pe = torch.zeros(num_nodes, k)
        
    return pe

# ==========================================
# Dataset 类定义
# ==========================================
class PowerFlowData(InMemoryDataset):   
    """
    适配 N-1 数据的 Dataset
    """
    partial_file_names = [
        "edge_features.npy",
        "node_features.npy",
        "labels.npy"
    ]
    split_order = {"train": 0, "val": 1, "test": 2}
    
    # 定义位置编码的维度
    PE_DIM = 8 

    def __init__(self, 
                root: str, 
                case: str = '118v_n1_train', # 修改默认值以匹配新文件名
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
            print("Computing normalization statistics...")
            # 1. 计算 P, Q, e, f 的统计量
            y_stats = data.y
            mean_y = torch.mean(y_stats, dim=0, keepdim=True)
            std_y = torch.std(y_stats, dim=0, keepdim=True)

            # 策略: e, f 不归一化
            mean_y[:, 2] = 0.0
            mean_y[:, 3] = 0.0
            std_y[:, 2] = 1.0
            std_y[:, 3] = 1.0

            # 2. 计算 Input X 的统计量
            other_stats = data.x[:, 4:] # Gii, Bii, PE...
            mean_other = torch.mean(other_stats, dim=0, keepdim=True) 
            std_other = torch.std(other_stats, dim=0, keepdim=True)   

            self.xymean = torch.cat([mean_y, mean_other], dim=1)
            self.xystd = torch.cat([std_y, std_other], dim=1)

        # 应用归一化
        data.x = (data.x - self.xymean) / (self.xystd + 1e-7)
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
        # 对应你的文件名格式: case118v_n1_train_edge_features.npy
        return ["case"+f"{self.case}"+"_"+name for name in self.partial_file_names]

    @property
    def processed_file_names(self) -> List[str]:
        return [
            "case"+f"{self.case}"+"_processed_train_ybus_pe.pt",
            "case"+f"{self.case}"+"_processed_val_ybus_pe.pt",
            "case"+f"{self.case}"+"_processed_test_ybus_pe.pt",
        ]
    
    def len(self):
        return self.slices['x'].shape[0]-1

    def process(self):
        assert len(self.raw_paths) % 3 == 0
        raw_paths_per_case = [[self.raw_paths[i], self.raw_paths[i+1],self.raw_paths[i+2]] for i in range(0, len(self.raw_paths), 3)]
        all_case_data = [[],[],[]] # Train, Val, Test

        for case_idx, raw_paths in enumerate(raw_paths_per_case):
            print(f"Processing raw files: {raw_paths}")
            
            # 1. 加载 object 数组
            edge_features_raw = np.load(raw_paths[0], allow_pickle=True)
            node_features_raw = np.load(raw_paths[1], allow_pickle=True)
            labels_raw = np.load(raw_paths[2], allow_pickle=True)
            
            total_samples = len(node_features_raw)
            print(f"Total samples found: {total_samples}")

            # 2. 计算切分
            if self.split is not None:
                split_len = [int(total_samples * i) for i in self.split]
                split_len[-1] = total_samples - sum(split_len[:-1])
            else:
                split_len = [total_samples, 0, 0]

            start_idx = 0
            split_edge_lists = []
            split_node_lists = []
            split_label_lists = [] # [新增]
            for length in split_len:
                end_idx = start_idx + length
                split_edge_lists.append(edge_features_raw[start_idx:end_idx])
                split_node_lists.append(node_features_raw[start_idx:end_idx])
                split_label_lists.append(labels_raw[start_idx:end_idx]) # [新增]
                start_idx = end_idx

            # 3. 逐集处理
            for split_idx in range(3):
                current_edges = split_edge_lists[split_idx]
                current_nodes = split_node_lists[split_idx]
                current_labels = split_label_lists[split_idx] # [新增]
                
                data_list = []
                
                for i in range(len(current_nodes)):
                    # =================================================
                    # [修复核心] 强制转换为 float32，避免 object 报错
                    # =================================================
                    try:
                        raw_node_np = current_nodes[i].astype(np.float32)
                        raw_edge_np = current_edges[i].astype(np.float32)
                        raw_label = int(current_labels[i])
                    except ValueError as e:
                        print(f"Skipping sample {i} due to conversion error: {e}")
                        continue
                        
                    raw_node = torch.from_numpy(raw_node_np)
                    raw_edge = torch.from_numpy(raw_edge_np)
                    # =================================================

                    # B. 提取标签和特征
                    bus_type = raw_node[:, 1].long()
                    vm_true = raw_node[:, 2]
                    va_true = raw_node[:, 3]
                    p_true = raw_node[:, 4]
                    q_true = raw_node[:, 5]
                    g_ii = raw_node[:, 6]
                    b_ii = raw_node[:, 7]
                    
                    va_rad = va_true * (torch.pi / 180.0)
                    e_true = vm_true * torch.cos(va_rad)
                    f_true = vm_true * torch.sin(va_rad)
                    
                    y = torch.stack([p_true, q_true, e_true, f_true], dim=-1)

                    x_base = torch.zeros(raw_node.shape[0], 6)
                    x_base[:, 4] = g_ii
                    x_base[:, 5] = b_ii
                    
                    mask = torch.zeros(raw_node.shape[0], 6) 
                    
                    is_slack = (bus_type == 0)
                    is_pv = (bus_type == 1)
                    is_pq = (bus_type == 2)

                    x_base[is_slack, 0:2] = 0.0
                    x_base[is_slack, 2] = e_true[is_slack]
                    x_base[is_slack, 3] = f_true[is_slack]
                    mask[is_slack, 0:2] = 1.0
                    
                    x_base[is_pv, 0] = p_true[is_pv]
                    x_base[is_pv, 1] = 0.0
                    x_base[is_pv, 2] = vm_true[is_pv]
                    x_base[is_pv, 3] = 0.0
                    mask[is_pv, 1:4] = 1.0

                    x_base[is_pq, 0] = p_true[is_pq]
                    x_base[is_pq, 1] = q_true[is_pq]
                    x_base[is_pq, 2] = 1.0 
                    x_base[is_pq, 3] = 0.0 
                    mask[is_pq, 2:4] = 1.0

                    # C. 处理边特征与位置编码 (PE)
                    curr_edge_index = raw_edge[:, 0:2].T.long()
                    curr_edge_attr = raw_edge[:, 2:] 
                    
                    num_nodes = raw_node.shape[0]
                    
                    # 动态计算 PE
                    pe = compute_laplacian_pe(curr_edge_index, num_nodes, k=self.PE_DIM)
                    
                    x_final = torch.cat([x_base, pe], dim=-1)
                    mask_pe = torch.zeros(num_nodes, self.PE_DIM)
                    mask_final = torch.cat([mask, mask_pe], dim=-1)

                    data = Data(
                        x=x_final,
                        y=y,
                        bus_type=bus_type,
                        pred_mask=mask_final,
                        target_vm=vm_true,
                        edge_index=curr_edge_index,
                        edge_attr=curr_edge_attr,
                        label=torch.tensor([raw_label], dtype=torch.long) # [关键新增]
                    )
                    data_list.append(data)

                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]
                    
                all_case_data[split_idx].extend(data_list)

        for idx, case_data in enumerate(all_case_data):
            if len(case_data) > 0:
                data, slices = self.collate(case_data)
                print(f"Saving {len(case_data)} samples to {self.processed_paths[idx]}...")
                torch.save((data, slices), self.processed_paths[idx])
            else:
                print(f"Warning: Split {idx} is empty, skipping save.")
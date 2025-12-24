"""
这个版本是基于Ybus的数据处理的代码
输入
P,Q,e,f,Bii,Gii
"""
import os
from typing import Callable, Optional, List, Tuple, Union

import torch
import numpy as np
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset

def polar_to_rect(vm, va_degree):
    """辅助函数：极坐标转直角坐标"""
    va_rad = va_degree * np.pi / 180.0
    e = vm * np.cos(va_rad)
    f = vm * np.sin(va_rad)
    return e, f

# 反归一化函数
def denormalize(input, mean, std):
    return input*(std.to(input.device)+1e-7) + mean.to(input.device)

# 数据增强函数，对于data中的bus_type，它会进行随机的变化，从而进行数据增强，哪怕是有问题的，在本代码中没有被调用
def random_bus_type(data: Data) -> Data:
    " data.bus_type -> randomize "
    data.bus_type = torch.randint_like(data.bus_type, low=0, high=2)
    return data

#   PowflowData类的定义
class PowerFlowData(InMemoryDataset):   
    """
    加载并处理基于Ybus的潮流数据

    x(input): [P,Q,e,f,Gii,Bii]
    y(label): [P,Q,e,f]
    edge_attr:[Gij,Bij]
    其中ef不做归一化，PQBG都需要做归一化
    """
    # 期望的输入的数据的名字，其中会根据后面的系统的名字，进行相应的名字的拼接
    partial_file_names = [
        "edge_features.npy",
        "node_features.npy",
    ]
    # 用于将字符串用于映射
    split_order = {"train": 0,"val": 1,"test": 2}
    mixed_cases = ['118v2','14v2',]

    # 这是有关PowerFlowData的类型定义的初始化部分
    def __init__(self, 
                root: str, 
                case: str = '14', 
                split: Optional[List[float]] = None, 
                task: str = "train", 
                transform: Optional[Callable] = None, 
                pre_transform: Optional[Callable] = None, 
                pre_filter: Optional[Callable] = None,
                normalize=True,
                xymean=None,
                xystd=None,
                edgemean=None,
                edgestd=None):

        # 这一步是做参数的检查，保证数据被分成三类，并且是train，val，test中的一种
        assert len(split) == 3
        assert task in ["train", "val", "test"]

        self.normalize = normalize
        self.case = case  
        self.split = split
        self.task = task
        # 必须先调用父类初始化，这会触发 raw_file_names 和 processed_file_names 的检查
        super().__init__(root, transform, pre_transform, pre_filter) 
        self.mask = torch.tensor([])

        # 赋值归一化参数
        if xymean is not None and xystd is not None:
            self.xymean, self.xystd = xymean, xystd
            print('xymean, xystd assigned.')
        else:
            self.xymean, self.xystd = None, None

        if edgemean is not None and edgestd is not None:
            self.edgemean, self.edgestd = edgemean, edgestd
            print('edgemean, edgestd assigned.')
        else:
            self.edgemean, self.edgestd = None, None

        # 加载处理好的数据
        path = self.processed_paths[self.split_order[self.task]]
        print(f"Loading processed data from: {path}")
        loaded_data = torch.load(path, weights_only=False)
        self.data, self.slices = self._normalize_dataset(*loaded_data)

    def get_data_dimensions(self):
        # 返回: Node Input Dim, Node Output Dim, Edge Dim
        return self[0].x.shape[1], self[0].y.shape[1], self[0].edge_attr.shape[1]

    # 返归一化相关的内容
    def get_data_means_stds(self):
        assert self.normalize == True   
        return self.xymean[:1, :], self.xystd[:1, :], self.edgemean[:1, :], self.edgestd[:1, :]

    def _normalize_dataset(self, data, slices):
        if not self.normalize:
            return data, slices

        if self.xymean is None or self.xystd is None:
            # 这里的逻辑稍微复杂一点，因为 x 和 y 的维度不一样了
            # x: [P, Q, e, f, Gii, Bii] (6维)
            # y: [P, Q, e, f] (4维)
            # 1. 计算P Q e f的统计量
            y_stats = data.y
            mean_y = torch.mean(y_stats, dim=0, keepdim=True) # [1, 4]
            std_y = torch.std(y_stats, dim=0, keepdim=True)   # [1, 4]

            # [策略] 强制 e(2), f(3) 不归一化
            mean_y[:, 2] = 0.0
            mean_y[:, 3] = 0.0
            std_y[:, 2] = 1.0
            std_y[:, 3] = 1.0

            # 2. 计算 Gii, Bii 的统计量 (基于输入 x 的后两列)
            # x 的结构是 [P_in, Q_in, e_in, f_in, Gii, Bii]
            # 我们取最后两列
            gb_stats = data.x[:, 4:]
            mean_gb = torch.mean(gb_stats, dim=0, keepdim=True) # [1, 2]
            std_gb = torch.std(gb_stats, dim=0, keepdim=True)   # [1, 2]

            # 3. 拼接成完整的 xymean/xystd (6维)
            # 用于归一化 x: [P, Q, e, f, Gii, Bii]
            self.xymean = torch.cat([mean_y, mean_gb], dim=1)
            self.xystd = torch.cat([std_y, std_gb], dim=1)

            print(f"✅ 节点归一化参数已生成 (Dim=6):")
            print(f"   P/Q: Mean={mean_y[0,:2].numpy()}, Std={std_y[0,:2].numpy()}")
            print(f"   e/f: Mean={mean_y[0,2:].numpy()}, Std={std_y[0,2:].numpy()} (Should be 0/1)")
            print(f"   Gii/Bii: Mean={mean_gb[0].numpy()}, Std={std_gb[0].numpy()}")

        # 应用归一化 (x 使用 6维参数)
        data.x = (data.x - self.xymean) / (self.xystd + 1e-7)
        # 应用归一化 (y 使用前4维参数)
        data.y = (data.y - self.xymean[:, :4]) / (self.xystd[:, :4] + 1e-7)

        # --- 2. 边特征归一化 ---
        if self.edgemean is None or self.edgestd is None:
            mean = torch.mean(data.edge_attr, dim=0, keepdim=True)
            std = torch.std(data.edge_attr, dim=0, keepdim=True)
            self.edgemean, self.edgestd = mean, std
            
        data.edge_attr = (data.edge_attr - self.edgemean) / (self.edgestd + 1e-7)

        return data, slices
    
    # 这三个方法是InMemoryDataset要求的标准接口，去目录下面查找哪些文件
    @property
    def raw_file_names(self) -> List[str]:
        if self.case != 'mixed':
            return ["case"+f"{self.case}"+"_"+name for name in self.partial_file_names]
        else:
            return ["case"+f"{case}"+"_"+name for case in self.mixed_cases for name in self.partial_file_names]
    # 处理后的数据应该叫什么名字，并保存在指定的目录下面
    @property
    def processed_file_names(self) -> List[str]:
        return [
            "case"+f"{self.case}"+"_processed_train_Ybus.pt",
            "case"+f"{self.case}"+"_processed_val_Ybus.pt",
            "case"+f"{self.case}"+"_processed_test_Ybus.pt",
        ]
    
    # 返回数据集中的图样本的总数
    def len(self):
        return self.slices['x'].shape[0]-1

    # def get(self, idx: int) -> Data: # override
    #     return self.data[idx]
    
    # 先修改process函数
    def process(self):
        
        assert len(self.raw_paths) % 2 == 0
        raw_paths_per_case = [[self.raw_paths[i], self.raw_paths[i+1],] for i in range(0, len(self.raw_paths), 2)]
        all_case_data = [[],[],[]]

        for case, raw_paths in enumerate(raw_paths_per_case):
            print(f"Processing raw files:{raw_paths}")

            edge_features = torch.from_numpy(np.load(raw_paths[0])).float()
            node_features = torch.from_numpy(np.load(raw_paths[1])).float()

            # 检查node_feature是8列
            if node_features.shape[-1]!= 8:
                raise ValueError(f"Expected 8 node features (P,Q,Vm,Va,P_net,Q_net,Gii,Bii), got {node_features.shape[2]}")
            

            assert self.split is not None
            if self.split is not None:
                split_len = [int(len(node_features) * i) for i in self.split]

            split_edge_features = torch.split(edge_features, split_len, dim=0)
            split_node_features = torch.split(node_features, split_len, dim=0)

            for idx in range(len(split_edge_features)):
                # 原始数据
                # [index , type , vm , va , P , Q , Gii , Bii]
                raw_data = split_node_features[idx] # 形状为(N_sub , num_buses ,8)
                bus_type = raw_data[:,:,1].long()

                # 构建target_y
                # y = [P Q e f]
                vm_true = raw_data[:, :, 2]
                va_true = raw_data[:, :, 3]
                p_true = raw_data[:, :, 4]
                q_true = raw_data[:, :, 5]
                g_ii = raw_data[:, :, 6]
                b_ii = raw_data[:, :, 7]
                
                # 转换直角坐标
                va_rad = va_true * (torch.pi / 180.0)
                e_true = vm_true * torch.cos(va_rad)
                f_true = vm_true * torch.sin(va_rad)

                # 构建y
                y = torch.stack([p_true, q_true, e_true, f_true], dim=-1)

                # 构建 input X
                x = torch.zeros(y.shape[0], y.shape[1], 6)
                x[:, :, 4] = g_ii
                x[:, :, 5] = b_ii

                mask = torch.zeros_like(x)

                # 辅助索引
                is_slack = (bus_type == 0)
                is_pv = (bus_type == 1)
                is_pq = (bus_type == 2)

                # 1. Slack (Type 0)
                # 已知: e, f (保持真值)
                # 未知: P, Q (填0, mask=1)
                x[is_slack, 0] = 0.0 # P
                x[is_slack, 1] = 0.0 # Q
                x[is_slack, 2] = e_true[is_slack] # e (Known)
                x[is_slack, 3] = f_true[is_slack] # f (Known)
                mask[is_slack, 0] = 1.0
                mask[is_slack, 1] = 1.0

                # 2. PV (Type 1)
                # 已知: P, Vm
                # 未知: Q, e, f (e,f 受 Vm 约束，但也视为未知需要预测)
                x[is_pv, 0] = p_true[is_pv] # P (Known)
                x[is_pv, 1] = 0.0           # Q (Unknown)
                # e, f 初值猜测: 设相角为0 => e=Vm, f=0
                x[is_pv, 2] = vm_true[is_pv] 
                x[is_pv, 3] = 0.0
                
                mask[is_pv, 1] = 1.0 # Predict Q
                mask[is_pv, 2] = 1.0 # Predict e
                mask[is_pv, 3] = 1.0 # Predict f

                # 3. PQ (Type 2)
                # 已知: P, Q
                # 未知: e, f
                x[is_pq, 0] = p_true[is_pq] # P (Known)
                x[is_pq, 1] = q_true[is_pq] # Q (Known)
                # e, f 初值猜测: Flat start => e=1, f=0
                x[is_pq, 2] = 1.0
                x[is_pq, 3] = 0.0
                
                mask[is_pq, 2] = 1.0 # Predict e
                mask[is_pq, 3] = 1.0 # Predict f

                # 获取边的特征
                # 获取边特征 [Gij, Bij]
                # 注意：raw_edge_features 只有4列 [from, to, G, B]
                e_feat_raw = split_edge_features[idx]
                edge_index = e_feat_raw[:, :, 0:2].transpose(1, 2).long() # [B, 2, E]
                edge_attr = e_feat_raw[:, :, 2:] # [B, E, 2]

                # 构建数据集
                data_list = []
                for i in range(len(x)):
                    data = Data(
                        x=x[i],
                        y=y[i],
                        bus_type=bus_type[i],
                        pred_mask=mask[i],
                        target_vm=vm_true[i],
                        edge_index=edge_index[i],
                        edge_attr=edge_attr[i]
                    )
                    data_list.append(data)

                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]
                    
                all_case_data[idx].extend(data_list)

        for idx, case_data in enumerate(all_case_data):
            data, slices = self.collate(case_data)
            print(f"Saving processed data to {self.processed_paths[idx]}...")
            torch.save((data, slices), self.processed_paths[idx])


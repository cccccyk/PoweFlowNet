"""
之前BG版本的时候用于处理数据的代码
"""
import os
from typing import Callable, Optional, List, Tuple, Union

import torch
import numpy as np
import torch.utils.data as data
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix, dense_to_sparse
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid

# ok 也就是说用的是x文件的数据，其中导入的分别是编号，节点类型，电压幅值，电压相角，有功功率，无功功率
feature_names_from_files = [
    'index',                # starting from 0 
    'type',                 # 
    'e',    # 
    'f', # 
    'Pd',                   # 
    'Qd',                   # 
]

# ok 对应的是使用edge_features的文件，其中该文件的一个向量有7位，其中只有前4位是有用的，分别代表from_bus,tu_bus,r,x
edge_feature_names_from_files = [
    'from_bus',             # 
    'to_bus',               #
    'g pu',                 # 
    'b pu',                 # 
]
def polar_to_rect(vm, va_degree):
    """辅助函数：极坐标转直角坐标"""
    va_rad = va_degree * np.pi / 180.0
    e = vm * np.cos(va_rad)
    f = vm * np.sin(va_rad)
    return e, f

# 数据增强函数，对于data中的bus_type，它会进行随机的变化，从而进行数据增强，哪怕是有问题的
def random_bus_type(data: Data) -> Data:
    " data.bus_type -> randomize "
    data.bus_type = torch.randint_like(data.bus_type, low=0, high=2)
    
    return data
    
# 反归一化函数
def denormalize(input, mean, std):
    return input*(std.to(input.device)+1e-7) + mean.to(input.device)

#   PowflowData类的定义
class PowerFlowData(InMemoryDataset):   # 首先它集成来自于Geometric的InMemorydataset的类型，将所有的.npy格式的数据转换成.pt格式的数据，加速内存的访问
    """PowerFlowData(InMemoryDataset)

    Parameters:
        root (str, optional) – Root directory where the dataset should be saved. (optional: None)
        pre_filter (callable)- A function 

    Comments:
        we actually do not need adjacency matrix, since we can use edge_index to represent the graph from `edge_features`

    Returns:
        class instance of PowerFlowData
    """
    # 期望的输入的数据的名字，其中会根据后面的系统的名字，进行相应的名字的拼接
    partial_file_names = [
        "edge_features.npy",
        "node_features.npy",
    ]
    # 用于将字符串用于映射
    split_order = {
        "train": 0,
        "val": 1,
        "test": 2
    }
    mixed_cases = [
        '118v2',
        '14v2',
    ]
    # 这个是对应每个节点的掩码的内容，我的每个节点的信息是全的，然后不同节点的需要掩码的内容是不一样的，比如PQ节点，PQ一直，需要预测其他两个维度的数据
    slack_mask = (0, 0, 1, 1) # 1 = need to predict, 0 = no need to predict
    gen_mask = (0, 1, 0, 1) 
    load_mask = (1, 1, 0, 0)
    bus_type_mask = (slack_mask, gen_mask, load_mask)

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
        self.case = case  # THIS MUST BE EXECUTED BEFORE super().__init__() since it is used in raw_file_names and processed_file_names
        self.split = split
        self.task = task
        super().__init__(root, transform, pre_transform, pre_filter) # self.process runs here 从data/processed文件下面去查找对应的处理后的.pt文件，如果该文件存在，什么都不做，直接跳过，如果改文件不存在，把对于的.npy文件去读取，并且将结果保存为.pt文件
        self.mask = torch.tensor([])
        # assign mean,std if specified 这几句是处理归一化相关的内容
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


        loaded_data = torch.load(self.processed_paths[self.split_order[self.task]], weights_only=False)
        self.data, self.slices = self._normalize_dataset(*loaded_data)

    # 告诉main.py我数据长什么样，方便main.py构建出一个输入层和输出层大小正确的神经网络
    # 返回的内容包括节点特征维度，样本的label特征维度，样本的边的特征的维度
    def get_data_dimensions(self):
        return self[0].x.shape[1], self[0].y.shape[1], self[0].edge_attr.shape[1]

    # 返归一化相关的内容
    def get_data_means_stds(self):
        assert self.normalize == True   # 安全检查，如果一开始就没归一化，那么归一化也没意义
        return self.xymean[:1, :], self.xystd[:1, :], self.edgemean[:1, :], self.edgestd[:1, :]

    def _normalize_dataset(self, data, slices):
        if not self.normalize:
            return data, slices

        if self.xymean is None or self.xystd is None:
            # data.y的结构是[P,Q,e,f]
            xy = data.y 
            mean = torch.mean(xy, dim=0, keepdim=True)
            std = torch.std(xy, dim=0, keepdim=True)
            
            mean[:,2] = 0.0
            mean[:,3] = 0.0
            std[:,2] = 1.0
            std[:,3] = 1.0

            self.xymean, self.xystd = mean, std
            print(f"✅ 已重置 e,f 的归一化参数: Mean={self.xymean}, Std={self.xystd}")

        # 应用归一化
        data.x = (data.x - self.xymean) / (self.xystd + 1e-7)
        data.y = (data.y - self.xymean) / (self.xystd + 1e-7)

        # 这一步对边的数据做归一化
        if self.edgemean is None or self.edgestd is None:
            mean = torch.mean(data.edge_attr, dim=0, keepdim=True)
            std = torch.std(data.edge_attr, dim=0, keepdim=True)
            self.edgemean, self.edgestd = mean, std
        data.edge_attr = (data.edge_attr - self.edgemean) / (self.edgestd + 0.0000001)

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
            "case"+f"{self.case}"+"_processed_train_v4.pt",
            "case"+f"{self.case}"+"_processed_val_v4.pt",
            "case"+f"{self.case}"+"_processed_test_v4.pt",
        ]
    
    # 返回数据集中的图样本的总数
    def len(self):
        return self.slices['x'].shape[0]-1

    # def get(self, idx: int) -> Data: # override
    #     return self.data[idx]

    def process(self):
        # then use from_scipy_sparse_matrix()
        assert len(self.raw_paths) % 2 == 0
        # 把同一个节点系统的数据打包成列表
        raw_paths_per_case = [[self.raw_paths[i], self.raw_paths[i+1],] for i in range(0, len(self.raw_paths), 2)]
        # 创建三个空列表，用于存储训练、验证、测试集
        all_case_data = [[],[],[]]
        # 加载.npy文件
        for case, raw_paths in enumerate(raw_paths_per_case):
            # process multiple cases (if specified) e.g. cases = [14, 118]
            edge_features = torch.from_numpy(np.load(raw_paths[0])).float()
            node_features = torch.from_numpy(np.load(raw_paths[1])).float()

            assert self.split is not None
            if self.split is not None:
                split_len = [int(len(node_features) * i) for i in self.split]

            # 划分数据集
            split_edge_features = torch.split(edge_features, split_len, dim=0)
            split_node_features = torch.split(node_features, split_len, dim=0)

            # 循环创建各个集合
            for idx in range(len(split_edge_features)):

                
                # 把后四列输入给y
                raw_y = split_node_features[idx][:, :, 2:] # shape (N, n_ndoes, 4); Vm, Va, P, Q
                # 这一步是指把节点类型给选出来，后面的掩码要用到, 后面的话，要根据bus_type来做相应的掩码的工作
                bus_type = split_node_features[idx][:, :, 1].type(torch.long) # shape (N, n_nodes)
                
                # 构建 true_y
                vm_true = raw_y[:, :, 0]
                va_true = raw_y[:, :, 1]
                p_true = raw_y[:, :, 2]
                q_true = raw_y[:, :, 3]

                va_rad = va_true * (torch.pi / 180.0)
                e_true = vm_true * torch.cos(va_rad)
                f_true = vm_true * torch.sin(va_rad)

                # 构建新的[P,Q,e,f]形式的y
                y = torch.stack([p_true, q_true, e_true, f_true], dim=-1)
                 # [新增] 哨兵打印！
                if idx == 0:
                    print("\n" + "!"*50)
                    print("DEBUG: 正在执行 PowerFlowData.process ...")
                    print(f"DEBUG: 正在写入 target_vm, vm_true shape: {vm_true.shape}")
                    print("!"*50 + "\n")
                
                x = y.clone()
                # 创建掩码，mask:1代表是自由变量，0代表是已知的
                mask = torch.zeros_like(x)

                # 用布尔变量来进行相应的掩码内容的填充
                is_slack = (bus_type == 0)
                is_pv = (bus_type == 1)
                is_pq = (bus_type == 2)
                
                # 平衡节点节点,P,Q未知，但是后面的知道
                x[is_slack,0] = 0.0
                x[is_slack,1] = 0.0
                mask[is_slack,0] = 1.0
                mask[is_slack,1] = 1.0

                # PV节点，P和V已知
                x[is_pv,2] = vm_true[is_pv]
                x[is_pv,3] = 0.0
                x[is_pv,1] = 0.0
                mask[is_pv,1] = 1.0
                mask[is_pv,2] = 1.0
                mask[is_pv,3] = 1.0

                # PV节点，相角和Q是未知的，
                x[is_pq,2] = 1.0
                x[is_pq,3] = 0.0
                mask[is_pq,2] = 1.0
                mask[is_pq,3] = 1.0

                # 获取边的特征
                e_feat = split_edge_features[idx]

                #转换为Data列表
                data_list = [
                    Data(
                        x=x[i],
                        y=y[i],
                        bus_type=bus_type[i],
                        pred_mask=mask[i],
                        target_vm=vm_true[i],
                        edge_index=e_feat[i,:,0:2].T.to(torch.long),
                        edge_attr=e_feat[i,:,2:],
                    )for i in range(len(x))
                ]

                if self.pre_filter is not None:  # filter out some data
                    data_list = [data for data in data_list if self.pre_filter(data)]

                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]
                    
                all_case_data[idx].extend(data_list)

        for idx, case_data in enumerate(all_case_data):
            data, slices = self.collate(case_data)
            torch.save((data, slices), self.processed_paths[idx])


def main():
    try:
        # shape = (N, n_edges, 7)       (from, to, ...)
        edge_features = np.load("data/raw/case118v3_edge_features.npy")
        # shape = (N, n_nodes, n_nodes)
        adj_matrix = np.load("data/raw/case118v3_adjacency_matrix.npy")
        # shape = (N, n_nodes, 9)
        node_features_x = np.load("data/raw/case118v3_node_features_x.npy")
        # shape = (N, n_nodes, 8)
        node_features_y = np.load("data/raw/case118v3_node_features_y.npy")
    except FileNotFoundError:
        print("File not found.")

    print(f"edge_features.shape = {edge_features.shape}")
    print(f"adj_matrix.shape = {adj_matrix.shape}")
    print(f"node_features_x.shape = {node_features_x.shape}")
    print(f"node_features_y.shape = {node_features_y.shape}")

    trainset = PowerFlowData(root="data", case='118',
                             split=[.5, .2, .3], task="train")
    train_loader = torch_geometric.loader.DataLoader(
        trainset, batch_size=12, shuffle=True)
    print(len(trainset))
    print(trainset[0])
    print(next(iter(train_loader)))
    pass


if __name__ == "__main__":
    main()
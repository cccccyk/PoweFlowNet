"""
this file defines the class of PowerFlowData, which is used to load the data of Power Flow
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
    'voltage magnitude',    # 
    'voltage angle degree', # 
    'Pd',                   # 
    'Qd',                   # 
    # 'Gs',                   # - equivalent to Pd, Qd                    8,
    # 'Bs',                   # - equivalent to Pd, Qd                    9,
    # 'Pg'                    # - removed
]

# ok 对应的是使用edge_features的文件，其中该文件的一个向量有7位，其中只有前4位是有用的，分别代表from_bus,tu_bus,r,x
edge_feature_names_from_files = [
    'from_bus',             # 
    'to_bus',               #
    'r pu',                 # 
    'x pu',                 # 
]

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

        # 这一步把数据加载到内存，一个实例的数据就准备好了
        # self.data, self.slices = self._normalize_dataset(
        #     *torch.load(self.processed_paths[self.split_order[self.task]]))  # necessary, do not forget!
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

    # 这一步是执行归一化
    def _normalize_dataset(self, data, slices, ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 如果在创建实例的时候设置了=False，这个函数会返回未修改的原始数据
        if not self.normalize:
            # TODO an actual mask, perhaps never necessary though
            return data, slices

        # 这一段是在做节点数据的归一化
        # normalizing
        # for node attributes
        if self.xymean is None or self.xystd is None:   # 只有当正在初始化trainset的时候，才会出现xymean是None的情况，
            xy = data.y # name 'xy' is from legacy. Shape (N, 4)
            mean = torch.mean(xy, dim=0, keepdim=True)
            std = torch.std(xy, dim=0, keepdim=True)
            # 在构建训练集的时候，保存下来训练集的归一化的参数，包括 mean 和 std
            self.xymean, self.xystd = mean, std
            # + 0.0000001 to avoid NaN's because of division by zero
        # 这一步是对节点的数据，包括输入的数据，以及label做归一化
        data.x = (data.x - self.xymean) / (self.xystd + 0.0000001)
        data.y = (data.y - self.xymean) / (self.xystd + 0.0000001)
        # for edge attributes

        # 这一步对边的数据做归一化
        if self.edgemean is None or self.edgestd is None:
            mean = torch.mean(data.edge_attr, dim=0, keepdim=True)
            std = torch.std(data.edge_attr, dim=0, keepdim=True)
            self.edgemean, self.edgestd = mean, std
        data.edge_attr = (data.edge_attr - self.edgemean) / (self.edgestd + 0.0000001)

        # deprecated: adding the mask
        # where x and y are unequal, the network must predict
        # 1 where value changed, 0 where it did not change
        # unequal = (data.x[:, 4:] != data.y).float()
        # data.prediction_mask = unequal
        # data.x = torch.concat([data.x, unequal], dim=1)

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
            "case"+f"{self.case}"+"_processed_train.pt",
            "case"+f"{self.case}"+"_processed_val.pt",
            "case"+f"{self.case}"+"_processed_test.pt",
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
                # shape of element in split_xx: [N, n_edges/n_nodes, n_features]
                # for each case, process train, val, test split
                # y是节点的所有的完整的特征
                y = split_node_features[idx][:, :, 2:] # shape (N, n_ndoes, 4); Vm, Va, P, Q
                # 这一步是对输入的数据做相应的掩码，生成输入的x向量
                bus_type = split_node_features[idx][:, :, 1].type(torch.long) # shape (N, n_nodes)
                bus_type_mask = torch.tensor(self.bus_type_mask)[bus_type] # shape (N, n_nodes, 4)
                x = y.clone()*(1.-bus_type_mask) # shape (N, n_nodes, 4)
                e = split_edge_features[idx] # shape (N, n_edges, 4)
                # 遍历split后的每一个样本，转换为data对象
                if idx == 0:
                    print("\n" + "!"*50)
                    print("DEBUG: 正在执行 PowerFlowData.process ...")
                    print()
                    print("!"*50 + "\n")
                data_list = [
                    Data(
                        x=x[i],
                        y=y[i],
                        bus_type=bus_type[i],
                        pred_mask=bus_type_mask[i],
                        edge_index=e[i, :, 0:2].T.to(torch.long),
                        edge_attr=e[i, :, 2:],
                    ) for i in range(len(x))
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
        edge_features = np.load("data/raw/case118v2_edge_features.npy")
        # shape = (N, n_nodes, n_nodes)
        adj_matrix = np.load("data/raw/case118v2_adjacency_matrix.npy")
        # shape = (N, n_nodes, 9)
        node_features_x = np.load("data/raw/case118v2_node_features_x.npy")
        # shape = (N, n_nodes, 8)
        node_features_y = np.load("data/raw/case118v2_node_features_y.npy")
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
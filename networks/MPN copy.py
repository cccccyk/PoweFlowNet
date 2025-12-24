import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, LayerNorm
from torch_geometric.utils import to_dense_batch

class MaskEmbdMultiMPN_Transformer(nn.Module):
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # 将data从输入的维度，从6维度，扩充到hidden_dim的维度（所以其实我真的觉得hidden_num在64维度其实就够了，上128的话训练的资源占比太大了）
        self.input_encoder = nn.Linear(nfeature_dim, hidden_dim)
        # 将mask从输入的维度，扩充到hidden_dim的维度，扩充相应的信息
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 用于NNConv的权重学习，将边的输入扩充为一个hidden*hidden的矩阵，用于权重的学习……但是我感觉这个太占显存了
        def create_edge_net(in_c, out_c):
            return nn.Sequential(
                nn.Linear(efeature_dim, hidden_dim), 
                nn.GELU(),
                nn.Linear(hidden_dim, in_c * out_c)
            )

        # convs是保存了图神经网络的层
        for _ in range(n_gnn_layers):
            self.convs.append(
                NNConv(hidden_dim, hidden_dim, 
                       create_edge_net(hidden_dim, hidden_dim), 
                       aggr='add')
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 图神经后面接了一个transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4,            # 4头注意力
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True 
        )

        # 这里定义了transformer的num_layers=1层
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1) # 1层通常足够，也可以设为2
        
        self.global_norm = nn.LayerNorm(hidden_dim)

        # 这个是一个预测头
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)

    def is_directed(self, edge_index):
        if edge_index.shape[1] == 0: return False
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack([edge_index[1,:], edge_index[0,:]], dim=0)
            edge_index = torch.cat([edge_index, edge_index_dup], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        return edge_index, edge_attr
    
    def forward(self, data):
        # 读取数据
        x = data.x
        # 读取mask数据
        mask = data.pred_mask.float()
        # 读取边的信息
        edge_index = data.edge_index
        edge_features = data.edge_attr
        batch = data.batch # 获取 batch 索引 [0, 0, ..., 1, 1, ...]
        
        # 这个的意思是，输入的数据经过掩码和扩容的头，对数据进行了融合和丰富
        h = self.input_encoder(x) + self.mask_embd(mask)
        # 这里对于边信息进行了处理
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        # 在这里进行了图神经网络的操作
        for i, conv in enumerate(self.convs):
            identity = h
            h = conv(h, edge_index, edge_features)
            h = self.norms[i](h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = h + identity # Residual
        # 这个是把图-序列转换，让GNN的输出能够被transformer处理
        h_dense, mask_batch = to_dense_batch(h, batch)
        
        h_global = self.transformer(h_dense, src_key_padding_mask=(~mask_batch))
        # 将序列重新转换回图
        h_global_flat = h_global[mask_batch]
        
        h = h + h_global_flat
        h = self.global_norm(h)

        out = self.output_decoder(h)

        return out
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, LayerNorm
from torch_geometric.utils import to_dense_batch

class GNN_Transformer_Block(nn.Module):
    """
    一个混合块：包含若干层局部 GNN 和 一层全局 Transformer
    """
    def __init__(self, hidden_dim, efeature_dim, n_gnn_layers=2, nhead=4, dropout_rate=0.0):
        super().__init__()
        
        # --- 局部 GNN 部分 ---
        self.convs = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        
        def create_edge_net(in_c, out_c):
            return nn.Sequential(
                nn.Linear(efeature_dim, hidden_dim), 
                nn.GELU(),
                nn.Linear(hidden_dim, in_c * out_c)
            )

        for _ in range(n_gnn_layers):
            self.convs.append(
                NNConv(hidden_dim, hidden_dim, create_edge_net(hidden_dim, hidden_dim), aggr='add')
            )
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))

        # --- 全局 Transformer 部分 ---
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.trans_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr, batch):
        h = x
        
        # 1. 局部物理细化 (GNN)
        for i, conv in enumerate(self.convs):
            identity = h
            h = conv(h, edge_index, edge_attr)
            h = self.gnn_norms[i](h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = h + identity # 局部残差

        # 2. 全局信息同步 (Transformer)
        h_dense, mask_batch = to_dense_batch(h, batch)
        # 注意：Transformer 内部自带残差，所以我们直接运行
        h_global = self.transformer_layer(h_dense, src_key_padding_mask=(~mask_batch))
        h_global_flat = h_global[mask_batch]
        
        # 3. 混合残差：将全局信息融合回主干
        h = h + h_global_flat
        h = self.trans_norm(h)
        
        return h

class MaskEmbdMultiMPN_Iterative(nn.Module):
    """
    [迭代式混合架构]
    通过多次循环 "局部-全局" 交互来精细化潮流解。
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        # 这里的 n_gnn_layers 我们重新定义为：循环的次数 (Blocks)
        # 比如 n_gnn_layers=3, 代表运行 3 次 [GNNs + Transformer]
        self.num_iterations = 2
        
        self.hidden_dim = hidden_dim
        
        # 1. Input Projector
        self.input_encoder = nn.Linear(nfeature_dim, hidden_dim)
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. 迭代混合层
        self.iterative_blocks = nn.ModuleList()
        for _ in range(self.num_iterations):
            self.iterative_blocks.append(
                GNN_Transformer_Block(
                    hidden_dim=hidden_dim, 
                    efeature_dim=efeature_dim, 
                    n_gnn_layers=2, # 每个 block 内部跑 2 层 GNN
                    nhead=4,
                    dropout_rate=dropout_rate
                )
            )

        # 3. Output Decoder
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def is_directed(self, edge_index):
        if edge_index.shape[1] == 0: return False
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack([edge_index[1,:], edge_index[0,:]], dim=0)
            edge_index = torch.cat([edge_index, edge_index_dup], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        return edge_index, edge_attr

    def forward(self, data):
        x, mask, edge_index, edge_attr, batch = \
            data.x, data.pred_mask.float(), data.edge_index, data.edge_attr, data.batch
        
        # 1. Embedding
        h = self.input_encoder(x) + self.mask_embd(mask)
        edge_index, edge_attr = self.undirect_graph(edge_index, edge_attr)

        # 2. 迭代细化 (循环逻辑)
        # 每一轮 block 都会让 h 变得更精准
        for block in self.iterative_blocks:
            h = block(h, edge_index, edge_attr, batch)

        # 3. Decoder
        out = self.output_decoder(h)
        return out
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, LayerNorm
from torch_geometric.utils import to_dense_batch

# 1. 基础 GNN 算子 (InteractionLayer)
class InteractionLayer(MessagePassing):
    """
    MLP-based GNN layer, learns non-linear interactions between (xi, xj, edge).
    Memory efficient (O(H)).
    """
    def __init__(self, hidden_dim, edge_dim, dropout_rate=0.0):
        super().__init__(aggr='add')
        
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(tmp)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(tmp)
    
    def forward(self, x, edge_index, edge_attr):
        # 注意：这里我们不加 Residual，因为 GPSLayer 外部会统一加
        x_norm = self.norm(x)
        out = self.propagate(edge_index, x=x_norm, edge_attr=edge_attr)
        return self.dropout(out)

# 2. 混合层 (PhysicsGPSLayer)
class PhysicsGPSLayer(nn.Module):
    """
    GraphGPS-style layer: Parallel GNN + Transformer
    H' = H + GNN(H) + Attn(H) + FFN(H)
    """
    def __init__(self, hidden_dim, edge_dim, nhead=4, dropout=0.0):
        super().__init__()
        
        # Local Physics (GNN)
        self.local_gnn = InteractionLayer(hidden_dim, edge_dim, dropout)
        
        # Global Context (Transformer)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        h = x
        
        # --- A. Parallel Processing (Local + Global) ---
        # 1. Local GNN
        h_local = self.local_gnn(h, edge_index, edge_attr)
        
        # 2. Global Attention
        h_dense, mask_batch = to_dense_batch(h, batch)
        # mask_batch: True for valid nodes, False for padding
        # key_padding_mask: True for ignore (so we pass ~mask_batch)
        h_attn_dense, _ = self.self_attn(h_dense, h_dense, h_dense, key_padding_mask=(~mask_batch))
        h_attn = h_attn_dense[mask_batch] # Flatten back
        h_attn = self.dropout(h_attn)
        
        # 3. First Residual Fusion
        # h = h + Local + Global
        h = h + h_local + h_attn
        h = self.norm1(h)
        
        # --- B. FFN ---
        h = h + self.ffn(h)
        h = self.norm2(h)
        
        return h

# 3. 主模型 (MaskEmbdMultiMPN_GPS)
class MaskEmbdMultiMPN_GPS(nn.Module):
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, 
                 n_gnn_layers=4, nhead=4, dropout_rate=0.0, **kwargs):
        super().__init__()
        
        # 1. Embedding
        # 注意: nfeature_dim 自动包含了 PE 的维度 (6 + 8 = 14)
        self.input_encoder = nn.Linear(nfeature_dim, hidden_dim)
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. GPS Layers
        self.layers = nn.ModuleList()
        for _ in range(n_gnn_layers):
            self.layers.append(
                PhysicsGPSLayer(hidden_dim, efeature_dim, nhead, dropout_rate)
            )
            
        # 3. Decoder (双头输出 e, f 效果更好，也可以用单头)
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # [修改] 删除 self.output_decoder，改为两个独立的头
        self.head_e = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # 只输出 e
        )
        
        self.head_f = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # 只输出 f
        )


    def is_directed(self, edge_index):
        if edge_index.shape[1] == 0: return False
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack([edge_index[1,:], edge_index[0,:]], dim=0)
            edge_index = torch.cat([edge_index, edge_index_dup], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        return edge_index, edge_attr

    def forward(self, data):
        # x 包含了 PE，维度是 14
        x, mask = data.x, data.pred_mask.float()
        edge_index, edge_attr = data.edge_index, data.edge_attr
        batch = data.batch
        
        # Embedding
        h = self.input_encoder(x) + self.mask_embd(mask)
        edge_index, edge_attr = self.undirect_graph(edge_index, edge_attr)
        
        # GPS Layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, batch)
            
        # Output
        out = self.output_decoder(h)

        # [修改] 分别预测
        # e = self.head_e(h)
        # f = self.head_f(h)

        # 拼接回 [N, 2]
        # out = torch.cat([e, f], dim=-1)
        return out
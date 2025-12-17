import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, TAGConv, GCNConv, ChebConv ,GATConv ,NNConv ,GATv2Conv ,GINEConv, LayerNorm, GENConv ,LayerNorm
from torch_geometric.utils import degree ,softmax
import torch.nn.functional as F

# 用于将边的信息进行传递，具体怎么用我还没理解，但是他是基于PyG里面的一个比较重要的类的，即messagepassing
class EdgeAggregation(MessagePassing):
    """MessagePassing for aggregating edge features
    """
    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        super().__init__(aggr='add')
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim

        # 这里是一个MLP模型，将节点*2+边的维度扩展到一个output_dim，这个步骤应该是要把边的信息学习进去
        # self.linear = nn.Linear(nfeature_dim, output_dim) 
        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim*2 + efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def message(self, x_i, x_j, edge_attr):
        # 这一段是把节点，节点，边的向量拼成了一个长条的向量
        '''
        x_j:        shape (N, nfeature_dim,)
        edge_attr:  shape (N, efeature_dim,)
        '''
        return self.edge_aggr(torch.cat([x_i, x_j, edge_attr], dim=-1)) # PNAConv style
    
    def forward(self, x, edge_index, edge_attr):
        '''
        input:
            x:          shape (N, num_nodes, nfeature_dim,)
            edge_attr:  shape (N, num_edges, efeature_dim,)
            
        output:
            out:        shape (N, num_nodes, output_dim,)
        '''
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # no self loop because NO EDGE ATTR FOR SELF LOOP
        
        # Step 2: Calculate the degree of each node.计算各个节点的度，未来防止度很高的节点，在聚合时因为接受了太多消息导致特征爆炸
        row, col = edge_index   # 获取图的连接关系

        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 
        
        # Step 3: Feature transformation. 
        # x = self.linear(x) # no feature transformation
        
        # Step 4: Propagation，OK 这是一个继承自父类propagate类的函数
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)
        #   no bias here
        # 最终返回的是一个新的节点的特征矩阵
        return out

# 模拟母线的工作，母线不仅仅只和邻居交谈，相反，母线的工作是感知所有节点的功率缺口，然后来平衡整个系统
# 但是似乎在实际的使用中并没有用到
class SlackAggregation(MessagePassing):
    """
    Edge aggregation for slack bus
    
    """
    def __init__(self, nfeature_dim, hidden_dim, flow='to_slack'):
        assert flow in ['to_slack', 'from_slack']
        super().__init__(aggr='mean',
                         flow='target_to_source' if flow=='to_slack' else 'source_to_target')
        self.nfeature_dim = nfeature_dim

        # self.linear = nn.Linear(nfeature_dim, output_dim) 
        self.mlp = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nfeature_dim)
        )
        
    def message(self, x_j):
        '''
        x_j:        shape (N, nfeature_dim,)
        '''
        return self.mlp(x_j)
    
    def update(self, aggregated):
        return aggregated
    
    # 创建一个新的虚拟的图，该图会忽略电网的真实物理拓扑
    def recreate_slack_graph(self, bus_type, batch):
        """
        bus_type: (N,) {0,1,2}
        batch: (N,) [0,0,0,...,1,1,1,.,,,]
        """
        num_nodes = len(bus_type)
        slack_mask = bus_type == 0 # shape (N,)
        slack_indices = slack_mask.nonzero(as_tuple=False).squeeze()
        batch_indices_of_slack = batch[slack_indices]
        
        valid_connections = batch_indices_of_slack[:, None] == batch[None, :]
            # shape (num_slack, N)
        from_nodes = slack_indices[:, None].expand(-1, num_nodes)[valid_connections]
        to_nodes = torch.arange(num_nodes, device=from_nodes.device)[None, :].expand(slack_indices.size(0), -1)[valid_connections]
        
        # filter out self connections
        not_self_connections = from_nodes != to_nodes
        from_nodes = from_nodes[not_self_connections]
        to_nodes = to_nodes[not_self_connections]
        
        slack_edge_index = torch.stack([from_nodes, to_nodes], dim=0) # shape (2, -1)
        
        return slack_edge_index
    
    def forward(self, x, bus_type, batch):
        '''
        input:
            x:          shape (num_nodes, nfeature_dim,)
            bus_type:   shape (num_nodes,)
            batch:      shape (num_nodes,)
            
        process:
            PV,PQ nodes ---(info)---> slack
            
        output:
            x':        shape num_nodes, nfeature_dim,)
        '''
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # no self loop because NO EDGE ATTR FOR SELF LOOP
        
        # Step 2: Calculate the degree of each node.
        slack_edge_index = self.recreate_slack_graph(bus_type, batch)
        row, col = slack_edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 
        
        # Step 3: Feature transformation. 
        # x = self.linear(x) # no feature transformation
        
        # Step 4: Propagation
        out = self.propagate(x=x, edge_index=slack_edge_index, norm=norm)
        #   no bias here
        
        return out
    
# MPN网络，混合型的图神经网络？包括一个消息传递阶段，以及一个GCN阶段
class MPN(nn.Module):
    """Wrapped Message Passing Network
        - One-time Message Passing to aggregate edge features into node features
        - Multiple Conv layers
    """
    # 这些定义相关的数值，会在第一次调用定义这个函数的时候用，具体在train函数里面
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)    # 创建有关边的内容的信息
        self.convs = nn.ModuleList()    #这里的意思是创建一个空的层列表，用TAG来进行相应的填充，其中第一层和中间层是hidden_dim-hidden_dim，最后的输出层是不一样的

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))

    def is_directed(self, edge_index):  
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    # 如果图是有向的，那么会自动的添加反向的边，反正就是无向的图训练起来效果会比较好
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    # 
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        x = self.edge_aggr(x, edge_index, edge_features)    # 消息传递
        for i in range(len(self.convs)-1):
            # x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        return x
    
class SkipMPN(nn.Module):
    """Wrapped Message Passing Network
        - * Added skip connection
        - One-time Message Passing to aggregate edge features into node features
        - Multiple Conv layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        self.convs = nn.ModuleList()

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        x = self.edge_aggr(x, edge_index, edge_features)
        for i in range(len(self.convs)-1):
            # x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        # skip connection
        x = input_x + x
        
        return x
    
class MaskEmbdMPN(nn.Module):
    """Wrapped Message Passing Network
        - * Added embedding for mask
        - One-time Message Passing to aggregate edge features into node features
        - Multiple Conv layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        self.convs = nn.ModuleList()

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nfeature_dim)
        )

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        
        x = self.mask_embd(mask) + x
        
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        x = self.edge_aggr(x, edge_index, edge_features)
        for i in range(len(self.convs)-1):
            # x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        return x

class MultiMPN(nn.Module):
    """Wrapped Message Passing Network
        - Multi-step mixed MP+Conv
        - No convolution layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        # self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        # self.convs = nn.ModuleList()
        self.layers = nn.ModuleList()

        if n_gnn_layers == 1:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        # self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, output_dim))

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        for i in range(len(self.layers)-1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)
        
        return x

class MaskEmbdMultiMPN(nn.Module):
    """Wrapped Message Passing Network
        - Mask Embedding
        - Multi-step mixed MP+Conv
        - No convolution layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        # self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        # self.convs = nn.ModuleList()
        self.layers = nn.ModuleList()

        if n_gnn_layers == 1:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
            
        else:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))
            

        # 中间层的时候，节点的信息已经不是4维度的，而是hidden_dim维度了，直接进行循环就行
        for l in range(n_gnn_layers-2):
            self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        # self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        # self.slack_aggr = SlackAggregation(hidden_dim, hidden_dim, 'to_slack')
        # self.slack_propagate = SlackAggregation(hidden_dim, hidden_dim, 'from_slack')
        self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, output_dim))
        
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nfeature_dim)
        )
        self.dropout = nn.Dropout(self.dropout_rate, inplace=False)

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        if edge_index.shape[1] == 0:
            # no edge at all, only single nodes. automatically undirected
            return False
        # next line: if there is the reverse of the first edge does not exist, then directed. 
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        # assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        # x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        assert data.x.shape[-1] == 4
        x = data.x # (N, 4)
        input_x = x # problem if there is inplace operation on x, so pay attention
        bus_type = data.bus_type.long()
        batch = data.batch
        mask = data.pred_mask.float() # indicating which features to predict (==1)
        edge_index = data.edge_index
        edge_features = data.edge_attr
                
        x = self.mask_embd(mask) + x
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        for i in range(len(self.layers)-1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = self.dropout(x)
            x = nn.ReLU()(x)
        
        # slack aggr
        # x = x + self.slack_aggr(x, bus_type=bus_type, batch=batch)
        # x = x + self.slack_propagate(x, bus_type=bus_type, batch=batch)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)
        
        return x
    
class MaskEmbdMultiMPN_NoMP(nn.Module):
    """Wrapped Message Passing Network
        - Mask Embedding
        - Multi-step mixed MP+Conv
        - No convolution layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        # self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        # self.convs = nn.ModuleList()
        self.layers = nn.ModuleList()

        if n_gnn_layers == 1:
            # self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
            
        else:
            # self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        for l in range(n_gnn_layers-2):
            # self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
            
        # self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, output_dim))
        
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nfeature_dim)
        )

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
                
        x = self.mask_embd(mask) + x
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        for i in range(len(self.layers)-1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)
        
        return x

class WrappedMultiConv(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels, K, **kwargs):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(ChebConv(in_channels, out_channels, K, normalization=None, **kwargs))
        
    def forward(self, x, edge_index_list, edge_weights_list):
        out = 0.
        for i in range(self.num_convs):
            edge_index = edge_index_list[i]
            edge_weights = edge_weights_list[i]
            out += self.convs[i](x, edge_index, edge_weights)

        return out

class MultiConvNet(nn.Module):
    """Wrapped Message Passing Network
        - No Message Passing to aggregate edge features into node features
        - Multi-level parallel Conv layers for different edge features
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        assert efeature_dim == 5
        efeature_dim = efeature_dim - 3 # should be 2, only these two are meaningful
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_trans = nn.Sequential(
            nn.Linear(efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, efeature_dim)
        )
        self.convs = nn.ModuleList()

        if n_gnn_layers == 1:
            self.convs.append(WrappedMultiConv(efeature_dim, nfeature_dim, output_dim, K=K))
        else:
            self.convs.append(WrappedMultiConv(efeature_dim, nfeature_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(WrappedMultiConv(efeature_dim, hidden_dim, hidden_dim, K=K))
            
        self.convs.append(WrappedMultiConv(efeature_dim, hidden_dim, output_dim, K=K))

    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack(
                [edge_index[1,:], edge_index[0,:]],
                dim = 0
            )   # (2, E)
            edge_index = torch.cat(
                [edge_index, edge_index_dup],
                dim = 1
            )   # (2, 2*E)
            edge_attr = torch.cat(
                [edge_attr, edge_attr],
                dim = 0
            )   # (2*E, fe)
            
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying. 
        input_x = x # problem if there is inplace operation on x, so pay attention
        mask = data.x[:, -self.nfeature_dim:]# last few dimensions: mask.
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)
        
        edge_features = edge_features[:, :2] + self.edge_trans(edge_features[:, :2]) # only take the first two meaningful features
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, 
                              edge_index_list=[edge_index]*self.efeature_dim,
                              edge_weights_list=[edge_features[:,i] for i in range(self.efeature_dim)])
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = self.convs[-1](x=x, 
                              edge_index_list=[edge_index]*self.efeature_dim,
                              edge_weights_list=[edge_features[:,i] for i in range(self.efeature_dim)])
        
        return x
    
    
class MPN_simplenet(nn.Module):
    """Wrapped Message Passing Network
        - One-time Message Passing to aggregate edge features into node features
        - Multiple Conv layers
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        self.convs = nn.ModuleList()

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        x = self.edge_aggr(x, edge_index, edge_features)
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        return x
    
class MaskEmbdMultiMPN_NNConv(nn.Module):
    """
    Pure NNConv architecture.
    Directly maps edge features (r, x) to convolution weights.
    Compatible with existing PowerFlowNet framework.
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        # K 参数在这里保留是为了兼容 train.py 的调用接口，虽然 NNConv 不需要它
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.dropout_rate = dropout_rate
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() # <--- 新增：归一化层列表

        # --- 辅助函数：生成边映射网络 (Edge Network) ---
        # NNConv 的核心：把物理参数 [r, x] 转换成权重矩阵
        # 输入: efeature_dim (2)
        # 输出: in_channels * out_channels (卷积核大小)
        def create_edge_net(in_c, out_c):
            return nn.Sequential(
                nn.Linear(efeature_dim, 64), # 将 r,x 映射到 16 维隐藏层
                nn.ReLU(),
                nn.Linear(64, in_c * out_c)  # 生成权重矩阵
            )

        # --- 1. 第一层 (Input -> Hidden) ---
        # 输入维度是 4 (nfeature_dim)
        self.layers.append(NNConv(nfeature_dim, hidden_dim, 
                                  create_edge_net(nfeature_dim, hidden_dim), 
                                  aggr='add'))
        self.norms.append(nn.LayerNorm(hidden_dim)) # <--- 新增

        # --- 2. 中间层 (Hidden -> Hidden) ---
        for l in range(n_gnn_layers - 2):
            self.layers.append(NNConv(hidden_dim, hidden_dim, 
                                      create_edge_net(hidden_dim, hidden_dim), 
                                      aggr='add'))
            self.norms.append(nn.LayerNorm(hidden_dim)) # <--- 新增
            
        # --- 3. 最后一层 (Hidden -> Output) ---
        # 输出维度是 4 (output_dim)
        self.layers.append(NNConv(hidden_dim, output_dim, 
                                  create_edge_net(hidden_dim, output_dim), 
                                  aggr='add'))
        
        # --- 掩码嵌入模块 ---
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nfeature_dim)
        )
        self.dropout = nn.Dropout(self.dropout_rate, inplace=False)

    # --- 辅助函数：处理无向图 ---
    def is_directed(self, edge_index):
        if edge_index.shape[1] == 0: return False
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack([edge_index[1,:], edge_index[0,:]], dim = 0)
            edge_index = torch.cat([edge_index, edge_index_dup], dim = 1)
            edge_attr = torch.cat([edge_attr, edge_attr], dim = 0)
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        # 1. 维度检查 (确保 data.x 是 4 维)
        assert data.x.shape[-1] == 4
        
        # 2. 解包数据
        x = data.x 
        mask = data.pred_mask.float()
        edge_index = data.edge_index
        edge_features = data.edge_attr # (E, 2) -> r, x
        
        # 3. 掩码嵌入 (Mask Embedding)
        # 学习 "未知" 的含义
        x = self.mask_embd(mask) + x
        
        # 4. 处理图方向
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        # 5. NNConv 循环 (前 N-1 层)
        # NNConv 需要 (x, edge_index, edge_attr)
        for i in range(len(self.layers)-1):
            x = self.layers[i](x, edge_index, edge_features) 
            x = self.norms[i](x) # <--- 关键修改：在这里加归一化！
            x = self.dropout(x)
            x = nn.ReLU()(x)
        # 6. 最后一层 (无激活函数)
        # 直接输出回归结果 (Vm, Va, P, Q)
        x = self.layers[-1](x, edge_index, edge_features)

        return x
    
class MaskEmbdMultiMPN_NNConv_v3(nn.Module):
    """
    [Improved NNConv with Residuals]
    
    Improvements over original:
    1. Input Projection: Maps 4-dim input to hidden-dim immediately.
    2. Residual Connections: x = x + block(x). Allows deeper networks (6-8 layers).
    3. GELU Activation: Smoother gradients for regression.
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # --- 1. 输入投影层 ---
        # 将原始特征 (P, Q, e, f) 映射到高维空间，方便后续处理
        self.input_encoder = nn.Linear(nfeature_dim, hidden_dim)
        
        # 掩码嵌入 (学习 "未知" 的含义)
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # --- 2. GNN 核心层 (带残差) ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 辅助函数：生成由边特征控制的权重矩阵
        # Input: efeature_dim (G, B) -> Output: hidden * hidden
        def create_edge_net(in_c, out_c):
            return nn.Sequential(
                nn.Linear(efeature_dim, hidden_dim), 
                nn.GELU(),
                nn.Linear(hidden_dim, in_c * out_c)
            )

        # 堆叠 N 层
        for _ in range(n_gnn_layers):
            self.convs.append(
                NNConv(hidden_dim, hidden_dim, 
                       create_edge_net(hidden_dim, hidden_dim), 
                       aggr='add')
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # --- 3. 输出解码层 ---
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)

    # 图处理辅助函数 (保持不变)
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
        x = data.x 
        mask = data.pred_mask.float()
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        # 1. 预处理：映射到 Hidden Space
        # 融合 Input 和 Mask 信息
        h = self.input_encoder(x) + self.mask_embd(mask)
        
        # 处理无向图
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        # 2. GNN 循环 (Residual Block)
        for i, conv in enumerate(self.convs):
            identity = h  # 保存输入用于残差
            
            # Conv
            h = conv(h, edge_index, edge_features)
            
            # Post-processing
            h = self.norms[i](h)
            h = F.gelu(h)
            h = self.dropout(h)
            
            # 【关键】残差连接
            h = h + identity
        
        # 3. 输出解码
        out = self.output_decoder(h)

        return out
    
class MaskEmbdMultiMPN_NNConv_v2(nn.Module):
    """
    [Physics-Weighted Multi-Hop NNConv]
    
    Architecture:
    1. Mask Embedding: Learn the meaning of missing values.
    2. NNConv Layers: Map edge attributes (r, x) to dynamic weights.
    3. K-Hop Accumulation: Inside each layer, propagate K times and SUM the results
       (like TAGConv) to capture global topology without over-smoothing.
    4. Residual Connections: Allow deep stacking (e.g., 8 layers).
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.dropout_rate = dropout_rate
        self.K = K  # 这里的 K 代表每一层内部的一波“连跳”次数
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # --- 辅助函数：生成边映射网络 (Edge Network) ---
        # 之前实验证明，中间层设为 64 能显著提升 V 和 Q 的精度
        def create_edge_net(in_c, out_c):
            return nn.Sequential(
                nn.Linear(efeature_dim, 64), # 2 -> 64 (宽瓶颈)
                nn.ReLU(),
                nn.Linear(64, in_c * out_c)
            )
        
        # self.hop_weights = nn.Parameter(torch.ones(self.K)) # 可学习的系数
        # self.hop_weights = nn.Parameter(torch.ones(n_gnn_layers, self.K))
        # --- 1. 第一层 (Input -> Hidden) ---
        # 这一层负责维度提升，只传 1 跳
        self.layers.append(NNConv(nfeature_dim, hidden_dim, 
                                  create_edge_net(nfeature_dim, hidden_dim), 
                                  aggr='add'))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # --- 2. 中间层 (Hidden -> Hidden) ---
        # 这些是主力层，我们将在 forward 中对其应用 K-hop 累加逻辑
        for l in range(n_gnn_layers - 2):
            self.layers.append(NNConv(hidden_dim, hidden_dim, 
                                      create_edge_net(hidden_dim, hidden_dim), 
                                      aggr='add'))
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        # --- 3. 最后一层 (Hidden -> Output) ---
        # 负责输出最终结果
        self.layers.append(NNConv(hidden_dim, output_dim, 
                                  create_edge_net(hidden_dim, output_dim), 
                                  aggr='add'))
        
        # --- 掩码嵌入模块 ---
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nfeature_dim)
        )
        self.dropout = nn.Dropout(self.dropout_rate, inplace=False)


    # 到这里结束

    # --- 辅助函数：处理无向图 ---
    def is_directed(self, edge_index):
        if edge_index.shape[1] == 0: return False
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack([edge_index[1,:], edge_index[0,:]], dim = 0)
            edge_index = torch.cat([edge_index, edge_index_dup], dim = 1)
            edge_attr = torch.cat([edge_attr, edge_attr], dim = 0)
            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    
    def forward(self, data):
        # 1. 维度检查
        assert data.x.shape[-1] == 4
        
        x = data.x 
        mask = data.pred_mask.float()
        edge_index = data.edge_index
        edge_features = data.edge_attr
        
        # 2. 掩码嵌入
        x = self.mask_embd(mask) + x
        
        # 3. 处理无向图
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        # 4. 循环层
        for i in range(len(self.layers)-1):
            identity = x # 保存旧状态，用于 ResNet 残差
            
            # === [核心修改] K-hop 累加逻辑 (仿 TAGConv) ===
            # 只有在输入输出维度一致（中间层）时才启用
            if x.shape[-1] == self.hidden_dim:
                
                # A. 初始化累加器 (对应 TAGConv 的 k=0 项，即保留原始信息)
                # 这能有效防止过度平滑，因为原始特征永远被保留了一份
                hop_accum = x 
                
                # B. 当前跳的信号
                current_signal = x
                
                # C. 连续跳跃 K 次
                for k in range(self.K):
                    # 传播 1 跳 (利用物理权重 r,x)
                    current_signal = self.layers[i](current_signal, edge_index, edge_features)
                    
                    # 激活：在每一跳之间加非线性，增强表达能力
                    current_signal = torch.relu(current_signal)
                    
                    # 累加：把这一跳“看到”的信息加到总量里
                    # Sum(0-hop + 1-hop + ... + K-hop)
                    hop_accum = hop_accum + current_signal
                    # hop_accum = hop_accum + current_signal * self.hop_weights[k]
                    # hop_accum = hop_accum + current_signal * self.hop_weights[i, k]
                
                # D. 这一层的输出就是累加结果
                out = hop_accum
                
            else:
                # 第一层 (4->128)，只传 1 次，不做循环
                out = self.layers[i](x, edge_index, edge_features)

            # === 标准后处理 ===
            out = self.norms[i](out) # LayerNorm 稳定数值
            out = self.dropout(out)
            out = nn.ReLU()(out)
            
            # === 层间残差连接 (ResNet) ===
            # 这是为了支持深层网络 (Deep GNN)
            if out.shape == identity.shape:
                x = identity + out 
            else:
                x = out
        
        # 5. 最后一层 (无激活，直接输出物理量)
        x = self.layers[-1](x, edge_index, edge_features)
        
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, LayerNorm
from torch_geometric.utils import to_dense_batch

class MaskEmbdMultiMPN_Transformer_Large(nn.Module):
    """
    [Heavy-Duty GNN + Transformer Hybrid]
    
    Structure (Sandwich):
    1. Embedding
    2. Pre-GNN (Local Physics extraction)
    3. Deep Transformer (Global Sync & Reference propagation)
    4. Post-GNN (Physics Refinement based on global context)
    5. Decoder
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, 
                 n_gnn_layers, K ,transformer_layers=4, dropout_rate=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # 1. Input Projection
        self.input_encoder = nn.Linear(nfeature_dim, hidden_dim)
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 辅助函数：生成边权重
        def create_edge_net(in_c, out_c):
            return nn.Sequential(
                nn.Linear(efeature_dim, hidden_dim), 
                nn.GELU(),
                nn.Linear(hidden_dim, in_c * out_c)
            )

        # 2. Pre-GNN Layers (前置 GNN，负责提取初步物理特征)
        self.pre_convs = nn.ModuleList()
        self.pre_norms = nn.ModuleList()
        for _ in range(n_gnn_layers):
            self.pre_convs.append(
                NNConv(hidden_dim, hidden_dim, create_edge_net(hidden_dim, hidden_dim), aggr='add')
            )
            self.pre_norms.append(nn.LayerNorm(hidden_dim))

        # 3. Deep Transformer (加深！负责全局对齐)
        # 这里把 num_layers 提出来作为参数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8,            # 头数增加到 8，捕捉更细微的关系
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True     # Pre-Norm 结构，训练深层网络更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 4. Post-GNN Layers (后置 GNN，负责利用全局信息精修物理)
        # 通常 2 层就够了
        self.post_convs = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        for _ in range(2): 
            self.post_convs.append(
                NNConv(hidden_dim, hidden_dim, create_edge_net(hidden_dim, hidden_dim), aggr='add')
            )
            self.post_norms.append(nn.LayerNorm(hidden_dim))

        # 5. Output Decoder
        self.output_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)

    # ... (保持 is_directed 和 undirect_graph 不变) ...
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
        x = data.x 
        mask = data.pred_mask.float()
        edge_index = data.edge_index
        edge_features = data.edge_attr
        batch = data.batch
        
        # --- A. Embedding ---
        h = self.input_encoder(x) + self.mask_embd(mask)
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        # --- B. Pre-GNN (Local Physics) ---
        for i, conv in enumerate(self.pre_convs):
            identity = h
            h = conv(h, edge_index, edge_features)
            h = self.pre_norms[i](h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = h + identity # ResNet

        # --- C. Deep Global Transformer ---
        # 1. To Dense
        h_dense, mask_batch = to_dense_batch(h, batch)
        
        # 2. Attention (Mask logic: True means ignore)
        h_global = self.transformer(h_dense, src_key_padding_mask=(~mask_batch))
        
        # 3. To Sparse
        h_global_flat = h_global[mask_batch]
        
        # 4. Residual Connection (Global info added to Local features)
        h = h + h_global_flat

        # --- D. Post-GNN (Physics Refinement) ---
        # 拿着全局信息再次进行物理校验
        for i, conv in enumerate(self.post_convs):
            identity = h
            h = conv(h, edge_index, edge_features)
            h = self.post_norms[i](h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = h + identity

        # --- E. Output ---
        out = self.output_decoder(h)

        return out

class MaskEmbdMultiMPN_Transformer(nn.Module):
    """
    [GNN + Transformer Hybrid Architecture]
    
    Philosophy:
    1. GNN (NNConv): Handles local physical constraints (Kirchhoff's Laws).
    2. Transformer: Handles global dependencies (Reference Angle Propagation & Global Power Balance).
    
    Structure:
    Input -> Embedding -> [NNConv ResBlocks] -> [Global Transformer] -> Decoder -> Output
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # 1. Input Projection
        self.input_encoder = nn.Linear(nfeature_dim, hidden_dim)
        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 2. Local Physics Processor (GNN Layers)
        # 我们依然保留 NNConv 来处理局部物理，但层数可以不用那么多
        # self.convs 是GNN的部分，里面塞了n_gnn_layers个NNConv
        self.convs = nn.ModuleList()

        # 这个里面是一个归一化层
        self.norms = nn.ModuleList()
        
        def create_edge_net(in_c, out_c):
            return nn.Sequential(
                nn.Linear(efeature_dim, hidden_dim), 
                nn.GELU(),
                nn.Linear(hidden_dim, in_c * out_c)
            )

        for _ in range(n_gnn_layers):
            self.convs.append(
                NNConv(hidden_dim, hidden_dim, 
                       create_edge_net(hidden_dim, hidden_dim), 
                       aggr='add')
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 3. Global Information Processor (Transformer)
        # 这是一个标准的 Transformer Encoder 层
        # batch_first=True: 输入格式为 [Batch, Seq_Len, Dim]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4,            # 4头注意力
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True 
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1) # 1层通常足够，也可以设为2
        
        # 4. Transformer 后的归一化
        self.global_norm = nn.LayerNorm(hidden_dim)

        # 5. Output Decoder，这是一个MLP的预测头
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
        
        # --- A. Embedding ---，用全连接层，对输入和掩码进行内容的丰富
        h = self.input_encoder(x) + self.mask_embd(mask)
        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        # --- B. Local GNN Propagation (NNConv ResNet) ---
        for i, conv in enumerate(self.convs):
            identity = h
            h = conv(h, edge_index, edge_features)
            h = self.norms[i](h)
            h = F.gelu(h)
            h = self.dropout(h)
            h = h + identity # Residual

        # --- C. Global Transformer Attention ---
        # 1. 转换为 Dense Batch: [Batch_Size, Max_Nodes, Hidden_Dim]
        # to_dense_batch 会自动填充 padding (mask_batch 用于指示哪些是填充点)
        h_dense, mask_batch = to_dense_batch(h, batch)
        
        # 2. Transformer 全局交互
        # src_key_padding_mask: 告诉 Transformer 忽略填充点
        # 注意: PyTorch Transformer 的 mask 逻辑是 True 代表忽略(masked)
        # mask_batch 里 True 代表有效节点，False 代表填充
        # 所以传入 Transformer 时要取反 (~mask_batch)
        h_global = self.transformer(h_dense, src_key_padding_mask=(~mask_batch))
        
        # 3. 还原为 Sparse Batch (Flatten)
        # 只取有效节点的数据
        h_global_flat = h_global[mask_batch]
        
        # 4. 残差连接：融合局部信息和全局信息
        h = h + h_global_flat
        h = self.global_norm(h)

        # --- D. Output ---
        out = self.output_decoder(h)

        return out
    

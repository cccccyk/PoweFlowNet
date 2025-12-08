import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, TAGConv, GCNConv, ChebConv ,GATConv  ,NNConv
from torch_geometric.utils import degree

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
            # self.layers.append(GATConv(hidden_dim, output_dim, heads=4, concat=False))
        else:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))
            # self.layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))

        # 中间层的时候，节点的信息已经不是4维度的，而是hidden_dim维度了，直接进行循环就行
        for l in range(n_gnn_layers-2):
            self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))
            # self.layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            
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
    

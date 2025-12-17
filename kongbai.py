
class PowerImbalance(MessagePassing):

    base_sn = 100 # kva
    base_voltage = 345 # kv
    base_ohm = 1190.25 # v**2/sn
    def __init__(self, xymean, xystd, edgemean, edgestd, reduction='mean'):
        super().__init__(aggr='add', flow='target_to_source')
        if xymean.shape[0] > 1:
            xymean = xymean[0:1]
        if xystd.shape[0] > 1:
            xystd = xystd[0:1]
        self.xymean = xymean
        self.xystd = xystd
        self.edgemean = edgemean
        self.edgestd = edgestd
        
    def de_normalize(self, x, edge_attr):
        self.xymean = self.xymean.to(x.device)
        self.xystd = self.xystd.to(x.device)
        self.edgemean = self.edgemean.to(x.device)
        self.edgestd = self.edgestd.to(x.device)
        return x * self.xystd + self.xymean, edge_attr * self.edgestd + self.edgemean
    
    def is_directed(self, edge_index):
        'determine if a graph id directed by reading only one edge'
        return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
    def undirect_graph(self, edge_index, edge_attr):
        """transform a directed graph (index, attr) into undirect by duplicating and reversing the directed edges

        Arguments:
            edge_index -- shape (2, E)
            edge_attr -- shape (E, fe)
        """
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
    
    def message(self, x_i, x_j, edge_attr):

        r_x = edge_attr[:, 0:2] # (num_edges, 2)
        r, x = r_x[:, 0:1], r_x[:, 1:2]

        g_ij = r / (r**2 + x**2)
        b_ij = -x / (r**2 + x**2)
        
        ym_ij = torch.sqrt(g_ij**2+b_ij**2)
        ya_ij = torch.acos(g_ij/ym_ij)
        vm_i = x_i[:, 0:1] # (num_edges, 1)
        va_i = 1/180.*torch.pi*x_i[:, 1:2] # (num_edges, 1)
        vm_j = x_j[:, 0:1] # (num_edges, 1)
        va_j = 1/180.*torch.pi*x_j[:, 1:2] # (num_edges, 1)
        e_i = vm_i * torch.cos(va_i)
        f_i = vm_i * torch.sin(va_i)
        e_j = vm_j * torch.cos(va_j)
        f_j = vm_j * torch.sin(va_j)

        Pji = g_ij*(e_i*e_j-e_i**2+f_i*f_j-f_i**2) + b_ij*(f_i*e_j-e_i*f_j)
        Qji = g_ij*(f_i*e_j-e_i*f_j) + b_ij*(-e_i*e_j+e_i**2-f_i*f_j+f_i**2)
        
        
        return torch.cat([Pji, Qji], dim=-1) # (num_edges, 2)
    
    def update(self, aggregated, x):
        """calculate power imbalance at each node

        Arguments:
            aggregated -- output of aggregation,    (num_nodes, 2)
            x -- node features                      (num_nodes, 6)
            
        Return:
            dPi|dQi: (num_nodes, 2)
        
        Formula:
        $$
            \Delta P_i = \sum_{j\in N_i} P_{ji} - P_{ij}
        $$
        """

        dPi = - aggregated[:, 0:1] + x[:, 2:3] # (num_nodes, 1)
        dQi = - aggregated[:, 1:2] + x[:, 3:4] # (num_nodes, 1)

        return torch.cat([dPi, dQi], dim=-1) # (num_nodes, 2)
        
    def forward(self, x, edge_index, edge_attr):
        """calculate power imbalance at each node

        Arguments:
            x -- _description_
            edge_index -- _description_
            edge_attr -- _description_
        
        Return:
            dPQ: torch.float
        
        Formula:
        $$
            \Delta P_i = \sum_{j\in N_i} P_{ji} - P_{ij}
        $$
        """
        if self.is_directed(edge_index):
            edge_index, edge_attr = self.undirect_graph(edge_index, edge_attr)
        x, edge_attr = self.de_normalize(x, edge_attr)    # correct, gecontroleerd. 
      
        dPQ = self.propagate(edge_index, x=x, edge_attr=edge_attr) # (num_nodes, 2)
        dPQ = dPQ.square().sum(dim=-1) # (num_nodes, 1)
        mean_dPQ = dPQ.mean()
        
        return mean_dPQ
    
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import networkx


class Masked_L2_loss(nn.Module):
    """
    Custom loss function for the masked L2 loss.

    Args:
        output (torch.Tensor): The output of the neural network model.
        target (torch.Tensor): The target values.
        mask (torch.Tensor): The mask for the target values.

    Returns:
        torch.Tensor: The masked L2 loss.
    """

    def __init__(self, regularize=True, regcoeff=1):
        super(Masked_L2_loss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.regularize = regularize
        self.regcoeff = regcoeff

    def forward(self, output, target, mask):

        masked = mask.type(torch.bool)

        # output = output * mask
        # target = target * mask
        outputl = torch.masked_select(output, masked)
        targetl = torch.masked_select(target, masked)

        loss = self.criterion(outputl, targetl)

        if self.regularize:
            masked = (1 - mask).type(torch.bool)
            output_reg = torch.masked_select(output, masked)
            target_reg = torch.masked_select(target, masked)
            loss = loss + self.regcoeff * self.criterion(output_reg, target_reg)

        return loss
    
class MaskedL2V2(nn.Module):
    """
    Args:
        - regularize (bool): not used
        - regcoeff (float): not used
    """
    def __init__(self, regularize=False, regcoeff=1):
        super(MaskedL2V2, self).__init__()
        
    def forward(self, output, target, mask):
        """ output.shape == target.shape == mask.shape == [N, F]
        ---
        F == 6 for legacy version. 
        
        F_0: vm
        F_1: va_degree
        F_2: Pd
        F_3: Qd
        F_4: ?
        F_5: ?
        
        """
        error = F.mse_loss(output, target, reduction='none') # (N, F)
        error = (error * mask.float()).sum(dim=0) / mask.sum(dim=0).clamp(min=1e-6) # (F,)
        loss_terms = {}
        loss_terms['total'] = (error * mask.sum(dim=0).clamp(min=1e-6)).sum() / mask.sum().clamp(min=1e-6) # mean of all prediction
        loss_terms['balanced total'] = error.mean() # mean of the average error of each feature
        loss_terms['vm'] = error[0]
        loss_terms['va'] = error[1]
        loss_terms['p'] = error[2]
        loss_terms['q'] = error[3]
        
        return loss_terms
    
class MaskedL1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask):
        error = F.l1_loss(output, target, reduction='none') # (N, F)
        error = (error * mask.float()).sum(dim=0) / mask.sum(dim=0).clamp(min=1e-6) # (F,)
        loss_terms = {}
        loss_terms['total'] = (error * mask.sum(dim=0).clamp(min=1e-6)).sum() / mask.sum().clamp(min=1e-6) # mean of all prediction
        loss_terms['balanced total'] = error.mean() # mean of the average error of each feature
        loss_terms['vm'] = error[0]
        loss_terms['va'] = error[1]
        loss_terms['p'] = error[2]
        loss_terms['q'] = error[3]
        
        return loss_terms


class PowerImbalance(MessagePassing):
    """Power Imbalance Loss Class

    Arguments:
        xymean: mean of the node features
        xy_std: standard deviation of the node features
        reduction: (str) 'sum' or 'mean' (node/batch-wise). P and Q are always added. 
        
    Input:
        x: node features        -- (N, 6)
        edge_index: edge index  -- (2, num_edges)
        edge_attr: edge features-- (num_edges, 2)
    """
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
        """calculate injected power Pji
        
        Formula:
        $$
        P_{ji} = V_m^i*V_m^j*Y_{ij}*\cos(V_a^i-V_a^j-\theta_{ij})
                -(V_m^i)^2*Y_{ij}*\cos(-\theta_{ij})
        $$
        $$
        Q_{ji} = V_m^i*V_m^j*Y_{ij}*\sin(V_a^i-V_a^j-\theta_{ij})
                -(V_m^i)^2*Y_{ij}*\sin(-\theta_{ij})
        $$
        
        Input:
            x_i: (num_edges, 6)
            x_j: (num_edges, 6)
            edge_attr: (num_edges, 2)
        
        Return:
            Pji|Qji: (num_edges, 2)
        """
        r_x = edge_attr[:, 0:2] # (num_edges, 2)
        r, x = r_x[:, 0:1], r_x[:, 1:2]
        # zm_ij = torch.norm(r_x, p=2, dim=-1, keepdim=True) # (num_edges, 1) NOTE (r**2+x**2)**0.5 should be non-zero
        # za_ij = torch.acos(edge_attr[:, 0:1] / zm_ij) # (num_edges, 1)
        # ym_ij = 1/(zm_ij + 1e-6)        # (num_edges, 1)
        # ya_ij = -za_ij      # (num_edges, 1)    
        # g_ij = ym_ij * torch.cos(ya_ij) # (num_edges, 1)
        # b_ij = ym_ij * torch.sin(ya_ij) # (num_edges, 1)
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
        
        ####### my (incomplete) method #######
        # Pji = vm_i * vm_j * ym_ij * torch.cos(va_i - va_j - ya_ij) \
        #         - vm_i**2 * ym_ij * torch.cos(-ya_ij)
        # Qji = vm_i * vm_j * ym_ij * torch.sin(va_i - va_j - ya_ij) \
        #         - vm_i**2 * ym_ij * torch.sin(-ya_ij)
        
        ####### standard method #######
        # cannot be done since there's not complete information about whole neighborhood. 
        
        ####### another reference method #######
        # Pji = vm_i * vm_j * (g_ij*torch.cos(va_i-va_j)+b_ij*torch.sin(va_i-va_j))
        # Qji = vm_i * vm_j * (g_ij*torch.sin(va_i-va_j)-b_ij*torch.cos(va_i-va_j))
        
        ####### reference method 3 #######
        # Pji = g_ij*(vm_i**2 - vm_i*vm_j*torch.cos(va_i-va_j)) \
        #     - b_ij*(vm_i*vm_j*torch.sin(va_i-va_j))
        # Qji = b_ij*(- vm_i**2 + vm_i*vm_j*torch.cos(va_i-va_j)) \
        #     - g_ij*(vm_i*vm_j*torch.sin(va_i-va_j))
            
        ###### another mine ######
        Pji = g_ij*(e_i*e_j-e_i**2+f_i*f_j-f_i**2) + b_ij*(f_i*e_j-e_i*f_j)
        Qji = g_ij*(f_i*e_j-e_i*f_j) + b_ij*(-e_i*e_j+e_i**2-f_i*f_j+f_i**2)
        
        # --- DEBUG ---
        # self._dPQ = torch.cat([Pji, Qji], dim=-1) # (num_edges, 2)
        # --- DEBUG ---
        
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
        # TODO check if the aggregated result is correct
        
        # --- DEBUG ---
        # self.node_dPQ = self._is_i.float() @ self._dPQ # correct, gecontroleerd.
        # --- DEBUG ---
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
        # --- per unit --- 
        # edge_attr[:, 0:2] = edge_attr[:, 0:2]/self.base_ohm
        # x[:, 2:4] = x[:, 2:4]/self.base_sn
        # --- DEBUG ---
        # self._edge_index = edge_index
        # self._is_i = torch.arange(14).view((14,1)).expand((14, 20)).long() == edge_index[0:1,:]
        # self._is_j = torch.arange(14).view((14,1)).expand((14, 20)).long() == edge_index[1:2,:]
        # --- DEBUG ---        
        dPQ = self.propagate(edge_index, x=x, edge_attr=edge_attr) # (num_nodes, 2)
        dPQ = dPQ.square().sum(dim=-1) # (num_nodes, 1)
        mean_dPQ = dPQ.mean()
        
        return mean_dPQ
''' 
# class PowerImbalance(MessagePassing):
#     """Power Imbalance Loss Class
    
#     Arguments:
#         xymean: mean of the node features
#         xy_std: standard deviation of the node features
#         reduction: (str) 'sum' or 'mean' (node/batch-wise). 
#     """
    
#     def __init__(self, xymean, xystd, edgemean, edgestd, reduction='mean'):
#         super().__init__(aggr='add', flow='target_to_source')
#         if xymean.shape[0] > 1: xymean = xymean[0:1]
#         if xystd.shape[0] > 1: xystd = xystd[0:1]
        
#         self.register_buffer('xymean', xymean)
#         self.register_buffer('xystd', xystd)
#         self.register_buffer('edgemean', edgemean)
#         self.register_buffer('edgestd', edgestd)
#         self.reduction = reduction
        
#     def de_normalize(self, x, edge_attr):
#         """反归一化"""
#         return x * self.xystd + self.xymean, edge_attr * self.edgestd + self.edgemean
    
#     def is_directed(self, edge_index):
#         if edge_index.shape[1] == 0: return False
#         return edge_index[0,0] not in edge_index[1,edge_index[0,:] == edge_index[1,0]]
    
#     def undirect_graph(self, edge_index, edge_attr):
#         edge_index_dup = torch.stack([edge_index[1,:], edge_index[0,:]], dim=0)
#         edge_index = torch.cat([edge_index, edge_index_dup], dim=1)
#         edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
#         return edge_index, edge_attr
    
#     def message(self, x_i, x_j, edge_attr):
#         """
#         计算支路功率流 Pji, Qji
#         输入 edge_attr 已经是 [G, B]
#         """
#         # --- 修改点：直接读取 G 和 B ---
#         # 假设 edge_attr 的第0列是 G，第1列是 B
#         g_ij = edge_attr[:, 0:1] 
#         b_ij = edge_attr[:, 1:2] 
        
#         # 获取节点电压
#         vm_i = x_i[:, 0:1] 
#         va_i = torch.deg2rad(x_i[:, 1:2]) # 确保输入是角度制并转为弧度
#         vm_j = x_j[:, 0:1] 
#         va_j = torch.deg2rad(x_j[:, 1:2]) 
        
#         # 转换为直角坐标形式 V = e + jf
#         e_i = vm_i * torch.cos(va_i)
#         f_i = vm_i * torch.sin(va_i)
#         e_j = vm_j * torch.cos(va_j)
#         f_j = vm_j * torch.sin(va_j)

#         # 计算支路功率流 Pji, Qji
#         # 公式推导：S_ji = V_i * conj(Y_ij * (V_i - V_j)) 忽略对地支路
#         # 或者基于标准支路流公式
#         term1 = e_i * e_j + f_i * f_j # Vi dot Vj
#         term2 = f_i * e_j - e_i * f_j # Vi cross Vj (2D)
#         v_sq = e_i**2 + f_i**2        # Vi squared     
        
#         # 注意：这里的物理公式必须与 pandapower 生成数据时隐含的物理模型一致
#         # 如果是标准的 Pi 模型且忽略对地电容：
#         Pji = g_ij * (v_sq - term1) - b_ij * term2
#         Qji = -b_ij * (v_sq - term1) - g_ij * term2
        
#         # 建议使用下面这组标准公式：
#         Pji_calc = g_ij * (v_sq - term1) - b_ij * term2
#         Qji_calc = -b_ij * (v_sq - term1) - g_ij * term2

#         return torch.cat([Pji_calc, Qji_calc], dim=-1)
    
#     def update(self, aggregated, x):
#         """
#         aggregated: sum(P_ji), sum(Q_ji) 流出功率的总和
#         x: 目标注入功率 (P_gen - P_load)
#         """
#         p_target = x[:, 2:3]
#         q_target = x[:, 3:4]
        
#         # 节点功率平衡方程：
#         # P_injected = Sum(P_flowing_out_to_neighbors)
#         # Error = P_target - P_calculated
#         dPi = p_target - aggregated[:, 0:1] 
#         dQi = q_target - aggregated[:, 1:2] 

#         return torch.cat([dPi, dQi], dim=-1)
        
#     def forward(self, x, edge_index, edge_attr):
#         if self.is_directed(edge_index):
#             edge_index, edge_attr = self.undirect_graph(edge_index, edge_attr)
            
#         # 1. 反归一化
#         # 因为数据生成时已经是G/B，所以这里的 mean/std 也是对应 G/B 的，直接反归一化即可得到真实的 G/B
#         x_denorm, edge_attr_denorm = self.de_normalize(x, edge_attr)
        
#         # 2. 计算物理误差
#         dPQ = self.propagate(edge_index, x=x_denorm, edge_attr=edge_attr_denorm)
        
#         # 3. 计算 MSE Loss
#         loss = dPQ.pow(2).mean()
        
#         return loss
'''
    
class MixedMSEPoweImbalance(nn.Module):
    """mixed mse and power imbalance loss
    
    loss = alpha * mse_loss + (1-alpha) * power_imbalance_loss
    """
    def __init__(self, xymean, xystd, edgemean, edgestd, alpha=0.5, reduction='mean'):
        super().__init__()
        assert alpha <= 1. and alpha >= 0
        self.power_imbalance = PowerImbalance(xymean, xystd, edgemean, edgestd, reduction)
        self.mse_loss_fn = nn.MSELoss(reduction=reduction)
        self.alpha = alpha
    
    def forward(self, x, edge_index, edge_attr, y):
        power_imb_loss = self.power_imbalance(x, edge_index, edge_attr)
        mse_loss = self.mse_loss_fn(x, y)
        loss = self.alpha * mse_loss + (1-self.alpha) * 0.020*power_imb_loss
        
        return loss

# 计算物理引擎
class RectangularPowerImbalance(MessagePassing):
    def __init__(self, xymean ,xystd ,edgemean ,edgestd):
        super().__init__(aggr='add',flow='target_to_source')
        self.register_buffer('xymean', xymean)
        self.register_buffer('xystd', xystd)
        self.register_buffer('edgemean', edgemean)
        self.register_buffer('edgestd', edgestd)

    def message(self, x_i,x_j,edge_attr):
        # 输入 x_i = e , x_j = f
        e_i, f_i = x_i[:, 0:1], x_i[:, 1:2]
        e_j, f_j = x_j[:, 0:1], x_j[:, 1:2]
        g_ij, b_ij = edge_attr[:, 0:1], edge_attr[:, 1:2]

        # 直角坐标系下的支路功率流公式
        term1 = e_i * e_j + f_i * f_j
        term2 = f_i * e_j - e_i * f_j 
        v_sq = e_i**2 + f_i**2
        
        # 计算流出功率
        P_flow = g_ij * (v_sq - term1) - b_ij * term2
        Q_flow = -b_ij * (v_sq - term1) - g_ij * term2
        
        return torch.cat([P_flow, Q_flow], dim=-1)
    
    def forward(self, pred_ef, input_pq, edge_index, edge_attr):
        # pred_ef 是模型的输出，但是已经是物理值，不需要归一化
        # input_pq 是真实的值，但是是归一化后的值，需要反归一化
        # 1. 对于PQ反归一化
        pq_mean = self.xymean[:,:2]
        pq_std = self.xystd[:,:2]
        real_pq = input_pq * (pq_std + 1e-7) + pq_mean
        # 对于 BG 反归一化
        real_edge = edge_attr * (self.edgestd + 1e-7) + self.edgemean

        # 处理无向图（变为双向）
         # 3. 处理无向图 (变为双向)
        if edge_index.shape[1] > 0:
            if edge_index[0,0] not in edge_index[1,edge_index[0,:]==edge_index[1,0]]:
                edge_index_dup = torch.stack([edge_index[1], edge_index[0]], dim=0)
                edge_index = torch.cat([edge_index, edge_index_dup], dim=1)
                real_edge = torch.cat([real_edge, real_edge], dim=0)
        
            # 计算P_calc
            calc_pq_flow = self.propagate(edge_index , x=pred_ef , edge_attr = real_edge)
        else : 
            calc_pq_flow = torch.zeros_like(real_pq)
        diff = real_pq + calc_pq_flow

        return diff.pow(2).mean()

class RectangularPureMSELoss(nn.Module):
    '''
    专门用于预测e,f设计的损失函数
    '''
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
    
    def forward(self,pred,target,mask):
        '''
        pred:[N,2]
        target:[N,4]
        mask:[N,4]
        '''
        target_ef = target[:,2:]    # 提取对应的e和f
        mask_ef = mask[:,2:]      # 提取需要预测的e和f
        squared_diff = (pred - target_ef) ** 2
        masked_loss = squared_diff * mask_ef

        loss = masked_loss.sum() / (mask_ef.sum() + 1e-6)

        return  loss
    
class RectangularMixedLoss(nn.Module):
    def __init__(self, xymean, xystd, edgemean, edgestd, alpha=0.8, beta=0.2, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # MSE 权重
        self.beta = beta    # 物理不平衡 权重
        self.gamma = gamma  # PV节点电压约束 权重

        self.mse_loss_fn = RectangularPureMSELoss()
        self.phys_engine = RectangularPowerImbalance(xymean, xystd, edgemean, edgestd)

        self.register_buffer('ef_mean', xymean[:, 2:])
        self.register_buffer('ef_std', xystd[:, 2:])

    def forward(self, pred_ef, target_y, mask, edge_index, edge_attr, bus_type, target_vm):
        # 获得纯mse的数值loss
        loss_mse = self.mse_loss_fn(pred_ef, target_y, mask)
        # 或得物理一致性损失
        input_pq = target_y[:, :2]
        loss_phys = self.phys_engine(pred_ef, input_pq, edge_index, edge_attr)
        # 节点电压约束（可选）
        ef_real = pred_ef * (self.ef_std + 1e-7) + self.ef_mean
        vm_sq_pred = ef_real[:, 0]**2 + ef_real[:, 1]**2
        vm_sq_target = target_vm ** 2

        # 只计算 PV 节点 (type=1)
        is_pv = (bus_type == 1)
        if is_pv.any():
            loss_pv = (vm_sq_pred[is_pv] - vm_sq_target[is_pv]).abs().mean()
        else:
            loss_pv = torch.tensor(0.0, device=pred_ef.device)
            
        # --- 总损失 ---
        # 如果 beta=0，就退化为带 PV 约束的 MSE；如果 gamma=0，就是纯 MSE + 物理
        total_loss = self.alpha * loss_mse + self.beta * loss_phys + self.gamma * loss_pv

        # 返回所有分项，方便打印日志
        return total_loss, loss_mse, loss_phys, loss_pv


def main():
    # TODO import trainset, select an data.y, calculate the imbalance
    # trainset = PowerFlowData(root='~/data/volume_2/power_flow_dataset', case='14', split=[.5, .3, .2], task='train')
    # sample = trainset[3]
    loss_fn = PowerImbalance(0, 1)
    x = torch.arange(18).reshape((3, 6)).float()
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ]).long()
    edge_attr = torch.tensor([
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0]
    ]).float()
    
    loss = loss_fn(x, edge_index, edge_attr)
    # loss = loss_fn(sample.y, sample.edge_index, sample.edge_attr)
    print(loss)
    
class Weighted_Masked_L2_loss(nn.Module):
    """
    针对不同物理量 (V, theta, P, Q) 赋予不同权重的 Loss
    """
    def __init__(self, weights=[1.0, 10.0, 10.0, 1.0]): 
        # 默认给 Q (index 3) 5倍权重
        super(Weighted_Masked_L2_loss, self).__init__()
        self.weights = torch.tensor(weights)
        self.criterion = nn.MSELoss(reduction='none') # 必须用 none，以便我们手动加权

    def forward(self, output, target, mask):
        # 1. 计算所有位置的平方误差
        # shape: [Batch, N, 4]
        squared_diff = (output - target) ** 2
        
        # 2. 确保权重在正确的设备上
        if self.weights.device != output.device:
            self.weights = self.weights.to(output.device)
            
        # 3. 应用权重
        # weights shape: [4] -> 广播到 [Batch, N, 4]
        # 让 Q 的误差被放大，P, V, Theta 保持原样
        weighted_diff = squared_diff * self.weights 
        
        # 4. 应用掩码 (只计算未知量)
        masked_loss = weighted_diff * mask
        
        # 5. 求平均
        # 除以 mask 中 1 的总数，防止 loss 随 batch size 变大
        loss = masked_loss.sum() / (mask.sum() + 1e-6)
        
        return loss
if __name__ == '__main__':
    main()
import os
import torch
import numpy as np
import logging
from torch_geometric.loader import DataLoader

# 引入你的数据处理和参数解析
from datasets.PowerFlowData import PowerFlowData
from utils.argument_parser import argument_parser
from utils.evaluation import load_model

# 引入你的模型定义 (根据你当前使用的模型修改这里)
from networks.MPN import MaskEmbdMultiMPN_NNConv_v2 , MaskEmbdMultiMPN

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def denormalize(val, mean, std):
    """反归一化：将模型输出还原为真实物理值"""
    return val * std + mean

@torch.no_grad()
def main():
    # 1. 获取参数
    args = argument_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # === 配置区域 ===
    # 你想查看测试集里的第几个样本？(0 ~ 3000+)
    SAMPLE_IDX = 0  
    # 你的模型运行ID (用于加载权重)
    RUN_ID = '20251204-1383' # <--- 请替换成你训练好的 run_id
    # ===============

    # 2. 加载数据参数 (均值/标准差)
    data_dir = args.data_dir
    data_param_path = os.path.join(data_dir, 'params', f'data_params_{RUN_ID}.pt')
    if not os.path.exists(data_param_path):
        logger.error(f"找不到数据参数文件: {data_param_path}")
        return

    data_param = torch.load(data_param_path, map_location='cpu')
    xymean, xystd = data_param['xymean'].to(device), data_param['xystd'].to(device)

    # 3. 加载数据集 (只加载 test set)
    logger.info(f"Loading Test Data from {args.case}...")
    testset = PowerFlowData(root=data_dir, case=args.case,
                            split=[.5, .2, .3], task='test',
                            xymean=xymean.cpu(), xystd=xystd.cpu())
    
    # 4. 提取单个样本
    if SAMPLE_IDX >= len(testset):
        logger.error(f"索引越界！测试集只有 {len(testset)} 个样本。")
        return
    
    data = testset[SAMPLE_IDX].to(device)
    
    # 5. 初始化并加载模型
    # 注意：这里要用你当前训练的模型类
    node_in_dim, node_out_dim, edge_dim = testset.get_data_dimensions()
    # model = MaskEmbdMultiMPN_NNConv_v2(
    #     nfeature_dim=node_in_dim,
    #     efeature_dim=edge_dim,
    #     output_dim=node_out_dim,
    #     hidden_dim=args.hidden_dim,
    #     n_gnn_layers=args.n_gnn_layers,
    #     K=args.K,
    #     dropout_rate=args.dropout_rate
    # ).to(device)

    model = MaskEmbdMultiMPN(
        nfeature_dim=node_in_dim,
        efeature_dim=edge_dim,
        output_dim=node_out_dim,
        hidden_dim=args.hidden_dim,
        n_gnn_layers=args.n_gnn_layers,
        K=args.K,
        dropout_rate=args.dropout_rate
    ).to(device)

    model, _ = load_model(model, RUN_ID, device)
    model.eval()

    # 6. 模型预测
    logger.info(f"Running inference on Sample #{SAMPLE_IDX}...")
    
    # 增加 batch 维度 (因为模型通常期望 batch input)
    # PyG 的 data 对象可以直接喂进去，因为它会自动处理 batch=1 的情况
    out = model(data)

    # 7. 数据还原 (反归一化)
    # 预测值
    pred_real = denormalize(out, xymean, xystd)
    # 真实值 (data.y 已经是归一化的，需要还原)
    truth_real = denormalize(data.y, xymean, xystd)
    
    # 8. 打印详细对比
    print_comparison(data, pred_real, truth_real)

def print_comparison(data, pred, truth):
    """
    格式化打印对比结果
    """
    num_nodes = data.num_nodes
    bus_type = data.bus_type.cpu().numpy() # 0=Slack, 1=PV, 2=PQ
    
    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()
    
    print("\n" + "="*100)
    print(f"{'Node':<5} | {'Type':<6} | {'Vm (Pred / True)':<20} | {'Va (Pred / True)':<20} | {'P (Pred / True)':<20} | {'Q (Pred / True)':<20}")
    print("-" * 100)

    # 统计 Slack 节点的误差
    slack_p_err = 0.0

    for i in range(num_nodes):
        b_type = int(bus_type[i])
        type_str = "Slack" if b_type == 0 else ("PV" if b_type == 1 else "PQ")
        
        # 格式化数值
        vm_str = f"{pred[i,0]:.4f} / {truth[i,0]:.4f}"
        va_str = f"{pred[i,1]:.2f} / {truth[i,1]:.2f}"
        p_str  = f"{pred[i,2]:.4f} / {truth[i,2]:.4f}"
        q_str  = f"{pred[i,3]:.4f} / {truth[i,3]:.4f}"
        
        # 如果是 Slack 节点，高亮显示 P
        prefix = " "
        if b_type == 0: 
            prefix = ">>" # 标记
            slack_p_err = abs(pred[i,2] - truth[i,2])
        
        print(f"{prefix}{i:<4} | {type_str:<6} | {vm_str:<20} | {va_str:<20} | {p_str:<20} | {q_str:<20}")

    print("="*100)
    print(f"\n[Analysis for Sample]")
    print(f"Slack Node P Prediction: {pred[np.where(bus_type==0)][0][2]:.4f} p.u.")
    print(f"Slack Node P Ground Truth: {truth[np.where(bus_type==0)][0][2]:.4f} p.u.")
    print(f"Slack Node P Absolute Error: {slack_p_err:.4f} p.u.")
    
    ref_val = abs(truth[np.where(bus_type==0)][0][2])
    if ref_val > 1e-5:
        print(f"Slack Node P Relative Error: {(slack_p_err / ref_val)*100:.2f}%")

if __name__ == "__main__":
    main()
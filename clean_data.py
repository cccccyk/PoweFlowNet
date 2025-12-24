import numpy as np
import os

# 配置
CASE_NAME = 'case118v_n1_train'
DATA_DIR = 'data/raw'
BAD_INDICES_FILE = f"{DATA_DIR}/{CASE_NAME}_bad_indices.npy"

def clean_dataset():
    print("🧹 开始物理删除脏数据...")
    
    # 1. 加载黑名单
    if not os.path.exists(BAD_INDICES_FILE):
        print("没找到 bad_indices.npy，请先运行之前的检查脚本！")
        return
    
    bad_indices = np.load(BAD_INDICES_FILE)
    print(f"待删除的索引: {bad_indices}")
    
    # 2. 加载原始数据
    edge_path = os.path.join(DATA_DIR, f"{CASE_NAME}_edge_features.npy")
    node_path = os.path.join(DATA_DIR, f"{CASE_NAME}_node_features.npy")
    labels_path = os.path.join(DATA_DIR, f"{CASE_NAME}_labels.npy") # 如果有的话
    
    edges = np.load(edge_path, allow_pickle=True)
    nodes = np.load(node_path, allow_pickle=True)
    
    print(f"原始形状: Nodes {nodes.shape}, Edges {edges.shape}")
    
    # 3. 删除
    # np.delete 返回一个新的数组
    nodes_clean = np.delete(nodes, bad_indices, axis=0)
    edges_clean = np.delete(edges, bad_indices, axis=0)
    
    # 处理 Labels (如果存在)
    if os.path.exists(labels_path):
        labels = np.load(labels_path, allow_pickle=True)
        labels_clean = np.delete(labels, bad_indices, axis=0)
        np.save(labels_path, labels_clean) # 覆盖保存
        print("Labels 已清理。")
        
    print(f"清理后形状: Nodes {nodes_clean.shape}, Edges {edges_clean.shape}")
    
    # 4. 覆盖保存 (直接覆盖原文件，保持文件名不变，方便后续代码复用)
    # 务必确认你不需要那个脏数据了
    np.save(edge_path, edges_clean)
    np.save(node_path, nodes_clean)
    
    print("✅ 文件已覆盖保存。数据现在是干净的了！")
    
    # 5. 清理旧的 processed 文件，强制 Dataset 重新处理
    os.system(f"rm data/processed/{CASE_NAME}_processed_*.pt")
    print("🗑️ 已删除旧的 .pt 缓存文件，下次运行会自动重新生成。")

if __name__ == "__main__":
    clean_dataset()
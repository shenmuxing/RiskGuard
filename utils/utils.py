import sys

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from tqdm import tqdm
import pickle
import numpy as np
import random
import torch
from torch_geometric.utils.map import map_index
from torch import Tensor
from typing import Optional
import torch.nn.functional as F

is_debug = False

import torch.multiprocessing as mp
import tempfile

def create_adjacency_matrix_sparse(G_day):
    from_nodes = np.array(G_day[0])
    to_nodes = np.array(G_day[1])
    values = np.array(G_day[2])
    num_nodes = int(max(max(from_nodes), max(to_nodes)) + 1)
    
    # 使用 PyTorch 的 COO 格式稀疏张量
    indices = torch.tensor([from_nodes, to_nodes], dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float)
    
    adjacency_matrix = torch.sparse_coo_tensor(indices, values, torch.Size([num_nodes, num_nodes]))
    return adjacency_matrix

def create_relation_encoding(G_day):
    from_nodes = np.array(G_day[0])
    to_nodes = np.array(G_day[1])
    values = np.array(G_day[2])
    relation_encoding = np.zeros((len(values), 1)) 
    relation_encoding[:, 0] = values
    return torch.tensor(relation_encoding).float()

def create_entity_index(feat):
    num_nodes = feat.shape[0]
    entity_index = np.arange(num_nodes)
    return torch.tensor(entity_index).long()

def get_graph_from_data(X, G, day):
    feat = X[day]
    if day >= len(G):  # 确保 day 不超出 G 的索引范围
        return None
        raise IndexError("Day index out of range for G.")
    adjacency_matrix = create_adjacency_matrix_sparse(G[day])
    point_e = None  # 根据你的需求生成
    relation_e = create_relation_encoding(G[day])
    entity_index = create_entity_index(feat)
    return feat, adjacency_matrix, point_e, relation_e, entity_index

def create_point_encoding(feat):
    point_e = np.sum(feat, axis=1)
    return torch.tensor(point_e).float()

def get_label_from_data(y, split=0.8):
    time, users = y.shape
    indices = np.arange(time)
    split_idx = int(split * time)
    train_mask = indices[:split_idx]
    test_mask = indices[split_idx:]
    train_label = y[train_mask]
    test_label = y[test_mask]
    return train_mask, test_mask, train_label, test_label

# 设置共享策略为文件系统
# mp.set_sharing_strategy('file_system')

# 创建临时目录
temp_dir = '/home/zhaojingye/tmp/torch_tmp'
os.makedirs(temp_dir, exist_ok=True)
os.environ['TMPDIR'] = temp_dir
tempfile.tempdir = temp_dir

def set_seed(seed):
    """
    设置所有随机种子以确保结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
set_seed(42)

def kth_largest(arr, k):
    if k < 1 or k > len(arr):
        return None
    # k变成从0开始的索引
    k = len(arr) - k
    return np.partition(arr, k)[k]

# %%
# 数据加载和预处理
def load_data(data_file):
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)
    
    if 'llm' not in data_file:
        data_dict['X_account'] = data_dict['X_account']
    return data_dict

def z_score_normalize(X, mean=None, std=None):
    if mean is None or std is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
    
    mean = np.zeros_like(mean)
    std = np.ones_like(std)
    return (X - mean) / (std + 1e-8), mean, std

def compute_base_probs(y,clip_min = 0.1, clip_max = 0.9):
    base_probs = np.zeros(2)
    base_probs[0] = np.clip(np.mean(y),clip_min,clip_max)
    # base_probs[0] = np.mean(y)
    base_probs[1] = 1 - base_probs[0]
    return base_probs

def get_size_in_gb(obj):
    return sys.getsizeof(obj) / (1024 ** 3)


def calculate_predictions(y_stock, y_account, G, num_account):
    mask_for_stock = (y_stock == 1)
    days, stocks = np.where(mask_for_stock)
    
    y_account_total_predict = np.zeros_like(y_account)
    y_account_total_true = np.asanyarray(y_account)

    for day, stock in zip(days, stocks):
        G_day = G[day]
        mask_for_G_day = (G_day[:2] == stock + num_account).any(axis=0)
        node_filtered_with_weight = G_day[:3, mask_for_G_day]
        y_predicted_with_confidence = np.zeros(y_account_total_predict.shape[1])
        y_predicted_with_confidence[node_filtered_with_weight[0].astype(int)] = np.abs(node_filtered_with_weight[2])
        y_account_total_predict[day] += y_predicted_with_confidence
    
    return y_account_total_predict.reshape(-1),y_account_total_true.reshape(-1)


def adjust_logits(logits, base_probs,tau = 1.0):
    """
    base_probs: the base probabilities for each class, shape (num_classes)
    Adjust the logits for temperature scaling
    """
    # return logits
    logits = logits + torch.log(
        (base_probs**tau + 1e-12).type(torch.float32)
    )

    return logits



def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    r"""Converts indices to a mask representation.

    Args:
        index (Tensor): The indices.
        size (int, optional): The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.

    Example:
        >>> index = torch.tensor([1, 3, 5])
        >>> index_to_mask(index)
        tensor([False,  True, False,  True, False,  True])

        >>> index_to_mask(index, size=7)
        tensor([False,  True, False,  True, False,  True, False])
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def mask_to_index(mask: Tensor) -> Tensor:
    r"""Converts a mask to an index representation.

    Args:
        mask (Tensor): The mask.

    Example:
        >>> mask = torch.tensor([False, True, False])
        >>> mask_to_index(mask)
        tensor([1])
    """
    return mask.nonzero(as_tuple=False).view(-1)


# 辅助函数
def step_probabilities(one_hop_target_nodes, gradients, learning_rate=0.1, delta=0.01):
    """
    计算更新后的概率分布
    
    参数:
    one_hop_target_nodes (torch.Tensor): 原始概率分布 (4, num_edges), 分别是 (dst, raw_sample_prob, edge_weight, edge_id )
    gradients (torch.Tensor): 梯度,(num_edges, 2), 分别是 (abs_gradients, edge_id)
    learning_rate (float): 学习率
    delta (float): 平滑因子
    
    返回:
    torch.Tensor: 更新后的概率分布
    """
    # 计算步长
    raw_probabilities = one_hop_target_nodes[1,:]
    gradients = gradients[:, 0]
    step = (raw_probabilities + delta / len(raw_probabilities)) * torch.exp(- (learning_rate * gradients) / (1 + delta))

    # 应用 softmax 函数
    new_probabilities = (1+delta) * F.softmax(step, dim=0) - delta/len(raw_probabilities)
    
    return new_probabilities

def get_original_node_indices(edge_index_to_update, nodes):
    original_node_indices = nodes[edge_index_to_update]
    return original_node_indices

def process_edges_and_gradients( edge_attrs_to_update, one_hop_target_nodes):
    """
        处理边和梯度信息，输入为 tensor，并返回torch.tensor类型的结果。
        
        参数:
        edge_attrs_to_update: tensor: 形状为(2,update_edges), abs_gradients和 edge_ids
        one_hop_target_nodes: 一跳目标节点张量, 形状为(4, num_edges) 分别是 (dst, raw_sample_prob, edge_weight, edge_id )
        
        返回:
        new_abs_gradients: torch.tensor，处理后的绝对梯度
    """

    # 创建边字典
    # edge_dict = {(src.item(), dst.item()): grad.item() for src, dst, grad in zip(edge_index_to_update[0], edge_index_to_update[1], abs_gradients)}
    edge_ids_to_update = edge_attrs_to_update[1,:].long() 
    edge_ids_one_hop = one_hop_target_nodes[3,:].long()
    ids_map, ids_mask = map_index(edge_ids_one_hop,edge_ids_to_update,inclusive=False)

    if ids_map.shape[0] == 0:
        return None
    
    # 初始化新的绝对梯度张量
    new_abs_gradients = torch.full((one_hop_target_nodes.shape[1],2), float('nan')).to(one_hop_target_nodes.device)
    new_abs_gradients[:,1] = one_hop_target_nodes[3,:]
    new_abs_gradients[ids_mask,0] = edge_attrs_to_update[0,ids_map]

    # 计算均值并填充NaN值
    # mean_value = torch.nanmean(new_abs_gradients[:,0])
    # mean_value = torch.mean(edge_attrs_to_update[0])
    mean_value = torch.tensor(0.0)
    if torch.isnan(mean_value):
        mean_value = torch.tensor(0.0)
    new_abs_gradients = torch.nan_to_num(new_abs_gradients, nan=mean_value.item())
    
    return new_abs_gradients


def generate_stock_features(model, dataloader, X_stock,stock_hidden_dim, account_num, time_window):
    model.eval()
    device = next(model.parameters()).device

    num_stocks = X_stock.shape[1]
    all_stock_features = torch.zeros(X_stock.shape[0], X_stock.shape[1], stock_hidden_dim).to(device)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader),mininterval=10):

            account_features, stock_features, edge_indices, edge_weights, edge_ids, point_encs, \
                y_days, learning_weights, center_nodes, nodes, last_day = batch
            if batch_idx % 1000 == 0:
                print(f"account_features.shape:{account_features.shape}")
                print(f"stock_features.shape:{stock_features.shape}")
                print(f"length of edge_indices:{len(edge_indices)}")
                print(f"edge_weights[-1].shape:{edge_weights[-1].shape}")
                print(f"y_days.shape:{y_days.shape}")
                print(f"center_nodes.shape:{center_nodes.shape}") 
            account_features = account_features.to(device)
            stock_features = stock_features.to(device)
            edge_ids = edge_ids.to(device)
            point_encs = point_encs.bool().to(device)
            y_day = y_days.reshape(-1,).to(device)

            nodes = nodes.to(device)
            center_nodes = center_nodes.to(device)
            node_to_predict_mask, _ = map_index(center_nodes, nodes, inclusive=True)
            # node_to_predict_mask = index_to_mask(node_to_predict_mask, nodes.shape[0]).to(device)
            learning_weights = learning_weights.to(device) if learning_weights is not None else None

            center_nodes = center_nodes.to(device)

            stock_features = model.get_stock_features(X_stock=stock_features, X_account=account_features, edge_indices=edge_indices,
                           point_enc=point_encs, edge_weights=edge_weights, node_to_predict_mask=node_to_predict_mask,
                           edge_ids=edge_ids, learning_weight=learning_weights)
            
            if last_day == time_window -1:
                all_stock_features[:stock_features.size(0),center_nodes - account_num, :] = stock_features[:,node_to_predict_mask,:].detach() #.squeeze(1)
            else:
                all_stock_features[last_day, center_nodes - account_num, :] = stock_features[-1, node_to_predict_mask, :].detach() #.view(-1)

    return all_stock_features.cpu()

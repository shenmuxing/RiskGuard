import os
# 防随机
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from tqdm import tqdm
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected,add_self_loops, degree, remove_self_loops,subgraph
from torch_geometric.utils.map import map_index
import time
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from collections import defaultdict
import logging
from typing import Optional
import gc
import sys
import argparse

from torch.utils.tensorboard import SummaryWriter

from utils.utils import *
from utils.logger import *
from models import *
from data import *

def parse_args():
    parser = argparse.ArgumentParser(description='Model Training Parameters')
    
    # 必需参数
    parser.add_argument('--data_file', type=str, required=True, help='Path to data file')
    
    # 可选参数
    parser.add_argument('--probability_learning_rate', type=float, default=0, help='Probability learning rate')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to run on')
    parser.add_argument('--delta', type=float, default=1, help='Delta value')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_epochs', type=int, default=6, help='Number of epochs')
    parser.add_argument('--time_window', type=int, default=2, help='Time window')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--clip_min', type=float, default=0.1, help='Minimum clip value')
    parser.add_argument('--clip_max', type=float, default=0.9, help='Maximum clip value')
    parser.add_argument('--tau', type=float, default=1, help='Tau value')
    parser.add_argument('--stock_alpha', type=float, default=0.1, help='Stock alpha')
    parser.add_argument('--stock_beta', type=float, default=0.1, help='Stock beta')
    parser.add_argument('--account_alpha', type=float, default=0.6, help='Account alpha')
    parser.add_argument('--account_beta', type=float, default=0.1, help='Account beta')
    parser.add_argument('--stock_batch_size', type=int, default=256, help='Stock batch size')
    parser.add_argument('--account_batch_size', type=int, default=256, help='Account batch size')
    parser.add_argument('--stock_num_neighbors', type=str, default='4000,200', 
                        help='Stock number of neighbors (format: n1,n2)')
    parser.add_argument('--account_num_neighbors', type=str, default='200,1000', 
                        help='Account number of neighbors (format: n1,n2)')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold value')
    
    return parser.parse_args()

# 模型训练函数
def train_model(model, dataloader, optimizer, criterion, num_epochs, device='cpu', model_type="stock", 
                base_probs=None,tau = 1,debug_train_size = 0,probability_learning_rate = 0,delta = 1,
                test_dataloader = None):
    writer = SummaryWriter() if is_debug else None

    model.to(device)
    start_time = time.time()
    
    # Early stopping variables
    best_loss = float('inf')
    patience = 2
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.1)
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 1000 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}")
                if batch_idx % 9 == 0:
                    gc.collect()
            if is_debug:
                if batch_idx > debug_train_size:
                    break
            account_features, stock_features, edge_indices, edge_weights, edge_ids, point_encs, \
                y_days, learning_weights, center_nodes, nodes, last_day = batch
            
            if __name__ == "__main__" and batch_idx == 0 and epoch == 0:
                print(f"account_features.shape:{account_features.shape}")
                print(f"stock_features.shape:{stock_features.shape}")
                print(f"length of edge_indices:{len(edge_indices)}")
                print(f"edge_weights[-1].shape:{edge_weights[-1].shape}")
                print(f"y_days.shape:{y_days.shape}")
                print(f"center_nodes.shape:{center_nodes.shape}")
                print(f"learning_weiths.unque:{torch.unique(learning_weights)}")

            account_features = account_features.to(device)
            stock_features = stock_features.to(device)
            edge_ids = edge_ids.to(device)
            point_encs = point_encs.bool().to(device)
            y_day = y_days.reshape(-1,).to(device)

            center_nodes = center_nodes.to(device)
            nodes = nodes.to(device)

            node_to_predict_mask, _ = map_index(center_nodes, nodes, inclusive=True)
            # node_to_predict_mask = index_to_mask(node_to_predict_mask, nodes.shape[0]).to(device)
            learning_weights = learning_weights.to(device) if learning_weights is not None else None

            center_nodes = center_nodes.to("cpu")

            optimizer.zero_grad()

            y_pred = model(X_stock=stock_features, X_account=account_features, edge_indices=edge_indices,
                           point_enc=point_encs, edge_weights=edge_weights, node_to_predict_mask=node_to_predict_mask,
                           edge_ids=edge_ids, learning_weight=learning_weights)
  
            if base_probs is not None:
                y_pred = adjust_logits(logits=y_pred, base_probs=base_probs, tau=tau)
            loss = criterion(y_pred, y_day)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            if is_debug and __name__ == "__main__" and epoch == 0 and batch_idx == 0:
                for name, param in model.named_parameters():
                    print(name)

            if is_debug:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'{name}.grad', param.grad, epoch * len(dataloader) + batch_idx)

                writer.add_histogram("model.last_hidden_spatial_layer.conv_grad.edge_grad_storage.grad",
                                    model.last_hidden_spatial_layer.conv_grad.edge_grad_storage.grad,
                                    epoch * len(dataloader) + batch_idx)
            with torch.no_grad():
                abs_gradients = model.last_hidden_spatial_layer.conv_grad.edge_grad_storage.grad.abs().detach().view(1,-1)
                # abs_gradients = model.last_hidden_spatial_layer.conv_grad.edge_grad_storage.abs().detach()
                edge_index_with_grad = model.last_hidden_spatial_layer.conv_grad.edge_index.detach()
                edge_ids = model.last_hidden_spatial_layer.conv_grad.edge_ids.detach()
                edge_attrs = torch.concat((abs_gradients, edge_ids), dim=0)
                _, edge_attrs_to_update = remove_self_loops(edge_index=edge_index_with_grad, edge_attr=edge_attrs.t())
                edge_attrs_to_update = edge_attrs_to_update.t() # shape: (2, num_edges)

                edge_ids = edge_attrs_to_update[1,:].long()
                if edge_ids.numel() == 0:
                    continue
                values = edge_attrs_to_update[0,:]
                edge_id_min = edge_ids.min()
                indices = edge_ids - edge_id_min
                bincount = torch.bincount(input = indices, weights = values)
                edge_attrs_to_update = torch.stack((bincount[indices], edge_ids), dim=0)
                # edge_index_to_update = get_original_node_indices(edge_index_to_update, nodes)
                for center_node in center_nodes:
                    center_node = center_node.item()
                    if (last_day, center_node) not in dataloader.dataset.sampler.sampling_weights:
                        continue
                    one_hop_target_nodes = (dataloader.dataset.sampler.sampling_weights[(last_day, center_node)]).to(device)
                    if round(dataloader.dataset.sampler.alpha * one_hop_target_nodes.size(1)) == one_hop_target_nodes.size(1):
                        continue
                    new_abs_gradients = process_edges_and_gradients(
                                                                    edge_attrs_to_update=edge_attrs_to_update,
                                                                    one_hop_target_nodes=one_hop_target_nodes, 
                                                                    )
                    if new_abs_gradients is None:
                        continue
                    new_probs = step_probabilities(one_hop_target_nodes=one_hop_target_nodes, gradients=new_abs_gradients,
                                                    learning_rate=probability_learning_rate, delta=delta)
                    
                    dataloader.dataset.sampler.update_sampling_weights(day=last_day, center_node=center_node,
                                                                    target_nodes=one_hop_target_nodes[0], 
                                                                    sampling_weights=new_probs.reshape(-1,),
                                                                    edge_weights=one_hop_target_nodes[2],
                                                                    edge_ids=one_hop_target_nodes[3])
        

            optimizer.step()
            total_loss += loss.item()


        # continue
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        if model_type == "account" and ((epoch + 1) in [1,4,5,6,10]) and (test_dataloader is not None):
            test_model(model = model, dataloader = test_dataloader, device=device, model_type="account",base_probs = base_probs,threshold = None) # base_probs=y_account_train_base_probs)
            print(f"Epoch {epoch+1}: Finished testing.")


        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered. No improvement for {patience} epochs.")
                break
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed. Total time: {training_time:.2f} seconds")
        
    # 在训练循环结束时，关闭SummaryWriter
    if is_debug:
        writer.close()

def test_model(model, dataloader, device='cpu', model_type="stock",base_probs= None,threshold = 0.5):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    torch.cuda.empty_cache()
    gc.collect()
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
            # continue
            account_features = account_features.to(device)
            stock_features = stock_features.to(device)
            edge_ids = edge_ids.to(device)
            point_encs = point_encs.bool().to(device)
            y_day = y_days.reshape(-1,).to(device)
            nodes = nodes.to(device)
            center_nodes = center_nodes.to(device)
            node_to_predict_mask, _ = map_index(center_nodes, nodes, inclusive=True)
            learning_weights = learning_weights.to(device) if learning_weights is not None else None

            center_nodes = center_nodes.to("cpu")

            y_pred = model(X_stock=stock_features, X_account=account_features, edge_indices=edge_indices,
                           point_enc=point_encs, edge_weights=edge_weights, node_to_predict_mask=node_to_predict_mask,
                           edge_ids=edge_ids, learning_weight=learning_weights)
            
            # 将预测结果和标签添加到列表中
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(y_day.cpu().numpy())
            account_features = None
            stock_features = None
            edge_indices = None
            edge_weights = None
            edge_ids = None
            point_encs = None
            y_days = None
            learning_weights = None
            center_nodes = None
            nodes = None
            last_day = None
            del account_features, stock_features, edge_indices, edge_weights, edge_ids, point_encs, \
                y_days, learning_weights, center_nodes, nodes, last_day
            
            if batch_idx % 2000 == 0 and model_type == "account":
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.1)
            elif batch_idx % 10 == 0 and model_type == "stock":
                torch.cuda.empty_cache()
                gc.collect()    
                time.sleep(0.1)
    # return None,None,None,None,None
    # 将列表转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    print("all_preds:",all_preds)
    print("positive of all_preds:",all_preds[:,1].sum()/len(all_preds))
    print("all_labels:",all_labels)
    print("positive of all_labels:",all_labels.sum()/len(all_labels))

    all_preds_class = (all_preds[:,1]>threshold).astype(int)
    # all_preds_class = np.argmax(all_preds, axis=1)
    print(f"Threshold:{threshold:.4f}") 
    f1 = f1_score(all_labels, all_preds_class, average='micro')
    roc_auc = roc_auc_score(all_labels, all_preds[:, 1])

    print(f"Model Type: {model_type}")
    print(f"F1: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return 


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

def main(data_file,logger = None,probability_learning_rate = 0,device = "cuda:1",delta = 1,learning_rate = 5e-5,\
          hidden_dim = 256, hidden_layers = 2, num_epochs = 6 ,time_window = 2, train_ratio = 0.8,dropout = 0.1,\
          clip_min = 0.1,clip_max = 0.9,tau = 1,stock_alpha = 0.1, stock_beta = 0.1, account_alpha = 0.6, account_beta = 0.1,\
           stock_batch_size = 256, account_batch_size = 256,stock_num_neighbors = (4000,200), account_num_neighbors = (200,1000) ,\
            num_workers = 10,threshold = 0.5):

    debug_train_size = 10 
    debug_total_days = 29

    num_epochs = num_epochs
    stock_hidden_dim = hidden_dim
    stock_hidden_layers = hidden_layers
    account_hidden_dim = hidden_dim
    account_hidden_layers = hidden_layers
    stock_alpha = stock_alpha
    stock_beta = stock_beta
    account_alpha = account_alpha
    account_beta = account_beta

    stock_batch_size = stock_batch_size

    account_batch_size = account_batch_size

    stock_num_neighbors = stock_num_neighbors
    account_num_neighbors = account_num_neighbors

    num_workers = num_workers

    time_window = time_window

    learning_rate = learning_rate

    threshold = threshold

    print("Experiment:")

    data_dict = load_data(data_file)
    
    # 分割数据集

    if is_debug:
        train_ratio = 0.6
        
        train_size = int(debug_total_days * train_ratio)
        # 股票数据
        X_stock_train, X_stock_test = data_dict['X_stock'][:train_size], data_dict['X_stock'][train_size:debug_total_days]
        y_stock_train, y_stock_test = data_dict['y_stock'][:train_size], data_dict['y_stock'][train_size:debug_total_days]

        y_stock_train_base_probs = compute_base_probs(y_stock_train[0],clip_min = clip_min,clip_max = clip_max) # 只使用第一个时间步的标签计算基准概率
        
        # 账户数据
        X_account_train, X_account_test = data_dict['X_account'][:train_size], data_dict['X_account'][train_size:debug_total_days]
        # mask_account_train, mask_account_test = (X_account_train[:,:,1:7] != 0).any(axis=2),(X_account_test[:,:,1:7] != 0).any(axis=2)

        y_account_train, y_account_test = data_dict['y_account'][:train_size], data_dict['y_account'][train_size:debug_total_days]
        
        # y_account_train_base_probs = compute_base_probs(y_account_train[mask_account_train]) # 只使用第一个时间步的标签计算基准概率
        y_account_train_base_probs = compute_base_probs(y_account_train[0],clip_min=clip_min,clip_max=clip_max) # 只使用第一个时间步的标签计算基准概率

        # 图数据
        G_train, G_test = data_dict['G'][:train_size], data_dict['G'][train_size:debug_total_days]

    else:
        train_ratio = train_ratio

        train_size = int(data_dict['X_stock'].shape[0] * train_ratio)

        # 股票数据
        X_stock_train, X_stock_test = data_dict['X_stock'][:train_size], data_dict['X_stock'][train_size:]
        y_stock_train, y_stock_test = data_dict['y_stock'][:train_size], data_dict['y_stock'][train_size:]

        y_stock_train_base_probs = compute_base_probs(y_stock_train[:train_size],clip_min=clip_min,clip_max=clip_max) # 计算基准概率
        
        # 账户数据
        X_account_train, X_account_test = data_dict['X_account'][:train_size], data_dict['X_account'][train_size:]

        y_account_train, y_account_test = data_dict['y_account'][:train_size], data_dict['y_account'][train_size:]
        
        # y_account_train_base_probs = compute_base_probs(y_account_train[mask_account_train]) # 只使用第一个时间步的标签计算基准概率
        y_account_train_base_probs = compute_base_probs(y_account_train[:train_size],clip_min=clip_min,clip_max=clip_max) # 计算基准概率
    
        # 图数据
        G_train, G_test = data_dict['G'][:train_size], data_dict['G'][train_size:]
        
    # 标准化股票数据
    X_stock_train_norm, stock_mean, stock_std = z_score_normalize(X_stock_train)
    X_stock_test_norm, _, _ = z_score_normalize(X_stock_test, stock_mean, stock_std)
    
    # 标准化账户数据
    X_account_train_norm, account_mean, account_std = z_score_normalize(X_account_train)
    X_account_test_norm, _, _ = z_score_normalize(X_account_test, account_mean, account_std)
    
    # 标准化图数据
    edge_weights = np.concatenate([g[2] for g in G_train])
    edge_mean, edge_std = edge_weights.mean(), edge_weights.std()
    
    for t in range(len(G_train)):
        G_train[t][2] = (G_train[t][2] - edge_mean) / (edge_std + 1e-8)
    for t in range(len(G_test)):
        G_test[t][2] = (G_test[t][2] - edge_mean) / (edge_std + 1e-8)
    
    account_num = X_account_train.shape[1]
    stock_num = X_stock_train.shape[1]
    entity_index = [
        list(range(account_num)),
        list(range(account_num, account_num + stock_num))
    ]

    account_dim = X_account_train.shape[2]
    stock_dim = X_stock_train.shape[2]
    data_dict = None
    X_account_train = None
    X_stock_train = None
    X_account_test = None
    X_stock_test = None
    del data_dict, X_account_train, X_stock_train, X_account_test, X_stock_test
    gc.collect()

    print("X_stock_train_norm.shape, X_stock_test_norm.shape:",X_stock_train_norm.shape, X_stock_test_norm.shape)
    print("y_stock_train.shape, y_stock_test.shape:",y_stock_train.shape, y_stock_test.shape)
    print("X_account_train_norm.shape, X_account_test_norm.shape:",X_account_train_norm.shape, X_account_test_norm.shape)
    print("y_account_train.shape, y_account_test.shape:",y_account_train.shape, y_account_test.shape)
    print("G_train[0].shape, G_test[0].shape:",G_train[0].shape, G_test[0].shape)
    print("y_stock_train_base_probs:", y_stock_train_base_probs)
    print("y_account_train_base_probs:", y_account_train_base_probs)

    # 打印各个数据占用的内存（GB）
    print("\n各数据占用的内存（GB）:")
    print(f"X_stock_train_norm: {get_size_in_gb(X_stock_train_norm):.4f} GB")
    print(f"X_stock_test_norm: {get_size_in_gb(X_stock_test_norm):.4f} GB")
    print(f"y_stock_train: {get_size_in_gb(y_stock_train):.4f} GB")
    print(f"y_stock_test: {get_size_in_gb(y_stock_test):.4f} GB")
    print(f"X_account_train_norm: {get_size_in_gb(X_account_train_norm):.4f} GB")
    print(f"X_account_test_norm: {get_size_in_gb(X_account_test_norm):.4f} GB")
    print(f"y_account_train: {get_size_in_gb(y_account_train):.4f} GB")
    print(f"y_account_test: {get_size_in_gb(y_account_test):.4f} GB")
    print(f"G_train: {get_size_in_gb(G_train):.4f} GB")
    print(f"G_test: {get_size_in_gb(G_test):.4f} GB")

    if train_ratio == 0.8:
        stock_train_dataset = StockAccountDataset(X_stock_train_norm, X_account_train_norm, G_train, y_stock_train, account_num,
                                                time_window=time_window, alpha=stock_alpha, beta=stock_beta, num_neighbors=stock_num_neighbors, 
                                                center_type="stock",batch_size = stock_batch_size,temp_file_path = data_file[:-4] + "_train_dataset.pkl")
    else:
        stock_train_dataset = StockAccountDataset(X_stock_train_norm, X_account_train_norm, G_train, y_stock_train, account_num,
                                                time_window=time_window, alpha=stock_alpha, beta=stock_beta, num_neighbors=stock_num_neighbors, 
                                                center_type="stock",batch_size = stock_batch_size,temp_file_path = None)        

    stock_model = HTGTStockPrediction(entity_index,norm = "batch",account_dim=account_dim,stock_dim = stock_dim ,hidden_layers = stock_hidden_layers,
                                      hidden_dim = stock_hidden_dim,dropout = dropout)
    stock_criterion = nn.CrossEntropyLoss()
    stock_optimizer = torch.optim.Adam(stock_model.parameters(), lr=learning_rate)

    y_stock_train_base_probs = torch.tensor(y_stock_train_base_probs).to(device)
    gc.collect()
    # stock train model

    
    stock_train_dataloader = DataLoader(stock_train_dataset, batch_size= 1, shuffle=False, collate_fn=collate_fn,
                                        num_workers=num_workers,
                                        # multiprocessing_context='spawn'
                                        )

    train_model(stock_model, stock_train_dataloader, stock_optimizer, stock_criterion, num_epochs, \
                device=device, model_type="stock",base_probs = y_stock_train_base_probs,tau= tau,\
                    probability_learning_rate=probability_learning_rate,debug_train_size=debug_train_size,delta = 1)

    new_stock_features_train = generate_stock_features(stock_model, stock_train_dataloader, X_stock_train_norm,stock_hidden_dim=stock_hidden_dim,
                                                        account_num=account_num,time_window=time_window)

    
    stock_optimizer.zero_grad()
    torch.cuda.empty_cache()
    gc.collect()

# %%
    if train_ratio == 0.8:
        stock_test_dataset = StockAccountDataset(X_stock_test_norm, X_account_test_norm, G_test, y_stock_test, account_num,
                                                time_window=time_window, alpha=stock_alpha, beta=stock_beta, num_neighbors=stock_num_neighbors,
                                                center_type="stock",batch_size = stock_batch_size,temp_file_path = data_file[:-4] + "_test_dataset.pkl")
    else:
        stock_test_dataset = StockAccountDataset(X_stock_test_norm, X_account_test_norm, G_test, y_stock_test, account_num,
                                                time_window=time_window, alpha=stock_alpha, beta=stock_beta, num_neighbors=stock_num_neighbors,
                                                center_type="stock",batch_size = stock_batch_size,temp_file_path =  None)        
        stock_test_dataloader = DataLoader(stock_test_dataset, batch_size= 1, shuffle=False, collate_fn=collate_fn,
                                           num_workers=num_workers,
                                        #    multiprocessing_context='spawn'
                                           )

    new_stock_features_test = generate_stock_features(stock_model, stock_test_dataloader, X_stock_test_norm,
                                                      stock_hidden_dim=stock_hidden_dim, account_num=account_num,time_window=time_window)

    stock_model.zero_grad()
    torch.cuda.empty_cache()
    gc.collect()

    account_train_dataset = stock_train_dataset
    account_train_dataset.reset(new_stock_features_train.numpy(), X_account_train_norm, G_train, y_account_train, account_num,
                                time_window=time_window, center_type="account",num_neighbors=account_num_neighbors, 
                                alpha=account_alpha,beta = account_beta, batch_size= account_batch_size)
    
    account_test_dataset = stock_test_dataset
    account_test_dataset.reset(new_stock_features_test.numpy(), X_account_test_norm, G_test, y_account_test, account_num,
                                time_window=time_window, center_type="account",num_neighbors=account_num_neighbors, 
                                alpha=account_alpha,beta = account_beta, batch_size= account_batch_size)
    
    account_train_dataloader = DataLoader(account_train_dataset, batch_size= 1, shuffle=False, collate_fn=collate_fn,
                                          num_workers=num_workers,
                                          )
    account_test_dataloader = DataLoader(account_test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                         num_workers =num_workers,
                                         )
    
    stock_dim = new_stock_features_train.size(2)

    account_model = HTGTAccountPrediction(entity_index,stock_dim=stock_dim,account_dim=account_dim,hidden_layers=account_hidden_layers,
                                          hidden_dim =account_hidden_dim,norm = "batch",dropout = dropout)
    
    account_criterion = nn.CrossEntropyLoss()
    account_optimizer = torch.optim.Adam(account_model.parameters(), lr=learning_rate)
    
    y_account_train_base_probs = torch.tensor(y_account_train_base_probs).to(device)
    gc.collect()

    train_model(account_model, account_train_dataloader, account_optimizer, account_criterion, num_epochs = num_epochs,\
                device =device, model_type="account",base_probs=y_account_train_base_probs,tau=tau,
                probability_learning_rate=probability_learning_rate,debug_train_size=debug_train_size,delta = delta,\
                    test_dataloader = account_test_dataloader)

    account_optimizer.zero_grad()
    torch.cuda.empty_cache()
    gc.collect()
    account_test_results = test_model(account_model, account_test_dataloader, device=device, model_type="account",base_probs = y_account_train_base_probs,threshold = threshold) 

    account_model.save_model("models/"+data_file[-8:-4] + "_account_model.pth")
    stock_model.save_model("models/data_"+data_file[-8:-4] + "_stock_model.pth")

    print(f"{data_file} Runing successfully")

if __name__ == '__main__':
    class LoggerConfig:
        def __init__(self):
            self.log_level = logging.DEBUG
            self.log_file = os.path.join('logs', 'app.log')
            self.log_max_size = 5 * 1024 * 1024  # 5 MB
            self.log_backup_count = 3
    
    # 设置logger
    cfg = LoggerConfig()
    logger = setup_logger(cfg)
    
    # 解析命令行参数
    args = parse_args()
    
    # 处理特殊参数
    args_dict = vars(args)
    args_dict['stock_num_neighbors'] = tuple(map(int, args.stock_num_neighbors.split(',')))
    args_dict['account_num_neighbors'] = tuple(map(int, args.account_num_neighbors.split(',')))
    args_dict['logger'] = logger
    
    # 调用main函数
    main(**args_dict)
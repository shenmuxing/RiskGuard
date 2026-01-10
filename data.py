import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import index_to_mask, mask_to_index
from torch_geometric.utils import to_undirected,add_self_loops, degree, remove_self_loops,subgraph
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pickle
# EdgeWeightSampler
class EdgeWeightSampler:
    def __init__(self, num_nodes, num_days, alpha, beta, 
                 num_accounts, num_stocks, sample_type,sample_for_test = True):
        self.num_nodes = num_nodes
        self.num_days = num_days
        self.num_accounts = num_accounts
        self.num_stocks = num_stocks
        self.sample_type = sample_type # account or stock
        self.sample_for_test = sample_for_test

        self.alpha = alpha
        self.beta = beta
        self.sampling_weights = {}
        self.edge_indices = {}
        self.edge_id_map = {}
        self.next_edge_id = 0

    def reset(self, num_nodes, num_days, alpha, beta, num_accounts, 
              num_stocks, sample_type,sample_for_test = True):
        self.num_nodes = num_nodes
        self.num_days = num_days
        self.num_accounts = num_accounts
        self.num_stocks = num_stocks
        self.sample_type = sample_type # account or stock
        self.sample_for_test = sample_for_test

        self.alpha = alpha
        self.beta = beta

    def get_edge_id(self, day, src, dst):
        key = (day, src, dst)
        if key not in self.edge_id_map:
            self.edge_id_map[key] = self.next_edge_id
            self.edge_id_map[(day, dst, src)] = self.next_edge_id
            self.next_edge_id += 1
        return self.edge_id_map[key]

    def update_sampling_weights(self, day, center_node, target_nodes,
                                edge_weights = None,edge_ids = None,
                                sampling_weights=None):
        if sampling_weights is None:
            sampling_weights = torch.ones(len(target_nodes)) / len(target_nodes)
        if edge_ids is None:
            edge_ids = torch.as_tensor([self.get_edge_id(day, center_node, target.item()) for target in target_nodes])
        self.sampling_weights[(day, center_node)] = torch.stack([target_nodes.detach().cpu(), sampling_weights.detach().cpu(), 
                                                                 edge_weights.detach().cpu(), edge_ids.detach().cpu()])

    def sample_neighbors(self, nodes, day,num_samples_first, num_samples_second):
        """
        number_samples_first: 第一层邻居的最大数量
        """

        # first_hops 和 learning_weights 长度一定相等，因此进行合并, 总之包括5列  0 表示 src ,1 表示 dst,
        #  2 表示 learning_weights, 3 表示 edge_weight, 4 表示 edge_id
        # seconde_hops 同上面的想法
        first_hops = []

        first_neighbors = torch.zeros(size = (self.num_nodes if self.sample_type == "account" else self.num_accounts,),dtype = torch.bool)
        second_neighbors = index_to_mask(nodes.long(),
                                         size = self.num_accounts if self.sample_type == "account" else self.num_nodes)

        nodes = nodes.tolist()
        for node in nodes:
            if (day, node) in self.sampling_weights:
                # target_nodes: (4, num_neighbors), 其中0 为节点id, 1 为抽样概率, 2 为边权重, 3 为 edge_id
                target_nodes = self.sampling_weights[(day, node)] 

                k = round(target_nodes.size(1) * self.alpha)
                if k == target_nodes.size(1):
                    num_remaining = 0
                    top_indices = torch.arange(k)
                elif k == 0:
                    num_remaining = int(np.ceil((target_nodes.size(1)) * self.beta))
                    top_indices = torch.tensor([],dtype = torch.long)
                else:
                    num_remaining = round((target_nodes.size(1)) * self.beta)
                    _, top_indices = torch.topk(target_nodes[1], k)

                top_alpha_neighbors = target_nodes[:,top_indices] # (4, k)

                first_hop = torch.ones(5, k + num_remaining, dtype = torch.float)
                first_hop[0,:].fill_(node)
                first_hop[1,0:k] = top_alpha_neighbors[0]
                first_hop[3:,0:k] = top_alpha_neighbors[2:]

                if num_remaining > 0:
                    remaining_mask = torch.ones(target_nodes.size(1), dtype=torch.bool)
                    remaining_mask[top_indices] = False
                    remaining_indices = mask_to_index(remaining_mask)
                    perm = torch.randperm(remaining_indices.size(0))

                    # 使用 torch.multinomial 进行抽样
                    # selected_indices = torch.multinomial(remaining_mask, num_remaining, replacement=False)
                    selected_remaining_indices = remaining_indices[perm[:num_remaining]]
                    selected_remaining = target_nodes[:,selected_remaining_indices]

                    first_hop[1,k:] = selected_remaining[0]
                    first_hop[2,k:].fill_((1 - self.alpha) / self.beta)
                    first_hop[3:,k:] = selected_remaining[2:]

                first_hops.append(first_hop)
                first_neighbors[first_hop[1,:].long()] = True
            else:
                if not self.sample_for_test:
                    second_neighbors[node] = False
        
        center_nodes = mask_to_index(second_neighbors)
        if len(first_hops) == 0:
            return center_nodes,None,center_nodes
        
        # 利用bool类型只有0和1两种状态，即可将重复节点去掉
        if first_neighbors.sum() < num_samples_first:
            first_neighbors = mask_to_index(first_neighbors)
        else:
            first_neighbors = torch.multinomial(first_neighbors.float(), num_samples_first, replacement=False)

        second_hops = []
        # second_hops 恐怕只能先用数组存储
        for neighbor in first_neighbors:
            if (day, neighbor.item()) in self.sampling_weights:
                second_target_nodes = self.sampling_weights[(day, neighbor.item())]
                if second_target_nodes.size(1) >= num_samples_second:
                    second_hop = torch.ones(5, num_samples_second, dtype = torch.float)
                    second_hop[0,:] = neighbor.item()
                    second_indices = torch.randperm(second_target_nodes.size(1))[:num_samples_second]
                    second_hop[1,:] = second_target_nodes[0,second_indices]
                    second_hop[3:,:] = second_target_nodes[2:,second_indices]
                else:
                    second_hop = torch.ones(5, second_target_nodes.size(1), dtype = torch.float)
                    second_hop[0,:] = neighbor.item()
                    second_hop[1,:] = second_target_nodes[0,:]
                    second_hop[3:,:] = second_target_nodes[2:,:]

                second_hops.append(second_hop)
                second_neighbors[second_hop[1,:].long()] = True

        second_neighbors = mask_to_index(second_neighbors)
        all_neighbors = torch.cat([first_neighbors,second_neighbors],dim = 0)

        if len(second_hops) == 0:
            # 这个不用去重，因为不可能重复
            sampled_edges = torch.concat(first_hops, dim=1)
            # 用to_undirected 转化为无向图
            sampled_edges_index, sampled_edges_attrs = to_undirected(edge_index = sampled_edges[:2,:].long(),
                                                                     edge_attr = sampled_edges[2:,:].t().float(),reduce = "max")
            sampled_edges = torch.cat([sampled_edges_index,sampled_edges_attrs.t()],dim = 0)
            return all_neighbors, sampled_edges,center_nodes
        else:
            # 这里得去重，使用to_undirected 即可去重
            first_hops = torch.concat(first_hops, dim=1)
            second_hops = torch.concat(second_hops, dim=1)
            sampled_edges = torch.concat([first_hops, second_hops], dim=1)
            # 用to_undirected 转化为无向图
            sampled_edges_index, sampled_edges_attrs = to_undirected(edge_index = sampled_edges[:2,:].long(),\
                                                                     edge_attr = sampled_edges[2:,:].t().float(),reduce = "max")
            sampled_edges = torch.cat([sampled_edges_index,sampled_edges_attrs.t()],dim = 0)
            return all_neighbors, sampled_edges,center_nodes

# %%

# StockAccountDataset 类定义
class StockAccountDataset(Dataset):
    def __init__(self, X_stock, X_account, G, y, account_num, time_window, center_type='account',\
                  num_neighbors=(10, 5), alpha=0.1, beta=0.1,batch_size = 1,temp_file_path = None):
        """
        batch_size: 由Dataset来判断batch_size的大小
        """

        self.X_stock = torch.as_tensor(X_stock, dtype=torch.float32)
        self.X_account = torch.as_tensor(X_account, dtype=torch.float32)
        self.G = G
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.account_num = account_num
        self.num_stocks = X_stock.shape[1]
        self.total_days = X_stock.shape[0]
        self.total_nodes = account_num + self.num_stocks
        
        self.time_window = time_window
        self.center_type = center_type
        self.num_neighbors = num_neighbors
        
        self.batch_size = batch_size

        self.sampler = EdgeWeightSampler(num_nodes = self.total_nodes,num_days = self.total_days,alpha =  alpha,beta = beta,\
                                         num_accounts = self.account_num,num_stocks = self.num_stocks,sample_type = center_type)
        self.build_graph(temp_file_path)

        if self.center_type == 'account':
            self.center_nodes = torch.arange(self.account_num)
            self.center_node_mask = index_to_mask(self.center_nodes, size=self.account_num)
        else:
            self.center_nodes = torch.arange(self.account_num, self.account_num + self.num_stocks)
            self.center_node_mask = index_to_mask(self.center_nodes - self.account_num, size=self.num_stocks)

        # self.now_idx = torch.tensor([0])
        self.get_map()
    def get_map(self):
        self.map = {}
        length = len(self)
        self.now_idx = 0
        for idx in range(length):
            center_idxs = []
            start_idx = self.now_idx  # 包括当前idx
        
            last_idx = -1
            for i in range(self.batch_size):
                overall_now_idx = start_idx + i

                now_idx = overall_now_idx % len(self.center_nodes)
                if now_idx < last_idx:
                    break
                center_idxs.append(now_idx)
                last_idx = now_idx
                self.now_idx += 1

            start_day = (start_idx // self.num_stocks) if self.center_type == "stock" else (start_idx // self.account_num)

            self.map[idx] = {
                "start_day" : start_day,
                "center_idxs" : center_idxs,
            }
    def build_graph(self, temp_file_path=None):
        self.edge_index = defaultdict(list)
        self.edge_attr = defaultdict(list)
        temp_sampling_weights = defaultdict(lambda: (set(), dict()))
        
        # 如果提供了临时文件路径且文件存在，则尝试加载
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                with open(temp_file_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.edge_index = saved_data['edge_index']
                    self.edge_attr = saved_data['edge_attr']
                    temp_sampling_weights = saved_data['temp_sampling_weights']
            except Exception as e:
                print(f"Error loading from temp file: {e}")
                # 如果加载失败，继续执行原始的构建过程
                self._build_graph_original(temp_sampling_weights)
        else:
            # 如果没有提供临时文件或文件不存在，执行原始的构建过程
            self._build_graph_original(temp_sampling_weights)
            
            # 如果提供了临时文件路径，保存中间结果
            if temp_file_path:
                try:
                    with open(temp_file_path, 'wb') as f:
                        pickle.dump({
                            'edge_index': dict(self.edge_index),
                            'edge_attr': dict(self.edge_attr),
                            'temp_sampling_weights': dict(temp_sampling_weights)
                        }, f)
                except Exception as e:
                    print(f"Error saving to temp file: {e}")

        # 继续处理temp_sampling_weights
        for (day, node), (neighbors, weights) in temp_sampling_weights.items():
            target_nodes = torch.as_tensor(list(neighbors))
            edge_weights = torch.as_tensor([weights[n] for n in neighbors])
            self.sampler.update_sampling_weights(day, node, target_nodes, edge_weights=edge_weights)
        temp_sampling_weights = None
        del temp_sampling_weights

        # 转换为无向图
        for day in range(self.total_days):
            self.edge_index[day], self.edge_attr[day] = to_undirected(
                edge_index=torch.tensor(self.edge_index[day], dtype=torch.long).t(),
                edge_attr=torch.tensor(self.edge_attr[day], dtype=torch.float))

    def _build_graph_original(self, temp_sampling_weights):
        """原始的图构建过程"""
        for day, day_data in enumerate(self.G):
            src, dst, weight = day_data  # 直接解包三个行向量

            # 使用 zip 将行向量转换为边的列表
            edges = list(zip(src, dst))
            self.edge_index[day] = edges
            self.edge_attr[day] = weight.tolist()

            # 批量更新临时权重信息
            for (s, d), w in zip(edges, weight):
                temp_sampling_weights[(day, s)][0].add(d)
                temp_sampling_weights[(day, s)][1][d] = w
                temp_sampling_weights[(day, d)][0].add(s)
                temp_sampling_weights[(day, d)][1][s] = w

    def reset(self, X_stock, X_account, G, y, account_num, time_window, \
                 center_type='account', num_neighbors=(10, 5), alpha=0.1, beta=0.1,batch_size = 1):
        """
        直接reset,输入的是kwargs属性
        """
        self.X_stock = torch.as_tensor(X_stock, dtype=torch.float32)
        self.X_account = torch.as_tensor(X_account, dtype=torch.float32)
        self.G = G
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.account_num = account_num
        self.num_stocks = X_stock.shape[1]
        self.total_days = X_stock.shape[0]
        self.total_nodes = account_num + self.num_stocks
        
        self.time_window = time_window
        self.center_type = center_type
        self.num_neighbors = num_neighbors
        
        self.batch_size = batch_size

        self.sampler.reset(num_nodes = self.total_nodes,num_days = self.total_days,alpha =  alpha,beta = beta,\
                             num_accounts = self.account_num,num_stocks = self.num_stocks,sample_type = center_type)

        if self.center_type == 'account':
            self.center_nodes = torch.arange(self.account_num)
            self.center_node_mask = index_to_mask(self.center_nodes, size=self.account_num)
        else:
            self.center_nodes = torch.arange(self.account_num, self.account_num + self.num_stocks)
            self.center_node_mask = index_to_mask(self.center_nodes - self.account_num, size=self.num_stocks)

        # self.now_idx = torch.tensor([0])
        self.get_map()

    def __len__(self):
        if len(self.center_nodes) % self.batch_size == 0:
            return (len(self.center_nodes)  // self.batch_size) * (self.total_days - self.time_window + 1)           
        else:
            return (len(self.center_nodes) // self.batch_size) * (self.total_days - self.time_window + 1)  + self.total_days - self.time_window + 1    

    def build_subgraph(self, nodes, day):
        
        sub_edge_index, sub_edge_attr = subgraph(
            nodes,
            self.edge_index[day],
            self.edge_attr[day],
            relabel_nodes=True,
            return_edge_mask=False,
            num_nodes = self.total_nodes
        )

        return sub_edge_index, sub_edge_attr
    
    def sample_subgraph(self, center_nodes, day):
        sampled_nodes, sampled_edges,center_nodes = self.sampler.sample_neighbors(center_nodes, day,num_samples_first= self.num_neighbors[0],
                                                                                  num_samples_second= self.num_neighbors[1])

        if sampled_edges is None:
            return sampled_nodes, torch.tensor([],dtype=torch.long).view(2,0), torch.tensor([],dtype=torch.float).view(3,0), center_nodes

        # return sampled_nodes, sampled_edges[:2,:].long(), sampled_edges[2:,:].t().float(), center_nodes
    
        sub_edge_index, sub_edge_attr = subgraph(
            sampled_nodes,
            edge_index = sampled_edges[:2,:].long(),
            edge_attr = sampled_edges[2:,:].t().float(),
            relabel_nodes=True,
            return_edge_mask=False,
            num_nodes = self.total_nodes
        )

        return sampled_nodes,sub_edge_index, sub_edge_attr.t(), center_nodes
    
    def __getitem__(self, idx):
        
        start_day = self.map[idx]["start_day"]
        center_idxs = self.map[idx]["center_idxs"]
        
        center_nodes = self.center_nodes[center_idxs]
        
        last_day = start_day + self.time_window - 1
        nodes, edge_index,edge_attrs,center_nodes = self.sample_subgraph(center_nodes, last_day)

        learning_weights = edge_attrs[0]
        edge_weight = edge_attrs[1]
        edge_ids  = edge_attrs[2]

        edge_indices = []
        edge_weights = []

        account_mask = nodes < self.account_num

        point_enc = account_mask.long()
        account_num = point_enc.sum()
        stock_num = nodes.shape[0] - account_num
        nodes_account_mask = nodes[account_mask]
        account_features = torch.empty(self.time_window,account_num, self.X_account[0].size(1),dtype = torch.float32)
        stock_features = torch.empty(self.time_window,stock_num, self.X_stock[0].size(1),dtype = torch.float32)

        nodes_stock_mask = nodes[~account_mask] - self.account_num
        for day in range(start_day, start_day + self.time_window - 1):
            current_edge_index, current_edge_weight = self.build_subgraph(nodes, day)

            account_features[day - start_day] = self.X_account[day][nodes_account_mask]
            stock_features[day - start_day] = self.X_stock[day][nodes_stock_mask]
            
            edge_indices.append(current_edge_index)
            edge_weights.append(current_edge_weight)

        account_features[-1] = self.X_account[last_day][nodes_account_mask]
        stock_features[-1] = self.X_stock[last_day][nodes_stock_mask]
        
        edge_indices.append(edge_index)
        edge_weights.append(edge_weight)

        # point_encs.append(point_enc)
        # center_node_mask   = index_to_mask(center_nodes, self.total_nodes)
        if self.center_type == "stock":
            # center_node_mask = index_to_mask(center_nodes - self.account_num, self.num_stocks)
            # self.center_node_mask.fill_(False)
            # self.center_node_mask[center_nodes - self.account_num] = True
            y = self.y[last_day][center_nodes - self.account_num].unsqueeze(0)
        else:
            # self.center_node_mask.fill_(False)
            # self.center_node_mask[center_nodes] = True
            y = self.y[last_day][center_nodes].unsqueeze(0)

        learning_weights = torch.tensor(learning_weights)

        return (
            account_features, 
            stock_features, 
            edge_indices, 
            edge_weights, 
            edge_ids, 
            point_enc, # point_enc 并不能直接用在account_features和stock_features上。
            y,
            learning_weights, 
            center_nodes, 
            nodes, 
            last_day
        )
    
def collate_fn(batch):
    """
    一个简单的 collate 函数，不对批次进行任何处理。
    
    Args:
        batch (list): 包含单个元素的列表，因为 batch_size 始终为 1。
    
    Returns:
        tuple: 返回批次中的单个元素。
    """
    # assert len(batch) == 1, "批次大小应该始终为 1"
    return batch[0]


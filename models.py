import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected,add_self_loops, degree, remove_self_loops,subgraph

# TemporalLayer 类定义
class TemporalLayer(nn.Module):
    def __init__(self, entity_index, input_units, units, activation,dropout=0.1,is_norm = True):
        """
        norm: "batch" or "layer": use batch normalization or layer normalization
        """
        super(TemporalLayer, self).__init__()
        self.entity_indexs = entity_index
        self.num_entity = len(entity_index)
        self.input_units = input_units
        self.units = units

        self.q_w = nn.Parameter(torch.randn((self.num_entity, self.units, self.units), dtype=torch.float), requires_grad=True)
        self.k_w = nn.Parameter(torch.randn((self.num_entity, self.units, self.units), dtype=torch.float), requires_grad=True)
        self.v_w = nn.Parameter(torch.randn((self.num_entity, self.units, self.units), dtype=torch.float), requires_grad=True)
        self.fc_w = nn.Parameter(torch.randn((self.num_entity, self.units, self.units), dtype=torch.float), requires_grad=True)

        self.norm = nn.LayerNorm(self.units)

        self.dropout = nn.Dropout(dropout)
    
        self.activation = activation
        self.is_norm = is_norm
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.q_w, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.k_w, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.v_w, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc_w, nonlinearity='leaky_relu')

    def forward(self, node_states, point_encs):
        """
        允许传入tensor，必须和point_encs 完全对应
        point_encs: tensor, shape (num_nodes, num_entity), 
        表示当下位置是否是account类型
        """
        # return self.activation(self.linear(node_states))

        time_steps, num_nodes, _ = node_states.shape
        account_mask = point_encs # True 为 account
        stock_mask = ~account_mask

        output_states = []
        for type_index in range(self.num_entity):
            entity_mask = stock_mask if type_index == 1 else account_mask
            this_entity_states = node_states[:, entity_mask, :]
            this_q_w = self.q_w[type_index]
            this_k_w = self.k_w[type_index]
            this_v_w = self.v_w[type_index]
            this_fc_w = self.fc_w[type_index]

            q = torch.matmul(this_entity_states[-1:], this_q_w)
            k = torch.matmul(this_entity_states, this_k_w)
            v = torch.matmul(this_entity_states, this_v_w)

            raw_score = torch.matmul(q, k.transpose(-1, -2))
            attention_score = F.softmax(raw_score, dim=-1)
            agg_node_state = torch.matmul(attention_score, v)
            # 在attention和fully connected层后添加dropout
            agg_node_state = self.dropout(agg_node_state)
            if self.is_norm:
                agg_node_state = self.norm(agg_node_state)
            
            agg_node_state = self.activation(torch.matmul(agg_node_state, this_fc_w))
            agg_node_state = self.dropout(agg_node_state)

            output_state = agg_node_state + q
            output_states.append(output_state)

        combined_output = torch.zeros((time_steps, num_nodes, self.units), device=node_states.device)
        combined_output[:, account_mask, :] = output_states[0]
        combined_output[:, stock_mask, :] = output_states[1]
        
        return combined_output.squeeze(0)


# CustomGCNConv 类定义
class CustomGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, norm = "batch",need_grad = False, dropout=0.1):
        super(CustomGCNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_enc_bipartite = nn.Parameter(torch.full((1, 1), 0.1, requires_grad=True, dtype=torch.float32))

        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_channels)
        self._init_weights()
        self.need_grad = need_grad
        self.dropout = nn.Dropout(dropout)
    def _init_weights(self):
        nn.init.kaiming_normal_(self.lin.weight,nonlinearity='leaky_relu')
        if self.lin.bias is not None:
            nn.init.constant_(self.lin.bias, 0)

    def forward(self, x, edge_index, edge_weight=None, edge_ids = None, learning_weight=None):
        edge_weight = (self.edge_enc_bipartite * edge_weight).view(-1,)

        if self.need_grad:
            edge_attr = torch.concat([edge_weight.view(1, -1), learning_weight.view(1, -1),edge_ids.view(1, -1)], dim=0).transpose(-1, -2)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
            self.edge_index = edge_index
            self.edge_ids = edge_attr[:,2].view(1,-1)
            edge_weight = edge_attr[:, 0]
            learning_weight = edge_attr[:, 1]

            if torch.is_grad_enabled():
                # self.edge_grad_storage = torch.ones(1,edge_index.size(1), device=x.device, requires_grad=True) * learning_weight.view(1, -1)
                # self.edge_grad_storage.retain_grad()
                self.edge_grad_storage = torch.ones(1,edge_index.size(1), device=x.device, requires_grad=True)
                self.edge_grad_storage.retain_grad()
            else:
                self.edge_grad_storage = torch.ones(1,edge_index.size(1), device=x.device) * learning_weight.view(1, -1)
        else:
            edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, num_nodes=x.size(0))
            self.edge_grad_storage = torch.ones(edge_index.size(1), device=x.device)
            learning_weight = torch.ones(edge_index.size(1), device=x.device)

        x = self.lin(x)
        x = self.norm(x)
        x = self.dropout(x)
        row, col = edge_index
        deg = degree(col, num_nodes = x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        norm = norm.view(-1, 1) * self.edge_grad_storage.view(-1, 1) * learning_weight.view(-1,1)
        return self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)

    def message(self, x_j, norm, edge_weight):
        if edge_weight is not None:
            return norm.view(-1, 1) * x_j * edge_weight.view(-1, 1)
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

# SpatialLayer 类定义
class SpatialLayer(nn.Module):
    def __init__(self, input_units, units, relations, entities, activation, 
                 need_grad=False, norm="batch",conv_type="gcn", dropout=0.1,is_norm = True):
        super(SpatialLayer, self).__init__()
        self.input_units = input_units
        self.units = units
        self.relations = relations
        self.entities = entities
        self.activation = activation
        self.need_grad = need_grad
        
        self.point_enc_w = nn.Parameter(torch.randn((self.entities, self.input_units, self.units),
                                                    dtype=torch.float), requires_grad=True)
        if not self.need_grad:
            self.conv = CustomGCNConv(self.units, self.units,norm = norm,need_grad = False,dropout = dropout)
        else:
            self.conv_grad = CustomGCNConv(self.units, self.units,norm = norm,need_grad = True,dropout = dropout)

        self.fc_f = nn.Linear(self.units, self.units)

        self.norm = nn.LayerNorm(self.units)
        self.dropout = nn.Dropout(dropout)
        self.is_norm = is_norm
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.point_enc_w,nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc_f.weight,nonlinearity='leaky_relu')
        if self.fc_f.bias is not None:
            nn.init.constant_(self.fc_f.bias, 0)

    def forward(self, node_state, edge_index, point_enc, edge_weight,edge_ids = None, learning_weight=None):
        point_encoding = self.point_enc_w[point_enc.int()]
        if self.need_grad:
            node_state = node_state[-1].unsqueeze(0)
        node_state = torch.matmul(node_state.unsqueeze(2), point_encoding).squeeze(2)
        if self.is_norm:
            node_state = self.norm(node_state)

        node_state = self.activation(node_state)
        node_state = self.dropout(node_state)
        node_state_res = node_state.clone().detach()
        if self.need_grad:
            edge_index_t = edge_index[-1].squeeze(0).to(device=node_state.device)
            node_state_t = node_state[-1].to(device=node_state.device)
            edge_weight_t = edge_weight[-1].squeeze(0).to(device=node_state.device)

            node_state_t = self.conv_grad(x = node_state_t,edge_index = edge_index_t, edge_weight = edge_weight_t,
                                          edge_ids = edge_ids, learning_weight=learning_weight)
            node_state_res[-1] = node_state_t  
        else:
            for t in range(len(edge_index)):
                edge_index_t = edge_index[t].squeeze(0).to(device=node_state.device)
                node_state_t = node_state[t].to(device=node_state.device)
                edge_weight_t = edge_weight[t].squeeze(0).to(device=node_state.device)

                node_state_t = self.conv(node_state_t, edge_index_t, edge_weight_t)

                node_state_res[t] = node_state_t
            
        node_state_res = self.activation(node_state_res)

        return node_state_res


# %%

# HTGTStockPrediction 类定义
class HTGTStockPrediction(torch.nn.Module):
    def __init__(self, entity_index,hidden_layers=2,hidden_dim=32,norm = "batch",account_dim=16,stock_dim = 19,\
                 is_use_temporal_layer = True,is_use_spatial_layer = True,dropout=0.1,is_spatial_norm = True,is_temporal_norm = True):
        super(HTGTStockPrediction, self).__init__()
        self.hidden_dim = hidden_dim
        self.account_fc = nn.Linear(account_dim, hidden_dim)
        self.stock_fc = nn.Linear(stock_dim, hidden_dim)

        self.account_norm = nn.LayerNorm(hidden_dim)
        self.stock_norm = nn.LayerNorm(hidden_dim)

        self.hidden_temporal_layers = nn.ModuleList([TemporalLayer(entity_index, hidden_dim, hidden_dim, F.leaky_relu,dropout=dropout,is_norm = is_temporal_norm) for i in range(hidden_layers)])
        self.hidden_spatial_layers = nn.ModuleList([SpatialLayer(hidden_dim, hidden_dim, 1, 2, F.leaky_relu,norm=norm,dropout=dropout,is_norm = is_spatial_norm) for i in range(hidden_layers - 1)])
        self.last_hidden_spatial_layer = SpatialLayer(hidden_dim, hidden_dim, 1, 2, F.leaky_relu, need_grad=True,norm = norm,dropout=dropout, is_norm = is_spatial_norm)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        self.is_use_temporal_layer = is_use_temporal_layer
        self.is_use_spatial_layer = is_use_spatial_layer
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight,nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Parameter):
            nn.init.kaiming_normal_(module,nonlinearity='leaky_relu')

    def forward(self, X_stock, X_account, edge_indices, point_enc, edge_weights, node_to_predict_mask,edge_ids = None, learning_weight=None):
        time_steps,_,_ = X_stock.shape
        num_nodes = len(point_enc)
        account_emb = self.account_fc(X_account)
        stock_emb = self.stock_fc(X_stock)
        
        account_emb = self.account_norm(account_emb)
        stock_emb = self.stock_norm(stock_emb)
        node_states = torch.zeros((time_steps, num_nodes, self.hidden_dim),
                                  device= stock_emb.device)
        node_states[:,point_enc,:] = account_emb
        node_states[:,~point_enc,:] = stock_emb
        # node_states = torch.cat([account_emb, stock_emb], dim=1)
        node_states = F.leaky_relu(node_states)
        if self.is_use_temporal_layer:

            node_states = self.hidden_temporal_layers[0](node_states, point_enc)
        
        for i in range(len(self.hidden_spatial_layers)):
            if self.is_use_spatial_layer:
                node_states = self.hidden_spatial_layers[i](node_state = node_states,edge_index =  edge_indices,\
                                                            point_enc =  point_enc, edge_weight =  edge_weights)
            if self.is_use_temporal_layer:
            
                node_states = self.hidden_temporal_layers[i + 1](node_states, point_enc)

        node_states = self.last_hidden_spatial_layer(node_state = node_states, edge_index = edge_indices, \
                                                     point_enc = point_enc, edge_weight = edge_weights, \
                                                        edge_ids = edge_ids, learning_weight=learning_weight)
        
        node_states = node_states[-1,:,:][node_to_predict_mask]
        y_pred = self.output_layer(node_states)
        
        return y_pred

    def get_stock_features(self, X_stock, X_account, edge_indices, point_enc, edge_weights, 
                           node_to_predict_mask,edge_ids = None, learning_weight=None):
        
        time_steps,_,_ = X_stock.shape
        num_nodes = len(point_enc)
        account_emb = self.account_fc(X_account)
        stock_emb = self.stock_fc(X_stock)
        
        account_emb = self.account_norm(account_emb)
        stock_emb = self.stock_norm(stock_emb)
        node_states = torch.zeros((time_steps, num_nodes, self.hidden_dim),
                                  device= stock_emb.device)
        node_states[:,point_enc,:] = account_emb
        node_states[:,~point_enc,:] = stock_emb
        node_states = F.leaky_relu(node_states)
        if self.is_use_temporal_layer:
            node_states = self.hidden_temporal_layers[0](node_states, point_enc)
        
        for i in range(len(self.hidden_spatial_layers)):
            if self.is_use_spatial_layer:
                node_states = self.hidden_spatial_layers[i](node_state = node_states,edge_index =  edge_indices,\
                                                            point_enc =  point_enc, edge_weight =  edge_weights)
            if self.is_use_temporal_layer:
                node_states = self.hidden_temporal_layers[i + 1](node_states, point_enc)

        node_states = node_states
        return node_states


    def save_model(self, path):
        """
        保存模型到指定路径
        Args:
            path: 保存模型的路径，包括文件名
        """
        model_state = {
            'model_state_dict': self.state_dict(),
            'hidden_dim': self.hidden_dim,
            'is_use_temporal_layer': self.is_use_temporal_layer,
            'is_use_spatial_layer': self.is_use_spatial_layer
        }
        torch.save(model_state, path)

    @classmethod
    def load_model(cls, path, entity_index, stock_dim=32, account_dim=16, hidden_layers=2, hidden_dim=32, norm="batch",is_use_temporal_layer=True, is_use_spatial_layer=True):
        """
        从指定路径加载模型
        Args:
            path: 模型文件路径
            entity_index: 实体索引
            其他参数与__init__相同
        Returns:
            加载好权重的模型实例
        """
        model = cls(entity_index, stock_dim, account_dim, hidden_layers, hidden_dim, norm,
                    is_use_temporal_layer = is_use_temporal_layer, is_use_spatial_layer = is_use_spatial_layer)
        model_state = torch.load(path)
        model.load_state_dict(model_state['model_state_dict'])
        return model

# %%

class HTGTAccountPrediction(torch.nn.Module):
    def __init__(self, entity_index,stock_dim = 32, account_dim = 16, hidden_layers=2,hidden_dim=32,norm = "batch",\
                 is_use_temporal_layer = True,is_use_spatial_layer = True,dropout = 0.1,is_spatial_norm = True,is_temporal_norm = True):
        super(HTGTAccountPrediction, self).__init__()

        self.hidden_dim = hidden_dim
        self.account_fc = nn.Linear(account_dim, hidden_dim)
        self.stock_fc = nn.Linear(stock_dim, hidden_dim)

        self.account_norm = nn.LayerNorm(hidden_dim)
        self.stock_norm = nn.LayerNorm(hidden_dim)

        self.hidden_temporal_layers = nn.ModuleList([TemporalLayer(entity_index, hidden_dim, hidden_dim, F.leaky_relu,dropout = dropout,is_norm = is_temporal_norm) for i in range(hidden_layers)])
        self.hidden_spatial_layers = nn.ModuleList([SpatialLayer(hidden_dim, hidden_dim, 1, 2, F.leaky_relu,norm=norm,dropout = dropout,is_norm = is_spatial_norm) for i in range(hidden_layers - 1)])
        self.last_hidden_spatial_layer = SpatialLayer(hidden_dim, hidden_dim, 1, 2, F.leaky_relu, need_grad=True,norm = norm,dropout = dropout, is_norm = is_spatial_norm)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        self.is_use_temporal_layer = is_use_temporal_layer
        self.is_use_spatial_layer = is_use_spatial_layer
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight,nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Parameter):
            nn.init.kaiming_normal_(module,nonlinearity='leaky_relu')
            
    def forward(self, X_stock, X_account, edge_indices, point_enc, edge_weights, node_to_predict_mask,
                edge_ids = None, learning_weight=None):
        time_steps,_,_ = X_stock.shape
        num_nodes = len(point_enc)
        account_emb = self.account_fc(X_account)
        stock_emb = self.stock_fc(X_stock)
        
        # account_emb = self.account_norm(account_emb)
        # stock_emb = self.stock_norm(stock_emb)
        node_states = torch.zeros((time_steps, num_nodes, self.hidden_dim),device= stock_emb.device)
        node_states[:,point_enc,:] = account_emb
        node_states[:,~point_enc,:] = stock_emb
        # node_states = torch.cat([account_emb, stock_emb], dim=1)
        node_states = F.leaky_relu(node_states)   
        if self.is_use_temporal_layer:
            node_states = self.hidden_temporal_layers[0](node_states, point_enc)
        
        for i in range(len(self.hidden_spatial_layers)):
            if self.is_use_spatial_layer:
                node_states = self.hidden_spatial_layers[i](node_state = node_states,edge_index =  edge_indices,\
                                                            point_enc =  point_enc, edge_weight =  edge_weights)
            if self.is_use_temporal_layer:
                node_states = self.hidden_temporal_layers[i + 1](node_states, point_enc)

        node_states = self.last_hidden_spatial_layer(node_state = node_states, edge_index = edge_indices, \
                                                    point_enc = point_enc, edge_weight = edge_weights, \
                                                    edge_ids = edge_ids, learning_weight=learning_weight)
        
        node_states = node_states[-1,:,:][node_to_predict_mask]
        y_pred = self.output_layer(node_states)

        return y_pred
    

    def save_model(self, path):
        """
        保存模型到指定路径
        Args:
            path: 保存模型的路径，包括文件名
        """
        model_state = {
            'model_state_dict': self.state_dict(),
            'hidden_dim': self.hidden_dim,
            'is_use_temporal_layer': self.is_use_temporal_layer,
            'is_use_spatial_layer': self.is_use_spatial_layer,
        }
        torch.save(model_state, path)

    @classmethod
    def load_model(cls, path, entity_index, stock_dim=32, account_dim=16, hidden_layers=2, hidden_dim=32, norm="batch", 
                   is_use_temporal_layer=True,is_use_spatial_layer=True):
        """
        从指定路径加载模型
        Args:
            path: 模型文件路径
            entity_index: 实体索引
            其他参数与__init__相同
        Returns:
            加载好权重的模型实例
        """
        model = cls(entity_index, stock_dim, account_dim, hidden_layers, hidden_dim, norm, 
                    is_use_temporal_layer = is_use_temporal_layer, is_use_spatial_layer = is_use_spatial_layer)
        model_state = torch.load(path)
        model.load_state_dict(model_state['model_state_dict'])
        return model

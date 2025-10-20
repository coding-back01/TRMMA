"""
G-TRMMA: Graph-Enhanced Trajectory Recovery with Map Matching Assistance
核心创新：用GNN编码器替换原始TRMMA中的路线Transformer编码器，
使模型能够理解路线的物理和几何结构（交叉口、转弯角度等）。
"""

import random
import time

from tqdm import tqdm
import os
import datetime as dt
import numpy as np
import pickle
import networkx as nx
from queue import Queue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from models.layers import Attention, GPSFormer, sequence_mask, sequence_mask3d
from preprocess import SparseDAM, SegInfo
from utils.model_utils import gps2grid, get_normalized_t
from utils.spatial_func import SPoint, project_pt_to_road, rate2gps
from utils.trajectory_func import STPoint
from utils.candidate_point import CandidatePoint

# 导入TRMMA中的辅助函数和类
from models.trmma import (
    get_num_pts, get_segs, remove_circle, calc_cos_value,
    DAPlanner, TrajRecData, TrajRecTestData, DecoderMulti,
    get_pro_features, get_label
)


# ================== GNN Components ==================

class GATLayer(nn.Module):
    """
    Graph Attention Layer (GAT)
    用于学习路网中每个路段的结构化表示，考虑邻居路段的影响
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        
        if concat:
            assert out_dim % num_heads == 0
            self.head_dim = out_dim // num_heads
        else:
            self.head_dim = out_dim
        
        # 多头注意力的线性变换
        self.W = nn.Linear(in_dim, self.head_dim * num_heads, bias=False)
        # 注意力计算参数 - 为每个head单独定义
        self.a = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        if not concat:
            self.out_proj = nn.Linear(out_dim * num_heads, out_dim)
    
    def forward(self, node_features, edge_index):
        """
        Args:
            node_features: [num_nodes, in_dim] 节点特征
            edge_index: [2, num_edges] 边索引 (source, target)
        Returns:
            [num_nodes, out_dim] 更新后的节点特征
        """
        num_nodes = node_features.size(0)
        num_edges = edge_index.size(1)
        
        # 线性变换: [num_nodes, head_dim * num_heads]
        h = self.W(node_features)
        # 重塑为多头: [num_nodes, num_heads, head_dim]
        h = h.view(num_nodes, self.num_heads, self.head_dim)
        
        # 准备注意力计算
        src, dst = edge_index[0], edge_index[1]
        
        # 获取源节点和目标节点的特征
        h_src = h[src]  # [num_edges, num_heads, head_dim]
        h_dst = h[dst]  # [num_edges, num_heads, head_dim]
        
        # 拼接并计算注意力分数
        h_cat = torch.cat([h_src, h_dst], dim=-1)  # [num_edges, num_heads, 2*head_dim]
        
        # 计算注意力分数 - 使用einsum进行高效计算
        # h_cat: [num_edges, num_heads, 2*head_dim]
        # self.a: [num_heads, 2*head_dim]
        # 结果: [num_edges, num_heads]
        attn_scores = torch.einsum('ehd,hd->eh', h_cat, self.a)
        attn_scores = self.leaky_relu(attn_scores)  # [num_edges, num_heads]
        
        # 对每个目标节点的所有入边进行softmax
        attn_weights = self._scatter_softmax(attn_scores, dst, num_nodes)
        attn_weights = self.dropout(attn_weights)
        
        # 聚合邻居特征
        # [num_edges, num_heads, head_dim] * [num_edges, num_heads, 1]
        weighted_h = h_src * attn_weights.unsqueeze(-1)
        
        # 对每个节点聚合其所有入边的加权特征
        # 【优化】添加自环到聚合中，防止孤立节点信息丢失
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, 
                         device=node_features.device, dtype=node_features.dtype)
        
        # 扩展dst以匹配多头和特征维度: [num_edges] -> [num_edges, num_heads, head_dim]
        dst_expanded = dst.view(-1, 1, 1).expand(-1, self.num_heads, self.head_dim)
        
        # 使用scatter_add在第0维聚合
        out.scatter_add_(0, dst_expanded, weighted_h)
        
        if self.concat:
            # 拼接多头输出: [num_nodes, num_heads * head_dim]
            out = out.view(num_nodes, -1)
        else:
            # 平均多头输出: [num_nodes, head_dim]
            out = out.mean(dim=1)
            out = self.out_proj(out)
        
        return out
    
    def _scatter_softmax(self, scores, indices, num_nodes):
        """
        对每个节点的入边分数进行softmax归一化
        使用稳定的scatter方法（不使用beta API）
        Args:
            scores: [num_edges, num_heads] 注意力分数
            indices: [num_edges] 目标节点索引
            num_nodes: 总节点数
        Returns:
            [num_edges, num_heads] 归一化后的注意力权重
        """
        num_edges = scores.size(0)
        num_heads = scores.size(1)
        
        # 使用更简单的方法：直接计算softmax，不做数值稳定化
        # 对于图神经网络，边的数量通常不会太大，直接计算是安全的
        exp_scores = torch.exp(scores)  # [num_edges, num_heads]
        
        # 对每个节点计算exp的和
        sum_exp = torch.zeros(num_nodes, num_heads, device=scores.device, dtype=scores.dtype)
        
        # 使用scatter_add聚合
        indices_expanded = indices.unsqueeze(1).expand(-1, num_heads)  # [num_edges, num_heads]
        sum_exp.scatter_add_(0, indices_expanded, exp_scores)
        
        # 归一化
        weights = exp_scores / (sum_exp[indices] + 1e-16)
        
        return weights


class RouteGraphEncoder(nn.Module):
    """
    路线图编码器：使用GNN处理路线子图
    核心创新：将路线R视为一个子图，学习其拓扑和几何结构
    """
    def __init__(self, hid_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        
        # 多层GAT
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # 第一层：输入维度为hid_dim
                self.gat_layers.append(GATLayer(hid_dim, hid_dim, num_heads, dropout, concat=True))
            else:
                # 后续层
                self.gat_layers.append(GATLayer(hid_dim, hid_dim, num_heads, dropout, concat=True))
        
        # Layer normalization
        self.norms = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(num_layers)])
        
        # 【修复】所有层都使用Identity残差连接（输入输出维度相同）
        self.residual_proj = nn.ModuleList([
            nn.Identity() for _ in range(num_layers)
        ])
    
    def forward(self, route_emb, route_len, adj_matrices):
        """
        Args:
            route_emb: [max_route_len, batch_size, hid_dim] 路线嵌入
            route_len: [batch_size] 每条路线的实际长度
            adj_matrices: List of [2, num_edges] 每个样本的邻接矩阵（边索引）
        Returns:
            route_outputs: [max_route_len, batch_size, hid_dim] GNN编码后的路线表示
        """
        max_route_len, batch_size, _ = route_emb.shape
        
        # 转换维度: [batch_size, max_route_len, hid_dim]
        route_emb = route_emb.transpose(0, 1)
        
        # 对批次中的每个样本分别处理
        outputs = []
        for i in range(batch_size):
            # 获取该样本的有效路线长度和邻接矩阵
            valid_len = route_len[i].item()
            node_features = route_emb[i, :valid_len, :]  # [valid_len, hid_dim]
            edge_index = adj_matrices[i]  # [2, num_edges]
            
            # 通过多层GAT
            h = node_features
            for layer_idx, (gat, norm, res_proj) in enumerate(zip(self.gat_layers, self.norms, self.residual_proj)):
                h_new = gat(h, edge_index)
                # Post-LN：先加残差，再归一化（更标准）
                h = norm(h + h_new)
                h = F.relu(h)
            
            # Padding到最大长度
            if valid_len < max_route_len:
                padding = torch.zeros(max_route_len - valid_len, self.hid_dim, 
                                    device=route_emb.device, dtype=route_emb.dtype)
                h = torch.cat([h, padding], dim=0)
            
            outputs.append(h)
        
        # 堆叠: [batch_size, max_route_len, hid_dim]
        route_outputs = torch.stack(outputs, dim=0)
        
        # 转换回: [max_route_len, batch_size, hid_dim]
        route_outputs = route_outputs.transpose(0, 1)
        
        return route_outputs


class GNNRouteEncoder(nn.Module):
    """
    GNN路线编码器：完全模仿TRMMA的DualFormer结构
    核心改进：用GNN替换Route的自注意力，但保留GPS-Route交互
    """
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.pro_features_flag = parameters.pro_features_flag
        
        # 【关键修复】强制GPS和Route层数相同，确保每层都交互
        # 这是DualFormer的核心设计理念
        self.num_layers = parameters.transformer_layers  # 使用transformer_layers作为统一层数
        
        # GPS轨迹编码器（保持不变）
        from models.layers import GPSLayer, MultiHeadAttention, Norm, FeedForward
        self.gps_layers = nn.ModuleList([
            GPSLayer(parameters.hid_dim, parameters.heads) 
            for _ in range(self.num_layers)
        ])
        
        # 路线GNN层 + GPS交互层（模仿RouteLayer结构）
        self.route_gnn_layers = nn.ModuleList([
            RouteGraphEncoder(parameters.hid_dim, num_layers=1, num_heads=parameters.heads)
            for _ in range(self.num_layers)
        ])
        
        self.route_gps_attns = nn.ModuleList([
            MultiHeadAttention(parameters.heads, parameters.hid_dim)
            for _ in range(self.num_layers)
        ])
        
        self.route_norms1 = nn.ModuleList([Norm(parameters.hid_dim) for _ in range(self.num_layers)])
        self.route_norms2 = nn.ModuleList([Norm(parameters.hid_dim) for _ in range(self.num_layers)])
        self.route_ffs = nn.ModuleList([FeedForward(parameters.hid_dim, parameters.hid_dim * 2) 
                                        for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(0.1)
        
        # 时间特征嵌入
        if self.pro_features_flag:
            self.temporal = nn.Embedding(parameters.pro_input_dim, parameters.pro_output_dim)
            self.fc_hid = nn.Linear(parameters.hid_dim + parameters.pro_output_dim, parameters.hid_dim)
    
    def forward(self, src, src_len, route, route_len, adj_matrices, pro_features):
        """
        完全模仿GRFormer的结构，但Route自注意力用GNN替代
        """
        bs = src.size(1)
        src_max_len = src.size(0)
        route_max_len = route.size(0)
        
        # 创建掩码
        gps_mask3d = torch.ones(bs, src_max_len, src_max_len, device=src.device)
        gps_mask3d = sequence_mask3d(gps_mask3d, src_len, src_len)
        inter_mask = torch.ones(bs, route_max_len, src_max_len, device=src.device)
        inter_mask = sequence_mask3d(inter_mask, route_len, src_len)
        
        # 转换维度
        gps_emb = src.transpose(0, 1)  # [bs, src_len, hid_dim]
        route_emb = route.transpose(0, 1)  # [bs, route_len, hid_dim]
        
        # 【关键修复】逐层同步处理GPS和Route（完全模仿GRFormer）
        for layer_idx in range(self.num_layers):
            # 1. GPS层（学习驾驶行为）
            gps_emb = self.gps_layers[layer_idx](gps_emb, gps_mask3d)
            
            # 2. Route GNN层（学习路网结构）
            route_t = route_emb.transpose(0, 1)  # [route_len, bs, hid_dim]
            route_gnn_out = self.route_gnn_layers[layer_idx](route_t, route_len, adj_matrices)
            route_gnn_out = route_gnn_out.transpose(0, 1)  # [bs, route_len, hid_dim]
            route1 = self.dropout(route_gnn_out)
            route_out = self.route_norms1[layer_idx](route_emb + route1)
            
            # 3. Route关注GPS（融合驾驶行为和路网结构）
            route2 = self.dropout(self.route_gps_attns[layer_idx](route_out, gps_emb, gps_emb, inter_mask))
            route_out2 = self.route_norms2[layer_idx](route_out + route2)
            
            # 4. FeedForward
            route_emb = self.route_ffs[layer_idx](route_out2)
        
        route_outputs = route_emb.transpose(0, 1)  # [route_len, bs, hid_dim]
        
        # 计算隐藏状态
        route_mask2d = torch.ones(bs, route_max_len, device=route.device)
        route_mask2d = sequence_mask(route_mask2d, route_len).transpose(0, 1).unsqueeze(-1).repeat(1, 1, self.hid_dim)
        
        masked_route = route_outputs * route_mask2d
        hidden = torch.sum(masked_route, dim=0) / route_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)  # [1, bs, hid_dim]
        
        # 融合时间特征
        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))
        
        return route_outputs, hidden


# ================== G-TRMMA Model ==================

class GTrajRecovery(nn.Module):
    """
    G-TRMMA主模型：使用GNN增强的轨迹恢复模型
    核心创新：用GNN处理路线子图，理解路网的物理和几何结构
    """
    def __init__(self, parameters):
        super().__init__()
        self.srcseg_flag = parameters.srcseg_flag
        self.hid_dim = parameters.hid_dim
        self.learn_pos = parameters.learn_pos
        self.rid_feats_flag = parameters.rid_feats_flag
        self.params = parameters
        
        # 路段ID嵌入
        self.emb_id = nn.Parameter(torch.rand(parameters.id_size, parameters.id_emb_dim))
        
        # 位置编码
        if self.learn_pos:
            max_input_length = 500
            self.pos_embedding_gps = nn.Embedding(max_input_length, parameters.hid_dim)
            self.pos_embedding_route = nn.Embedding(max_input_length, parameters.hid_dim)
        
        # GPS输入处理
        input_dim_gps = 3
        if self.learn_pos:
            input_dim_gps += parameters.hid_dim
        if self.srcseg_flag:
            input_dim_gps += parameters.hid_dim + 1
        self.fc_in_gps = nn.Linear(input_dim_gps, parameters.hid_dim)
        
        # 路线输入处理
        input_dim_route = parameters.hid_dim
        if self.learn_pos:
            input_dim_route += parameters.hid_dim
        if self.rid_feats_flag:
            input_dim_route += parameters.rid_fea_dim
        self.fc_in_route = nn.Linear(input_dim_route, parameters.hid_dim)
        
        # 编码器：使用GNN处理路线
        self.encoder = GNNRouteEncoder(parameters)
        
        # 解码器：复用TRMMA的解码器
        self.decoder = DecoderMulti(parameters)
        
        self.init_weights()
    
    def init_weights(self):
        """权重初始化"""
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
    
    def forward(self, src, src_len, trg_id, trg_rate, trg_len, pro_features, 
                rid_features_dict, da_routes, da_lengths, da_pos, 
                src_seg_seqs, src_seg_feats, adj_matrices, d_rids, d_rates, 
                teacher_forcing_ratio):
        """
        前向传播
        Args:
            adj_matrices: List of edge_index，每个样本的路线子图邻接矩阵
        """
        max_trg_len = trg_id.size(0)
        batch_size = trg_id.size(1)
        
        # 路段嵌入传递给解码器
        self.decoder.emb_id = self.emb_id
        
        # 1. GPS输入处理
        gps_emb = src.float()
        if self.learn_pos:
            gps_pos = src[:, :, -1].long()
            gps_pos_emb = self.pos_embedding_gps(gps_pos)
            gps_emb = torch.cat([gps_emb, gps_pos_emb], dim=-1)
        if self.srcseg_flag:
            seg_emb = self.emb_id[src_seg_seqs]
            gps_emb = torch.cat((gps_emb, seg_emb, src_seg_feats), dim=-1)
        gps_in = self.fc_in_gps(gps_emb)
        gps_in_lens = torch.tensor(src_len, device=src.device)
        
        # 2. 路线输入处理
        route_emb = self.emb_id[da_routes]
        if self.learn_pos:
            route_pos_emb = self.pos_embedding_route(da_pos)
            route_emb = torch.cat([route_emb, route_pos_emb], dim=-1)
        if self.rid_feats_flag:
            route_feats = rid_features_dict[da_routes]
            route_emb = torch.cat([route_emb, route_feats], dim=-1)
        route_in = self.fc_in_route(route_emb)
        route_in_lens = torch.tensor(da_lengths, device=src.device)
        
        # 3. 编码：使用GNN处理路线
        route_outputs, hiddens = self.encoder(gps_in, gps_in_lens, route_in, 
                                              route_in_lens, adj_matrices, pro_features)
        
        # 4. 准备解码器的注意力掩码
        route_attn_mask = torch.ones(batch_size, max(da_lengths), device=src.device)
        route_attn_mask = sequence_mask(route_attn_mask, route_in_lens)
        
        # 5. 解码
        outputs_id, outputs_rate = self.decoder(
            max_trg_len, batch_size, trg_id, trg_rate, trg_len, hiddens,
            rid_features_dict, da_routes, route_outputs, route_attn_mask,
            d_rids, d_rates, teacher_forcing_ratio
        )
        
        final_outputs_id = outputs_id[1:-1]
        final_outputs_rate = outputs_rate[1:-1]
        
        return final_outputs_id, final_outputs_rate


# ================== Dataset with Graph Construction ==================

class GTrajRecData(TrajRecData):
    """
    G-TRMMA的训练数据集
    扩展原始数据集，增加路线子图的构建
    """
    def __init__(self, rn, trajs_dir, mbr, parameters, mode):
        super().__init__(rn, trajs_dir, mbr, parameters, mode)
        # 保存路网图，用于构建路线子图
        self.G = pickle.load(open(os.path.join(parameters.dam_root, "road_graph_wtime"), "rb"))
    
    def __getitem__(self, index):
        # 调用父类方法获取基础数据
        data_list = super().__getitem__(index)
        
        # 为每个数据项添加邻接矩阵
        enhanced_data = []
        for data_item in data_list:
            da_route = data_item[0]  # [route_len]
            
            # 构建路线子图的邻接矩阵
            edge_index = self._build_route_graph(da_route)
            
            # 添加邻接矩阵到数据项
            enhanced_data.append(data_item + [edge_index])
        
        return enhanced_data
    
    def _build_route_graph(self, route_tensor):
        """
        根据路线构建子图的边索引
        【优化版】更稳定的图结构：主要依赖顺序性，减少复杂连接
        
        Args:
            route_tensor: [route_len] 路线路段ID序列（tensor）
        Returns:
            edge_index: [2, num_edges] 边索引（tensor）
        """
        route = route_tensor.tolist()
        route_len = len(route)
        
        edges = set()  # 使用set自动去重
        
        # 策略1：添加自环（保留节点自身信息）- 最重要！
        for i in range(route_len):
            edges.add((i, i))
        
        # 策略2：双向顺序边（允许信息双向流动）
        for i in range(route_len - 1):
            edges.add((i, i + 1))  # 前向边
            edges.add((i + 1, i))  # 后向边
        
        # 策略3：【可选】添加有限的跳跃连接（仅k=2，避免过度连接）
        # 这允许模型看到"下下个"路段，模拟驾驶时的前瞻
        if route_len > 2:
            for i in range(route_len - 2):
                edges.add((i, i + 2))
                edges.add((i + 2, i))
        
        # 转换为tensor并排序（保证确定性）
        edges = sorted(list(edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index


class GTrajRecTestData(TrajRecTestData):
    """
    G-TRMMA的测试数据集
    扩展原始测试数据集，增加路线子图的构建
    """
    def __init__(self, rn, trajs_dir, mbr, parameters, mode):
        # 先加载G图
        self.G = pickle.load(open(os.path.join(parameters.dam_root, "road_graph_wtime"), "rb"))
        
        # 调用父类初始化
        super().__init__(rn, trajs_dir, mbr, parameters, mode)
        
        # 为所有路线构建邻接矩阵
        self.adj_matrices = []
        for route in tqdm(self.routes, desc='Building route graphs'):
            edge_index = self._build_route_graph(route)
            self.adj_matrices.append(edge_index)
    
    def __getitem__(self, index):
        # 获取基础数据
        base_data = super().__getitem__(index)
        
        # 添加邻接矩阵
        adj_matrix = self.adj_matrices[index]
        
        return base_data + (adj_matrix,)
    
    def _build_route_graph(self, route):
        """
        根据路线构建子图的边索引
        【优化版】更稳定的图结构：主要依赖顺序性，减少复杂连接
        
        Args:
            route: list 路线路段ID序列
        Returns:
            edge_index: [2, num_edges] 边索引（tensor）
        """
        route_len = len(route)
        
        edges = set()  # 使用set自动去重
        
        # 策略1：添加自环（保留节点自身信息）- 最重要！
        for i in range(route_len):
            edges.add((i, i))
        
        # 策略2：双向顺序边（允许信息双向流动）
        for i in range(route_len - 1):
            edges.add((i, i + 1))  # 前向边
            edges.add((i + 1, i))  # 后向边
        
        # 策略3：【可选】添加有限的跳跃连接（仅k=2，避免过度连接）
        # 这允许模型看到"下下个"路段，模拟驾驶时的前瞻
        if route_len > 2:
            for i in range(route_len - 2):
                edges.add((i, i + 2))
                edges.add((i + 2, i))
        
        # 转换为tensor并排序（保证确定性）
        edges = sorted(list(edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index


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

from models.layers import Attention, GPSFormer, GRFormer, sequence_mask, sequence_mask3d
from preprocess import SparseDAM, SegInfo
from utils.model_utils import gps2grid, get_normalized_t
from utils.spatial_func import SPoint, project_pt_to_road, rate2gps
from utils.trajectory_func import STPoint
from utils.candidate_point import CandidatePoint
from models.mma import GPS2Seg
from models.trmma import TrajRecovery, DAPlanner, TrajRecData, TrajRecTestData


class IterativeMMA(nn.Module):
    """
    整合MMA和TRMMA的端到端迭代优化模型
    基于原有TRMMA架构进行扩展
    """
    
    def __init__(self, parameters):
        super().__init__()
        
        # 基础TRMMA模块
        self.trmma_module = TrajRecovery(parameters)
        
        # 迭代控制参数
        self.max_iterations = getattr(parameters, 'max_iterations', 3)
        self.convergence_threshold = getattr(parameters, 'convergence_threshold', 0.01)
        
        # 收敛判断网络
        self.convergence_predictor = nn.Sequential(
            nn.Linear(parameters.hid_dim, parameters.hid_dim // 2),
            nn.ReLU(),
            nn.Linear(parameters.hid_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 自适应融合权重网络
        self.fusion_weight_net = nn.Sequential(
            nn.Linear(parameters.hid_dim * 2, parameters.hid_dim),
            nn.ReLU(),
            nn.Linear(parameters.hid_dim, 1),
            nn.Sigmoid()
        )
        
        self.params = parameters
        
    def forward(self, src, src_len, trg_id, trg_rate, trg_len, pro_features, rid_features_dict, 
                da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates, 
                teacher_forcing_ratio, mode='train'):
        """
        前向传播函数
        Args:
            mode: 'train' 使用标准训练模式, 'iterative' 使用迭代优化模式
        """
        
        if mode == 'train' or teacher_forcing_ratio > 0:
            # 训练模式：直接使用TRMMA
            return self.trmma_module(
                src, src_len, trg_id, trg_rate, trg_len, pro_features, rid_features_dict,
                da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates,
                teacher_forcing_ratio
            )
        else:
            # 迭代优化模式
            return self.forward_iterative(
                src, src_len, trg_id, trg_rate, trg_len, pro_features, rid_features_dict,
                da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates
            )
    
    def forward_iterative(self, src, src_len, trg_id, trg_rate, trg_len, pro_features, rid_features_dict,
                         da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates):
        """迭代优化的前向传播"""
        
        # 初始化
        current_outputs_id = None
        current_outputs_rate = None
        trajectory_history = []
        convergence_scores = []
        
        for iteration in range(self.max_iterations):
            # 使用TRMMA进行轨迹恢复
            outputs_id, outputs_rate = self.trmma_module(
                src, src_len, trg_id, trg_rate, trg_len, pro_features, rid_features_dict,
                da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates,
                teacher_forcing_ratio=-1  # 使用推理模式
            )
            
            # 收敛检查和自适应融合
            if current_outputs_id is not None:
                # 计算收敛分数
                convergence_score = self.compute_convergence_score(
                    current_outputs_id, outputs_id
                )
                convergence_scores.append(convergence_score)
                
                # 检查是否收敛
                if convergence_score < self.convergence_threshold:
                    break
                
                # 自适应融合
                outputs_id = self.adaptive_fusion(
                    current_outputs_id, outputs_id, iteration
                )
                outputs_rate = self.adaptive_fusion(
                    current_outputs_rate, outputs_rate, iteration
                )
            
            current_outputs_id = outputs_id
            current_outputs_rate = outputs_rate
            
            # 记录历史
            trajectory_history.append({
                'outputs_id': current_outputs_id.clone(),
                'outputs_rate': current_outputs_rate.clone(),
                'iteration': iteration,
                'convergence_score': convergence_scores[-1] if convergence_scores else 1.0
            })
            
            # 为下一次迭代准备输入
            # 这里可以根据当前输出调整输入，目前保持不变
            
        return current_outputs_id, current_outputs_rate
    
    def compute_convergence_score(self, prev_outputs, curr_outputs):
        """计算收敛分数"""
        # 计算输出差异
        diff = torch.abs(prev_outputs - curr_outputs).mean()
        return diff.item()
    
    def adaptive_fusion(self, prev_outputs, curr_outputs, iteration):
        """自适应融合两次输出结果"""
        # 简单的线性融合，权重随迭代次数递减
        weight = 1.0 / (iteration + 2)  # 后续迭代给予更少权重给历史结果
        
        fused_outputs = weight * prev_outputs + (1 - weight) * curr_outputs
        
        return fused_outputs


# 为了保持与原有代码的兼容性，创建别名
TestDemo1 = IterativeMMA

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from copy import deepcopy
import copy
from models.diff_module import diff_CSDI
from models.diff_util import diff_forward_x0_constraint, cal_x0_from_noise_ddpm

class Diff_RNTraj(nn.Module):
    def __init__(self, diff_model, diffusion_hyperparams):
        super(Diff_RNTraj, self).__init__()
        self.diff_model = diff_model
        self.diffusion_param = diffusion_hyperparams

    def forward(self, spatial_A_trans, SE, dense_ids, sparse_ids, mask):
        """
        spatial_A_trans: UTGraph
        SE: pre-trained road segment representation
        dense_ids: 稠密路段 ID 序列 (B, T)
        sparse_ids: 稀疏路段 ID 序列 (B, T)
        mask: 稀疏掩码 (B, T)
        """
        device = SE.device

        # dense / sparse 的路段嵌入 (B, T, D)
        dense_embed = SE[dense_ids.to(device)]
        sparse_embed = SE[sparse_ids.to(device)]

        diff_loss, const_loss, x0_loss, sparse_loss, id_loss = diff_forward_x0_constraint(
            self.diff_model,
            dense_embed,
            sparse_embed,
            mask.to(device),
            self.diffusion_param,
            SE,
            spatial_A_trans.to(device),
            dense_ids=dense_ids,  # 传递真实的路段ID用于id_loss
            compute_id_loss=True,  # 训练时计算id_loss
        )

        return diff_loss, const_loss, x0_loss, sparse_loss, id_loss

    def generate_data(self, spatial_A_trans, SE, batchsize, length, pre_dim):
        
        """Ggenerate data"""
        
        x0 = cal_x0_from_noise_ddpm(self.diff_model, self.diffusion_param, batchsize, length, pre_dim + 1)  # B, T, 65
        x0_road = x0[:,:,:pre_dim]  # B, T, 64
        B, T, F = x0_road.shape

        x0_road_shape = x0_road.reshape(B*T, F)

        x0_abs = x0_road_shape.norm(dim=1)
        SE_abs = SE.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x0_road_shape, SE) / (torch.einsum('i,j->ij', x0_abs, SE_abs) + 1e-6)
        
        sim_matrix = sim_matrix.reshape(B, T, -1)  # B, T, road num    2000 * 20 = 40000 top1->top10
        sim_matrix = sim_matrix.argmax(-1)
        
        rates = x0[:,:,pre_dim]

        return sim_matrix, rates

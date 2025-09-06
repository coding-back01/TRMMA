import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from models.layers import Encoder as Transformer, Attention, sequence_mask, sequence_mask3d
from utils.model_utils import gps2grid, get_normalized_t


class GPS2SegData(Dataset):  # 定义GPS轨迹到路段映射的数据集类，继承自PyTorch的Dataset

    def __init__(self, rn, trajs_dir, mbr, parameters, mode):  # 初始化函数，接收路网、轨迹目录、最小边界矩形、参数和模式
        self.parameters = parameters  # 存储模型参数
        self.rn = rn  # 存储路网对象
        self.mbr = mbr  # 存储最小边界矩形对象
        self.grid_size = parameters.grid_size  # 存储网格大小参数
        self.time_span = parameters.time_span  # 存储时间跨度参数

        self.src_grid_seqs, self.src_gps_seqs, self.src_temporal_feas = [], [], []  # 初始化源网格序列、GPS序列和时间特征列表
        self.trg_rids = []  # 初始化目标路段ID列表
        if mode == 'train':  # 如果是训练模式
            file = os.path.join(trajs_dir, 'train.pkl')  # 设置训练数据文件路径
        elif mode == 'valid':  # 如果是验证模式
            file = os.path.join(trajs_dir, 'valid.pkl')  # 设置验证数据文件路径
        elif mode == 'test':  # 如果是测试模式
            file = os.path.join(trajs_dir, 'test_output.pkl')  # 设置测试数据文件路径
        else:  # 如果模式不匹配
            raise NotImplementedError  # 抛出未实现异常
        trajs = pickle.load(open(file, "rb"))  # 从pickle文件加载轨迹数据
        if parameters.small and mode == 'train':  # 如果使用小数据集且为训练模式
            idx_group = 0  # 设置组索引为0
            num_group = 5  # 设置总组数为5
            num_k = len(trajs) // num_group  # 计算每组的轨迹数量
            trajs = trajs[num_k * idx_group: num_k * (idx_group + 1)]  # 选择指定组的轨迹数据

        self.trajs = trajs  # 存储轨迹数据
        self.keep_ratio = parameters.init_ratio  # 设置保留比例为初始比例
        if mode in ['train']:  # 如果是训练模式
            self.ds_type = 'random'  # 设置数据采样类型为随机
        else:  # 如果不是训练模式
            self.ds_type = 'fixed'  # 设置数据采样类型为固定

    def __len__(self):  # 定义数据集长度方法
        return len(self.trajs)  # 返回轨迹数据的长度

    def __getitem__(self, index):  # 定义获取单个数据项的方法
        traj = self.trajs[index]  # 获取指定索引的轨迹
        if self.ds_type == 'random':  # 如果是随机采样
            length = len(traj.pt_list)  # 获取轨迹点列表的长度
            keep_index = [0] + sorted(random.sample(range(1, length - 1), int((length - 2) * self.keep_ratio))) + [length - 1]  # 随机选择保留的点索引，确保首尾点被保留
        elif self.ds_type == 'fixed':  # 如果是固定采样
            keep_index = traj.low_idx  # 使用轨迹预定义的低频索引
        else:  # 如果采样类型不匹配
            raise NotImplementedError  # 抛出未实现异常
        src_list = np.array(traj.pt_list, dtype=object)  # 将轨迹点列表转换为numpy数组
        src_list = src_list[keep_index].tolist()  # 根据保留索引选择轨迹点并转换为列表

        src_gps_seq, src_grid_seq, trg_rid = self.get_seqs(src_list)  # 从轨迹点列表获取GPS序列、网格序列和目标路段ID

        trg_candis = self.rn.get_trg_segs(src_gps_seq, self.parameters.candi_size, self.parameters.search_dist, self.parameters.beta)  # 从路网获取目标候选路段
        candi_label, candi_id, candi_feat, candi_mask = self.get_candis_feats(trg_candis, trg_rid)  # 获取候选路段的标签、ID、特征和掩码

        src_grid_seq = torch.tensor(src_grid_seq)  # 将网格序列转换为PyTorch张量

        return src_grid_seq, trg_rid, candi_label, candi_id, candi_feat, candi_mask  # 返回处理后的数据

    def get_candis_feats(self, ls_candi, trg_id):  # 定义获取候选路段特征的方法
        candi_id = []  # 初始化候选路段ID列表
        candi_feat = []  # 初始化候选路段特征列表
        candi_onehot = []  # 初始化候选路段独热编码列表
        candi_mask = []  # 初始化候选路段掩码列表
        for candis, trg in zip(ls_candi, trg_id):  # 遍历每个位置的候选路段和目标路段
            candi_mask.append([1] * len(candis) + [0] * (self.parameters.candi_size - len(candis)))  # 创建掩码，有效候选为1，填充部分为0
            tmp_id = []  # 临时ID列表
            tmp_feat = []  # 临时特征列表
            tmp_onehot = [0] * self.parameters.candi_size  # 初始化独热编码向量
            for candi in candis:  # 遍历当前位置的所有候选路段
                tmp_id.append(candi.eid)  # 添加候选路段的边ID
                tmp_feat.append([candi.err_weight, candi.cosv, candi.cosv_pre, candi.cosf, candi.cosl, candi.cos1, candi.cos2, candi.cos3, candi.cosp])  # 添加候选路段的特征向量
            tmp_id.extend([0] * (self.parameters.candi_size - len(candis)))  # 用0填充ID列表到固定长度
            tmp_feat.extend([[0] * len(tmp_feat[0])] * (self.parameters.candi_size - len(candis)))  # 用零向量填充特征列表到固定长度
            if trg in tmp_id:  # 如果目标路段在候选列表中
                idx = tmp_id.index(trg)  # 获取目标路段在候选列表中的索引
                tmp_onehot[idx] = 1  # 在独热编码中标记目标路段位置为1
            candi_id.append(tmp_id)  # 添加当前位置的候选ID列表
            candi_feat.append(tmp_feat)  # 添加当前位置的候选特征列表
            candi_onehot.append(tmp_onehot)  # 添加当前位置的独热编码
        candi_onehot = torch.tensor(candi_onehot)  # 将独热编码列表转换为PyTorch张量
        candi_id = torch.tensor(candi_id) + 1  # 将候选ID列表转换为PyTorch张量并加1（避免0索引）
        candi_feat = torch.tensor(candi_feat)  # 将候选特征列表转换为PyTorch张量
        candi_mask = torch.tensor(candi_mask, dtype=torch.float32)  # 将掩码列表转换为浮点型PyTorch张量
        return candi_onehot, candi_id, candi_feat, candi_mask  # 返回处理后的候选路段信息

    def get_seqs(self, ds_pt_list):  # 定义从轨迹点列表获取序列的方法
        ls_gps_seq = []  # 初始化GPS序列列表
        ls_grid_seq = []  # 初始化网格序列列表
        mm_eids = []  # 初始化地图匹配边ID列表
        time_interval = self.time_span  # 获取时间间隔
        first_pt = ds_pt_list[0]  # 获取第一个轨迹点作为时间基准
        for ds_pt in ds_pt_list:  # 遍历轨迹点列表中的每个点
            ls_gps_seq.append([ds_pt.lat, ds_pt.lng])  # 添加GPS坐标到GPS序列
            if self.parameters.gps_flag:  # 如果使用GPS标志
                locgrid_xid = (ds_pt.lat - self.rn.minLat) / (self.rn.maxLat - self.rn.minLat)  # 计算归一化的纬度网格坐标
                locgrid_yid = (ds_pt.lng - self.rn.minLon) / (self.rn.maxLon - self.rn.minLon)  # 计算归一化的经度网格坐标
            else:  # 如果不使用GPS标志
                locgrid_xid, locgrid_yid = gps2grid(ds_pt, self.mbr, self.grid_size)  # 使用网格转换函数计算网格坐标
            t = get_normalized_t(first_pt, ds_pt, time_interval)  # 计算归一化的时间特征
            ls_grid_seq.append([locgrid_xid, locgrid_yid, t])  # 添加网格坐标和时间特征到网格序列
            mm_eids.append(ds_pt.data['candi_pt'].eid)  # 添加候选点的边ID到地图匹配边ID列表

        return ls_gps_seq, ls_grid_seq, mm_eids  # 返回GPS序列、网格序列和地图匹配边ID列表


class Encoder(nn.Module):  # 定义编码器类，继承自nn.Module

    def __init__(self, parameters):  # 初始化方法，接收参数对象
        super().__init__()  # 调用父类构造函数
        self.hid_dim = parameters.hid_dim  # 设置隐藏层维度

        input_dim = 3  # 设置输入维度为3（经度、纬度、时间）
        self.fc_in = nn.Linear(input_dim, parameters.hid_dim)  # 创建输入全连接层，将3维输入映射到隐藏维度
        self.transformer = Transformer(parameters.hid_dim, parameters.transformer_layers, heads=4)  # 创建Transformer层，设置隐藏维度、层数和注意力头数

    def forward(self, src, src_len):  # 前向传播方法，接收源序列和序列长度
        # src = [batch size, src len, 3]  # 源序列形状注释
        # src_len = [batch size]  # 序列长度形状注释
        max_src_len = src.size(1)  # 获取序列的最大长度
        bs = src.size(0)  # 获取批次大小

        src_len = torch.tensor(src_len, device=src.device)  # 将序列长度转换为张量并移动到相同设备

        mask3d = torch.ones(bs, max_src_len, max_src_len, device=src.device)  # 创建3维掩码张量，用于Transformer注意力
        mask2d = torch.ones(bs, max_src_len, device=src.device)  # 创建2维掩码张量，用于输出掩码

        mask3d = sequence_mask3d(mask3d, src_len, src_len)  # 根据序列长度生成3维序列掩码
        mask2d = sequence_mask(mask2d, src_len).unsqueeze(-1).repeat(1, 1, self.hid_dim)  # 生成2维序列掩码并扩展到隐藏维度

        src = self.fc_in(src)  # 通过输入全连接层处理源序列
        outputs = self.transformer(src, mask3d)  # 通过Transformer处理序列，使用3维掩码

        assert outputs.size(1) == max_src_len  # 断言输出序列长度与输入序列长度一致
        outputs = outputs * mask2d  # 应用2维掩码到输出，将填充位置置零

        return outputs  # 返回编码后的输出


class GPS2Seg(nn.Module):

    def __init__(self, parameters):
        super().__init__()  # 调用父类构造函数
        self.direction_flag = parameters.direction_flag  # 是否使用方向信息的标志
        self.attn_flag = parameters.attn_flag  # 是否使用注意力机制的标志
        self.only_direction = parameters.only_direction  # 是否只使用方向信息的标志

        self.emb_id = nn.Embedding(parameters.id_size, parameters.id_emb_dim)  # 候选段ID的嵌入层
        self.encoder = Encoder(parameters)  # GPS轨迹编码器

        fc_id_out_input_dim = parameters.hid_dim  # 全连接层输入维度初始化
        if self.direction_flag:  # 如果使用方向信息
            fc_id_out_input_dim += 9  # 增加9维方向特征
        if self.only_direction:  # 如果只使用方向信息
            fc_id_out_input_dim = 9  # 输入维度设为9
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, parameters.hid_dim)  # 候选段特征输出层

        mlp_dim = parameters.hid_dim * 2  # MLP输入维度初始化
        if self.attn_flag:  # 如果使用注意力机制
            self.attn = Attention(parameters.hid_dim)  # 初始化注意力层
            mlp_dim += parameters.hid_dim  # 增加注意力输出维度

        self.prob_out = nn.Sequential(  # 概率输出网络
            nn.Linear(mlp_dim, parameters.hid_dim * 2),  # 第一个全连接层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(parameters.hid_dim * 2, 1),  # 第二个全连接层，输出1维
            nn.Sigmoid()  # Sigmoid激活函数，输出概率
        )

        self.params = parameters  # 保存参数
        self.hid_dim = parameters.hid_dim  # 保存隐藏维度

        self.init_weights()  # 初始化权重

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)  # 获取输入到隐藏层权重
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)  # 获取隐藏到隐藏层权重
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)  # 获取偏置参数

        for t in ih:  # 遍历输入到隐藏层权重
            nn.init.xavier_uniform_(t)  # 使用Xavier均匀初始化
        for t in hh:  # 遍历隐藏到隐藏层权重
            nn.init.orthogonal_(t)  # 使用正交初始化
        for t in b:  # 遍历偏置参数
            nn.init.constant_(t, 0)  # 初始化为0

    def forward(self, src, src_len, candi_ids, candi_feats, candi_masks):
        candi_num = candi_ids.shape[-1]  # 获取候选段数量

        candi_embedding = self.emb_id(candi_ids)  # 获取候选段ID嵌入
        if self.direction_flag:  # 如果使用方向信息
            candi_embedding = torch.cat([candi_embedding, candi_feats], dim=-1)  # 拼接方向特征
        if self.only_direction:  # 如果只使用方向信息
            candi_embedding = candi_feats  # 只使用方向特征
        candi_vec = self.fc_id_out(candi_embedding)  # 通过全连接层得到候选段向量

        src = src.float()  # 转换为浮点数类型
        encoder_outputs = self.encoder(src, src_len)  # 编码GPS轨迹
        if self.attn_flag:  # 如果使用注意力机制
            _, context = self.attn(encoder_outputs, candi_vec, candi_vec, candi_masks)  # 计算注意力上下文
            encoder_outputs = torch.cat((encoder_outputs, context), dim=-1)  # 拼接编码器输出和上下文

        output_multi = encoder_outputs.unsqueeze(-2).repeat(1, 1, candi_num, 1)  # 扩展编码器输出以匹配候选段数量

        outputs_id = self.prob_out(torch.cat((output_multi, candi_vec), dim=-1)).squeeze(-1)  # 计算每个候选段的概率
        outputs_id = outputs_id.masked_fill(candi_masks == 0, 0)  # 使用掩码填充无效候选段

        return outputs_id  # 返回候选段概率

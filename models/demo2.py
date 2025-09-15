import os
import pickle
import random

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

from models.layers import Attention, GPSFormer, GRFormer, sequence_mask, sequence_mask3d
from utils.model_utils import gps2grid, get_normalized_t, AttrDict
from utils.spatial_func import SPoint


# ==================== 数据集（端到端） ====================

class E2ETrajData(Dataset):
    """
    端到端训练数据集：在 TRMMA 的样本粒度上（低频点对之间的 gap）构造样本，
    同时返回 MMA 所需的候选集与标签，保证两端输入齐备并可端到端训练。
    """

    def __init__(self, rn, trajs_dir, mbr, parameters, mode):
        self.parameters = parameters
        self.rn = rn
        self.mbr = mbr  # MBR of all trajectories
        self.grid_size = parameters.grid_size
        self.time_span = parameters.time_span
        self.mode = mode
        self.keep_ratio = parameters.keep_ratio
        if mode == 'train':
            # 训练阶段从 init_ratio 开始，便于后续按 decay_ratio 衰减到 keep_ratio
            self.keep_ratio = getattr(parameters, 'init_ratio', parameters.keep_ratio)

        if mode == 'train':
            file = os.path.join(trajs_dir, 'train.pkl')
        elif mode == 'valid':
            file = os.path.join(trajs_dir, 'valid.pkl')
        else:
            raise NotImplementedError
        trajs = pickle.load(open(file, "rb"))
        if parameters.small and mode == 'train':
            idx_group = 0
            num_group = 5
            num_k = len(trajs) // num_group
            trajs = trajs[num_k * idx_group: num_k * (idx_group + 1)]
        self.trajs = trajs

        # 为与 TRMMA 的最终输出一致：在非 train 模式下，预先构建 groups 与 src_mms
        self.groups = []
        self.src_mms = []
        if self.mode != 'train':
            for serial, traj in enumerate(self.trajs):
                low_idx = traj.low_idx
                src_list = np.array(traj.pt_list, dtype=object)
                src_list = src_list[low_idx].tolist()

                # 低频观测点的匹配结果 (gps, seg, rate)
                src_mm = []
                for pt in src_list:
                    candi_pt = pt.data['candi_pt']
                    src_mm.append([[candi_pt.lat, candi_pt.lng], candi_pt.eid, candi_pt.rate])
                self.src_mms.append(src_mm)

                # 记录该轨迹的 gap 数量（每个 gap 对应一个样本）
                for p1_idx, p2_idx in zip(low_idx[:-1], low_idx[1:]):
                    if (p1_idx + 1) < p2_idx:
                        self.groups.append(serial)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        traj = self.trajs[index]

        # 以论文相同方式下采样，构造 (p1, p2) 的 gap 样本
        if self.mode == 'train':
            length = len(traj.pt_list)
            keep_index = [0] + sorted(random.sample(range(1, length - 1), int((length - 2) * self.keep_ratio))) + [length - 1]
        else:
            keep_index = traj.low_idx
        src_list = np.array(traj.pt_list, dtype=object)
        src_list = src_list[keep_index].tolist()

        trg_list = traj.pt_list

        data = []
        for p1, p1_idx, p2, p2_idx in zip(src_list[:-1], keep_index[:-1], src_list[1:], keep_index[1:]):
            # 仅对低频点对之间存在 gap 的片段构造样本
            if (p1_idx + 1) < p2_idx:
                tmp_src_list = [p1, p2]

                # 基础序列（与 TRMMA 一致）：
                ls_grid_seq, ls_gps_seq, hours, tmp_seg_seq = self._get_src_seq(tmp_src_list)
                features = self._get_pro_features(tmp_src_list, hours)

                # MMA 候选：对两个端点生成候选集（保持与 MMA 训练一致的9维候选特征）
                trg_candis = self.rn.get_trg_segs(ls_gps_seq, self.parameters.candi_size, self.parameters.search_dist, self.parameters.beta)
                candi_onehot, candi_ids, candi_feats, candi_masks = self._build_mma_candidates(trg_candis, tmp_seg_seq)

                # TRMMA 监督：目标路段序列与速率（完整高频序列的映射）
                mm_eids, mm_rates = self._get_trg_seq(trg_list[p1_idx: p2_idx + 1])
                path = traj.cpath[p1.cpath_idx: p2.cpath_idx + 1]

                da_route = [self.rn.valid_edge_one[item] for item in path]
                src_seg_seq = [self.rn.valid_edge_one[item] for item in tmp_seg_seq]
                src_seg_feat = self._get_src_seg_feat(ls_gps_seq, tmp_seg_seq)
                label = self._get_label([self.rn.valid_edge_one[item] for item in path], mm_eids[1:-1])

                # 打包张量（与现有风格一致）
                da_route = torch.tensor(da_route)
                src_grid_seq = torch.tensor(ls_grid_seq)
                src_pro_fea = torch.tensor(features)
                src_seg_seq = torch.tensor(src_seg_seq)
                src_seg_feat = torch.tensor(src_seg_feat)

                trg_rid = torch.tensor(mm_eids)
                trg_rate = torch.tensor(mm_rates)
                label = torch.tensor(label, dtype=torch.float32)
                d_rid = trg_rid[-1]
                d_rate = trg_rate[-1]

                # MMA 所需的候选与标签
                candi_onehot = torch.tensor(candi_onehot)
                candi_ids = torch.tensor(candi_ids) + 1  # 与 MMA 一致：+1 避免0索引
                candi_feats = torch.tensor(candi_feats)
                candi_masks = torch.tensor(candi_masks, dtype=torch.float32)

                # 返回顺序遵循 TRMMA，再追加 MMA 字段，便于 collate 对齐
                data.append([
                    da_route, src_grid_seq, src_pro_fea, src_seg_seq, src_seg_feat, trg_rid, trg_rate, label, d_rid, d_rate,
                    candi_onehot, candi_ids, candi_feats, candi_masks
                ])

        return data

    # ======== 工具函数（保持与现有代码风格） ========

    def _get_src_seg_feat(self, gps_seq, seg_seq):
        feats = []
        for ds_pt, seg in zip(gps_seq, seg_seq):
            gps = SPoint(ds_pt[0], ds_pt[1])
            candi = self.rn.pt2seg(gps, seg)
            feats.append([candi.rate])
        return feats

    def _get_src_seq(self, ds_pt_list):
        hours = []
        ls_grid_seq = []
        ls_gps_seq = []
        first_pt = ds_pt_list[0]
        time_interval = self.time_span
        seg_seq = []
        for ds_pt in ds_pt_list:
            hours.append(ds_pt.time_arr.hour)
            t = get_normalized_t(first_pt, ds_pt, time_interval)
            ls_gps_seq.append([ds_pt.lat, ds_pt.lng])
            if self.parameters.gps_flag:
                locgrid_xid = (ds_pt.lat - self.rn.minLat) / (self.rn.maxLat - self.rn.minLat)
                locgrid_yid = (ds_pt.lng - self.rn.minLon) / (self.rn.maxLon - self.rn.minLon)
            else:
                locgrid_xid, locgrid_yid = gps2grid(ds_pt, self.mbr, self.grid_size)
            ls_grid_seq.append([locgrid_xid, locgrid_yid, t])
            seg_seq.append(ds_pt.data['candi_pt'].eid)

        return ls_grid_seq, ls_gps_seq, hours, seg_seq

    def _get_trg_seq(self, tmp_pt_list):
        mm_eids = []
        mm_rates = []
        for pt in tmp_pt_list:
            candi_pt = pt.data['candi_pt']
            mm_eids.append(self.rn.valid_edge_one[candi_pt.eid])
            mm_rates.append([candi_pt.rate])
        return mm_eids, mm_rates

    def _get_label(self, cpath, trg_rid):
        label = []
        pre_rid = -1
        pre_prob = []
        for rid in trg_rid:
            if rid == pre_rid:
                tmp = pre_prob
            else:
                idx = 0
                if rid in cpath:
                    idx = cpath.index(rid)
                tmp = [0] * len(cpath)
                tmp[idx] = 1
                pre_rid = rid
                pre_prob = tmp
            label.append(tmp)
        return label

    def _get_pro_features(self, ds_pt_list, hours):
        hour = np.bincount(hours).argmax()
        week = ds_pt_list[0].time_arr.weekday()
        if week in [5, 6]:
            hour += 24
        return hour

    def _build_mma_candidates(self, ls_candi, trg_id_seq):
        """构造 MMA 的候选集张量与 one-hot 标签（对齐 top-k 候选）。"""
        candi_id = []
        candi_feat = []
        candi_onehot = []
        candi_mask = []
        for candis, trg in zip(ls_candi, trg_id_seq):
            candi_mask.append([1] * len(candis) + [0] * (self.parameters.candi_size - len(candis)))
            tmp_id = []
            tmp_feat = []
            tmp_onehot = [0] * self.parameters.candi_size
            for candi in candis:
                tmp_id.append(candi.eid)
                tmp_feat.append([
                    getattr(candi, 'err_weight', getattr(candi, 'error', 0.0)),
                    getattr(candi, 'cosv', 0.0),
                    getattr(candi, 'cosv_pre', 0.0),
                    getattr(candi, 'cosf', 0.0),
                    getattr(candi, 'cosl', 0.0),
                    getattr(candi, 'cos1', 0.0),
                    getattr(candi, 'cos2', 0.0),
                    getattr(candi, 'cos3', 0.0),
                    getattr(candi, 'cosp', 0.0)
                ])
            tmp_id.extend([0] * (self.parameters.candi_size - len(candis)))
            tmp_feat.extend([[0] * len(tmp_feat[0])] * (self.parameters.candi_size - len(candis)))
            if trg in tmp_id:
                idx = tmp_id.index(trg)
                tmp_onehot[idx] = 1
            candi_id.append(tmp_id)
            candi_feat.append(tmp_feat)
            candi_onehot.append(tmp_onehot)
        return candi_onehot, candi_id, candi_feat, candi_mask


# ==================== MMA 子模型（复用现有定义） ====================

class MMAEncoder(nn.Module):
    """与 models.mma.Encoder 一致，简化导入依赖，避免循环引用。"""

    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.fc_in = nn.Linear(3, parameters.hid_dim)
        self.transformer = Encoder(parameters.hid_dim, parameters.transformer_layers, heads=4)

    def forward(self, src, src_len):
        max_src_len = src.size(1)
        bs = src.size(0)

        src_len = torch.tensor(src_len, device=src.device)

        mask3d = torch.ones(bs, max_src_len, max_src_len, device=src.device)
        mask2d = torch.ones(bs, max_src_len, device=src.device)

        mask3d = sequence_mask3d(mask3d, src_len, src_len)
        mask2d = sequence_mask(mask2d, src_len).unsqueeze(-1).repeat(1, 1, self.hid_dim)

        src = self.fc_in(src)
        outputs = self.transformer(src, mask3d)

        assert outputs.size(1) == max_src_len
        outputs = outputs * mask2d
        return outputs


from models.layers import Encoder  # 复用同名 Encoder


class GPS2Seg(nn.Module):
    """与 models.mma.GPS2Seg 相同定义，保留概率输出，便于 BCE 与软嵌入计算。"""

    def __init__(self, parameters):
        super().__init__()
        self.direction_flag = parameters.direction_flag
        self.attn_flag = parameters.attn_flag
        self.only_direction = parameters.only_direction

        self.emb_id = nn.Embedding(parameters.id_size, parameters.id_emb_dim)
        self.encoder = MMAEncoder(parameters)

        fc_id_out_input_dim = parameters.hid_dim
        if self.direction_flag:
            fc_id_out_input_dim += 9
        if self.only_direction:
            fc_id_out_input_dim = 9
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, parameters.hid_dim)

        mlp_dim = parameters.hid_dim * 2
        if self.attn_flag:
            self.attn = Attention(parameters.hid_dim)
            mlp_dim += parameters.hid_dim

        self.prob_out = nn.Sequential(
            nn.Linear(mlp_dim, parameters.hid_dim * 2),
            nn.ReLU(),
            nn.Linear(parameters.hid_dim * 2, 1),
            nn.Sigmoid()
        )

        self.params = parameters
        self.hid_dim = parameters.hid_dim

        # 与 MMA 保持一致的权重初始化
        self.init_weights()

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, src, src_len, candi_ids, candi_feats, candi_masks):
        candi_num = candi_ids.shape[-1]

        candi_embedding = self.emb_id(candi_ids)
        if self.direction_flag:
            candi_embedding = torch.cat([candi_embedding, candi_feats], dim=-1)
        if self.only_direction:
            candi_embedding = candi_feats
        candi_vec = self.fc_id_out(candi_embedding)

        src = src.float()
        encoder_outputs = self.encoder(src, src_len)
        if self.attn_flag:
            _, context = self.attn(encoder_outputs, candi_vec, candi_vec, candi_masks)
            encoder_outputs = torch.cat((encoder_outputs, context), dim=-1)

        output_multi = encoder_outputs.unsqueeze(-2).repeat(1, 1, candi_num, 1)
        outputs_id = self.prob_out(torch.cat((output_multi, candi_vec), dim=-1)).squeeze(-1)
        outputs_id = outputs_id.masked_fill(candi_masks == 0, 0)

        return output_multi, candi_vec, outputs_id


# ==================== TRMMA 子模型（带软段嵌入注入） ====================

class GPSEncoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.pro_features_flag = parameters.pro_features_flag
        self.hid_dim = parameters.hid_dim
        self.transformer = GPSFormer(parameters.hid_dim, parameters.transformer_layers, heads=parameters.heads)
        if self.pro_features_flag:
            self.temporal = nn.Embedding(parameters.pro_input_dim, parameters.pro_output_dim)
            self.fc_hid = nn.Linear(parameters.hid_dim + parameters.pro_output_dim, parameters.hid_dim)

    def forward(self, src, src_len, pro_features):
        bs = src.size(1)
        max_src_len = src.size(0)

        mask3d = torch.ones(bs, max_src_len, max_src_len, device=src.device)
        mask2d = torch.ones(bs, max_src_len, device=src.device)

        mask3d = sequence_mask3d(mask3d, src_len, src_len)
        mask2d = sequence_mask(mask2d, src_len).transpose(0, 1).unsqueeze(-1).repeat(1, 1, self.hid_dim)

        src = src.transpose(0, 1)
        outputs = self.transformer(src, mask3d)
        outputs = outputs.transpose(0, 1)

        assert outputs.size(0) == max_src_len

        outputs = outputs * mask2d
        hidden = torch.sum(outputs, dim=0) / src_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)

        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))

        return outputs, hidden


class GREncoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.pro_features_flag = parameters.pro_features_flag
        self.transformer = GRFormer(parameters.hid_dim, parameters.transformer_layers, heads=parameters.heads)
        if self.pro_features_flag:
            self.temporal = nn.Embedding(parameters.pro_input_dim, parameters.pro_output_dim)
            self.fc_hid = nn.Linear(parameters.hid_dim + parameters.pro_output_dim, parameters.hid_dim)

    def forward(self, src, src_len, route, route_len, pro_features):
        bs = src.size(1)
        src_max_len = src.size(0)
        route_max_len = route.size(0)

        mask3d = torch.ones(bs, src_max_len, src_max_len, device=src.device)
        route_mask3d = torch.ones(bs, route_max_len, route_max_len, device=src.device)
        route_mask2d = torch.ones(bs, route_max_len, device=src.device)
        inter_mask = torch.ones(bs, route_max_len, src_max_len, device=src.device)

        mask3d = sequence_mask3d(mask3d, src_len, src_len)
        route_mask3d = sequence_mask3d(route_mask3d, route_len, route_len)
        route_mask2d = sequence_mask(route_mask2d, route_len).transpose(0,1).unsqueeze(-1).repeat(1, 1, self.hid_dim)
        inter_mask = sequence_mask3d(inter_mask, route_len, src_len)

        src = src.transpose(0, 1)
        route = route.transpose(0, 1)
        outputs = self.transformer(src, route, mask3d, route_mask3d, inter_mask)
        outputs = outputs.transpose(0, 1)

        assert outputs.size(0) == route_max_len

        outputs = outputs * route_mask2d
        hidden = torch.sum(outputs, dim=0) / route_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)

        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))

        return outputs, hidden


class DecoderMulti(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.id_size = parameters.id_size
        self.emb_id = None
        self.dest_type = parameters.dest_type
        self.rate_flag = parameters.rate_flag
        self.prog_flag = parameters.prog_flag
        self.rid_feats_flag = parameters.rid_feats_flag

        rnn_input_dim = parameters.hid_dim
        if self.rid_feats_flag:
            rnn_input_dim += parameters.rid_fea_dim
        if self.rate_flag:
            rnn_input_dim += 1
        if self.dest_type in [1, 2]:
            rnn_input_dim += parameters.hid_dim
            if self.rid_feats_flag:
                rnn_input_dim += parameters.rid_fea_dim
            if self.rate_flag:
                rnn_input_dim += 1

        self.rnn = nn.GRU(rnn_input_dim, parameters.hid_dim)
        self.attn_route = Attention(parameters.hid_dim)
        if self.rate_flag:
            fc_rate_out_input_dim = parameters.hid_dim + parameters.hid_dim
            self.fc_rate_out = nn.Sequential(
                nn.Linear(fc_rate_out_input_dim, parameters.hid_dim * 2),
                nn.ReLU(),
                nn.Linear(parameters.hid_dim * 2, 1),
                nn.Sigmoid()
            )

    def decoding_step(self, input_id, input_rate, hidden, route_outputs,
                      route_attn_mask, d_rids, d_rates, rid_features_dict, dt, observed_emb, observed_mask):
        rnn_input = self.emb_id[input_id]
        if self.rid_feats_flag:
            rnn_input = torch.cat([rnn_input, rid_features_dict[input_id]], dim=-1)
        if self.rate_flag:
            rnn_input = torch.cat((rnn_input, input_rate), dim=-1)
        if self.dest_type in [1, 2]:
            embed_drids = self.emb_id[d_rids]
            rnn_input = torch.cat((rnn_input, embed_drids), dim=-1)
            if self.rid_feats_flag:
                rnn_input = torch.cat([rnn_input, rid_features_dict[input_id]], dim=-1)
            if self.rate_flag:
                rnn_input = torch.cat((rnn_input, d_rates), dim=-1)

        rnn_input = rnn_input.unsqueeze(0)
        output, hidden = self.rnn(rnn_input, hidden)

        query = hidden.permute(1, 0, 2)
        key = route_outputs.permute(1, 0, 2).unsqueeze(1)
        scores, weighted = self.attn_route(query, key, key, route_attn_mask.unsqueeze(1))
        prediction_id = scores.squeeze(1).masked_fill(route_attn_mask == 0, 0)
        weighted = weighted.permute(1, 0, 2)

        if self.rate_flag:
            rate_input = torch.cat((hidden, weighted), dim=-1).squeeze(0)
            prediction_rate = self.fc_rate_out(rate_input)
        else:
            prediction_rate = torch.ones((prediction_id.shape[0], 1), dtype=torch.float32, device=hidden.device) / 2

        return prediction_id, prediction_rate, hidden

    def forward(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len, hidden, rid_features_dict, routes, route_outputs, route_attn_mask, d_rids, d_rates, teacher_forcing_ratio):
        routes = routes.permute(1, 0)
        outputs_id = torch.zeros([max_trg_len, batch_size, routes.shape[1]], device=hidden.device)
        rate_out_dim = 1
        outputs_rate = torch.zeros([max_trg_len, batch_size, rate_out_dim], device=hidden.device)

        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]
        for t in range(1, max_trg_len):
            teacher_force = random.random() < teacher_forcing_ratio
            dt = None
            observed_emb = None
            observed_mask = None
            prediction_id, prediction_rate, hidden = self.decoding_step(input_id, input_rate, hidden, route_outputs, route_attn_mask, d_rids, d_rates, rid_features_dict, dt, observed_emb, observed_mask)

            outputs_id[t] = prediction_id
            outputs_rate[t] = prediction_rate

            if teacher_force:
                input_id = trg_id[t]
                input_rate = trg_rate[t]
            else:
                input_id = (F.one_hot(prediction_id.argmax(dim=1), routes.shape[1]) * routes).sum(-1)
                input_rate = prediction_rate

        mask_trg = torch.ones([batch_size, max_trg_len], device=outputs_id.device)
        mask_trg = sequence_mask(mask_trg, torch.tensor(trg_len, device=outputs_id.device))
        outputs_rate = outputs_rate.permute(1, 0, 2)
        outputs_rate = outputs_rate.masked_fill(mask_trg.unsqueeze(-1) == 0, 0)
        outputs_rate = outputs_rate.permute(1, 0, 2)
        return outputs_id, outputs_rate


class TrajRecoveryE2E(nn.Module):
    """
    TRMMA 改造版：在 GPS 编码输入端注入“软路段嵌入”（来自 MMA 的候选分布加权和），
    从而让 TRMMA 的损失可以通过该通道反向到 MMA。
    """

    def __init__(self, parameters):
        super().__init__()
        self.srcseg_flag = parameters.srcseg_flag
        self.hid_dim = parameters.hid_dim
        self.da_route_flag = parameters.da_route_flag
        self.learn_pos = parameters.learn_pos
        self.rid_feats_flag = parameters.rid_feats_flag
        self.params = parameters

        self.emb_id = nn.Parameter(torch.rand(parameters.id_size, parameters.id_emb_dim))
        if self.learn_pos:
            max_input_length = 500
            self.pos_embedding_gps = nn.Embedding(max_input_length, parameters.hid_dim)
            self.pos_embedding_route = nn.Embedding(max_input_length, parameters.hid_dim)

        # 与 TRMMA 不同点：GPS 输入额外拼接一份 hid_dim 的软段嵌入
        input_dim_gps = 3
        if self.learn_pos:
            input_dim_gps += parameters.hid_dim
        if self.srcseg_flag:
            input_dim_gps += parameters.hid_dim + 1
        input_dim_gps += parameters.hid_dim  # 软段嵌入
        self.fc_in_gps = nn.Linear(input_dim_gps, parameters.hid_dim)

        input_dim_route = parameters.hid_dim
        if self.learn_pos:
            input_dim_route += parameters.hid_dim
        if self.rid_feats_flag:
            input_dim_route += parameters.rid_fea_dim
        self.fc_in_route = nn.Linear(input_dim_route, parameters.hid_dim)

        if self.da_route_flag:
            self.encoder = GREncoder(parameters)
        else:
            self.encoder = GPSEncoder(parameters)
        self.decoder = DecoderMulti(parameters)

        # forward 内阶段计时（与 TRMMA 对齐）
        self.timer1, self.timer2, self.timer3, self.timer4, self.timer5, self.timer6 = [], [], [], [], [], []

        # 权重初始化（与 MMA/TRMMA 保持一致）
        self.init_weights()

    def init_weights(self):
        """
        与 MMA/TRMMA 相同的权重初始化策略：
        - RNN 的 weight_ih 用 Xavier 均匀
        - RNN 的 weight_hh 用正交
        - bias 全部置 0
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, src, src_len, trg_id, trg_rate, trg_len, pro_features, rid_features_dict, da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio, soft_seg_emb):
        t0 = time.time()

        max_trg_len = trg_id.size(0)
        batch_size = trg_id.size(1)

        self.decoder.emb_id = self.emb_id

        gps_emb = src.float()
        if self.learn_pos:
            gps_pos = src[:, :, -1].long()
            gps_pos_emb = self.pos_embedding_gps(gps_pos)
            gps_emb = torch.cat([gps_emb, gps_pos_emb], dim=-1)
        if self.srcseg_flag:
            seg_emb = self.emb_id[src_seg_seqs]
            gps_emb = torch.cat((gps_emb, seg_emb, src_seg_feats), dim=-1)
        # 注入来自 MMA 的软段嵌入（维度 [src_len, bs, hid_dim]）
        gps_emb = torch.cat([gps_emb, soft_seg_emb], dim=-1)
        gps_in = self.fc_in_gps(gps_emb)
        gps_in_lens = torch.tensor(src_len, device=src.device)

        self.timer1.append(time.time() - t0)
        t1 = time.time()

        if self.da_route_flag:
            route_emb = self.emb_id[da_routes]
            if self.learn_pos:
                route_pos_emb = self.pos_embedding_route(da_pos)
                route_emb = torch.cat([route_emb, route_pos_emb], dim=-1)
            if self.rid_feats_flag:
                route_feats = rid_features_dict[da_routes]
                route_emb = torch.cat([route_emb, route_feats], dim=-1)
            route_in = self.fc_in_route(route_emb)
            route_in_lens = torch.tensor(da_lengths, device=src.device)
            self.timer2.append(time.time() - t1)
            t2 = time.time()
            route_outputs, hiddens = self.encoder(gps_in, gps_in_lens, route_in, route_in_lens, pro_features)
            self.timer3.append(time.time() - t2)
            t3 = time.time()
        else:
            _, hiddens = self.encoder(gps_in, gps_in_lens, pro_features)
            route_in_lens = torch.tensor(da_lengths, device=src.device)
            route_outputs = self.emb_id[da_routes]

        route_attn_mask = torch.ones(batch_size, max(da_lengths), device=src.device)
        route_attn_mask = sequence_mask(route_attn_mask, route_in_lens)

        self.timer4.append(time.time() - (t3 if self.da_route_flag else t1))
        t4 = time.time()
        outputs_id, outputs_rate = self.decoder(max_trg_len, batch_size, trg_id, trg_rate, trg_len, hiddens, rid_features_dict, da_routes, route_outputs, route_attn_mask, d_rids, d_rates, teacher_forcing_ratio)

        final_outputs_id = outputs_id[1:-1]
        final_outputs_rate = outputs_rate[1:-1]
        self.timer5.append(time.time() - t4)
        self.timer6.append(time.time() - t0)
        return final_outputs_id, final_outputs_rate


# ==================== 端到端模型封装 ====================

class End2EndModel(nn.Module):
    """
    将 MMA 与 TRMMA 串联：
    - MMA 对低频端点生成候选概率分布，并形成“软段嵌入”；
    - TRMMAE2E 接收软段嵌入，进行解码恢复；
    - TRMMA 的损失通过软段嵌入反传到 MMA。
    """

    def __init__(self, mma_args: AttrDict, trmma_args: AttrDict):
        super().__init__()
        self.mma = GPS2Seg(mma_args)
        self.trmma = TrajRecoveryE2E(trmma_args)

    def forward(self,
                # MMA 输入
                mma_src_seqs, mma_src_lens, mma_candi_ids, mma_candi_feats, mma_candi_masks,
                # TRMMA 输入
                src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                pro_features, rid_features_dict, da_routes, da_lengths, da_pos,
                src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio):

        # 1) MMA 前向：两端点的候选概率分布
        mma_out_multi, mma_candi_vec, mma_probs = self.mma(mma_src_seqs, mma_src_lens, mma_candi_ids, mma_candi_feats, mma_candi_masks)
        # 软段嵌入（概率加权和），形状 [bs, src_len, hid]
        soft_seg_emb = (mma_probs.unsqueeze(-1) * mma_candi_vec).sum(dim=-2)
        # 转换为 TRMMA 期望的 [src_len, bs, hid]
        soft_seg_emb = soft_seg_emb.permute(1, 0, 2)

        # 2) TRMMA 前向：注入软嵌入
        out_ids, out_rates = self.trmma(src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                        pro_features, rid_features_dict, da_routes, da_lengths, da_pos,
                                        src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio,
                                        soft_seg_emb)

        return {
            'mma_probs': mma_probs,  # [bs, src_len, candi]
            'out_ids': out_ids,      # [trg_len-2, bs, da_len]
            'out_rates': out_rates   # [trg_len-2, bs, 1]
        }


# ==================== 损失函数（联合） ====================

class E2ELoss(nn.Module):
    """联合损失：MMA BCE + TRMMA BCE + TRMMA L1。"""

    def __init__(self, lambda_mma=1.0, lambda_id=10.0, lambda_rate=5.0):
        super().__init__()
        self.lambda_mma = lambda_mma
        self.lambda_id = lambda_id
        self.lambda_rate = lambda_rate
        self.crit_mma = nn.BCELoss(reduction='mean')
        self.crit_id = nn.BCELoss(reduction='sum')
        self.crit_rate = nn.L1Loss(reduction='sum')

    def forward(self, outputs, labels, lengths):
        # MMA
        mma_probs = outputs['mma_probs']  # [bs, 2, k]
        mma_onehot = labels['mma_onehot'].float()  # [bs, 2, k]
        loss_mma = self.crit_mma(mma_probs, mma_onehot) * mma_probs.shape[-1]

        # TRMMA
        out_ids = outputs['out_ids']       # [trg_len-2, bs, da_len]
        out_rates = outputs['out_rates']   # [trg_len-2, bs, 1]
        trg_labels = labels['trg_labels']  # 已与模型输出对齐（中间步），不再裁剪
        trg_rates = labels['trg_rates'][1:-1]               # [trg_len-2, bs, 1]

        trg_lengths = lengths['trg_lengths']
        da_lengths = lengths['da_lengths']
        trg_lengths_sub = [length - 2 for length in trg_lengths]
        denom_id = np.sum(np.array(trg_lengths_sub) * np.array(da_lengths))
        denom_id = max(1, denom_id)
        denom_rate = max(1, sum(trg_lengths_sub))

        loss_id = self.crit_id(out_ids, trg_labels) * self.lambda_id / denom_id
        loss_rate = self.crit_rate(out_rates, trg_rates) * self.lambda_rate / denom_rate

        total = self.lambda_mma * loss_mma + loss_id + loss_rate
        return total, {'mma': loss_mma.item(), 'id': loss_id.item(), 'rate': loss_rate.item()}



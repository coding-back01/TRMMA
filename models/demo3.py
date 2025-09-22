import os
import pickle
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset

from models.layers import Attention, GPSFormer, GRFormer, sequence_mask, sequence_mask3d, Encoder
from utils.model_utils import gps2grid, get_normalized_t, AttrDict
from utils.spatial_func import SPoint
from preprocess import SparseDAM, SegInfo
import networkx as nx
import datetime as dt


# ==================== 数据集（端到端，demo3隔离版） ====================

class E2E3TrajData(Dataset):
    """
    端到端训练数据集（demo3专用，避免与demo2互相引用）。
    每条轨迹切分为 gap 样本，提供：
      - 两端点候选集（id/feat/mask/onehot标签）
      - 路由（训练/验证用真值子路径；测试用规划路径）
      - 高频目标段序列与位置比率
    """

    def __init__(self, rn, trajs_dir, mbr, parameters, mode: str):
        self.parameters = parameters
        self.rn = rn
        self.mbr = mbr
        self.grid_size = parameters.grid_size
        self.time_span = parameters.time_span
        self.mode = mode
        self.keep_ratio = parameters.keep_ratio
        if mode == 'train':
            self.keep_ratio = getattr(parameters, 'init_ratio', parameters.keep_ratio)

        if mode == 'train':
            file = os.path.join(trajs_dir, 'train.pkl')
        elif mode == 'valid':
            file = os.path.join(trajs_dir, 'valid.pkl')
        elif mode == 'test':
            file = os.path.join(trajs_dir, 'test_output.pkl')
        else:
            raise NotImplementedError
        trajs = pickle.load(open(file, "rb"))
        if parameters.small and mode == 'train':
            idx_group = 0
            num_group = 5
            num_k = len(trajs) // num_group
            trajs = trajs[num_k * idx_group: num_k * (idx_group + 1)]
        self.trajs = trajs

        self.dam = None
        if self.mode != 'train':
            self.dam = DA3Planner(parameters.dam_root, parameters.id_size - 1, parameters.utc)

        self.groups = []
        self.src_mms = []
        if self.mode != 'train':
            for serial, traj in enumerate(self.trajs):
                low_idx = traj.low_idx
                src_list = np.array(traj.pt_list, dtype=object)
                src_list = src_list[low_idx].tolist()

                src_mm = []
                for pt in src_list:
                    candi_pt = pt.data['candi_pt']
                    src_mm.append([[candi_pt.lat, candi_pt.lng], candi_pt.eid, candi_pt.rate])
                self.src_mms.append(src_mm)

                for p1_idx, p2_idx in zip(low_idx[:-1], low_idx[1:]):
                    if (p1_idx + 1) < p2_idx:
                        self.groups.append(serial)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        traj = self.trajs[index]

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
            if (p1_idx + 1) < p2_idx:
                tmp_src_list = [p1, p2]

                ls_grid_seq, ls_gps_seq, hours, tmp_seg_seq = self._get_src_seq(tmp_src_list)
                features = self._get_pro_features(tmp_src_list, hours)

                trg_candis = self.rn.get_trg_segs(ls_gps_seq, self.parameters.candi_size, self.parameters.search_dist, self.parameters.beta)
                candi_onehot, candi_ids, candi_feats, candi_masks = self._build_selector_candidates(trg_candis, tmp_seg_seq)

                mm_eids, mm_rates = self._get_trg_seq(trg_list[p1_idx: p2_idx + 1])
                if self.mode in ['train', 'valid']:
                    path = traj.cpath[p1.cpath_idx: p2.cpath_idx + 1]
                else:
                    s1, s2 = tmp_seg_seq[0], tmp_seg_seq[1]
                    ts = getattr(p1, 'time', None)
                    path = self.dam.planning_multi([s1, s2], ts, mode=self.parameters.planner)

                da_route = [self.rn.valid_edge_one[item] for item in path]
                src_seg_seq = [self.rn.valid_edge_one[item] for item in tmp_seg_seq]
                src_seg_feat = self._get_src_seg_feat(ls_gps_seq, tmp_seg_seq)
                label = self._get_label([self.rn.valid_edge_one[item] for item in path], mm_eids[1:-1])

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

                candi_onehot = torch.tensor(candi_onehot)
                candi_ids = torch.tensor(candi_ids) + 1
                candi_feats = torch.tensor(candi_feats)
                candi_masks = torch.tensor(candi_masks, dtype=torch.float32)

                data.append([
                    da_route, src_grid_seq, src_pro_fea, src_seg_seq, src_seg_feat, trg_rid, trg_rate, label, d_rid, d_rate,
                    candi_onehot, candi_ids, candi_feats, candi_masks
                ])

        return data

    def _get_src_seg_feat(self, gps_seq, seg_seq):
        feats = []
        for ds_pt, seg in zip(gps_seq, seg_seq):
            candi = self.rn.pt2seg(SPoint(ds_pt[0], ds_pt[1]), seg)
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

    def _build_selector_candidates(self, ls_candi, trg_id_seq):
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
            if len(tmp_feat) == 0:
                tmp_feat.append([0] * 9)
            tmp_feat.extend([[0] * len(tmp_feat[0])] * (self.parameters.candi_size - len(candis)))
            if trg in tmp_id:
                idx = tmp_id.index(trg)
                tmp_onehot[idx] = 1
            candi_id.append(tmp_id)
            candi_feat.append(tmp_feat)
            candi_onehot.append(tmp_onehot)
        return candi_onehot, candi_id, candi_feat, candi_mask


# ==================== 规划器（demo3隔离版） ====================

def calc_cos_value(vec1, vec2):
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    a = vec1 * vec1
    b = vec2 * vec2
    c = vec1 * vec2
    denom = np.sqrt(a[0] + a[1]) * np.sqrt(b[0] + b[1])
    cos_value = (c[0] + c[1]) / denom if denom != 0 else 1.0
    return cos_value


class DA3Planner(object):
    def __init__(self, dam_root, id_size, utc):
        self.csm = SparseDAM(dam_root, id_size)
        self.seg_info = SegInfo(os.path.join(dam_root, "seg_info.csv"))
        self.G = pickle.load(open(os.path.join(dam_root, "road_graph_wtime"), "rb"))
        self.vehicle_num = np.load(os.path.join(dam_root, "vehicle_num_{}-48.npy".format(3600)))
        self.tz = dt.timezone(dt.timedelta(hours=utc))
        self.max_seq_len = 79
        self.freq_limit = 1
        self.dcsm_theta = 1
        self.no_path_cnt = 0

    def planning_multi(self, od, t, mode='da', segs_flag=False):
        pred = [od[0]]
        timestamp = t
        if timestamp is None:
            timestamp = 0
        else:
            timestamp = timestamp + self.seg_info.get_seg_travel_time(od[0])
        segs = []
        for i in range(len(od) - 1):
            o = od[i]
            d = od[i + 1]

            if pred[-1] != o:
                break

            if mode == 'da':
                col_d = self.csm.get_col(d)
                route = [o]
                seg_used = np.zeros(self.seg_info.seg_num, dtype=np.int32)
                seg_used[o] = 1
                seg_used[d] = 1

                while len(route) < self.max_seq_len and route[-1] != d:
                    out_segs = list(self.G.neighbors(route[-1]))
                    if len(out_segs) == 0:
                        break

                    nextseg = -1
                    next_max = -1
                    tie_cnt = 0
                    tie_nbrs = []
                    for seg in out_segs:
                        if seg == d:
                            nextseg = d
                            tie_cnt = 1
                            break
                        if seg_used[seg] >= self.freq_limit:
                            continue
                        curr_prob = col_d[seg]
                        if curr_prob is None:
                            curr_prob = 0
                        if curr_prob > next_max:
                            nextseg = seg
                            tie_cnt = 1
                            next_max = curr_prob
                            tie_nbrs = [seg]
                        elif curr_prob == next_max:
                            tie_cnt += 1
                            tie_nbrs.append(seg)

                    if tie_cnt != 1:
                        if tie_cnt == 0:
                            tie_nbrs = out_segs
                        if next_max < self.dcsm_theta:
                            nextseg, _ = self.break_tie_angle(route[-1], tie_nbrs, d)
                        else:
                            nextseg, flag = self.break_tie_traffic_flow(tie_nbrs, timestamp)
                            if flag:
                                nextseg, _ = self.break_tie_angle(route[-1], tie_nbrs, d)

                    if nextseg == -1:
                        break
                    route.append(nextseg)
                    timestamp += self.seg_info.get_seg_travel_time(nextseg)
                    seg_used[nextseg] += 1

                if route[-1] != d:
                    try:
                        _, route = nx.bidirectional_dijkstra(self.G, o, d, weight="time")
                    except nx.exception.NetworkXNoPath:
                        self.no_path_cnt += 1
                        route = [o, d]
                route = self.remove_circle(route)
            elif mode == 'time':
                try:
                    _, route = nx.bidirectional_dijkstra(self.G, o, d, weight="time")
                except nx.exception.NetworkXNoPath:
                    self.no_path_cnt += 1
                    route = [o, d]
            elif mode == 'length':
                try:
                    _, route = nx.bidirectional_dijkstra(self.G, o, d, weight="length")
                except nx.exception.NetworkXNoPath:
                    self.no_path_cnt += 1
                    route = [o, d]
            else:
                raise NotImplementedError
            pred = pred + route[1:]
            segs.append(route)
        if segs_flag:
            return pred, segs
        else:
            return pred

    def break_tie_angle(self, curr, tie_nbrs, d):
        curr_geo = self.seg_info.get_seg_geo(curr)
        curr_trg = curr_geo[2:]
        d_geo = self.seg_info.get_seg_geo(d)
        d_src = d_geo[:2]
        vec1 = d_src - curr_trg
        nextseg = -1
        next_max = -2
        tie_cnt = 0
        for seg in tie_nbrs:
            vec2 = self.seg_info.get_seg_vec(seg)
            cos_value = calc_cos_value(vec1, vec2)
            if cos_value > next_max:
                nextseg = seg
                tie_cnt = 1
                next_max = cos_value
            else:
                tie_cnt += 1
        flag = False
        return nextseg, flag

    def break_tie_traffic_flow(self, tie_nbrs, timestamp):
        idx, _ = self.get_time_idx2(timestamp)
        nextseg = -1
        next_max = -1
        tie_cnt = 0
        for seg in tie_nbrs:
            prob = self.vehicle_num[seg, idx]
            if prob > next_max:
                nextseg = seg
                tie_cnt = 1
                next_max = prob
            elif prob == next_max:
                tie_cnt += 1
        flag = False
        if tie_cnt > 1:
            flag = True
        return nextseg, flag

    def get_time_idx2(self, timestamp):
        time_arr = dt.datetime.fromtimestamp(timestamp, self.tz)
        if time_arr.weekday() in [0, 1, 2, 3, 4]:
            idx = time_arr.hour
        else:
            idx = time_arr.hour + 24
        t_r = (time_arr.minute * 60 + time_arr.second) * 1.0 / 3600
        return int(idx), t_r

    def remove_circle(self, path_fixed):
        cur = 0
        while cur < len(path_fixed):
            eid = path_fixed[cur]
            idx = []
            for i in range(cur, len(path_fixed)):
                if path_fixed[i] == eid:
                    idx.append(i)
            path_fixed = path_fixed[0: cur] + path_fixed[max(idx): ]
            cur += 1
        return path_fixed


# ==================== 选择器（MMA，可微） ====================

class Selector3Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.fc_in = nn.Linear(3, parameters.hid_dim)
        self.transformer = Encoder(parameters.hid_dim, parameters.transformer_layers, heads=4)

    def forward(self, src, src_len):
        max_src_len = src.size(1)
        bs = src.size(0)
        src_len_t = torch.tensor(src_len, device=src.device)
        mask3d = torch.ones(bs, max_src_len, max_src_len, device=src.device)
        mask2d = torch.ones(bs, max_src_len, device=src.device)
        mask3d = sequence_mask3d(mask3d, src_len_t, src_len_t)
        mask2d = sequence_mask(mask2d, src_len_t).unsqueeze(-1).repeat(1, 1, self.hid_dim)
        src = self.fc_in(src)
        outputs = self.transformer(src, mask3d)
        outputs = outputs * mask2d
        return outputs


class Selector3Head(nn.Module):
    """
    候选选择头（demo3）：
    - 输出候选 logits（未归一化，便于稳定的联合损失）
    - 提供 masked softmax 概率与 Gumbel-Softmax 权重（用于软段嵌入）
    """
    def __init__(self, parameters):
        super().__init__()
        self.params = parameters
        self.direction_flag = parameters.direction_flag
        self.attn_flag = parameters.attn_flag
        self.only_direction = parameters.only_direction
        self.use_gumbel = getattr(parameters, 'use_gumbel', True)
        self.tau = getattr(parameters, 'gumbel_tau', 1.0)

        self.emb_id = nn.Embedding(parameters.id_size, parameters.id_emb_dim)
        self.encoder = Selector3Encoder(parameters)

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
            nn.Linear(parameters.hid_dim * 2, 1)
        )

    def forward(self, src, src_len, candi_ids, candi_feats, candi_masks, tau: float = None):
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
        logits = self.prob_out(torch.cat((output_multi, candi_vec), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(candi_masks == 0, float('-inf'))

        probs = F.softmax(logits, dim=-1)

        # gumbel soft weights for soft segment embedding (train-time)
        if tau is None:
            tau = self.tau
        if self.use_gumbel and self.training:
            gumbel_w = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        else:
            gumbel_w = probs

        return output_multi, candi_vec, logits, probs, gumbel_w


# ==================== 重建分支（TRMMA，路由后缀掩码） ====================

class GPSEncoder3(nn.Module):
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
        outputs = outputs * mask2d
        hidden = torch.sum(outputs, dim=0) / src_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)
        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))
        return outputs, hidden


class GREncoder3(nn.Module):
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
        route_mask2d = sequence_mask(route_mask2d, route_len).transpose(0, 1).unsqueeze(-1).repeat(1, 1, self.hid_dim)
        inter_mask = sequence_mask3d(inter_mask, route_len, src_len)
        src = src.transpose(0, 1)
        route = route.transpose(0, 1)
        outputs = self.transformer(src, route, mask3d, route_mask3d, inter_mask)
        outputs = outputs.transpose(0, 1)
        outputs = outputs * route_mask2d
        hidden = torch.sum(outputs, dim=0) / route_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)
        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))
        return outputs, hidden


class Decoder3(nn.Module):
    """
    路径后缀掩码的段预测 + 位置比率预测。
    - 段预测：对 R 或其后缀进行注意力打分（概率），不输出全图。
    - 位置比率：MLP 回归并在损失加入平滑正则。
    """
    def __init__(self, parameters):
        super().__init__()
        self.id_size = parameters.id_size
        self.emb_id = None
        self.rate_flag = parameters.rate_flag
        self.rid_feats_flag = parameters.rid_feats_flag
        self.dest_type = parameters.dest_type
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
            self.fc_rate_out = nn.Sequential(
                nn.Linear(parameters.hid_dim * 2, parameters.hid_dim * 2),
                nn.ReLU(),
                nn.Linear(parameters.hid_dim * 2, 1),
                nn.Sigmoid()
            )

    def _decoding_step(self, input_id, input_rate, hidden, route_outputs, suffix_mask, d_rids, d_rates, rid_features_dict):
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
        scores, weighted = self.attn_route(query, key, key, suffix_mask.unsqueeze(1))
        pred_id_prob = scores.squeeze(1).masked_fill(suffix_mask == 0, 0)
        weighted = weighted.permute(1, 0, 2)
        if self.rate_flag:
            rate_input = torch.cat((hidden, weighted), dim=-1).squeeze(0)
            pred_rate = self.fc_rate_out(rate_input)
        else:
            pred_rate = torch.ones((pred_id_prob.shape[0], 1), dtype=torch.float32, device=hidden.device) / 2
        return pred_id_prob, pred_rate, hidden

    def forward(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len, hidden,
                rid_features_dict, routes, route_outputs, route_len, d_rids, d_rates,
                teacher_forcing_ratio, trg_labels=None, disable_suffix: bool = False):
        routes = routes.permute(1, 0)  # [bs, R]
        outputs_id = torch.zeros([max_trg_len, batch_size, routes.shape[1]], device=hidden.device)
        outputs_rate = torch.zeros([max_trg_len, batch_size, 1], device=hidden.device)
        # 初始化后缀指针（各样本从0位置开始）
        ptr = torch.zeros(batch_size, dtype=torch.long, device=hidden.device)

        # 若提供GT标签，预先计算每步的后缀起点（训练时可精确约束）
        gt_ptr_seq = None
        if trg_labels is not None:
            # trg_labels: [T_lab, bs, R]
            gt_idx = trg_labels.argmax(dim=-1)  # [T_lab, bs]
            # 每步允许索引 >= 上一步的 gt_idx
            gt_ptr_seq = torch.cummax(gt_idx, dim=0)[0]  # [T_lab, bs]
            # 对齐长度至 max_trg_len，末端重复最后一步，避免越界
            T_lab = gt_ptr_seq.size(0)
            if T_lab < max_trg_len:
                pad_steps = max_trg_len - T_lab
                last = gt_ptr_seq[-1:].repeat(pad_steps, 1)
                gt_ptr_seq = torch.cat([gt_ptr_seq, last], dim=0)

        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]
        for t in range(1, max_trg_len):
            if teacher_forcing_ratio > 0:
                use_teacher = random.random() < teacher_forcing_ratio
            else:
                use_teacher = False

            # 构建当前步的路由后缀掩码（可禁用，使用全R注意力做诊断）
            R = routes.shape[1]
            ar = torch.arange(R, device=hidden.device).unsqueeze(0).repeat(batch_size, 1)
            if disable_suffix:
                suffix_mask = torch.ones(batch_size, R, device=hidden.device)
            else:
                if gt_ptr_seq is not None and use_teacher:
                    cur_ptr = gt_ptr_seq[t - 1]
                else:
                    cur_ptr = ptr
                suffix_mask = (ar >= cur_ptr.unsqueeze(1)).float()
            # 长度掩码（按 route_len）
            len_mask = sequence_mask(torch.ones(batch_size, R, device=hidden.device), route_len).float()
            suffix_mask = suffix_mask * len_mask

            pred_id, pred_rate, hidden = self._decoding_step(input_id, input_rate, hidden, route_outputs, suffix_mask, d_rids, d_rates, rid_features_dict)
            outputs_id[t] = pred_id
            outputs_rate[t] = pred_rate

            if use_teacher:
                input_id = trg_id[t]
                input_rate = trg_rate[t]
                ptr = torch.max(ptr, gt_ptr_seq[t])
            else:
                # 依据预测分布更新指针（单调）
                pred_idx = pred_id.argmax(dim=-1)
                ptr = torch.max(ptr, pred_idx)
                input_id = (F.one_hot(pred_idx, routes.shape[1]) * routes).sum(-1)
                input_rate = pred_rate

        mask_trg = torch.ones([batch_size, max_trg_len], device=outputs_id.device)
        mask_trg = sequence_mask(mask_trg, torch.tensor(trg_len, device=outputs_id.device))
        outputs_rate = outputs_rate.permute(1, 0, 2)
        outputs_rate = outputs_rate.masked_fill(mask_trg.unsqueeze(-1) == 0, 0)
        outputs_rate = outputs_rate.permute(1, 0, 2)
        return outputs_id, outputs_rate


class Reconstructor3(nn.Module):
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

        # GPS 输入：3 + 可选pos + 源段嵌入 + 软段嵌入
        input_dim_gps = 3
        if self.learn_pos:
            input_dim_gps += parameters.hid_dim
        if self.srcseg_flag:
            input_dim_gps += parameters.hid_dim + 1
        input_dim_gps += parameters.hid_dim
        self.fc_in_gps = nn.Linear(input_dim_gps, parameters.hid_dim)

        input_dim_route = parameters.hid_dim
        if self.learn_pos:
            input_dim_route += parameters.hid_dim
        if self.rid_feats_flag:
            input_dim_route += parameters.rid_fea_dim
        self.fc_in_route = nn.Linear(input_dim_route, parameters.hid_dim)

        if self.da_route_flag:
            self.encoder = GREncoder3(parameters)
        else:
            self.encoder = GPSEncoder3(parameters)
        self.decoder = Decoder3(parameters)

    def forward(self, src, src_len, trg_id, trg_rate, trg_len, pro_features,
                rid_features_dict, da_routes, da_lengths, da_pos,
                src_seg_seqs, src_seg_feats, d_rids, d_rates,
                teacher_forcing_ratio, soft_seg_emb, trg_labels=None):
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
        gps_emb = torch.cat([gps_emb, soft_seg_emb], dim=-1)
        gps_in = self.fc_in_gps(gps_emb)
        gps_in_lens = torch.tensor(src_len, device=src.device)

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
            route_outputs, hiddens = self.encoder(gps_in, gps_in_lens, route_in, route_in_lens, pro_features)
        else:
            _, hiddens = self.encoder(gps_in, gps_in_lens, pro_features)
            route_in_lens = torch.tensor(da_lengths, device=src.device)
            route_outputs = self.emb_id[da_routes]

        out_ids, out_rates = self.decoder(max_trg_len, batch_size, trg_id, trg_rate, trg_len, hiddens,
                                          rid_features_dict, da_routes, route_outputs, route_in_lens,
                                          d_rids, d_rates, teacher_forcing_ratio, trg_labels=trg_labels,
                                          disable_suffix=getattr(self.params, 'disable_suffix', False))

        final_outputs_id = out_ids[1:-1]
        final_outputs_rate = out_rates[1:-1]
        return final_outputs_id, final_outputs_rate


# ==================== 端到端封装（带KL/熵/平滑） ====================

class End2End3Model(nn.Module):
    def __init__(self, args: AttrDict):
        super().__init__()
        self.selector = Selector3Head(args)
        self.reconstructor = Reconstructor3(args)
        self.eps = 1e-9

    def forward(self,
                # 选择器输入
                sel_src_seqs, sel_src_lens, sel_candi_ids, sel_candi_feats, sel_candi_masks,
                # 重建器输入
                src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                pro_features, rid_features_dict, da_routes, da_lengths, da_pos,
                src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio,
                tau=None, trg_rid_labels=None):

        sel_out_multi, sel_candi_vec, sel_logits, sel_probs, sel_weights = self.selector(
            sel_src_seqs, sel_src_lens, sel_candi_ids, sel_candi_feats, sel_candi_masks, tau=tau
        )
        soft_seg_emb = (sel_weights.unsqueeze(-1) * sel_candi_vec).sum(dim=-2)
        soft_seg_emb = soft_seg_emb.permute(1, 0, 2)

        out_ids, out_rates = self.reconstructor(src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                                pro_features, rid_features_dict, da_routes, da_lengths, da_pos,
                                                src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio,
                                                soft_seg_emb, trg_labels=trg_rid_labels)

        return {
            'selector_logits': sel_logits,     # [bs, 2, k]
            'selector_probs': sel_probs,       # [bs, 2, k]
            'out_ids': out_ids,                # [T-2, bs, R]
            'out_rates': out_rates             # [T-2, bs, 1]
        }


class E2E3Loss(nn.Module):
    def __init__(self, lambda_selector=1.0, lambda_id=10.0, lambda_rate=5.0,
                 lambda_kl=0.1, lambda_entropy=0.05, lambda_smooth=0.5, use_bce_logits=False):
        super().__init__()
        self.lambda_selector = lambda_selector
        self.lambda_id = lambda_id
        self.lambda_rate = lambda_rate
        self.lambda_kl = lambda_kl
        self.lambda_entropy = lambda_entropy
        self.lambda_smooth = lambda_smooth
        self.use_bce_logits = use_bce_logits
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='sum')
        self.l1 = nn.L1Loss(reduction='sum')
        self.eps = 1e-9
        # 限制KL计算的样本数，避免大批次时过慢
        self.kl_max_batch = 512

    def _entropy(self, p):
        p = p.clamp_min(self.eps)
        return -(p * torch.log(p)).sum(dim=-1)

    def _rate_smooth(self, rates, mask_len: List[int]):
        # rates: [T, bs, 1] -> [T, bs]
        r = rates.squeeze(-1)
        if r.shape[0] <= 1:
            return torch.tensor(0.0, device=rates.device)
        dr = (r[1:] - r[:-1]).abs()
        # mask by valid steps per batch
        Tm1, bs = dr.shape
        mask = torch.zeros(bs, Tm1, device=rates.device)
        # mask_len contains original trg_len; here loss computed on T-2 steps, so effective len-1
        eff = [max(0, l - 3) for l in mask_len]
        for i, l in enumerate(eff):
            if l > 0:
                mask[i, :l] = 1
        mask = mask.transpose(0, 1)
        return (dr * mask).sum()

    def _build_c_union(self, candi_ids):
        # candi_ids: [bs, 2, k]
        # return union set per batch (list of tensors with unique ids), and mapping idx in union per endpoint
        bs, two, k = candi_ids.shape
        unions = []
        maps = []
        for b in range(bs):
            ids0 = candi_ids[b, 0].tolist()
            ids1 = candi_ids[b, 1].tolist()
            union = []
            for x in ids0 + ids1:
                if x not in union:
                    union.append(x)
            unions.append(torch.tensor(union, device=candi_ids.device, dtype=torch.long))
            # map endpoint candidates to union index (-1 if padded 0 and not present)
            m0 = torch.tensor([union.index(x) if x in union else -1 for x in ids0], device=candi_ids.device)
            m1 = torch.tensor([union.index(x) if x in union else -1 for x in ids1], device=candi_ids.device)
            maps.append((m0, m1))
        return unions, maps

    def _proj_route_probs_to_union(self, routes, out_ids, unions):
        # routes: [R]; out_ids: [T, R]; unions: [U]
        # 先对时间维求和，得到每个 route 位置的总概率，再一次性映射到 union
        dist_r = out_ids.sum(dim=0)  # [R]
        U = unions.shape[0]
        if U == 0:
            return unions.new_zeros((0,), dtype=dist_r.dtype)
        # [R, U] 匹配矩阵
        matches = routes.unsqueeze(1).eq(unions.unsqueeze(0))
        agg = (matches.float() * dist_r.unsqueeze(1)).sum(dim=0)  # [U]
        s = agg.sum().clamp_min(self.eps)
        # 若无重叠，返回均匀分布
        out = torch.where(s > 0, agg / s, torch.full((U,), 1.0 / max(1, U), device=out_ids.device))
        return out

    def forward(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], lengths: Dict[str, List[int]],
                aux: Dict[str, torch.Tensor]):
        # selector loss
        sel_logits = outputs['selector_logits']  # [bs, 2, k]
        sel_probs = outputs['selector_probs']    # [bs, 2, k]
        sel_onehot = labels['selector_onehot'].float()  # [bs, 2, k]
        sel_mask = (sel_onehot.sum(dim=-1) > 0.5).float()
        if self.use_bce_logits:
            # BCE on logits with onehot labels (masked)
            loss_sel = self.bce_logits(sel_logits, sel_onehot)
            # zero out where no GT
            loss_sel = (loss_sel * sel_mask.unsqueeze(-1)).sum() / sel_mask.sum().clamp_min(1.0)
        else:
            sel_p_true = (sel_probs * sel_onehot).sum(dim=-1).clamp_min(self.eps)
            nll_sel = -torch.log(sel_p_true) * sel_mask
            loss_sel = nll_sel.sum() / sel_mask.sum().clamp_min(1.0)

        # decoder losses（时间维对齐：模型输出对应标签的中间步）
        out_ids = outputs['out_ids']         # [T_pred, bs, R]
        out_rates = outputs['out_rates']     # [T_pred, bs, 1]
        trg_labels = labels['trg_labels']    # [T_lab, bs, R] 或 [T_pred, bs, R]
        trg_rates = labels['trg_rates']      # [T_lab, bs, 1]

        T_pred = out_ids.size(0)
        # 对齐段标签（通常已是 T-2，与 T_pred 一致）
        T_lab_labels = trg_labels.size(0)
        if T_lab_labels >= T_pred + 2:
            trg_labels_aligned = trg_labels[1:-1]
        elif T_lab_labels == T_pred:
            trg_labels_aligned = trg_labels
        else:
            pad = T_pred - T_lab_labels
            trg_labels_aligned = torch.cat([trg_labels, trg_labels[-1:].repeat(pad, 1, 1)], dim=0)

        # 对齐比率标签（通常是 T，需要裁剪为 [1:-1] 与 T_pred 对齐）
        T_lab_rates = trg_rates.size(0)
        if T_lab_rates >= T_pred + 2:
            trg_rates_aligned = trg_rates[1:-1]
        elif T_lab_rates == T_pred:
            trg_rates_aligned = trg_rates
        else:
            pad = T_pred - T_lab_rates
            trg_rates_aligned = torch.cat([trg_rates, trg_rates[-1:].repeat(pad, 1, 1)], dim=0)

        step_mask = (trg_labels_aligned.sum(dim=-1) > 0.5).float()  # [T_pred, bs]
        p_true = (out_ids * trg_labels_aligned).sum(dim=-1).clamp_min(self.eps)
        nll = -torch.log(p_true)
        valid_steps = step_mask.sum().clamp_min(1.0)
        loss_id = (nll * step_mask).sum() * self.lambda_id / valid_steps

        denom_rate = max(1, sum([l - 2 for l in lengths['trg_lengths']]))
        loss_rate = self.l1(out_rates, trg_rates_aligned) * self.lambda_rate / denom_rate

        # KL 对齐（路由分布投影到候选并与 selector 对齐）
        candi_ids = aux['candi_ids']  # [bs, 2, k]
        routes = aux['routes']        # [bs, R]
        unions, maps = self._build_c_union(candi_ids)
        bs = sel_probs.shape[0]
        kl_sum = 0.0
        ent_sum = 0.0
        if bs > self.kl_max_batch:
            idx_subset = torch.randperm(bs, device=sel_probs.device)[:self.kl_max_batch].tolist()
        else:
            idx_subset = range(bs)
        for b in idx_subset:
            union = unions[b]
            if union.numel() == 0:
                continue
            # selector分布投到 union：端点平均
            m0, m1 = maps[b]
            k0 = (m0 >= 0)
            k1 = (m1 >= 0)
            Pm = torch.zeros(union.shape[0], device=sel_probs.device)
            if k0.any():
                Pm.index_add_(0, m0[k0], sel_probs[b, 0, k0])
            if k1.any():
                Pm.index_add_(0, m1[k1], sel_probs[b, 1, k1])
            Pm = Pm / Pm.sum().clamp_min(self.eps)

            # decoder分布：对 T 步在 R 上的分布聚合后投影到 union
            Pb = self._proj_route_probs_to_union(routes[b], out_ids[:, b], union)
            # KL(Pb || Pm)
            kl = (Pb.clamp_min(self.eps) * (torch.log(Pb.clamp_min(self.eps)) - torch.log(Pm.clamp_min(self.eps)))).sum()
            kl_sum = kl_sum + kl
            # 熵正则（selector 平均熵）
            ent = 0.5 * (self._entropy(sel_probs[b, 0]) + self._entropy(sel_probs[b, 1]))
            ent_sum = ent_sum + ent

        loss_kl = self.lambda_kl * (kl_sum / max(1, bs))
        loss_entropy = self.lambda_entropy * (ent_sum / max(1, bs))

        # 速率平滑
        loss_smooth = self.lambda_smooth * self._rate_smooth(out_rates, lengths['trg_lengths'])

        total = self.lambda_selector * loss_sel + loss_id + loss_rate + loss_kl + loss_entropy + loss_smooth
        return total, {
            'selector': loss_sel.item(),
            'id': loss_id.item(),
            'rate': loss_rate.item(),
            'kl': loss_kl.item() if isinstance(loss_kl, torch.Tensor) else float(loss_kl),
            'entropy': loss_entropy.item() if isinstance(loss_entropy, torch.Tensor) else float(loss_entropy),
            'smooth': loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else float(loss_smooth)
        }


__all__ = [
    'E2E3TrajData', 'End2End3Model', 'E2E3Loss', 'DA3Planner'
]



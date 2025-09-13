import os
import pickle
import random
import time

from tqdm import tqdm
import datetime as dt
import numpy as np
import networkx as nx
from queue import Queue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from models.layers import Encoder as Transformer, Attention, GPSFormer, GRFormer, sequence_mask, sequence_mask3d
from preprocess import SparseDAM, SegInfo
from utils.model_utils import gps2grid, get_normalized_t
from utils.spatial_func import SPoint, project_pt_to_road, rate2gps
from utils.trajectory_func import STPoint
from utils.candidate_point import CandidatePoint


# ==================== MMA Model Components ====================

class GPS2SegData(Dataset):
    """GPS轨迹到路段映射的数据集类"""

    def __init__(self, rn, trajs_dir, mbr, parameters, mode):
        self.parameters = parameters
        self.rn = rn
        self.mbr = mbr
        self.grid_size = parameters.grid_size
        self.time_span = parameters.time_span

        self.src_grid_seqs, self.src_gps_seqs, self.src_temporal_feas = [], [], []
        self.trg_rids = []
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
        self.keep_ratio = parameters.init_ratio
        if mode in ['train']:
            self.ds_type = 'random'
        else:
            self.ds_type = 'fixed'

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        traj = self.trajs[index]
        if self.ds_type == 'random':
            length = len(traj.pt_list)
            keep_index = [0] + sorted(random.sample(range(1, length - 1), int((length - 2) * self.keep_ratio))) + [length - 1]
        elif self.ds_type == 'fixed':
            keep_index = traj.low_idx
        else:
            raise NotImplementedError
        src_list = np.array(traj.pt_list, dtype=object)
        src_list = src_list[keep_index].tolist()

        src_gps_seq, src_grid_seq, trg_rid = self.get_seqs(src_list)

        trg_candis = self.rn.get_trg_segs(src_gps_seq, self.parameters.candi_size, self.parameters.search_dist, self.parameters.beta)
        candi_label, candi_id, candi_feat, candi_mask = self.get_candis_feats(trg_candis, trg_rid)

        src_grid_seq = torch.tensor(src_grid_seq)

        return src_grid_seq, trg_rid, candi_label, candi_id, candi_feat, candi_mask

    def get_candis_feats(self, ls_candi, trg_id):
        candi_id = []
        candi_feat = []
        candi_onehot = []
        candi_mask = []
        for candis, trg in zip(ls_candi, trg_id):
            candi_mask.append([1] * len(candis) + [0] * (self.parameters.candi_size - len(candis)))
            tmp_id = []
            tmp_feat = []
            tmp_onehot = [0] * self.parameters.candi_size
            for candi in candis:
                tmp_id.append(candi.eid)
                tmp_feat.append([candi.err_weight, candi.cosv, candi.cosv_pre, candi.cosf, candi.cosl, candi.cos1, candi.cos2, candi.cos3, candi.cosp])
            tmp_id.extend([0] * (self.parameters.candi_size - len(candis)))
            tmp_feat.extend([[0] * len(tmp_feat[0])] * (self.parameters.candi_size - len(candis)))
            if trg in tmp_id:
                idx = tmp_id.index(trg)
                tmp_onehot[idx] = 1
            candi_id.append(tmp_id)
            candi_feat.append(tmp_feat)
            candi_onehot.append(tmp_onehot)
        candi_onehot = torch.tensor(candi_onehot)
        candi_id = torch.tensor(candi_id) + 1
        candi_feat = torch.tensor(candi_feat)
        candi_mask = torch.tensor(candi_mask, dtype=torch.float32)
        return candi_onehot, candi_id, candi_feat, candi_mask

    def get_seqs(self, ds_pt_list):
        ls_gps_seq = []
        ls_grid_seq = []
        mm_eids = []
        time_interval = self.time_span
        first_pt = ds_pt_list[0]
        for ds_pt in ds_pt_list:
            ls_gps_seq.append([ds_pt.lat, ds_pt.lng])
            if self.parameters.gps_flag:
                locgrid_xid = (ds_pt.lat - self.rn.minLat) / (self.rn.maxLat - self.rn.minLat)
                locgrid_yid = (ds_pt.lng - self.rn.minLon) / (self.rn.maxLon - self.rn.minLon)
            else:
                locgrid_xid, locgrid_yid = gps2grid(ds_pt, self.mbr, self.grid_size)
            t = get_normalized_t(first_pt, ds_pt, time_interval)
            ls_grid_seq.append([locgrid_xid, locgrid_yid, t])
            mm_eids.append(ds_pt.data['candi_pt'].eid)

        return ls_gps_seq, ls_grid_seq, mm_eids


class Encoder(nn.Module):
    """MMA编码器"""

    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim

        input_dim = 3
        self.fc_in = nn.Linear(input_dim, parameters.hid_dim)
        self.transformer = Transformer(parameters.hid_dim, parameters.transformer_layers, heads=4)

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


class GPS2Seg(nn.Module):
    """MMA主模型"""

    def __init__(self, parameters):
        super().__init__()
        self.direction_flag = parameters.direction_flag
        self.attn_flag = parameters.attn_flag
        self.only_direction = parameters.only_direction

        self.emb_id = nn.Embedding(parameters.id_size, parameters.id_emb_dim)
        self.encoder = Encoder(parameters)

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

        self.init_weights()

    def init_weights(self):
        """初始化权重"""
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


# ==================== TRMMA Model Components ====================

def get_num_pts(time_span, time_interval):
    """计算插值点数量"""
    num_pts = 0
    if time_span % time_interval > time_interval / 2:
        num_pts = time_span // time_interval
    elif time_span > time_interval:
        num_pts = time_span // time_interval - 1
    return num_pts


def get_segs(o, d, rn):
    """获取路段序列"""
    def get_nbrs(rid, max_deps):
        rset = set()
        cset = set()
        q = Queue()
        q.put((rid, 0))
        rset.add(rid)
        while not q.empty():
            rid, dep = q.get()
            if dep == max_deps:
                continue
            if rid in cset:
                continue
            cset.add(rid)
            for nrid in rn.edgeDict[rid]:
                if nrid in rn.valid_edge:
                    rset.add(nrid)
                    q.put((nrid, dep + 1))
        return rset

    oset = set()
    dset = set()
    depth = 0
    while depth < 1000 and len(oset & dset) == 0:
        depth += 1
        oset = get_nbrs(o, depth)
        dset = get_nbrs(d, depth)

    res = list(oset | dset)
    res.remove(o)
    res.remove(d)
    res = [o] + res +[d]
    return res


def remove_circle(path_fixed):
    """移除路径中的环路"""
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


def calc_cos_value(vec1, vec2):
    """计算余弦值"""
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    a = vec1 * vec1
    b = vec2 * vec2
    c = vec1 * vec2
    denom = np.sqrt(a[0] + a[1]) * np.sqrt(b[0] + b[1])
    cos_value = (c[0] + c[1]) / denom if denom != 0 else 1.0
    return cos_value


class DAPlanner(object):
    """DA路径规划器"""
    
    def __init__(self, dam_root, id_size, utc):
        self.csm = SparseDAM(dam_root, id_size)
        self.seg_info = SegInfo(os.path.join(dam_root, "seg_info.csv"))
        self.G = pickle.load(open(os.path.join(dam_root, "road_graph_wtime"), "rb"))
        print("Segment Nodes: {}, Edges: {}".format(len(self.G.nodes), len(self.G.edges)))
        self.vehicle_num = np.load(os.path.join(dam_root, "vehicle_num_{}-48.npy".format(3600)))
        self.tz = dt.timezone(dt.timedelta(hours=utc))

        self.max_seq_len = 79
        self.freq_limit = 1
        self.dcsm_theta = 1

        self.no_path_cnt = 0

    def planning_multi_batch(self, ods, ts):
        preds = []
        for i, od in enumerate(ods):
            route = self.planning_multi(od, ts[i])
            preds.append(route)

        return preds

    def planning_multi(self, od, t, mode='da', segs_flag=False):
        pred = [od[0]]
        timestamp = t + self.seg_info.get_seg_travel_time(od[0])
        segs = []
        for i in range(len(od)-1):
            o = od[i]
            d = od[i+1]

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
                    except nx.exception.NetworkXNoPath as e:
                        self.no_path_cnt += 1
                        route = [o, d]
                route = remove_circle(route)
            elif mode == 'time':
                try:
                    _, route = nx.bidirectional_dijkstra(self.G, o, d, weight="time")
                except nx.exception.NetworkXNoPath as e:
                    self.no_path_cnt += 1
                    route = [o, d]
            elif mode == 'length':
                try:
                    _, route = nx.bidirectional_dijkstra(self.G, o, d, weight="length")
                except nx.exception.NetworkXNoPath as e:
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

    def get_interpolated_pts(self, src, trg, sub_seq, time_span, rn):
        num_pts = get_num_pts(trg.time - src.time, time_span)
        candi_d = trg.data['candi_pt']
        candi_o = src.data['candi_pt']
        pred_id = [candi_o.eid]
        pred_rate = [candi_o.rate]
        time_in = [src.time]
        forward_unit = self.seg_info.get_rn_distance(sub_seq, candi_o.rate, candi_d.rate) / (num_pts + 1)

        while num_pts > 0:
            forward_meter = forward_unit
            pointer = 0
            flag_find = False
            assert sub_seq[0] == pred_id[-1] and sub_seq[-1] == candi_d.eid
            while pointer < len(sub_seq) and flag_find == False:
                if pointer == 0:
                    try_meter = self.seg_info.get_seg_length(sub_seq[pointer]) * (1 - pred_rate[-1])
                elif pointer == (len(sub_seq) - 1):
                    try_meter = self.seg_info.get_seg_length(sub_seq[pointer]) * candi_d.rate
                else:
                    try_meter = self.seg_info.get_seg_length(sub_seq[pointer])
                if forward_meter > try_meter:
                    forward_meter -= try_meter
                    pointer += 1
                else:
                    flag_find = True

            if flag_find:
                id_tmp = sub_seq[pointer]
                rate_tmp = forward_meter / self.seg_info.get_seg_length(id_tmp)
                if pointer == 0:
                    rate_tmp += pred_rate[-1]
                sub_seq = sub_seq[pointer:]
            else:
                id_tmp = sub_seq[-1]
                start = 0
                if pred_id[-1] == id_tmp:
                    start = pred_rate[-1]
                unit_rate = (candi_d.rate - start) / (num_pts + 1)
                rate_tmp = unit_rate
                if pred_id[-1] == id_tmp:
                    rate_tmp += pred_rate[-1]
                sub_seq = sub_seq[-1:]

            pred_id.append(id_tmp)
            pred_rate.append(rate_tmp)
            time_in.append(time_in[-1] + time_span)
            num_pts -= 1
        res = [src]
        for eid, ratio, ts in zip(pred_id[1:], pred_rate[1:], time_in[1:]):
            projected = rate2gps(rn, eid, ratio)
            dist = 0.
            rate = ratio
            candi_pt = CandidatePoint(projected.lat, projected.lng, eid, dist, rate * self.seg_info.get_seg_length(eid), rate)
            pt = STPoint(projected.lat, projected.lng, ts, {'candi_pt': candi_pt})
            pt.time_arr = dt.datetime.fromtimestamp(ts, self.tz)
            res.append(pt)
        res.append(trg)
        return res


class TrajRecData(Dataset):
    """轨迹恢复训练数据集"""

    def __init__(self, rn, trajs_dir, mbr, parameters, mode):
        self.parameters = parameters
        self.rn = rn
        self.mbr = mbr
        self.grid_size = parameters.grid_size
        self.time_span = parameters.time_span
        self.mode = mode
        self.keep_ratio = parameters.keep_ratio

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

                ls_grid_seq, ls_gps_seq, hours, tmp_seg_seq = self.get_src_seq(tmp_src_list)
                features = get_pro_features(tmp_src_list, hours)

                mm_eids, mm_rates = self.get_trg_seq(trg_list[p1_idx: p2_idx + 1])
                path = traj.cpath[p1.cpath_idx: p2.cpath_idx + 1]

                da_route = [self.rn.valid_edge_one[item] for item in path]
                src_seg_seq = [self.rn.valid_edge_one[item] for item in tmp_seg_seq]
                src_seg_feat = self.get_src_seg_feat(ls_gps_seq, tmp_seg_seq)
                label = get_label([self.rn.valid_edge_one[item] for item in path], mm_eids[1:-1])

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

                data.append([da_route, src_grid_seq, src_pro_fea, src_seg_seq, src_seg_feat, trg_rid, trg_rate, label, d_rid, d_rate])

        return data

    def get_src_seg_feat(self, gps_seq, seg_seq):
        feats = []
        for ds_pt, seg in zip(gps_seq, seg_seq):
            gps = SPoint(ds_pt[0], ds_pt[1])
            candi = self.rn.pt2seg(gps, seg)
            feats.append([candi.rate])
        return feats

    def get_src_seq(self, ds_pt_list):
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

    def get_trg_seq(self, tmp_pt_list):
        mm_eids = []
        mm_rates = []
        for pt in tmp_pt_list:
            candi_pt = pt.data['candi_pt']
            mm_eids.append(self.rn.valid_edge_one[candi_pt.eid])
            mm_rates.append([candi_pt.rate])
        return mm_eids, mm_rates


def get_label(cpath, trg_rid):
    """获取标签"""
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


def get_pro_features(ds_pt_list, hours):
    """获取轨迹特征"""
    hour = np.bincount(hours).argmax()
    week = ds_pt_list[0].time_arr.weekday()
    if week in [5, 6]:
        hour += 24
    return hour


class TrajRecTestData(Dataset):
    """轨迹恢复测试数据集"""

    def __init__(self, rn, trajs_dir, mbr, parameters, mode):
        self.parameters = parameters
        self.rn = rn
        self.mbr = mbr
        self.grid_size = parameters.grid_size
        self.time_span = parameters.time_span

        self.dam = DAPlanner(parameters.dam_root, parameters.id_size - 1, parameters.utc)
        if parameters.eid_cate == 'gps2seg':
            inferred_segs = pickle.load(open(parameters.inferred_seg_path, "rb"))
            predict_id, _ = zip(*inferred_segs)
        elif parameters.eid_cate in ['mm', 'nn']:
            predict_id = pickle.load(open(os.path.join(trajs_dir, "test_{}_eids.pkl".format(parameters.eid_cate)), "rb"))
        else:
            predict_id = []

        self.src_grid_seqs, self.src_gps_seqs, self.src_pro_feas = [], [], []
        self.trg_gps_seqs, self.trg_rids, self.trg_rates = [], [], []
        self.src_seg_seq = []
        self.src_seg_feats = []
        self.src_time_seq = []
        self.trg_time_seq = []
        self.routes = []
        self.labels = []
        trajs = pickle.load(open(os.path.join(trajs_dir, 'test_output.pkl'), "rb"))

        self.groups = []
        self.src_mms = []
        route_time = 0
        for serial, traj in tqdm(enumerate(trajs), desc='traj num'):
            trg_list = traj.pt_list.copy()
            src_list = np.array(traj.pt_list, dtype=object)
            src_list = src_list[traj.low_idx].tolist()

            _, src_gps_seq, _, seg_seq, time_seq = self.get_src_seq(src_list)
            if parameters.eid_cate in ['mm', 'nn', 'gps2seg']:
                seg_seq = predict_id[serial]
            src_mm = []
            for seg, (lat, lng) in zip(seg_seq, src_gps_seq):
                projected, rate, dist = project_pt_to_road(self.rn, SPoint(lat, lng), seg)
                src_mm.append([[projected.lat, projected.lng], seg, rate])
            self.src_mms.append(src_mm)

            for p1, p1_idx, p2, p2_idx, s1, s2, ts, mmf1, mmf2 in zip(src_list[:-1], traj.low_idx[:-1], src_list[1:], traj.low_idx[1:], seg_seq[:-1], seg_seq[1:], time_seq[:-1], src_mm[:-1], src_mm[1:]):
                if (p1_idx + 1) < p2_idx:
                    tmp_seg_seq = [s1, s2]
                    tmp_src_list = [p1, p2]

                    ls_grid_seq, ls_gps_seq, hours, _, _ = self.get_src_seq(tmp_src_list)
                    features = get_pro_features(tmp_src_list, hours)

                    mm_gps_seq, mm_eids, mm_rates, trg_time = self.get_trg_seq(trg_list[p1_idx: p2_idx + 1])
                    path = traj.cpath[p1.cpath_idx: p2.cpath_idx + 1]

                    if parameters.eid_cate in ['mm', 'nn', 'gps2seg']:
                        t0 = time.time()
                        path = self.dam.planning_multi([s1, s2], ts, mode=parameters.planner)
                        route_time += time.time() - t0
                        mm_gps_seq[0] = mmf1[0]
                        mm_eids[0] = self.rn.valid_edge_one[mmf1[1]]
                        mm_rates[0] = [mmf1[2]]
                        mm_gps_seq[-1] = mmf2[0]
                        mm_eids[-1] = self.rn.valid_edge_one[mmf2[1]]
                        mm_rates[-1] = [mmf2[2]]

                    self.routes.append([self.rn.valid_edge_one[item] for item in path])
                    self.src_seg_seq.append([self.rn.valid_edge_one[item] for item in tmp_seg_seq])
                    self.src_seg_feats.append(self.get_src_seg_feat(ls_gps_seq, tmp_seg_seq))
                    self.labels.append(get_label([self.rn.valid_edge_one[item] for item in path], mm_eids[1:-1]))

                    self.trg_gps_seqs.append(mm_gps_seq)
                    self.trg_rids.append(mm_eids)
                    self.trg_rates.append(mm_rates)
                    self.src_grid_seqs.append(ls_grid_seq)
                    self.src_gps_seqs.append(ls_gps_seq)
                    self.src_pro_feas.append(features)
                    self.src_time_seq.append(time_seq)
                    self.trg_time_seq.append(trg_time)
                    self.groups.append(serial)
        print(route_time, route_time / len(trajs) * 1000, len(trajs) / route_time)

    def __len__(self):
        return len(self.src_grid_seqs)

    def __getitem__(self, index):
        src_grid_seq = self.src_grid_seqs[index]
        trg_gps_seq = self.trg_gps_seqs[index]
        trg_rid = self.trg_rids[index]
        trg_rate = self.trg_rates[index]
        da_route = self.routes[index]

        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.float32)

        src_seg_seq = self.src_seg_seq[index]
        src_seg_feat = self.src_seg_feats[index]
        src_seg_seq = torch.tensor(src_seg_seq)
        src_seg_feat = torch.tensor(src_seg_feat)

        src_grid_seq = torch.tensor(src_grid_seq)
        trg_gps_seq = torch.tensor(trg_gps_seq)
        trg_rid = torch.tensor(trg_rid)
        trg_rate = torch.tensor(trg_rate)
        src_pro_fea = torch.tensor(self.src_pro_feas[index])
        da_route = torch.tensor(da_route)

        d_rid = trg_rid[-1]
        d_rate = trg_rate[-1]
        trg_gps_seq = trg_gps_seq
        trg_rid = trg_rid
        trg_rate = trg_rate

        return da_route, src_grid_seq, src_pro_fea, src_seg_seq, src_seg_feat, trg_gps_seq, trg_rid, trg_rate, label, d_rid, d_rate

    def get_src_seg_feat(self, gps_seq, seg_seq):
        feats = []
        for ds_pt, seg in zip(gps_seq, seg_seq):
            gps = SPoint(ds_pt[0], ds_pt[1])
            candi = self.rn.pt2seg(gps, seg)
            feats.append([candi.rate])
        return feats

    def get_src_seq(self, ds_pt_list):
        timestamps = []
        hours = []
        ls_grid_seq = []
        ls_gps_seq = []
        first_pt = ds_pt_list[0]
        time_interval = self.time_span
        seg_seq = []
        for ds_pt in ds_pt_list:
            timestamps.append(ds_pt.time)
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

        return ls_grid_seq, ls_gps_seq, hours, seg_seq, timestamps

    def get_trg_seq(self, tmp_pt_list):
        mm_gps_seq = []
        mm_eids = []
        mm_rates = []
        time_arrs = []
        for pt in tmp_pt_list:
            time_arr = pt.time_arr
            time_arrs.append([time_arr.month + 1, time_arr.day + 1, 1 if time_arr.weekday() in [0, 1, 2, 3, 4] else 2, time_arr.hour + 1, time_arr.minute + 1, time_arr.second + 1])
            candi_pt = pt.data['candi_pt']

            mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
            mm_eids.append(self.rn.valid_edge_one[candi_pt.eid])
            mm_rates.append([candi_pt.rate])
        return mm_gps_seq, mm_eids, mm_rates, time_arrs


class GPSEncoder(nn.Module):
    """GPS编码器"""

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
    """图路径编码器"""

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
    """多任务解码器"""

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

            if teacher_forcing_ratio == -1 and self.prog_flag:
                for i in range(batch_size):
                    if t < trg_len[i]:
                        prev_idx = (input_id[i] == routes[i]).nonzero(as_tuple=True)[0][0]
                        tmp_flag = True
                        while tmp_flag:
                            cur_idx = prediction_id[i].argmax()
                            if cur_idx < prev_idx:
                                prediction_id[i, cur_idx] = 1e-6
                            else:
                                tmp_flag = False

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


class TrajRecovery(nn.Module):
    """轨迹恢复主模型"""

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
        input_dim_gps = 3
        if self.learn_pos:
            input_dim_gps += parameters.hid_dim
        if self.srcseg_flag:
            input_dim_gps += parameters.hid_dim + 1
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

        self.init_weights()

        self.timer1, self.timer2, self.timer3, self.timer4, self.timer5, self.timer6 = [], [], [], [], [], []

    def init_weights(self):
        """初始化权重"""
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, src, src_len, trg_id, trg_rate, trg_len, pro_features, rid_features_dict, da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio, outputs_multi, candi_vec, results):

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

        t4 = time.time()
        self.timer4.append(time.time() - t3)

        outputs_id, outputs_rate = self.decoder(max_trg_len, batch_size, trg_id, trg_rate, trg_len, hiddens, rid_features_dict, da_routes, route_outputs, route_attn_mask, d_rids, d_rates, teacher_forcing_ratio)

        final_outputs_id = outputs_id[1:-1]
        final_outputs_rate = outputs_rate[1:-1]

        t5 = time.time()
        self.timer5.append(time.time() - t4)
        self.timer6.append(t5 - t0)

        return final_outputs_id, final_outputs_rate

def get_results(predict_id, target_id, lengths):
    predict_id = predict_id.detach().cpu().tolist()  # 将预测结果从GPU移动到CPU并转换为Python列表格式

    results = []  # 初始化结果列表，用于存储处理后的预测和目标数据对
    for pred, trg, length in zip(predict_id, target_id, lengths):  # 遍历每个样本的预测值、目标值和有效长度
        results.append([pred[:length], trg])  # 截取预测序列的有效部分（根据length），与完整目标序列组成数据对并添加到结果列表
    return results  # 返回包含所有样本预测-目标数据对的结果列表


# ==================== 端到端迭代优化模型 ====================

class IterativeModel(nn.Module):
    """
    端到端迭代轨迹恢复模型
    实现: GPS -> MMA -> TRMMA -> 重新MMA -> 迭代优化
    """
    
    def __init__(self, mma_args, trmma_args, max_iterations=2):
        super().__init__()
        
        # 初始化两个子模型
        self.mma_model = GPS2Seg(mma_args)
        self.trmma_model = TrajRecovery(trmma_args)
        
        # 迭代参数
        self.max_iterations = max_iterations
        
        # 保存参数配置
        self.mma_args = mma_args
        self.trmma_args = trmma_args
        
        # 一致性损失权重（可调节）
        self.consistency_weight = 1.0
        
        # 路网对象和路段信息（用于GPS重构）
        self.rn = None
        self.seg_info = None
        
    def set_road_network(self, rn, seg_info=None):
        """设置路网对象和路段信息"""
        self.rn = rn
        self.seg_info = seg_info
        
    def forward(self, 
                # MMA输入
                src_seqs_mma, src_lengths_mma, candi_ids, candi_feats, candi_masks,
                # TRMMA输入  
                src_seqs_trmma, src_lengths_trmma, trg_rids, trg_rates, trg_lengths,
                src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos, 
                src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio,
                # 训练标志
                training=True):
        """
        前向传播：实现端到端的迭代优化
        """
        
        # ===== 第一次MMA：初始地图匹配 =====
        outputs_multi, candi_vec, output_ids = self.mma_model(src_seqs_mma, src_lengths_mma, candi_ids, candi_feats, candi_masks)
        
        candi_size = candi_ids.shape[-1]
        output_tmp = (F.one_hot(output_ids.argmax(-1), candi_size) * candi_ids).sum(dim=-1) - 1

        results = get_results(output_tmp, trg_rids, src_lengths_mma)

        # ===== TRMMA：轨迹补齐 =====
        trmma_output_ids, trmma_output_rates = self.trmma_model(
            src_seqs_trmma, src_lengths_trmma, trg_rids, trg_rates, trg_lengths,
            src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos, 
            src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio,
            outputs_multi, candi_vec, results
        )
        
        if not training or self.max_iterations <= 1:
            # 如果不训练或只要一次迭代，直接返回
            return {
                'initial_mma_output': initial_mma_output,
                'trmma_output_ids': trmma_output_ids,
                'trmma_output_rates': trmma_output_rates,
                'refined_mma_output': None,
                'consistency_loss': torch.tensor(0.0, device=initial_mma_output.device),
                'reconstructed_gps': None
            }
        
        # ===== 迭代优化部分 =====
        # 从TRMMA输出构造新的轨迹序列用于重新匹配
        
        # 检查必要的依赖是否已设置
        if self.rn is None:  # 检查路网对象是否已设置
            raise ValueError("Road network (rn) must be set before calling forward. Use set_road_network() method.")  # 抛出错误提示需要设置路网
        if self.seg_info is None:  # 检查路段信息是否已设置
            raise ValueError("Segment info must be set before calling forward. Use set_road_network() method.")  # 抛出错误提示需要设置路段信息
        
        # 获取TRMMA预测的路段ID序列
        # 基于get_results函数中的路段提取逻辑
        import torch.nn.functional as F  # 导入函数式接口用于one-hot编码
        predicted_seg_ids = []  # 初始化预测路段ID列表
        
        # 将TRMMA输出转换为路段ID
        output_tmp = (F.one_hot(trmma_output_ids.argmax(-1), da_routes.shape[0]) *   # 对预测ID进行one-hot编码
                     da_routes.permute(1, 0).unsqueeze(1).repeat(1, trmma_output_ids.shape[0], 1).permute(1, 0, 2)).sum(dim=-1)  # 与路线矩阵相乘得到实际路段ID
        
        # 转换为列表格式，每个批次一个序列
        output_tmp = output_tmp.permute(1, 0).detach()  # 转换维度顺序并分离梯度 [batch_size, seq_len]
        
        for b, pred_seg in enumerate(output_tmp):  # 遍历每个批次的预测路段
            if b < len(da_lengths):  # 检查批次索引是否在有效范围内
                length = da_lengths[b] if hasattr(da_lengths, '__len__') else len(pred_seg)  # 获取当前批次的序列长度
                predicted_seg_ids.append(pred_seg[:length])  # 截取有效长度的路段ID
        
        # 将预测的路段序列转换为GPS坐标
        from utils.evaluation_utils import toseq  # 导入路段到GPS序列转换函数
        reconstructed_gps_list = []  # 初始化重构GPS坐标列表
        for b, seg_ids in enumerate(predicted_seg_ids):  # 遍历每个批次的预测路段ID
            if b < len(da_lengths):  # 检查批次索引是否在有效范围内
                route_len = da_lengths[b]  # 获取当前批次的路线长度
                route = da_routes[:route_len, b]  # 获取当前批次的路线
                rates = trmma_output_rates[:len(seg_ids), b] if b < trmma_output_rates.shape[1] else torch.full((len(seg_ids),), 0.5, device=seg_ids.device)  # 获取通行率，如果不存在则使用默认值0.5
                # 使用toseq函数转换路段和通行率为GPS坐标
                gps_seq = toseq(self.rn, seg_ids.cpu().numpy(), rates.cpu().numpy(), route.cpu().numpy(), self.seg_info)  # 调用转换函数生成GPS序列，只在必要时转换为CPU
                reconstructed_gps_list.append(gps_seq)  # 将生成的GPS序列添加到列表中
        
        # 转换为tensor格式
        batch_size, max_seq_len, _ = src_seqs_mma.shape  # 获取批次大小和最大序列长度
        device = trmma_output_rates.device  # 获取设备信息
        reconstructed_gps = torch.zeros(batch_size, max_seq_len, 3, device=device)  # 初始化重构GPS张量，3维表示纬度、经度、时间
        for b, gps_seq in enumerate(reconstructed_gps_list):  # 遍历每个批次的GPS序列
            seq_len = min(len(gps_seq), max_seq_len)  # 获取实际序列长度，不超过最大长度
            for t in range(seq_len):  # 遍历序列中的每个时间步
                if t < len(gps_seq):  # 检查时间步是否在有效范围内
                    reconstructed_gps[b, t, 0] = gps_seq[t][0]  # 设置纬度
                    reconstructed_gps[b, t, 1] = gps_seq[t][1]  # 设置经度
                    reconstructed_gps[b, t, 2] = t * 0.1  # 设置时间特征，使用线性增长
        
        # 为重构的GPS序列生成新的候选路段
        new_candi_ids_list = []  # 初始化新候选路段ID列表
        new_candi_feats_list = []  # 初始化新候选路段特征列表
        new_candi_masks_list = []  # 初始化新候选路段掩码列表
        
        for b, gps_seq in enumerate(reconstructed_gps_list):  # 遍历每个批次的重构GPS序列
            # 转换候选路段为所需格式
            batch_candi_ids = []  # 初始化当前批次的候选路段ID
            batch_candi_feats = []  # 初始化当前批次的候选路段特征
            batch_candi_masks = []  # 初始化当前批次的候选路段掩码
            
            if len(gps_seq) > 0:  # 检查GPS序列是否非空
                # 使用路网的get_trg_segs函数生成候选路段
                trg_candis = self.rn.get_trg_segs(gps_seq, candi_ids.shape[-1],   # 调用路网函数获取目标候选路段
                                                 self.mma_args.search_dist, self.mma_args.beta)  # 使用搜索距离和beta参数
                
                for candis in trg_candis:  # 遍历每个时间步的候选路段
                    candi_ids_seq = []  # 初始化当前时间步的候选路段ID
                    candi_feats_seq = []  # 初始化当前时间步的候选路段特征
                    candi_mask_seq = []  # 初始化当前时间步的候选路段掩码
                    
                    if candis:  # 检查候选路段是否存在
                        for candi in candis[:candi_ids.shape[-1]]:  # 遍历候选路段，不超过最大候选数量
                            candi_ids_seq.append(candi.eid + 1)  # 添加候选路段ID，+1避免0索引
                            # 使用候选路段的特征 - 按照原作者的9维特征格式
                            candi_feats_seq.append([  # 构建9维特征向量
                                getattr(candi, 'err_weight', candi.error),  # 误差权重，如果不存在则使用error
                                getattr(candi, 'cosv', 0.0),      # cosv特征，默认0.0
                                getattr(candi, 'cosv_pre', 0.0),  # cosv_pre特征，默认0.0
                                getattr(candi, 'cosf', 0.0),      # cosf特征，默认0.0
                                getattr(candi, 'cosl', 0.0),      # cosl特征，默认0.0
                                getattr(candi, 'cos1', 0.0),      # cos1特征，默认0.0
                                getattr(candi, 'cos2', 0.0),      # cos2特征，默认0.0
                                getattr(candi, 'cos3', 0.0),      # cos3特征，默认0.0
                                getattr(candi, 'cosp', 0.0)       # cosp特征，默认0.0
                            ])
                            candi_mask_seq.append(1.0)  # 设置有效候选路段的掩码为1.0
                    
                    # 填充到固定长度
                    while len(candi_ids_seq) < candi_ids.shape[-1]:  # 当候选路段数量不足时进行填充
                        candi_ids_seq.append(0)  # 填充无效ID为0
                        candi_feats_seq.append([0.0] * 9)  # 填充无效特征为全0向量
                        candi_mask_seq.append(0.0)  # 填充无效掩码为0.0
                    
                    batch_candi_ids.append(candi_ids_seq[:candi_ids.shape[-1]])  # 截取到指定长度并添加到批次列表
                    batch_candi_feats.append(candi_feats_seq[:candi_ids.shape[-1]])  # 截取到指定长度并添加到批次列表
                    batch_candi_masks.append(candi_mask_seq[:candi_ids.shape[-1]])  # 截取到指定长度并添加到批次列表
            else:  # 如果GPS序列为空
                # 如果没有GPS序列，创建空的候选路段
                seq_len = src_lengths_mma[b] if b < len(src_lengths_mma) else max_seq_len  # 获取序列长度
                for t in range(seq_len):  # 遍历每个时间步
                    batch_candi_ids.append([0] * candi_ids.shape[-1])  # 添加全0的候选路段ID
                    batch_candi_feats.append([[0.0] * 9 for _ in range(candi_ids.shape[-1])])  # 添加全0的候选路段特征
                    batch_candi_masks.append([0.0] * candi_ids.shape[-1])  # 添加全0的候选路段掩码
            
            new_candi_ids_list.append(batch_candi_ids)  # 将当前批次的候选路段ID添加到总列表
            new_candi_feats_list.append(batch_candi_feats)  # 将当前批次的候选路段特征添加到总列表
            new_candi_masks_list.append(batch_candi_masks)  # 将当前批次的候选路段掩码添加到总列表
        
        # 转换为tensor格式
        device = candi_ids.device  # 获取设备信息
        batch_size, max_seq_len, candi_size = candi_ids.shape  # 获取批次大小、最大序列长度和候选路段数量
        new_candi_ids = torch.zeros(batch_size, max_seq_len, candi_size, dtype=torch.long, device=device)  # 初始化新候选路段ID张量
        new_candi_feats = torch.zeros(batch_size, max_seq_len, candi_size, 9, device=device)  # 初始化新候选路段特征张量，9维特征
        new_candi_masks = torch.zeros(batch_size, max_seq_len, candi_size, device=device)  # 初始化新候选路段掩码张量
        
        for b, (ids, feats, masks) in enumerate(zip(new_candi_ids_list, new_candi_feats_list, new_candi_masks_list)):  # 遍历每个批次的候选路段数据
            seq_len = min(len(ids), max_seq_len)  # 获取实际序列长度，不超过最大长度
            for t in range(seq_len):  # 遍历序列中的每个时间步
                if t < len(ids):  # 检查时间步是否在有效范围内
                    new_candi_ids[b, t] = torch.tensor(ids[t], device=device)  # 设置候选路段ID张量
                    new_candi_feats[b, t] = torch.tensor(feats[t], device=device)  # 设置候选路段特征张量
                    new_candi_masks[b, t] = torch.tensor(masks[t], device=device)  # 设置候选路段掩码张量
        # ===== 第二次MMA：重新地图匹配 =====
        refined_mma_output = self.mma_model(
            reconstructed_gps, src_lengths_mma, new_candi_ids, new_candi_feats, new_candi_masks
        )
        
        # ===== 计算一致性损失 =====
        consistency_loss = self._compute_consistency_loss(initial_mma_output, refined_mma_output, candi_masks)
        
        return {
            'initial_mma_output': initial_mma_output,
            'trmma_output_ids': trmma_output_ids,
            'trmma_output_rates': trmma_output_rates,
            'refined_mma_output': refined_mma_output,
            'consistency_loss': consistency_loss,
            'reconstructed_gps': reconstructed_gps
        }
    
    def _compute_consistency_loss(self, initial_output, refined_output, masks):
        """
        计算初始匹配和精炼匹配之间的一致性损失
        """
        if refined_output is None:
            return torch.tensor(0.0, device=initial_output.device)
        
        # 计算两次匹配结果的差异
        # 方法1：KL散度损失（鼓励refined结果更加确定）
        initial_probs = F.softmax(initial_output, dim=-1)
        refined_probs = F.softmax(refined_output, dim=-1)
        
        # 计算KL散度
        kl_loss = F.kl_div(refined_probs.log(), initial_probs, reduction='none')
        kl_loss = kl_loss.masked_fill(masks == 0, 0).sum() / masks.sum().clamp(min=1)
        
        # 方法2：熵损失（鼓励refined结果更加确定）
        refined_entropy = -(refined_probs * (refined_probs + 1e-8).log()).sum(dim=-1)
        entropy_loss = refined_entropy.masked_fill(masks.sum(dim=-1) == 0, 0).mean()
        
        consistency_loss = kl_loss + 0.1 * entropy_loss
        
        return consistency_loss


def compute_iterative_loss(outputs, 
                          # MMA标签
                          mma_labels,
                          # TRMMA标签  
                          trmma_labels, trmma_rates_labels,
                          # 损失权重
                          mma_weight=1.0, trmma_weight=1.0, consistency_weight=0.5):
    """
    计算迭代模型的总损失
    """
    device = outputs['initial_mma_output'].device
    
    # MMA损失（初始匹配）
    mma_criterion = nn.BCELoss(reduction='mean')
    initial_mma_loss = mma_criterion(
        torch.sigmoid(outputs['initial_mma_output']), 
        mma_labels.float()
    )
    
    # TRMMA损失
    trmma_id_criterion = nn.BCELoss(reduction='sum')
    trmma_rate_criterion = nn.L1Loss(reduction='sum')
    
    # 计算有效长度用于归一化
    valid_elements = (trmma_labels.sum(dim=-1) > 0).sum()
    if valid_elements == 0:
        valid_elements = 1
    
    trmma_id_loss = trmma_id_criterion(outputs['trmma_output_ids'], trmma_labels) / valid_elements
    trmma_rate_loss = trmma_rate_criterion(outputs['trmma_output_rates'], trmma_rates_labels) / valid_elements
    trmma_loss = trmma_id_loss + trmma_rate_loss
    
    # 精炼MMA损失（如果有）
    refined_mma_loss = torch.tensor(0.0, device=device)
    if outputs['refined_mma_output'] is not None:
        refined_mma_loss = mma_criterion(
            torch.sigmoid(outputs['refined_mma_output']), 
            mma_labels.float()
        )
    
    # 一致性损失
    consistency_loss = outputs['consistency_loss']
    
    # 总损失
    total_loss = (mma_weight * initial_mma_loss + 
                  trmma_weight * trmma_loss + 
                  consistency_weight * consistency_loss +
                  0.3 * refined_mma_loss)  # 精炼MMA损失权重较小
    
    return {
        'total_loss': total_loss,
        'initial_mma_loss': initial_mma_loss,
        'trmma_loss': trmma_loss,
        'refined_mma_loss': refined_mma_loss,
        'consistency_loss': consistency_loss
    }
import os
import pickle
import random

from tqdm import tqdm
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

from models.layers_demo2 import GPSFormer, GRFormer, sequence_mask, sequence_mask3d, Encoder, AttentionDemo2
from utils.model_utils import gps2grid, get_normalized_t, AttrDict
from utils.spatial_func import SPoint, project_pt_to_road, rate2gps
from utils.trajectory_func import STPoint
from utils.candidate_point import CandidatePoint
from preprocess import SparseDAM, SegInfo
import networkx as nx
import datetime as dt
from queue import Queue

# ==================== 规划器 ====================

def get_num_pts(time_span, time_interval):
    num_pts = 0
    if time_span % time_interval > time_interval / 2:
        num_pts = time_span // time_interval
    elif time_span > time_interval:
        num_pts = time_span // time_interval - 1
    return num_pts

def remove_circle(path_fixed):
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
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    a = vec1 * vec1
    b = vec2 * vec2
    c = vec1 * vec2
    denom = np.sqrt(a[0] + a[1]) * np.sqrt(b[0] + b[1])
    cos_value = (c[0] + c[1]) / denom if denom != 0 else 1.0
    return cos_value


class DAPlanner(object):
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
        timestamp = t
        if timestamp is None:
            timestamp = 0
        else:
            timestamp = timestamp + self.seg_info.get_seg_travel_time(od[0])
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
                    except nx.exception.NetworkXNoPath:
                        self.no_path_cnt += 1
                        route = [o, d]
                route = remove_circle(route)
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



# ==================== 数据集（端到端） ====================

class E2ETrajData(Dataset):
    """
    端到端训练数据集：按低频点对之间的 gap 构造样本，
    同时返回候选集与标签，保证两端输入齐备并可端到端训练。
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

        # 统一为按 gap 生成子样本的逻辑（与 trmma 一致），同时保留 demo2 的候选分支
        for p1, p1_idx, p2, p2_idx in zip(src_list[:-1], keep_index[:-1], src_list[1:], keep_index[1:]):
            if (p1_idx + 1) < p2_idx:
                tmp_src_list = [p1, p2]

                ls_grid_seq, ls_gps_seq, hours, tmp_seg_seq = self.get_src_seq(tmp_src_list)
                features = get_pro_features(tmp_src_list, hours)

                # 候选（针对该 gap 的两个低频点）
                trg_candis = self.rn.get_trg_segs(ls_gps_seq, self.parameters.candi_size, self.parameters.search_dist, self.parameters.beta)
                candi_onehot, candi_ids, candi_feats, candi_masks = self.build_selector_candidates(trg_candis, tmp_seg_seq)

                # 目标高频序列（该 gap 间的完整映射）
                mm_eids, mm_rates = self.get_trg_seq(trg_list[p1_idx: p2_idx + 1])
                path = traj.cpath[p1.cpath_idx: p2.cpath_idx + 1]

                da_route = [self.rn.valid_edge_one[item] for item in path]
                src_seg_seq = [self.rn.valid_edge_one[item] for item in tmp_seg_seq]
                src_seg_feat = self.get_src_seg_feat(ls_gps_seq, tmp_seg_seq)
                label = get_label([self.rn.valid_edge_one[item] for item in path], mm_eids[1:-1])

                # 打包张量
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

                # 候选与标签
                candi_onehot = torch.tensor(candi_onehot)
                candi_ids = torch.tensor(candi_ids) + 1
                candi_feats = torch.tensor(candi_feats)
                candi_masks = torch.tensor(candi_masks, dtype=torch.float32)

                # 返回顺序：先基础字段，再追加候选相关字段，便于 collate 对齐
                data.append([
                    da_route, src_grid_seq, src_pro_fea, src_seg_seq, src_seg_feat,
                    trg_rid, trg_rate, label, d_rid, d_rate,
                    candi_onehot, candi_ids, candi_feats, candi_masks
                ])

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

    def build_selector_candidates(self, ls_candi, trg_id_seq):
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

def get_label(cpath, trg_rid):
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
    hour = np.bincount(hours).argmax()
    week = ds_pt_list[0].time_arr.weekday()
    if week in [5, 6]:
        hour += 24
    return hour


class E2ETrajTestData(Dataset):

    def __init__(self, rn, trajs_dir, mbr, parameters):
        self.parameters = parameters
        self.rn = rn
        self.mbr = mbr
        self.grid_size = parameters.grid_size
        self.time_span = parameters.time_span

        self.dam = DAPlanner(parameters.dam_root, parameters.id_size - 1, parameters.utc)
        if parameters.eid_cate == 'gps2seg' and parameters.inferred_seg_path and os.path.exists(parameters.inferred_seg_path):
            inferred_segs = pickle.load(open(parameters.inferred_seg_path, "rb"))
            predict_id, _ = zip(*inferred_segs)
        elif parameters.eid_cate in ['mm', 'nn']:
            seg_file_path = os.path.join(trajs_dir, "test_{}_eids.pkl".format(parameters.eid_cate))
            if os.path.exists(seg_file_path):
                predict_id = pickle.load(open(seg_file_path, "rb"))
            else:
                predict_id = []
        else:
            predict_id = []

        self.src_grid_seqs, self.src_gps_seqs, self.src_pro_feas = [], [], []
        self.trg_gps_seqs, self.trg_rids, self.trg_rates = [], [], []
        self.src_seg_seq = []
        self.src_seg_feats = []
        self.routes = []
        self.labels = []
        self.d_rids, self.d_rates = [], []
        self.candi_onehots, self.candi_ids, self.candi_feats, self.candi_masks = [], [], [], []
        trajs = pickle.load(open(os.path.join(trajs_dir, 'test_output.pkl'), "rb"))

        self.groups = []
        self.src_mms = []
        for serial, traj in tqdm(enumerate(trajs), desc='traj num'):
            trg_list = traj.pt_list.copy()
            src_list = np.array(traj.pt_list, dtype=object)
            src_list = src_list[traj.low_idx].tolist()

            _, src_gps_seq, _, seg_seq, time_seq = self.get_src_seq(src_list)
            if parameters.eid_cate in ['mm', 'nn', 'gps2seg'] and predict_id:
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

                    trg_candis = self.rn.get_trg_segs(ls_gps_seq, self.parameters.candi_size, self.parameters.search_dist, self.parameters.beta)
                    candi_onehot, candi_ids, candi_feats, candi_masks = self.build_selector_candidates(trg_candis, tmp_seg_seq)

                    mm_gps_seq, mm_eids, mm_rates = self.get_trg_seq(trg_list[p1_idx: p2_idx + 1])
                    path = traj.cpath[p1.cpath_idx: p2.cpath_idx + 1]

                    if parameters.eid_cate in ['mm', 'nn', 'gps2seg']:
                        s1, s2 = tmp_seg_seq[0], tmp_seg_seq[1]
                        ts = getattr(p1, 'time', None)
                        path = self.dam.planning_multi([s1, s2], ts, mode=self.parameters.planner)
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
                    self.d_rids.append(mm_eids[-1])
                    self.d_rates.append(mm_rates[-1])
                    self.candi_onehots.append(candi_onehot)
                    self.candi_ids.append(candi_ids)
                    self.candi_feats.append(candi_feats)
                    self.candi_masks.append(candi_masks)
                    self.groups.append(serial)

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

        candi_onehot = torch.tensor(self.candi_onehots[index])
        candi_ids = torch.tensor(self.candi_ids[index]) + 1
        candi_feats = torch.tensor(self.candi_feats[index])
        candi_masks = torch.tensor(self.candi_masks[index], dtype=torch.float32)

        return [
            da_route, src_grid_seq, src_pro_fea, src_seg_seq, src_seg_feat, trg_gps_seq,
            trg_rid, trg_rate, label, d_rid, d_rate,
            candi_onehot, candi_ids, candi_feats, candi_masks
        ]

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
        for pt in tmp_pt_list:
            candi_pt = pt.data['candi_pt']
            mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
            mm_eids.append(self.rn.valid_edge_one[candi_pt.eid])
            mm_rates.append([candi_pt.rate])
        return mm_gps_seq, mm_eids, mm_rates

    def build_selector_candidates(self, ls_candi, trg_id_seq):
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


# ==================== 选择器分支 ====================

class SelectorEncoder(nn.Module):
    """序列编码器（候选选择分支）。复用通用 Encoder，避免循环引用。"""

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


class SelectorHead(nn.Module):
    """候选选择头：输出每个候选的选择概率及其向量表征。"""

    def __init__(self, parameters):
        super().__init__()
        self.direction_flag = parameters.direction_flag
        self.attn_flag = parameters.attn_flag
        self.only_direction = parameters.only_direction

        self.emb_id = nn.Embedding(parameters.id_size, parameters.id_emb_dim)
        self.encoder = SelectorEncoder(parameters)

        fc_id_out_input_dim = parameters.hid_dim
        if self.direction_flag:
            fc_id_out_input_dim += 9
        if self.only_direction:
            fc_id_out_input_dim = 9
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, parameters.hid_dim)

        mlp_dim = parameters.hid_dim * 2
        if self.attn_flag:
            self.attn = AttentionDemo2(parameters.hid_dim)
            mlp_dim += parameters.hid_dim

        self.prob_out = nn.Sequential(
            nn.Linear(mlp_dim, parameters.hid_dim * 2),
            nn.ReLU(),
            nn.Linear(parameters.hid_dim * 2, 1),
            nn.Sigmoid()
        )

        self.params = parameters
        self.hid_dim = parameters.hid_dim

        # 权重初始化
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
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


# ==================== 重建分支（带软段嵌入注入） ====================

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
        self.attn_route = AttentionDemo2(parameters.hid_dim)
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


class Reconstructor(nn.Module):
    """
    轨迹重建器：在 GPS 编码端注入“软路段嵌入”（来自候选分布加权和），
    使重建损失能通过该通道反向更新选择分支。
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

        # GPS 输入额外拼接一份 hid_dim 的软段嵌入
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

        # 权重初始化
        self.init_weights()

        # forward 内阶段计时
        self.timer1, self.timer2, self.timer3, self.timer4, self.timer5, self.timer6 = [], [], [], [], [], []

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
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

        t5 = time.time()
        self.timer5.append(time.time() - t4)
        self.timer6.append(t5 - t0)
        return final_outputs_id, final_outputs_rate


# ==================== 端到端模型封装 ====================

class End2EndModel(nn.Module):
    """
    端到端结构：
    - 选择器对端点生成候选概率分布，形成“软段嵌入”；
    - 重建器接收软段嵌入进行解码重建；
    - 重建损失可反向更新选择器。
    """

    def __init__(self, args: AttrDict):
        super().__init__()
        self.selector = SelectorHead(args)
        self.reconstructor = Reconstructor(args)

    def forward(self,
                # 统一GPS输入 (像TRMMA一样)
                src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                pro_features, rid_features_dict, da_routes, da_lengths, da_pos,
                src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio,
                # 选择器专用输入
                sel_candi_ids, sel_candi_feats, sel_candi_masks):

        # GPS数据形状处理：src_seqs应该是[src_len, bs, 3]，需要转换为[bs, src_len, 3]给selector
        sel_src_seqs = src_seqs.permute(1, 0, 2)  # [bs, src_len, 3]
        sel_src_lens = src_lengths

        # 1) 选择器前向：两端点的候选概率分布（已 masked softmax）
        sel_out_multi, sel_candi_vec, sel_probs = self.selector(sel_src_seqs, sel_src_lens, sel_candi_ids, sel_candi_feats, sel_candi_masks)
        # 软段嵌入（概率加权和），形状 [bs, src_len, hid]
        soft_seg_emb = (sel_probs.unsqueeze(-1) * sel_candi_vec).sum(dim=-2)
        # 转换为重建器期望的 [src_len, bs, hid]
        soft_seg_emb = soft_seg_emb.permute(1, 0, 2)

        # 2) 重建器前向：注入软嵌入，直接使用原始GPS数据[src_len, bs, 3]
        out_ids, out_rates = self.reconstructor(src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                                pro_features, rid_features_dict, da_routes, da_lengths, da_pos,
                                                src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio,
                                                soft_seg_emb)

        return {
            'selector_probs': sel_probs,  # [bs, src_len, candi]
            'out_ids': out_ids,      # [trg_len-2, bs, da_len]
            'out_rates': out_rates   # [trg_len-2, bs, 1]
        }


# ==================== 损失函数（联合） ====================

class E2ELoss(nn.Module):

    def __init__(self, lambda_selector=1.0, lambda_id=10.0, lambda_rate=5.0):
        super().__init__()
        self.lambda_selector = lambda_selector
        self.lambda_id = lambda_id
        self.lambda_rate = lambda_rate
        
        self.crit_selector = nn.BCELoss(reduction='mean')
        self.crit_id = nn.BCELoss(reduction='sum')
        self.crit_rate = nn.L1Loss(reduction='sum')

    def forward(self, outputs, labels, lengths):

        selector_probs = outputs['selector_probs']  # [bs, src_len, k]
        selector_onehot = labels['selector_onehot'].float()  # [bs, src_len, k]
        
        loss_selector = self.crit_selector(selector_probs, selector_onehot) * selector_onehot.shape[-1]

        out_ids = outputs['out_ids']       # [trg_len-2, bs, da_len]
        out_rates = outputs['out_rates']   # [trg_len-2, bs, 1]
        trg_labels = labels['trg_labels']  
        trg_rates = labels['trg_rates'][1:-1]  

        trg_lengths = lengths['trg_lengths']
        da_lengths = lengths['da_lengths']
        trg_lengths_sub = [length - 2 for length in trg_lengths]
        
        loss_id = self.crit_id(out_ids, trg_labels) * self.lambda_id / np.sum(np.array(trg_lengths_sub) * np.array(da_lengths))

        loss_rate = self.crit_rate(out_rates, trg_rates) * self.lambda_rate / sum(trg_lengths_sub) 

        total = self.lambda_selector * loss_selector + loss_id + loss_rate
        return total, {'selector': loss_selector.item(), 'id': loss_id.item(), 'rate': loss_rate.item()}



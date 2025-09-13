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


def get_num_pts(time_span, time_interval):
    num_pts = 0  # do not have to interpolate points
    if time_span % time_interval > time_interval / 2:
        # if the reminder is larger than half of the time interval
        num_pts = time_span // time_interval  # quotient
    elif time_span > time_interval:
        # if the reminder is smaller than half of the time interval and not equal to time interval
        num_pts = time_span // time_interval - 1
    return num_pts


def get_segs(o, d, rn):

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
        timestamp = t + self.seg_info.get_seg_travel_time(od[0])
        segs = []
        for i in range(len(od)-1):
            o = od[i]
            d = od[i+1]

            # extended path will not transitable
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
                    # dead road, cannot reach d
                    if len(out_segs) == 0:
                        break

                    # judge tie
                    nextseg = -1
                    next_max = -1
                    tie_cnt = 0
                    tie_nbrs = []
                    for seg in out_segs:
                        # d in neighbors, directly select d
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

            # assert 0 <= rate_tmp <= 1 and 1e-2 <= speed_tmp <= 35
            pred_id.append(id_tmp)
            pred_rate.append(rate_tmp)
            time_in.append(time_in[-1] + time_span)
            num_pts -= 1
        res = [src]
        for eid, ratio, ts in zip(pred_id[1:], pred_rate[1:], time_in[1:]):
            projected = rate2gps(rn, eid, ratio)
            dist = 0.
            rate = ratio
            # lat, lng = self.seg_info.get_gps(eid, ratio)
            # x = SPoint(lat, lng)
            # projected, rate, dist = project_pt_to_road(rn, x, eid)
            candi_pt = CandidatePoint(projected.lat, projected.lng, eid, dist, rate * self.seg_info.get_seg_length(eid), rate)
            pt = STPoint(projected.lat, projected.lng, ts, {'candi_pt': candi_pt})
            pt.time_arr = dt.datetime.fromtimestamp(ts, self.tz)
            res.append(pt)
        res.append(trg)
        return res


class TrajRecData(Dataset):

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


class TrajRecTestData(Dataset):

    def __init__(self, rn, trajs_dir, mbr, parameters, mode):
        self.parameters = parameters
        self.rn = rn
        self.mbr = mbr  # MBR of all trajectories
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

    def __init__(self, parameters):
        super().__init__()
        self.pro_features_flag = parameters.pro_features_flag
        self.hid_dim = parameters.hid_dim

        self.transformer = GPSFormer(parameters.hid_dim, parameters.transformer_layers, heads=parameters.heads)

        if self.pro_features_flag:
            self.temporal = nn.Embedding(parameters.pro_input_dim, parameters.pro_output_dim)
            self.fc_hid = nn.Linear(parameters.hid_dim + parameters.pro_output_dim, parameters.hid_dim)

    def forward(self, src, src_len, pro_features):
        # src = [src len, batch size, 3]
        bs = src.size(1)
        max_src_len = src.size(0)

        mask3d = torch.ones(bs, max_src_len, max_src_len, device=src.device)
        mask2d = torch.ones(bs, max_src_len, device=src.device)

        mask3d = sequence_mask3d(mask3d, src_len, src_len)
        mask2d = sequence_mask(mask2d, src_len).transpose(0, 1).unsqueeze(-1).repeat(1, 1, self.hid_dim)

        src = src.transpose(0, 1)
        outputs = self.transformer(src, mask3d)
        outputs = outputs.transpose(0, 1)  # [src len, bs, hid dim]

        assert outputs.size(0) == max_src_len

        outputs = outputs * mask2d
        hidden = torch.sum(outputs, dim=0) / src_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)

        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))

        return outputs, hidden


# 图路径编码器类，用于处理GPS轨迹和路径信息的联合编码
class GREncoder(nn.Module):

    def __init__(self, parameters):  # 初始化函数，接收参数配置
        super().__init__()  # 调用父类初始化函数
        self.hid_dim = parameters.hid_dim  # 隐藏层维度
        self.pro_features_flag = parameters.pro_features_flag  # 是否使用轨迹特征标志

        # Transformer编码器，用于处理GPS和路径的联合表示
        self.transformer = GRFormer(parameters.hid_dim, parameters.transformer_layers, heads=parameters.heads)  # 图路径Transformer模型

        if self.pro_features_flag:  # 如果使用轨迹特征
            # 时间特征嵌入层
            self.temporal = nn.Embedding(parameters.pro_input_dim, parameters.pro_output_dim)  # 时间特征嵌入
            # 特征融合全连接层
            self.fc_hid = nn.Linear(parameters.hid_dim + parameters.pro_output_dim, parameters.hid_dim)  # 隐藏状态融合层

    def forward(self, src, src_len, route, route_len, pro_features):  # 前向传播函数
        # src = [src len, batch size, 3]  # GPS轨迹输入
        # if only input trajectory, input dim = 2; elif input trajectory + behavior feature, input dim = 2 + n  # 输入维度说明
        # src_len = [batch size]  # 源序列长度
        bs = src.size(1)  # 批次大小
        src_max_len = src.size(0)  # GPS序列最大长度
        route_max_len = route.size(0)  # 路径序列最大长度

        # 创建注意力掩码矩阵
        mask3d = torch.ones(bs, src_max_len, src_max_len, device=src.device)  # GPS自注意力掩码
        route_mask3d = torch.ones(bs, route_max_len, route_max_len, device=src.device)  # 路径自注意力掩码
        route_mask2d = torch.ones(bs, route_max_len, device=src.device)  # 路径序列掩码
        inter_mask = torch.ones(bs, route_max_len, src_max_len, device=src.device)  # GPS-路径交互掩码

        # 应用序列长度掩码
        mask3d = sequence_mask3d(mask3d, src_len, src_len)  # 应用GPS序列掩码
        route_mask3d = sequence_mask3d(route_mask3d, route_len, route_len)  # 应用路径序列掩码
        route_mask2d = sequence_mask(route_mask2d, route_len).transpose(0,1).unsqueeze(-1).repeat(1, 1, self.hid_dim)  # 扩展路径掩码维度
        inter_mask = sequence_mask3d(inter_mask, route_len, src_len)  # 应用交互掩码

        # 调整输入维度顺序
        src = src.transpose(0, 1)  # 转换GPS输入维度顺序
        route = route.transpose(0, 1)  # 转换路径输入维度顺序
        # Transformer编码过程
        outputs = self.transformer(src, route, mask3d, route_mask3d, inter_mask)  # 通过Transformer编码
        outputs = outputs.transpose(0, 1)  # [src len, bs, hid dim]  # 恢复输出维度顺序

        # idx = [i for i in range(bs)]  # 批次索引（注释掉的代码）
        # hidden = outputs[[i - 1 for i in src_len], idx, :].unsqueeze(0)  # 获取最后时刻隐藏状态（注释掉的代码）
        assert outputs.size(0) == route_max_len  # 确保输出长度正确

        # 计算平均池化隐藏状态
        outputs = outputs * route_mask2d  # 应用掩码到输出
        hidden = torch.sum(outputs, dim=0) / route_len.unsqueeze(-1).repeat(1, self.hid_dim)  # 计算加权平均隐藏状态
        hidden = hidden.unsqueeze(0)  # 增加维度

        if self.pro_features_flag:  # 如果使用轨迹特征
            # 融合时间特征
            extra_emb = self.temporal(pro_features)  # 获取时间特征嵌入
            extra_emb = extra_emb.unsqueeze(0)  # 增加维度
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))  # 融合特征并激活

        return outputs, hidden  # 返回编码输出和隐藏状态


# 多任务解码器类，用于同时预测路段ID和通行率
class DecoderMulti(nn.Module):

    def __init__(self, parameters):
        super().__init__()

        self.id_size = parameters.id_size  # 路段ID总数
        self.emb_id = None # updated in encoder  # 路段ID嵌入层，在编码器中更新
        self.dest_type = parameters.dest_type  # 目的地类型
        self.rate_flag = parameters.rate_flag  # 是否预测通行率标志
        self.prog_flag = parameters.prog_flag  # 是否使用渐进式解码标志

        self.rid_feats_flag = parameters.rid_feats_flag  # 是否使用路段特征标志

        # self.temporal_flag = False  # 时间特征标志（注释掉）
        # self.prev_flag = False  # 历史信息标志（注释掉）
        # self.max_dist = 500  # 最大距离（注释掉）
        # self.src_len = 2  # 源序列长度（注释掉）

        # 计算RNN输入维度
        rnn_input_dim = parameters.hid_dim  # 基础隐藏维度
        if self.rid_feats_flag:  # 如果使用路段特征
            rnn_input_dim += parameters.rid_fea_dim  # 增加路段特征维度
        if self.rate_flag:  # 如果预测通行率
            rnn_input_dim += 1  # 增加通行率维度
        if self.dest_type in [1, 2]:  # 如果目的地类型为1或2
            rnn_input_dim += parameters.hid_dim  # 增加目的地嵌入维度
            if self.rid_feats_flag:  # 如果使用路段特征
                rnn_input_dim += parameters.rid_fea_dim  # 再次增加路段特征维度
            if self.rate_flag:  # 如果预测通行率
                rnn_input_dim += 1  # 再次增加通行率维度
        # if self.temporal_flag:  # 时间特征处理（注释掉）
        #     self.temporal_dist_embedding = nn.Embedding(self.max_dist, parameters.hid_dim // self.src_len)
        #     rnn_input_dim += parameters.hid_dim
        # self.fusion_mlp = nn.Sequential(  # 特征融合MLP（注释掉）
        #     nn.Linear(rnn_input_dim, 2 * parameters.hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(2 * parameters.hid_dim, parameters.hid_dim)
        # )
        
        # RNN解码器
        self.rnn = nn.GRU(rnn_input_dim, parameters.hid_dim)  # GRU循环神经网络

        # if self.prev_flag:  # 历史信息处理（注释掉）
        #     self.observed_mlp = nn.Linear(parameters.hid_dim + parameters.rid_fea_dim + 1, parameters.hid_dim)
        #     self.observed_attn = Attention(parameters.hid_dim)

        # 路径注意力机制
        self.attn_route = Attention(parameters.hid_dim)  # 路径注意力层

        # 通行率预测网络
        if self.rate_flag:  # 如果需要预测通行率
            fc_rate_out_input_dim = parameters.hid_dim + parameters.hid_dim  # 通行率预测输入维度
            self.fc_rate_out = nn.Sequential(  # 通行率预测全连接网络
                nn.Linear(fc_rate_out_input_dim, parameters.hid_dim * 2),  # 第一层线性变换
                nn.ReLU(),  # ReLU激活函数
                nn.Linear(parameters.hid_dim * 2, 1),  # 第二层线性变换
                nn.Sigmoid()  # Sigmoid激活函数，输出0-1之间的概率
            )

    # 单步解码函数
    def decoding_step(self, input_id, input_rate, hidden, route_outputs,
                      route_attn_mask, d_rids, d_rates, rid_features_dict, dt, observed_emb, observed_mask):

        rnn_input = self.emb_id[input_id]  # 获取输入路段ID的嵌入表示
        if self.rid_feats_flag:  # 如果使用路段特征
            rnn_input = torch.cat([rnn_input, rid_features_dict[input_id]], dim=-1)  # 拼接路段特征
        if self.rate_flag:  # 如果使用通行率
            rnn_input = torch.cat((rnn_input, input_rate), dim=-1)  # 拼接通行率信息
        if self.dest_type in [1, 2]:  # 如果目的地类型为1或2
            embed_drids = self.emb_id[d_rids]  # 获取目的地路段嵌入
            rnn_input = torch.cat((rnn_input, embed_drids), dim=-1)  # 拼接目的地嵌入
            if self.rid_feats_flag:  # 如果使用路段特征
                rnn_input = torch.cat([rnn_input, rid_features_dict[input_id]], dim=-1)  # 拼接路段特征
            if self.rate_flag:  # 如果使用通行率
                rnn_input = torch.cat((rnn_input, d_rates), dim=-1)  # 拼接目的地通行率
        # if self.temporal_flag:  # 时间特征处理（注释掉）
        #     rnn_input = torch.cat((rnn_input, dt), dim=-1)
        # rnn_input = self.fusion_mlp(rnn_input)  # 特征融合（注释掉）
        rnn_input = rnn_input.unsqueeze(0)  # 增加时间步维度

        # RNN前向传播
        output, hidden = self.rnn(rnn_input, hidden)  # RNN解码步骤

        # 历史信息注意力处理（注释掉的功能）
        if False:  # 历史观测信息处理分支（未启用）
            observed_emb = observed_emb.unsqueeze(1)  # 增加维度
            _, observed_weighted = self.observed_attn(hidden.permute(1, 0, 2), observed_emb, observed_emb, observed_mask.unsqueeze(1))  # 历史信息注意力
            query = hidden.permute(1, 0, 2) + observed_weighted  # 融合历史信息
        else:  # 当前使用的分支
            query = hidden.permute(1, 0, 2)  # 使用当前隐藏状态作为查询
        
        # 路径注意力计算
        key = route_outputs.permute(1, 0, 2).unsqueeze(1)  # 路径输出作为键值
        scores, weighted = self.attn_route(query, key, key, route_attn_mask.unsqueeze(1))  # 计算注意力分数和加权表示
        prediction_id = scores.squeeze(1).masked_fill(route_attn_mask == 0, 0)  # 应用掩码到预测分数
        weighted = weighted.permute(1, 0, 2)  # 调整加权表示维度

        # 通行率预测
        if self.rate_flag:  # 如果需要预测通行率
            rate_input = torch.cat((hidden, weighted), dim=-1).squeeze(0)  # 拼接隐藏状态和注意力加权表示
            prediction_rate = self.fc_rate_out(rate_input)  # 通过全连接网络预测通行率
        else:  # 如果不预测通行率
            prediction_rate = torch.ones((prediction_id.shape[0], 1), dtype=torch.float32, device=hidden.device) / 2  # 使用默认值0.5

        return prediction_id, prediction_rate, hidden  # 返回路段ID预测、通行率预测和隐藏状态

    # 前向传播函数
    def forward(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len, hidden, rid_features_dict, routes, route_outputs, route_attn_mask, d_rids, d_rates, teacher_forcing_ratio):

        # 初始化输出张量
        routes = routes.permute(1, 0)  # 调整路径维度顺序 [bs, seq len]
        outputs_id = torch.zeros([max_trg_len, batch_size, routes.shape[1]], device=hidden.device)  # 路段ID预测输出张量
        rate_out_dim = 1  # 通行率输出维度
        outputs_rate = torch.zeros([max_trg_len, batch_size, rate_out_dim], device=hidden.device)  # 通行率预测输出张量
        # states = torch.zeros([max_trg_len, batch_size, hidden.shape[-1]], device=hidden.device)  # 状态张量（注释掉）
        # states[0] = hidden  # 初始状态（注释掉）

        # 时间特征初始化（注释掉）
        # if self.temporal_flag:
        #     src_time_steps = torch.column_stack([torch.zeros(batch_size, device=hidden.device), torch.tensor(trg_len, device=hidden.device) - 1]).long()
        
        # 历史信息初始化（注释掉）
        # if self.prev_flag:
        #     observed_ids = trg_id[0, :].unsqueeze(-1)
        #     observed_rates = trg_rate[0, :].unsqueeze(-1)

        # 解码器的第一个输入是<sos>标记
        input_id = trg_id[0, :]  # 初始输入路段ID
        input_rate = trg_rate[0, :]  # 初始输入通行率
        
        # 逐步解码过程
        for t in range(1, max_trg_len):  # 从第1步开始解码（第0步是<sos>）
            # 决定是否使用教师强制
            teacher_force = random.random() < teacher_forcing_ratio  # 随机决定是否使用教师强制

            # 初始化时间和历史信息变量
            dt = None  # 时间差信息
            observed_emb = None  # 历史观测嵌入
            observed_mask = None  # 历史观测掩码
            
            # 时间特征处理（注释掉）
            # if self.temporal_flag:
            #     temporal_dist = torch.abs(
            #         torch.tensor(t, device=hidden.device).repeat(batch_size, self.src_len) - src_time_steps)
            #     temporal_dist = torch.where(temporal_dist >= self.max_dist, self.max_dist - 1, temporal_dist)
            #     dt = self.temporal_dist_embedding(temporal_dist).reshape(batch_size, -1)
            
            # 历史信息处理（注释掉）
            # if self.prev_flag:
            #     observed_len, _ = torch.min(torch.vstack([torch.tensor(t + 1, device=hidden.device).repeat(batch_size).unsqueeze(0), torch.tensor(trg_len, device=hidden.device).unsqueeze(0)]), dim=0)
            #     observed_mask = torch.ones(batch_size, t + 1, device=hidden.device)
            #     observed_mask = sequence_mask(observed_mask, observed_len)
            #     tmp_ids = torch.cat([d_rids.unsqueeze(-1), observed_ids], dim=1)
            #     tmp_rates = torch.cat([d_rates.unsqueeze(-1), observed_rates], dim=1)
            #     tmp_ids = tmp_ids.masked_fill(observed_mask == 0, 0)
            #     tmp_rates = tmp_rates.masked_fill(observed_mask.unsqueeze(-1) == 0, 0)
            #     observed_emb = torch.cat([self.emb_id[tmp_ids], rid_features_dict[tmp_ids], tmp_rates], dim=-1)
            #     observed_emb = self.observed_mlp(observed_emb)

            # 执行单步解码
            prediction_id, prediction_rate, hidden = self.decoding_step(input_id, input_rate, hidden, route_outputs, route_attn_mask, d_rids, d_rates, rid_features_dict, dt, observed_emb, observed_mask)

            # 渐进式解码约束处理
            if teacher_forcing_ratio == -1 and self.prog_flag:  # 如果使用渐进式解码
                for i in range(batch_size):  # 遍历批次中的每个样本
                    if t < trg_len[i]:  # 如果当前时间步小于目标长度
                        prev_idx = (input_id[i] == routes[i]).nonzero(as_tuple=True)[0][0]  # 找到前一个路段在路径中的索引
                        tmp_flag = True  # 临时标志
                        while tmp_flag:  # 循环直到找到合适的预测
                            cur_idx = prediction_id[i].argmax()  # 获取当前预测的最大值索引
                            if cur_idx < prev_idx:  # 如果预测索引小于前一个索引（违反渐进约束）
                                prediction_id[i, cur_idx] = 1e-6  # 将该预测设为很小的值
                            else:  # 如果满足渐进约束
                                tmp_flag = False  # 退出循环

            # 保存预测结果
            outputs_id[t] = prediction_id  # 保存路段ID预测
            outputs_rate[t] = prediction_rate  # 保存通行率预测
            # states[t] = hidden  # 保存隐藏状态（注释掉）

            # 获取预测中概率最高的标记
            # top1_id = prediction_id.argmax(1).unsqueeze(-1)  # 获取最高概率预测（注释掉）

            # 决定下一步的输入：教师强制 vs 模型预测
            if teacher_force:  # 如果使用教师强制
                input_id = trg_id[t]  # 使用真实的下一个路段ID
                input_rate = trg_rate[t]  # 使用真实的下一个通行率
            else:  # 如果使用模型预测
                input_id = (F.one_hot(prediction_id.argmax(dim=1), routes.shape[1]) * routes).sum(-1)  # 根据预测选择路段ID
                input_rate = prediction_rate  # 使用预测的通行率

            # 更新历史信息（注释掉）
            # if self.prev_flag:
            #     observed_ids = torch.cat([observed_ids, input_id.unsqueeze(-1)], dim=1)
            #     observed_rates = torch.cat([observed_rates, input_rate.unsqueeze(-1)], dim=1)

        # 应用序列长度掩码到输出
        mask_trg = torch.ones([batch_size, max_trg_len], device=outputs_id.device)  # 创建目标掩码
        mask_trg = sequence_mask(mask_trg, torch.tensor(trg_len, device=outputs_id.device))  # 应用序列长度掩码
        outputs_rate = outputs_rate.permute(1, 0, 2)  # 调整通行率输出维度 [batch size, seq len, 1]
        outputs_rate = outputs_rate.masked_fill(mask_trg.unsqueeze(-1) == 0, 0)  # 应用掩码到通行率输出
        outputs_rate = outputs_rate.permute(1, 0, 2)  # 恢复原始维度顺序
        return outputs_id, outputs_rate  # 返回路段ID预测和通行率预测


class TrajRecovery(nn.Module):  # 轨迹恢复模型类，继承自PyTorch的nn.Module

    def __init__(self, parameters):  # 初始化函数，接收参数配置
        super().__init__()  # 调用父类初始化函数
        self.srcseg_flag = parameters.srcseg_flag  # 是否使用源路段标志
        self.hid_dim = parameters.hid_dim  # 隐藏层维度

        self.da_route_flag = parameters.da_route_flag  # 是否使用DA路径标志

        self.learn_pos = parameters.learn_pos  # 是否学习位置编码标志
        self.rid_feats_flag = parameters.rid_feats_flag  # 是否使用路段特征标志

        self.params = parameters  # 保存参数配置

        self.emb_id = nn.Parameter(torch.rand(parameters.id_size, parameters.id_emb_dim))  # 路段ID嵌入参数
        if self.learn_pos:  # 如果需要学习位置编码
            max_input_length = 500  # 最大输入长度
            self.pos_embedding_gps = nn.Embedding(max_input_length, parameters.hid_dim)  # GPS位置嵌入层
            self.pos_embedding_route = nn.Embedding(max_input_length, parameters.hid_dim)  # 路径位置嵌入层
        input_dim_gps = 3  # GPS输入维度基础值
        if self.learn_pos:  # 如果学习位置编码
            input_dim_gps += parameters.hid_dim  # 增加位置编码维度
        if self.srcseg_flag:  # 如果使用源路段
            input_dim_gps += parameters.hid_dim + 1  # 增加路段嵌入和特征维度
        self.fc_in_gps = nn.Linear(input_dim_gps, parameters.hid_dim)  # GPS输入全连接层
        input_dim_route = parameters.hid_dim  # 路径输入维度基础值
        if self.learn_pos:  # 如果学习位置编码
            input_dim_route += parameters.hid_dim  # 增加位置编码维度
        if self.rid_feats_flag:  # 如果使用路段特征
            input_dim_route += parameters.rid_fea_dim  # 增加路段特征维度
        self.fc_in_route = nn.Linear(input_dim_route, parameters.hid_dim)  # 路径输入全连接层

        if self.da_route_flag:  # 如果使用DA路径
            self.encoder = GREncoder(parameters)  # 使用图路径编码器
        else:  # 否则
            self.encoder = GPSEncoder(parameters)  # 使用GPS编码器
        self.decoder = DecoderMulti(parameters)  # 多任务解码器

        self.init_weights()  # 初始化权重

        self.timer1, self.timer2, self.timer3, self.timer4, self.timer5, self.timer6 = [], [], [], [], [], []  # 计时器列表

    def init_weights(self):  # 权重初始化函数
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

    def forward(self, src, src_len, trg_id, trg_rate, trg_len, pro_features, rid_features_dict, da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio):  # 前向传播函数

        t0 = time.time()  # 记录开始时间

        max_trg_len = trg_id.size(0)  # 目标序列最大长度
        batch_size = trg_id.size(1)  # 批次大小

        # road representation
        self.decoder.emb_id = self.emb_id  # 将路段嵌入传递给解码器

        gps_emb = src.float()  # GPS嵌入，转换为浮点型
        if self.learn_pos:  # 如果学习位置编码
            gps_pos = src[:, :, -1].long()  # 获取GPS位置索引
            gps_pos_emb = self.pos_embedding_gps(gps_pos)  # 获取GPS位置嵌入
            gps_emb = torch.cat([gps_emb, gps_pos_emb], dim=-1)  # 拼接GPS数据和位置嵌入
        if self.srcseg_flag:  # 如果使用源路段
            seg_emb = self.emb_id[src_seg_seqs]  # 获取路段嵌入
            gps_emb = torch.cat((gps_emb, seg_emb, src_seg_feats), dim=-1)  # 拼接GPS、路段嵌入和路段特征
        gps_in = self.fc_in_gps(gps_emb)  # 通过全连接层处理GPS输入
        gps_in_lens = torch.tensor(src_len, device=src.device)  # 源序列长度张量

        self.timer1.append(time.time() - t0)  # 记录第一阶段耗时
        t1 = time.time()  # 记录时间点1

        # encoder_outputs 是输入序列的所有隐藏状态，包括前向和后向
        # hidden 是最终的前向和后向隐藏状态，通过线性层处理

        if self.da_route_flag:  # 如果使用DA路径
            route_emb = self.emb_id[da_routes]  # 获取路径嵌入
            if self.learn_pos:  # 如果学习位置编码
                route_pos_emb = self.pos_embedding_route(da_pos)  # 获取路径位置嵌入
                route_emb = torch.cat([route_emb, route_pos_emb], dim=-1)  # 拼接路径嵌入和位置嵌入
            if self.rid_feats_flag:  # 如果使用路段特征
                route_feats = rid_features_dict[da_routes]  # 获取路径特征
                route_emb = torch.cat([route_emb, route_feats], dim=-1)  # 拼接路径嵌入和特征
            route_in = self.fc_in_route(route_emb)  # 通过全连接层处理路径输入
            route_in_lens = torch.tensor(da_lengths, device=src.device)  # 路径序列长度张量

            self.timer2.append(time.time() - t1)  # 记录第二阶段耗时
            t2 = time.time()  # 记录时间点2

            route_outputs, hiddens = self.encoder(gps_in, gps_in_lens, route_in, route_in_lens, pro_features)  # 编码器前向传播

            self.timer3.append(time.time() - t2)  # 记录第三阶段耗时
            t3 = time.time()  # 记录时间点3
        else:  # 如果不使用DA路径
            _, hiddens = self.encoder(gps_in, gps_in_lens, pro_features)  # 仅使用GPS编码
            route_in_lens = torch.tensor(da_lengths, device=src.device)  # 路径序列长度张量
            route_outputs = self.emb_id[da_routes]  # 直接使用路径嵌入作为输出

        route_attn_mask = torch.ones(batch_size, max(da_lengths), device=src.device)  # 创建注意力掩码
        route_attn_mask = sequence_mask(route_attn_mask, route_in_lens)  # 应用序列掩码

        t4 = time.time()  # 记录时间点4
        self.timer4.append(time.time() - t3)  # 记录第四阶段耗时

        outputs_id, outputs_rate = self.decoder(max_trg_len, batch_size, trg_id, trg_rate, trg_len, hiddens, rid_features_dict, da_routes, route_outputs, route_attn_mask, d_rids, d_rates, teacher_forcing_ratio)  # 解码器前向传播

        final_outputs_id = outputs_id[1:-1]  # 去除开始和结束标记的ID输出
        final_outputs_rate = outputs_rate[1:-1]  # 去除开始和结束标记的速率输出

        t5 = time.time()  # 记录时间点5
        self.timer5.append(time.time() - t4)  # 记录第五阶段耗时
        self.timer6.append(t5 - t0)  # 记录总耗时

        return final_outputs_id, final_outputs_rate  # 返回最终的ID和速率输出


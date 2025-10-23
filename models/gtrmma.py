"""
G-TRMMA: Graph-Enhanced Trajectory Recovery with Map Matching Assistance
"""

import random
import time
from tqdm import tqdm
import os
import datetime as dt
import numpy as np
import pickle
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.nn import GATConv

from models.layers import Attention, sequence_mask, sequence_mask3d, GPSLayer, MultiHeadAttention, Norm, FeedForward
from preprocess import SparseDAM, SegInfo
from utils.model_utils import gps2grid, get_normalized_t
from utils.spatial_func import SPoint, project_pt_to_road


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


class TrajRecData(Dataset):
    
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


class TrajRecTestData(Dataset):
    
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

            for p1, p1_idx, p2, p2_idx, s1, s2, ts, mmf1, mmf2 in zip(
                src_list[:-1], traj.low_idx[:-1], src_list[1:], traj.low_idx[1:], 
                seg_seq[:-1], seg_seq[1:], time_seq[:-1], src_mm[:-1], src_mm[1:]
            ):
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

    def forward(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len, hidden, rid_features_dict, 
                routes, route_outputs, route_attn_mask, d_rids, d_rates, teacher_forcing_ratio):

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

            prediction_id, prediction_rate, hidden = self.decoding_step(
                input_id, input_rate, hidden, route_outputs, route_attn_mask, 
                d_rids, d_rates, rid_features_dict, dt, observed_emb, observed_mask
            )

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


class RouteGraphEncoder(nn.Module):
    def __init__(self, hid_dim, num_layers=1, num_heads=4, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        
        self.gat_layers = nn.ModuleList([
            GATConv(hid_dim, hid_dim // num_heads, heads=num_heads, dropout=dropout, concat=True, add_self_loops=False)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(num_layers)])
    
    def forward(self, route_emb, route_len, adj_matrices):
        batch_size, max_route_len, _ = route_emb.shape
        
        node_list = []
        edge_list = []
        offset = 0
        
        for i in range(batch_size):
            valid_len = route_len[i].item()
            node_list.append(route_emb[i, :valid_len, :])
            edge_list.append(adj_matrices[i] + offset)
            offset += valid_len
        
        batched_nodes = torch.cat(node_list, dim=0)
        batched_edges = torch.cat(edge_list, dim=1)
        
        h = batched_nodes
        for gat, norm in zip(self.gat_layers, self.norms):
            h_new = gat(h, batched_edges)
            h = norm(h + h_new)
        
        outputs = []
        node_idx = 0
        for i in range(batch_size):
            valid_len = route_len[i].item()
            sample_h = h[node_idx:node_idx + valid_len]
            
            if valid_len < max_route_len:
                padding = torch.zeros(max_route_len - valid_len, self.hid_dim, 
                                    device=route_emb.device, dtype=route_emb.dtype)
                sample_h = torch.cat([sample_h, padding], dim=0)
            outputs.append(sample_h)
            node_idx += valid_len
        
        route_outputs = torch.stack(outputs, dim=0)
        return route_outputs


class GNNRouteEncoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.pro_features_flag = parameters.pro_features_flag
        self.num_layers = parameters.transformer_layers
        
        self.gps_layers = nn.ModuleList([
            GPSLayer(parameters.hid_dim, parameters.heads) 
            for _ in range(self.num_layers)
        ])
        
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
        
        if self.pro_features_flag:
            self.temporal = nn.Embedding(parameters.pro_input_dim, parameters.pro_output_dim)
            self.fc_hid = nn.Linear(parameters.hid_dim + parameters.pro_output_dim, parameters.hid_dim)
    
    def forward(self, src, src_len, route, route_len, adj_matrices, pro_features):
        bs = src.size(1)
        src_max_len = src.size(0)
        route_max_len = route.size(0)
        
        gps_mask3d = torch.ones(bs, src_max_len, src_max_len, device=src.device)
        gps_mask3d = sequence_mask3d(gps_mask3d, src_len, src_len)
        inter_mask = torch.ones(bs, route_max_len, src_max_len, device=src.device)
        inter_mask = sequence_mask3d(inter_mask, route_len, src_len)
        
        gps_emb = src.transpose(0, 1)
        route_emb = route.transpose(0, 1)
        
        for layer_idx in range(self.num_layers):
            gps_emb = self.gps_layers[layer_idx](gps_emb, gps_mask3d)
            
            route_gnn_out = self.route_gnn_layers[layer_idx](route_emb, route_len, adj_matrices)
            route1 = self.dropout(route_gnn_out)
            route_out = self.route_norms1[layer_idx](route_emb + route1)
            
            route2 = self.dropout(self.route_gps_attns[layer_idx](route_out, gps_emb, gps_emb, inter_mask))
            route_out2 = self.route_norms2[layer_idx](route_out + route2)
            
            route_emb = self.route_ffs[layer_idx](route_out2)
        
        route_outputs = route_emb.transpose(0, 1)
        
        route_mask2d = torch.ones(bs, route_max_len, device=route.device)
        route_mask2d = sequence_mask(route_mask2d, route_len).transpose(0, 1).unsqueeze(-1).repeat(1, 1, self.hid_dim)
        
        masked_route = route_outputs * route_mask2d
        hidden = torch.sum(masked_route, dim=0) / route_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)
        
        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))
        
        return route_outputs, hidden


class GTrajRecovery(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.srcseg_flag = parameters.srcseg_flag
        self.hid_dim = parameters.hid_dim
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
        
        self.encoder = GNNRouteEncoder(parameters)
        self.decoder = DecoderMulti(parameters)
        
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
    
    def forward(self, src, src_len, trg_id, trg_rate, trg_len, pro_features, 
                rid_features_dict, da_routes, da_lengths, da_pos, 
                src_seg_seqs, src_seg_feats, adj_matrices, d_rids, d_rates, 
                teacher_forcing_ratio):
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
        
        route_emb = self.emb_id[da_routes]
        if self.learn_pos:
            route_pos_emb = self.pos_embedding_route(da_pos)
            route_emb = torch.cat([route_emb, route_pos_emb], dim=-1)
        if self.rid_feats_flag:
            route_feats = rid_features_dict[da_routes]
            route_emb = torch.cat([route_emb, route_feats], dim=-1)
        route_in = self.fc_in_route(route_emb)
        route_in_lens = torch.tensor(da_lengths, device=src.device)
        
        route_outputs, hiddens = self.encoder(gps_in, gps_in_lens, route_in, 
                                              route_in_lens, adj_matrices, pro_features)
        
        route_attn_mask = torch.ones(batch_size, max(da_lengths), device=src.device)
        route_attn_mask = sequence_mask(route_attn_mask, route_in_lens)
        
        outputs_id, outputs_rate = self.decoder(
            max_trg_len, batch_size, trg_id, trg_rate, trg_len, hiddens,
            rid_features_dict, da_routes, route_outputs, route_attn_mask,
            d_rids, d_rates, teacher_forcing_ratio
        )
        
        final_outputs_id = outputs_id[1:-1]
        final_outputs_rate = outputs_rate[1:-1]
        
        return final_outputs_id, final_outputs_rate


class GTrajRecData(TrajRecData):
    def __init__(self, rn, trajs_dir, mbr, parameters, mode):
        super().__init__(rn, trajs_dir, mbr, parameters, mode)
        self.G = pickle.load(open(os.path.join(parameters.dam_root, "road_graph_wtime"), "rb"))
        self.route_cache = {}
    
    def __getitem__(self, index):
        data_list = super().__getitem__(index)
        
        enhanced_data = []
        for data_item in data_list:
            da_route = data_item[0]
            route_key = tuple(da_route.tolist())
            
            if route_key not in self.route_cache:
                edge_index = self._build_route_graph_with_topology(route_key)
                self.route_cache[route_key] = edge_index
            else:
                edge_index = self.route_cache[route_key]
            
            enhanced_data.append(data_item + [edge_index])
        
        return enhanced_data
    
    def _build_route_graph_with_topology(self, route_tuple):
        route_len = len(route_tuple)
        edges = set()
        
        for i in range(route_len):
            edges.add((i, i))
        
        for i in range(route_len - 1):
            edges.add((i, i + 1))
            edges.add((i + 1, i))
        
        for i, rid_i in enumerate(route_tuple):
            if rid_i in self.G:
                neighbors = set(self.G.neighbors(rid_i)) | set(self.G.predecessors(rid_i))
                for j, rid_j in enumerate(route_tuple):
                    if i != j and abs(i - j) <= 2 and rid_j in neighbors:
                        edges.add((i, j))
        
        edges = sorted(list(edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index


class GTrajRecTestData(TrajRecTestData):
    def __init__(self, rn, trajs_dir, mbr, parameters, mode):
        super().__init__(rn, trajs_dir, mbr, parameters, mode)
        
        self.G = pickle.load(open(os.path.join(parameters.dam_root, "road_graph_wtime"), "rb"))
        
        self.adj_matrices = []
        for route in tqdm(self.routes, desc='Building graphs'):
            edge_index = self._build_route_graph_with_topology(route)
            self.adj_matrices.append(edge_index)
    
    def _build_route_graph_with_topology(self, route):
        route_len = len(route)
        edges = set()
        
        for i in range(route_len):
            edges.add((i, i))
        
        for i in range(route_len - 1):
            edges.add((i, i + 1))
            edges.add((i + 1, i))
        
        for i, rid_i in enumerate(route):
            if rid_i in self.G:
                neighbors = set(self.G.neighbors(rid_i)) | set(self.G.predecessors(rid_i))
                for j, rid_j in enumerate(route):
                    if i != j and abs(i - j) <= 2 and rid_j in neighbors:
                        edges.add((i, j))
        
        edges = sorted(list(edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def __getitem__(self, index):
        base_data = super().__getitem__(index)
        adj_matrix = self.adj_matrices[index]
        return base_data + (adj_matrix,)

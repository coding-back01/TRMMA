import random
import time
from tqdm import tqdm
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from models.layers import Attention, GPSFormer, GRFormer, sequence_mask, sequence_mask3d
from models.trmma import DAPlanner, get_num_pts, calc_cos_value
from utils.model_utils import gps2grid, get_normalized_t
from utils.spatial_func import SPoint, project_pt_to_road, rate2gps
from utils.trajectory_func import STPoint
from utils.candidate_point import CandidatePoint


class UTTData(Dataset):
    """UTT训练数据集"""
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
            keep_index = [0] + sorted(random.sample(range(1, length - 1), 
                                      int((length - 2) * self.keep_ratio))) + [length - 1]
        else:
            keep_index = traj.low_idx
        
        src_list = np.array(traj.pt_list, dtype=object)
        src_list = src_list[keep_index].tolist()
        trg_list = traj.pt_list

        data = []
        for p1, p1_idx, p2, p2_idx in zip(src_list[:-1], keep_index[:-1], 
                                           src_list[1:], keep_index[1:]):
            if (p1_idx + 1) < p2_idx:
                tmp_src_list = [p1, p2]
                
                ls_grid_seq, ls_gps_seq, hours, tmp_seg_seq = self.get_src_seq(tmp_src_list)
                features = self.get_pro_features(tmp_src_list, hours)
                
                mm_gps_seq, mm_eids_raw, mm_rates = self.get_trg_seq(trg_list[p1_idx: p2_idx + 1])
                
                candi_label, candi_id, candi_feat, candi_mask = self.get_candis_feats(
                    tmp_seg_seq, mm_eids_raw)
                
                path = traj.cpath[p1.cpath_idx: p2.cpath_idx + 1]
                path_valid = [self.rn.valid_edge_one[item] if item in self.rn.valid_edge_one else 0 
                             for item in path]
                
                src_grid_seq = torch.tensor(ls_grid_seq)
                src_pro_fea = torch.tensor(features)
                src_seg_seq = torch.tensor([self.rn.valid_edge_one[item] if item in self.rn.valid_edge_one else 0 
                                           for item in tmp_seg_seq])
                
                mm_eids_mapped = [self.rn.valid_edge_one[eid] if eid in self.rn.valid_edge_one else 0 
                                 for eid in mm_eids_raw]
                
                trg_gps_seq = torch.tensor(mm_gps_seq)
                trg_rid = torch.tensor(mm_eids_mapped)
                trg_rate = torch.tensor(mm_rates)
                
                path_tensor = torch.tensor(path_valid)
                
                d_rid = trg_rid[-1]
                d_rate = trg_rate[-1]
                
                data.append([src_grid_seq, src_pro_fea, src_seg_seq, 
                            trg_gps_seq, trg_rid, trg_rate, 
                            candi_label, candi_id, candi_feat, candi_mask,
                            path_tensor, d_rid, d_rate])
        
        return data

    def get_candis_feats(self, src_seg_seq, trg_eids):
        """获取每个GPS点的候选路段特征"""
        candi_id = []
        candi_feat = []
        candi_onehot = []
        candi_mask = []
        
        for seg_id, trg in zip(src_seg_seq, [trg_eids[0], trg_eids[-1]]):
            candis = []
            if seg_id in self.rn.valid_edge:
                candis.append(seg_id)
                if seg_id in self.rn.edgeDict:
                    for neighbor in self.rn.edgeDict[seg_id]:
                        if neighbor in self.rn.valid_edge and len(candis) < self.parameters.candi_size:
                            candis.append(neighbor)
            
            if len(candis) == 0:
                candis = [0]
            
            candi_mask.append([1] * len(candis) + [0] * (self.parameters.candi_size - len(candis)))
            tmp_id = []
            tmp_feat = []
            tmp_onehot = [0] * self.parameters.candi_size
            
            for candi_eid in candis:
                tmp_id.append(candi_eid)
                tmp_feat.append([0.0] * 9)
            
            tmp_id.extend([0] * (self.parameters.candi_size - len(candis)))
            tmp_feat.extend([[0] * 9] * (self.parameters.candi_size - len(candis)))
            
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

    def get_src_seq(self, ds_pt_list):
        """获取源序列特征"""
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
        """获取目标序列"""
        mm_gps_seq = []
        mm_eids_raw = []
        mm_rates = []
        
        for pt in tmp_pt_list:
            candi_pt = pt.data['candi_pt']
            mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
            mm_eids_raw.append(candi_pt.eid)
            mm_rates.append([candi_pt.rate])
        
        return mm_gps_seq, mm_eids_raw, mm_rates

    def get_pro_features(self, ds_pt_list, hours):
        """获取轨迹级别特征"""
        hour = np.bincount(hours).argmax()
        week = ds_pt_list[0].time_arr.weekday()
        if week in [5, 6]:
            hour += 24
        return hour


class UTTTestData(Dataset):
    """UTT测试数据集"""
    def __init__(self, rn, trajs_dir, mbr, parameters, mode):
        self.parameters = parameters
        self.rn = rn
        self.mbr = mbr
        self.grid_size = parameters.grid_size
        self.time_span = parameters.time_span

        self.dam = DAPlanner(parameters.dam_root, parameters.id_size - 1, parameters.utc)
        
        self.src_grid_seqs = []
        self.src_gps_seqs = []
        self.src_pro_feas = []
        self.src_seg_seq = []
        self.trg_gps_seqs = []
        self.trg_rids = []
        self.trg_rates = []
        self.paths = []
        self.groups = []
        self.src_mms = []
        
        trajs = pickle.load(open(os.path.join(trajs_dir, 'test_output.pkl'), "rb"))

        for serial, traj in tqdm(enumerate(trajs), desc='Loading test data'):
            trg_list = traj.pt_list.copy()
            src_list = np.array(traj.pt_list, dtype=object)
            src_list = src_list[traj.low_idx].tolist()

            _, src_gps_seq, _, seg_seq, time_seq = self.get_src_seq(src_list)
            
            src_mm = []
            for seg, (lat, lng) in zip(seg_seq, src_gps_seq):
                projected, rate, dist = project_pt_to_road(self.rn, SPoint(lat, lng), seg)
                src_mm.append([[projected.lat, projected.lng], seg, rate])
            self.src_mms.append(src_mm)

            for p1, p1_idx, p2, p2_idx, s1, s2, ts, mmf1, mmf2 in zip(
                    src_list[:-1], traj.low_idx[:-1], src_list[1:], traj.low_idx[1:],
                    seg_seq[:-1], seg_seq[1:], time_seq[:-1], src_mm[:-1], src_mm[1:]):
                
                if (p1_idx + 1) < p2_idx:
                    tmp_src_list = [p1, p2]
                    
                    ls_grid_seq, ls_gps_seq, hours, _, _ = self.get_src_seq(tmp_src_list)
                    features = self.get_pro_features(tmp_src_list, hours)
                    
                    mm_gps_seq, mm_eids_raw, mm_rates = self.get_trg_seq(trg_list[p1_idx: p2_idx + 1])
                    
                    planner_mode = getattr(parameters, 'planner', 'da')
                    path = self.dam.planning_multi([s1, s2], ts, mode=planner_mode)
                    
                    mm_gps_seq[0] = mmf1[0]
                    mm_eids_raw[0] = mmf1[1]
                    mm_rates[0] = [mmf1[2]]
                    mm_gps_seq[-1] = mmf2[0]
                    mm_eids_raw[-1] = mmf2[1]
                    mm_rates[-1] = [mmf2[2]]
                    
                    mm_eids_mapped = [self.rn.valid_edge_one[eid] if eid in self.rn.valid_edge_one else 0 
                                     for eid in mm_eids_raw]
                    
                    self.paths.append([self.rn.valid_edge_one[item] if item in self.rn.valid_edge_one else 0 
                                      for item in path])
                    self.src_seg_seq.append([self.rn.valid_edge_one[s1] if s1 in self.rn.valid_edge_one else 0, 
                                            self.rn.valid_edge_one[s2] if s2 in self.rn.valid_edge_one else 0])
                    self.trg_gps_seqs.append(mm_gps_seq)
                    self.trg_rids.append(mm_eids_mapped)
                    self.trg_rates.append(mm_rates)
                    self.src_grid_seqs.append(ls_grid_seq)
                    self.src_gps_seqs.append(ls_gps_seq)
                    self.src_pro_feas.append(features)
                    self.groups.append(serial)

    def __len__(self):
        return len(self.src_grid_seqs)

    def __getitem__(self, index):
        src_grid_seq = torch.tensor(self.src_grid_seqs[index])
        src_pro_fea = torch.tensor(self.src_pro_feas[index])
        src_seg_seq = torch.tensor(self.src_seg_seq[index])
        
        trg_gps_seq = torch.tensor(self.trg_gps_seqs[index])
        trg_rid = torch.tensor(self.trg_rids[index])
        trg_rate = torch.tensor(self.trg_rates[index])
        path = torch.tensor(self.paths[index])
        
        d_rid = trg_rid[-1]
        d_rate = trg_rate[-1]
        
        return (src_grid_seq, src_pro_fea, src_seg_seq, 
                trg_gps_seq, trg_rid, trg_rate, path, d_rid, d_rate)

    def get_src_seq(self, ds_pt_list):
        """获取源序列特征"""
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
        """获取目标序列"""
        mm_gps_seq = []
        mm_eids_raw = []
        mm_rates = []
        
        for pt in tmp_pt_list:
            candi_pt = pt.data['candi_pt']
            mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
            mm_eids_raw.append(candi_pt.eid)
            mm_rates.append([candi_pt.rate])
        
        return mm_gps_seq, mm_eids_raw, mm_rates

    def get_pro_features(self, ds_pt_list, hours):
        """获取轨迹级别特征"""
        hour = np.bincount(hours).argmax()
        week = ds_pt_list[0].time_arr.weekday()
        if week in [5, 6]:
            hour += 24
        return hour


class JointEncoder(nn.Module):
    """联合编码器"""
    def __init__(self, parameters, emb_id=None):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.pro_features_flag = parameters.pro_features_flag
        
        self.gps_transformer = GPSFormer(parameters.hid_dim, 
                                        parameters.transformer_layers, 
                                        heads=parameters.heads)
        
        if emb_id is not None:
            self.emb_id = emb_id
        else:
            self.emb_id = nn.Embedding(parameters.id_size, parameters.id_emb_dim)
        
        self.candi_fc = nn.Linear(9, parameters.hid_dim)
        self.fusion_norm = nn.LayerNorm(parameters.hid_dim)
        self.fusion_dropout = nn.Dropout(parameters.dropout)
        
        if self.pro_features_flag:
            self.temporal = nn.Embedding(parameters.pro_input_dim, parameters.pro_output_dim)
            self.fc_hid = nn.Linear(parameters.hid_dim + parameters.pro_output_dim, 
                                   parameters.hid_dim)

    def forward(self, src, src_len, src_seg_ids, candi_ids, candi_feats, 
                candi_masks, pro_features):
        bs = src.size(1)
        max_src_len = src.size(0)
        
        src_len_tensor = torch.tensor(src_len, device=src.device)
        
        # GPS编码
        mask3d = torch.ones(bs, max_src_len, max_src_len, device=src.device)
        mask2d = torch.ones(bs, max_src_len, device=src.device)
        mask3d = sequence_mask3d(mask3d, src_len_tensor, src_len_tensor)
        mask2d = sequence_mask(mask2d, src_len_tensor).transpose(0, 1).unsqueeze(-1)
        
        src = src.transpose(0, 1)
        gps_outputs = self.gps_transformer(src, mask3d)
        gps_outputs = gps_outputs.transpose(0, 1)
        gps_outputs = gps_outputs * mask2d
        
        # 候选路段编码
        candi_emb = self.emb_id(candi_ids)
        candi_feat_emb = self.candi_fc(candi_feats)
        candi_combined = candi_emb + candi_feat_emb
        
        # 注意力融合
        gps_outputs_bs = gps_outputs.permute(1, 0, 2)
        gps_query = gps_outputs_bs.unsqueeze(2)
        candi_keys = candi_combined
        
        scores = torch.matmul(gps_query, candi_keys.transpose(-2, -1)) / (self.hid_dim ** 0.5)
        scores = scores.squeeze(2)
        scores = scores.masked_fill(candi_masks == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        context = (candi_combined * attn_weights_expanded).sum(dim=2)
        
        fusion_outputs = gps_outputs_bs + context
        fusion_outputs = self.fusion_norm(fusion_outputs)
        fusion_outputs = self.fusion_dropout(fusion_outputs)
        fusion_outputs = fusion_outputs.permute(1, 0, 2)
        
        # 全局hidden state
        hidden = torch.sum(fusion_outputs * mask2d, dim=0) / src_len_tensor.unsqueeze(-1)
        hidden = hidden.unsqueeze(0)
        
        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))
        
        # 地图匹配概率
        fusion_for_mm = fusion_outputs.permute(1, 0, 2)
        fusion_expanded = fusion_for_mm.unsqueeze(2)
        mm_scores = torch.matmul(fusion_expanded, candi_combined.transpose(-2, -1))
        mm_scores = mm_scores.squeeze(2)
        mm_scores = mm_scores.masked_fill(candi_masks == 0, -1e9)
        mm_probs = torch.sigmoid(mm_scores)
        mm_probs = mm_probs.masked_fill(candi_masks == 0, 0)
        
        return fusion_outputs, hidden, mm_probs


class TopologyConstrainedDecoder(nn.Module):
    """带拓扑约束的解码器"""
    def __init__(self, parameters, rn, emb_id=None):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.id_size = parameters.id_size
        self.rn = rn
        self.emb_id = emb_id
        
        self.rate_flag = parameters.rate_flag
        self.dest_type = parameters.dest_type
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
        
        self.attn_W = nn.Linear(parameters.hid_dim * 2, parameters.hid_dim)
        self.attn_v = nn.Linear(parameters.hid_dim, 1, bias=False)
        
        if self.rate_flag:
            self.fc_rate_out = nn.Sequential(
                nn.Linear(parameters.hid_dim * 2, parameters.hid_dim * 2),
                nn.ReLU(),
                nn.Linear(parameters.hid_dim * 2, 1),
                nn.Sigmoid()
            )

    def forward(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len, 
                hidden, rid_features_dict, path, path_len, encoder_outputs,
                d_rids, d_rates, teacher_forcing_ratio):
        max_path_len = path.size(0)
        outputs_id = torch.zeros([max_trg_len, batch_size, max_path_len], 
                                device=hidden.device)
        outputs_rate = torch.zeros([max_trg_len, batch_size, 1], 
                                  device=hidden.device)
        
        path_mask = torch.ones(batch_size, max_path_len, device=hidden.device)
        path_mask = sequence_mask(path_mask, torch.tensor(path_len, device=hidden.device))
        
        path_emb = self.emb_id(path.permute(1, 0))
        
        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]
        
        for t in range(1, max_trg_len):
            teacher_force = random.random() < teacher_forcing_ratio
            
            rnn_input = self.emb_id(input_id)
            if self.rid_feats_flag:
                rnn_input = torch.cat([rnn_input, rid_features_dict[input_id]], dim=-1)
            if self.rate_flag:
                rnn_input = torch.cat((rnn_input, input_rate), dim=-1)
            if self.dest_type in [1, 2]:
                embed_drids = self.emb_id(d_rids)
                rnn_input = torch.cat((rnn_input, embed_drids), dim=-1)
                if self.rid_feats_flag:
                    rnn_input = torch.cat([rnn_input, rid_features_dict[d_rids]], dim=-1)
                if self.rate_flag:
                    rnn_input = torch.cat((rnn_input, d_rates), dim=-1)
            
            rnn_input = rnn_input.unsqueeze(0)
            
            output, hidden = self.rnn(rnn_input, hidden)
            
            query = hidden.squeeze(0)
            query_expanded = query.unsqueeze(1)
            combined = torch.cat([query_expanded.expand(-1, max_path_len, -1), path_emb], dim=-1)
            
            energy = torch.tanh(self.attn_W(combined))
            scores = self.attn_v(energy).squeeze(-1)
            scores = scores.masked_fill(path_mask == 0, -1e9)
            prediction_id = F.softmax(scores, dim=-1)
            
            outputs_id[t] = prediction_id
            
            if self.rate_flag:
                attn_weights = prediction_id.unsqueeze(1)
                weighted_path = torch.bmm(attn_weights, path_emb).squeeze(1)
                rate_input = torch.cat((query, weighted_path), dim=-1)
                prediction_rate = self.fc_rate_out(rate_input)
                outputs_rate[t] = prediction_rate
            else:
                outputs_rate[t] = 0.5
            
            if teacher_force:
                input_id = trg_id[t]
                input_rate = trg_rate[t]
            else:
                input_id = (F.one_hot(prediction_id.argmax(dim=1), max_path_len) * 
                          path.permute(1, 0)).sum(-1).long()
                input_rate = prediction_rate
        
        return outputs_id, outputs_rate


class UTT(nn.Module):
    """Unified Trajectory Transformer"""
    def __init__(self, parameters, rn=None):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.params = parameters
        
        self.emb_id = nn.Embedding(parameters.id_size, parameters.id_emb_dim)
        
        input_dim_gps = 3
        self.fc_in_gps = nn.Linear(input_dim_gps, parameters.hid_dim)
        
        self.encoder = JointEncoder(parameters, emb_id=self.emb_id)
        self.decoder = TopologyConstrainedDecoder(parameters, rn, emb_id=self.emb_id)
        
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

    def forward(self, src, src_len, src_seg_ids, candi_ids, candi_feats, candi_masks,
                trg_id, trg_rate, trg_len, pro_features, rid_features_dict, 
                path, path_len, d_rids, d_rates, teacher_forcing_ratio):
        max_trg_len = trg_id.size(0)
        batch_size = trg_id.size(1)
        
        gps_in = self.fc_in_gps(src.float())
        
        encoder_outputs, hidden, mm_probs = self.encoder(
            gps_in, src_len, src_seg_ids, candi_ids, candi_feats, 
            candi_masks, pro_features)
        
        outputs_id, outputs_rate = self.decoder(
            max_trg_len, batch_size, trg_id, trg_rate, trg_len,
            hidden, rid_features_dict, path, path_len, encoder_outputs,
            d_rids, d_rates, teacher_forcing_ratio)
        
        final_outputs_id = outputs_id[1:-1]
        final_outputs_rate = outputs_rate[1:-1]
        
        return mm_probs, final_outputs_id, final_outputs_rate


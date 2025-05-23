import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from models.layers import Encoder as Transformer, Attention, sequence_mask, sequence_mask3d
from utils.model_utils import gps2grid, get_normalized_t


class GPS2SegData(Dataset):

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

    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim

        input_dim = 3
        self.fc_in = nn.Linear(input_dim, parameters.hid_dim)
        self.transformer = Transformer(parameters.hid_dim, parameters.transformer_layers, heads=4)

    def forward(self, src, src_len):
        # src = [batch size, src len, 3]
        # src_len = [batch size]
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

        self.init_weights()  # learn how to init weights

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

        return outputs_id

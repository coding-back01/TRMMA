import argparse
import os
import pickle
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.demo3 import E2E3TrajData, End2End3Model, DA3Planner
from utils.map import RoadNetworkMapFull
from utils.model_utils import AttrDict
from utils.spatial_func import SPoint
from utils.mbr import MBR
from utils.model_utils import gps2grid
from utils.evaluation_utils import toseq


def collate_fn3(batch):
    data = []
    for item in batch:
        data.extend(item)

    da_routes, src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, trg_rids, trg_rates, \
    trg_rid_labels, d_rids, d_rates, candi_labels, candi_ids, candi_feats, candi_masks = zip(*data)

    import torch.nn.utils.rnn as rnn_utils
    import torch
    src_lengths = [len(seq) for seq in src_seqs]
    src_seqs = rnn_utils.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    src_pro_feas = torch.vstack(src_pro_feas).squeeze(-1)
    src_seg_seqs = rnn_utils.pad_sequence(src_seg_seqs, batch_first=True, padding_value=0)
    src_seg_feats = rnn_utils.pad_sequence(src_seg_feats, batch_first=True, padding_value=0)
    trg_lengths = [len(seq) for seq in trg_rids]
    trg_rids = rnn_utils.pad_sequence(trg_rids, batch_first=True, padding_value=0)
    trg_rates = rnn_utils.pad_sequence(trg_rates, batch_first=True, padding_value=0)

    da_lengths = [len(seq) for seq in da_routes]
    da_routes = rnn_utils.pad_sequence(da_routes, batch_first=True, padding_value=0)
    da_pos = [torch.tensor(list(range(1, item + 1))) for item in da_lengths]
    da_pos = rnn_utils.pad_sequence(da_pos, batch_first=True, padding_value=0)
    d_rids = torch.vstack(d_rids).squeeze(-1)
    d_rates = torch.vstack(d_rates)
    max_da = max(da_lengths)
    trg_rid_labels = list(trg_rid_labels)
    for i in range(len(trg_rid_labels)):
        if trg_rid_labels[i].shape[1] < max_da:
            tmp = torch.zeros(trg_rid_labels[i].shape[0], max_da - trg_rid_labels[i].shape[1]) + 1e-6
            trg_rid_labels[i] = torch.cat([trg_rid_labels[i], tmp], dim=-1)
    trg_rid_labels = rnn_utils.pad_sequence(trg_rid_labels, batch_first=True, padding_value=1e-6)

    candi_labels = rnn_utils.pad_sequence(candi_labels, batch_first=True, padding_value=0)
    candi_ids = rnn_utils.pad_sequence(candi_ids, batch_first=True, padding_value=0)
    candi_feats = rnn_utils.pad_sequence(candi_feats, batch_first=True, padding_value=0)
    candi_masks = rnn_utils.pad_sequence(candi_masks, batch_first=True, padding_value=0)

    return (
        src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths,
        trg_rids, trg_rates, trg_lengths, trg_rid_labels,
        da_routes, da_lengths, da_pos, d_rids, d_rates,
        candi_labels, candi_ids, candi_feats, candi_masks
    )


def main():
    parser = argparse.ArgumentParser(description='E2E3 inference')
    parser.add_argument('--city', type=str, default='porto')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else 'cpu')
    city = args.city
    if city in ["PT", "porto", "porto1", "porto2", "porto3", "porto4", "porto5", "porto7", "porto9", "pt1", "pt3", "pt5", "pt10", "pt20", "pt40", "pt60", "pt80"]:
        zone_range = [41.1395, -8.6911, 41.1864, -8.5521]
        ts = 15
        utc = 1
    elif city in ["beijing", "beijing1", "beijing2", "beijing3", "beijing4", "beijing5", "beijing7", "beijing9", "bj1", "bj3", "bj5", "bj10", "bj20", "bj40", "bj60", "bj80"]:
        zone_range = [39.7547, 116.1994, 40.0244, 116.5452]
        ts = 60
        utc = 0
    elif city in ["chengdu", "chengdu1", "chengdu2", "chengdu3", "chengdu4", "chengdu5", "chengdu7", "chengdu9", "cd1", "cd3", "cd5", "cd10", "cd20", "cd40", "cd60", "cd80"]:
        zone_range = [30.6443, 104.0288, 30.7416, 104.1375]
        ts = 12
        utc = 8
    elif city in ["xian", "xian1", "xian2", "xian3", "xian4", "xian5", "xian7", "xian9", "xa1", "xa3", "xa5", "xa10", "xa20", "xa40", "xa60", "xa80"]:
        zone_range = [34.2060, 108.9058, 34.2825, 109.0049]
        ts = 12
        utc = 8
    else:
        raise NotImplementedError

    map_root = os.path.join("data", args.city, "roadnet")
    rn = RoadNetworkMapFull(map_root, zone_range=zone_range, unit_length=50)
    dam = DA3Planner(os.path.join("data", args.city), rn.valid_edge_cnt_one - 1, utc)

    model = torch.load(args.model_path, map_location=device)
    model.eval()

    traj_root = os.path.join("data", args.city)
    dataset = E2E3TrajData(rn, traj_root, MBR(zone_range[0], zone_range[1], zone_range[2], zone_range[3]), AttrDict({'grid_size':50,'time_span':ts,'keep_ratio':0.125,'candi_size':10,'search_dist':50,'beta':15,'gps_flag':True,'small':False,'planner':'da','dam_root':os.path.join('data',args.city),'id_size':rn.valid_edge_cnt_one,'utc':utc}), 'test')

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn3, num_workers=2, pin_memory=True)

    rid_features_dict = torch.from_numpy(rn.get_rid_rnfea_dict(dam, ts)).to(device)

    data = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, \
            trg_rids, trg_rates, trg_lengths, trg_rid_labels, \
            da_routes, da_lengths, da_pos, d_rids, d_rates, \
            candi_onehots, candi_ids, candi_feats, candi_masks = batch

            sel_src_seqs = src_seqs.to(device, non_blocking=True)
            sel_src_lens = src_lengths
            sel_candi_ids = candi_ids.to(device, non_blocking=True)
            sel_candi_feats = candi_feats.to(device, non_blocking=True)
            sel_candi_masks = candi_masks.to(device, non_blocking=True)

            src_pro_feas = src_pro_feas.to(device, non_blocking=True)
            src_seqs_tr = sel_src_seqs.permute(1, 0, 2)
            trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
            trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)
            trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)

            da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
            da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
            d_rids = d_rids.to(device, non_blocking=True)
            d_rates = d_rates.to(device, non_blocking=True)

            src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
            src_seg_feats = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)

            outputs = model(
                sel_src_seqs, sel_src_lens, sel_candi_ids, sel_candi_feats, sel_candi_masks,
                src_seqs_tr, src_lengths, trg_rids, trg_rates, trg_lengths,
                src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos,
                src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio=-1,
                tau=None, trg_rid_labels=trg_rid_labels
            )

            out_ids = outputs['out_ids']
            out_rates = outputs['out_rates']
            routes_bs_len = da_routes.permute(1, 0)
            T = out_ids.shape[0]
            onehot = F.one_hot(out_ids.argmax(-1), da_routes.shape[0])
            expanded_routes = routes_bs_len.unsqueeze(1).repeat(1, T, 1).permute(1, 0, 2)
            output_tmp = (onehot * expanded_routes).sum(dim=-1)

            output_rates = out_rates.squeeze(2)
            gt_rates = trg_rates.squeeze(2)
            trg_lengths_sub = [length - 2 for length in trg_lengths]

            # collect
            pred_ids = output_tmp
            results = []
            predict_id = pred_ids.permute(1, 0).detach().cpu().tolist()
            predict_rate = output_rates.permute(1, 0).detach().cpu().tolist()
            target_id = trg_rids[1:-1].permute(1, 0).detach().cpu().tolist()
            target_rate = gt_rates[1:-1].permute(1, 0).detach().cpu().tolist()
            routes = da_routes.permute(1, 0).detach().cpu().tolist()
            for ps, pr, ti, tr, length, route, route_len in zip(predict_id, predict_rate, target_id, target_rate, trg_lengths_sub, routes, da_lengths):
                results.append([ps[:length], pr[:length], ti[:length], tr[:length], route[:route_len]])
            data.extend(results)

            if len(loader) >= 10 and (i + 1) % (len(loader) // 10) == 0:
                print("==> Test:", (i + 1) // (len(loader) // 10))

    save_dir = os.path.dirname(args.model_path)
    pickle.dump(data, open(os.path.join(save_dir, 'infer_output_e2e3.pkl'), "wb"))
    print('Saved to:', os.path.join(save_dir, 'infer_output_e2e3.pkl'))


if __name__ == '__main__':
    main()




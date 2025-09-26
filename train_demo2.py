import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.evaluation_utils import calc_metrics, toseq

import random
import time
import logging

import os
import argparse
import pickle

import torch
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.optim as optim
from utils.map import RoadNetworkMapFull
from utils.spatial_func import SPoint
from utils.mbr import MBR
from models.demo2 import E2ETrajData, E2ETrajTestData, End2EndModel, E2ELoss, DAPlanner
from utils.model_utils import AttrDict, gps2grid
from tqdm import tqdm
from collections import Counter
import numpy as np


def collate_fn(batch):
    data = []
    for item in batch:
        data.extend(item)

    da_routes, src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, trg_rids, trg_rates, \
    trg_rid_labels, d_rids, d_rates, candi_labels, candi_ids, candi_feats, candi_masks = zip(*data)

    src_lengths = [len(seq) for seq in src_seqs]
    src_seqs = rnn_utils.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    src_pro_feas = torch.vstack(src_pro_feas).squeeze(-1).long()
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


def collate_fn_test(batch):
    # 测试集的 __getitem__ 返回单样本（15 个字段）的列表，这里不要展平，直接按 batch 聚合
    da_routes, src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, trg_gps_seqs, trg_rids, trg_rates, \
    trg_rid_labels, d_rids, d_rates, candi_labels, candi_ids, candi_feats, candi_masks = zip(*batch)

    src_lengths = [len(seq) for seq in src_seqs]
    src_seqs = rnn_utils.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    # 可能为 0 维张量（标量），使用 stack 聚合到 [bs]，供 embedding 使用
    src_pro_feas = torch.vstack(src_pro_feas).squeeze(-1).long()
    src_seg_seqs = rnn_utils.pad_sequence(src_seg_seqs, batch_first=True, padding_value=0)
    src_seg_feats = rnn_utils.pad_sequence(src_seg_feats, batch_first=True, padding_value=0)
    trg_lengths = [len(seq) for seq in trg_gps_seqs]
    trg_gps_seqs = rnn_utils.pad_sequence(trg_gps_seqs, batch_first=True, padding_value=0)
    trg_rids = rnn_utils.pad_sequence(trg_rids, batch_first=True, padding_value=0)
    trg_rates = rnn_utils.pad_sequence(trg_rates, batch_first=True, padding_value=0)

    da_lengths = [len(seq) for seq in da_routes]
    da_routes = rnn_utils.pad_sequence(da_routes, batch_first=True, padding_value=0)
    da_pos = [torch.tensor(list(range(1, item + 1))) for item in da_lengths]
    da_pos = rnn_utils.pad_sequence(da_pos, batch_first=True, padding_value=0)
    # 末端 rid 处理，与训练时保持一致
    d_rids = torch.vstack(d_rids).squeeze(-1)
    d_rates = torch.vstack(d_rates)
    max_da = max(da_lengths)
    trg_rid_labels = list(trg_rid_labels)
    for i in range(len(trg_rid_labels)):
        if trg_rid_labels[i].shape[1] < max_da:
            tmp = torch.zeros(trg_rid_labels[i].shape[0], max_da - trg_rid_labels[i].shape[1])
            trg_rid_labels[i] = torch.cat([trg_rid_labels[i], tmp], dim=-1)
    trg_rid_labels = rnn_utils.pad_sequence(trg_rid_labels, batch_first=True, padding_value=0)

    candi_labels = rnn_utils.pad_sequence(candi_labels, batch_first=True, padding_value=0)
    candi_ids = rnn_utils.pad_sequence(candi_ids, batch_first=True, padding_value=0)
    candi_feats = rnn_utils.pad_sequence(candi_feats, batch_first=True, padding_value=0)
    candi_masks = rnn_utils.pad_sequence(candi_masks, batch_first=True, padding_value=0)

    return (
        src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths,
        trg_gps_seqs, trg_rids, trg_rates, trg_lengths, trg_rid_labels,
        da_routes, da_lengths, da_pos, d_rids, d_rates,
        candi_labels, candi_ids, candi_feats, candi_masks
    )

def train(model, iterator, optimizer, criterion, rid_features_dict, parameters, device):
    epoch_ttl_loss = 0
    epoch_selector_loss = 0
    epoch_id_loss = 0
    epoch_rate_loss = 0

    time_ttl = 0
    time_move = 0
    time_forward = 0
    time_loss = 0
    time_zero = 0
    time_gradient = 0
    time_update = 0
    time_ttl2 = 0

    t0 = time.time()

    model.train()
    for i, batch in enumerate(iterator):
        t1 = time.time()
        # 解包 batch
        src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, \
        trg_rids, trg_rates, trg_lengths, trg_rid_labels, \
        da_routes, da_lengths, da_pos, d_rids, d_rates, \
        candi_onehots, candi_ids, candi_feats, candi_masks = batch

        # 统一处理GPS数据，和TRMMA保持一致
        src_pro_feas = src_pro_feas.to(device, non_blocking=True)
        trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
        src_seqs = src_seqs.permute(1, 0, 2).to(device, non_blocking=True)  # [src_len, bs, 3] - 和TRMMA一致
        trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
        trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)
        src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
        src_seg_feats = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)
        da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
        da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
        d_rids = d_rids.to(device, non_blocking=True)
        d_rates = d_rates.to(device, non_blocking=True)
        
        # Selector专用数据
        sel_candi_ids = candi_ids.to(device, non_blocking=True)
        sel_candi_feats = candi_feats.to(device, non_blocking=True)
        sel_candi_masks = candi_masks.to(device, non_blocking=True)

        time_move += time.time() - t1
        t2 = time.time()

        outputs = model(
            src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
            src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos,
            src_seg_seqs, src_seg_feats, d_rids, d_rates, parameters.tf_ratio,
            sel_candi_ids, sel_candi_feats, sel_candi_masks
        )

        time_forward += time.time() - t2
        t3 = time.time()

        labels = {
            'selector_onehot': candi_onehots.to(device, non_blocking=True),
            'trg_labels': trg_rid_labels,
            'trg_rates': trg_rates
        }
        lengths = {
            'trg_lengths': trg_lengths,
            'da_lengths': da_lengths
        }

        ttl_loss, loss_dict = criterion(outputs, labels, lengths)

        time_loss += time.time() - t3
        t4 = time.time()

        optimizer.zero_grad(set_to_none=True)
        time_zero += time.time() - t4
        t5 = time.time()

        ttl_loss.backward()
        time_gradient += time.time() - t5
        t6 = time.time()

        optimizer.step()
        time_update += time.time() - t6

        epoch_ttl_loss += ttl_loss.item()

        epoch_selector_loss += loss_dict['selector']
        epoch_id_loss += loss_dict['id']
        epoch_rate_loss += loss_dict['rate']

        if len(iterator) >= 10 and (i + 1) % (len(iterator) // 10) == 0:
            print("==>{}: {}, {}, {}, {}".format((i + 1) // (len(iterator) // 10), epoch_ttl_loss / (i + 1), epoch_selector_loss / (i + 1), epoch_id_loss / (i + 1), epoch_rate_loss / (i + 1)))
        time_ttl2 += time.time() - t1
    time_ttl += time.time() - t0
    # print(time_ttl, time_ttl - time_ttl2, time_move, time_forward, time_loss, time_zero, time_gradient, time_update)
    # print(np.sum(model.trmma.timer6), np.sum(model.trmma.timer1), np.sum(model.trmma.timer2), np.sum(model.trmma.timer3), np.sum(model.trmma.timer4), np.sum(model.trmma.timer5))
    
    return epoch_ttl_loss / len(iterator), epoch_selector_loss / len(iterator), epoch_id_loss / len(iterator), epoch_rate_loss / len(iterator)


def evaluate(model, iterator, criterion, rid_features_dict, parameters, device):
    epoch_ttl_loss = 0
    epoch_selector_loss = 0
    epoch_id_loss = 0
    epoch_rate_loss = 0

    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, \
            trg_rids, trg_rates, trg_lengths, trg_rid_labels, \
            da_routes, da_lengths, da_pos, d_rids, d_rates, \
            candi_onehots, candi_ids, candi_feats, candi_masks = batch

            src_pro_feas = src_pro_feas.to(device, non_blocking=True)
            trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
            src_seqs = src_seqs.permute(1, 0, 2).to(device, non_blocking=True)  # [src_len, bs, 3] - 和TRMMA一致
            trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
            trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)

            da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
            da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
            d_rids = d_rids.to(device, non_blocking=True)
            d_rates = d_rates.to(device, non_blocking=True)

            src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
            src_seg_feats = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)
            
            # Selector专用数据
            sel_candi_ids = candi_ids.to(device, non_blocking=True)
            sel_candi_feats = candi_feats.to(device, non_blocking=True)
            sel_candi_masks = candi_masks.to(device, non_blocking=True)

            outputs = model(
                src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos,
                src_seg_seqs, src_seg_feats, d_rids, d_rates, 0,
                sel_candi_ids, sel_candi_feats, sel_candi_masks
            )

            labels = {
                'selector_onehot': candi_onehots.to(device, non_blocking=True),
                'trg_labels': trg_rid_labels,
                'trg_rates': trg_rates
            }
            lengths = {
                'trg_lengths': trg_lengths,
                'da_lengths': da_lengths
            }

            ttl_loss, loss_dict = criterion(outputs, labels, lengths)

            epoch_ttl_loss += ttl_loss.item()
            epoch_selector_loss += loss_dict['selector']
            epoch_id_loss += loss_dict['id']
            epoch_rate_loss += loss_dict['rate']

        print((epoch_ttl_loss) / (i + 1), epoch_selector_loss / (i + 1), epoch_id_loss / (i + 1), epoch_rate_loss / (i + 1))

    return epoch_ttl_loss / len(iterator), epoch_selector_loss / len(iterator), epoch_id_loss / len(iterator), epoch_rate_loss / len(iterator)


def get_results(predict_id, predict_rate, target_id, target_rate, target_gps, trg_len, routes, route_lengths, inverse_flag=True):

    if inverse_flag:
        predict_id = predict_id - 1
        target_id = target_id - 1
        routes = routes - 1

    predict_id = predict_id.permute(1, 0).detach().cpu().tolist()
    predict_rate = predict_rate.permute(1, 0).detach().cpu().tolist()
    target_gps = target_gps.permute(1, 0, 2).detach().cpu().tolist()
    target_id = target_id.permute(1, 0).detach().cpu().tolist()
    target_rate = target_rate.permute(1, 0).detach().cpu().tolist()
    routes = routes.permute(1, 0).detach().cpu().tolist()

    results = []
    for pred_seg, pred_rate, trg_id, trg_rate, trg_gps, length, route, route_len in zip(
            predict_id, predict_rate, target_id, target_rate, target_gps, trg_len, routes, route_lengths):
        results.append([
            pred_seg[:length], pred_rate[:length],
            trg_id[:length], trg_rate[:length], trg_gps[:length],
            route[:route_len]
        ])
    return results


def infer(model, iterator, rid_features_dict, device):
    data = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, \
            trg_gps_seqs, trg_rids, trg_rates, trg_lengths, trg_rid_labels, \
            da_routes, da_lengths, da_pos, d_rids, d_rates, \
            candi_onehots, candi_ids, candi_feats, candi_masks = batch

            src_pro_feas = src_pro_feas.to(device, non_blocking=True)
            trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
            src_seqs = src_seqs.permute(1, 0, 2).to(device, non_blocking=True)  # [src_len, bs, 3] - 和TRMMA一致
            trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
            trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)

            da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
            da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
            d_rids = d_rids.to(device, non_blocking=True)
            d_rates = d_rates.to(device, non_blocking=True)

            src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
            src_seg_feats = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)
            
            # Selector专用数据
            sel_candi_ids = candi_ids.to(device, non_blocking=True)
            sel_candi_feats = candi_feats.to(device, non_blocking=True)
            sel_candi_masks = candi_masks.to(device, non_blocking=True)

            outputs = model(
                src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos,
                src_seg_seqs, src_seg_feats, d_rids, d_rates, -1,
                sel_candi_ids, sel_candi_feats, sel_candi_masks
            )

            out_ids = outputs['out_ids']
            out_rates = outputs['out_rates']
            output_tmp = (F.one_hot(out_ids.argmax(-1), da_routes.shape[0]) * da_routes.permute(1, 0).unsqueeze(1).repeat(1, trg_rid_labels.shape[0], 1).permute(1, 0, 2)).sum(dim=-1)

            output_rates = out_rates.squeeze(2)
            gt_rates = trg_rates.squeeze(2)
            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2)
            trg_lengths_sub = [length - 2 for length in trg_lengths]

            results = get_results(
                output_tmp, output_rates, trg_rids[1:-1], gt_rates[1:-1], trg_gps_seqs[1:-1],
                trg_lengths_sub, da_routes, da_lengths
            )
            data.extend(results)

            if len(iterator) >= 10 and (i + 1) % (len(iterator) // 10) == 0:
                print("==> Test: {}".format((i + 1) // (len(iterator) // 10)))

    return data


def main():
    parser = argparse.ArgumentParser(description='E2E Selector+Reconstructor')
    parser.add_argument('--city', type=str, default='porto')
    parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')
    parser.add_argument('--tf_ratio', type=float, default=1, help='teaching ratio in float')
    parser.add_argument('--lambda_selector', type=float, default=1.0, help='weight for selector bce')
    parser.add_argument('--lambda1', type=float, default=10, help='weight for seg bce')
    parser.add_argument('--lambda2', type=float, default=5, help='weight for rate l1')
    parser.add_argument('--hid_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--transformer_layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument('--model_old_path', type=str, default='', help='old model path')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--eid_cate', type=str, default='gps2seg')
    parser.add_argument('--inferred_seg_path', type=str, default='')  # 推断分段路径，字符串类型，默认空
    parser.add_argument('--direction_flag', action='store_true', default=True)
    parser.add_argument('--attn_flag', action='store_true', default=True)
    parser.add_argument("--candi_size", type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--only_direction', action='store_true')
    parser.add_argument('--da_route_flag', action='store_true', default=True)
    parser.add_argument('--srcseg_flag', action='store_true', default=True)
    parser.add_argument('--gps_flag', action='store_true', default=False) 
    parser.add_argument('--planner', type=str, default='da')

    opts = parser.parse_args()
    print(opts)

    device = torch.device(f"cuda:{opts.gpu_id}" if torch.cuda.is_available() else 'cpu')
    print(f"Use GPU: cuda {opts.gpu_id}")

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    print('device', device)
    
    model_save_root = f'./model/E2E/{opts.city}/'
    model_save_path = model_save_root + 'E2E_' + opts.city + '_' + 'keep-ratio_' + str(opts.keep_ratio) + '_' + time.strftime("%Y%m%d_%H%M%S") + '/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=os.path.join(model_save_path, 'log.txt'),
                        filemode='a')

    city = opts.city
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

    print('Preparing data...')
    map_root = os.path.join("data", opts.city, "roadnet")
    rn = RoadNetworkMapFull(map_root, zone_range=zone_range, unit_length=50)

    args = AttrDict()
    args_dict = {
        'device': device,
        'transformer_layers': opts.transformer_layers,
        'heads': opts.heads,
        'tandem_fea_flag': True,
        'pro_features_flag': True,
        'srcseg_flag': opts.srcseg_flag,
        'da_route_flag': opts.da_route_flag,
        'rate_flag': True,
        'prog_flag': False,
        'dest_type': 2,
        'gps_flag': opts.gps_flag,
        'rid_feats_flag': True,
        'learn_pos': True,

        # constraint
        'search_dist': 50,
        'beta': 15,
        'gamma': 30,

        # extra info
        'rid_fea_dim': 18,
        'pro_input_dim': 48,
        'pro_output_dim': 8,

        # MBR
        'min_lat': zone_range[0],
        'min_lng': zone_range[1],
        'max_lat': zone_range[2],
        'max_lng': zone_range[3],

        # input
        'city': opts.city,
        'keep_ratio': opts.keep_ratio,
        'grid_size': 50,
        'time_span': ts,

        # model
        'hid_dim': opts.hid_dim,
        'id_emb_dim': opts.hid_dim,
        'dropout': 0.1,
        'id_size': rn.valid_edge_cnt_one,

        # train
        'lambda1': opts.lambda1,
        'lambda2': opts.lambda2,
        'n_epochs': opts.epochs,
        'batch_size': opts.batch_size,
        'learning_rate': opts.lr,
        'lr_step': 2,
        'lr_decay': 0.8,
        'tf_ratio': opts.tf_ratio,
        'decay_flag': True,
        'decay_ratio': 0.9,
        'clip': 1,
        'log_step': 1,

        'utc': utc,
        'small': opts.small,
        'dam_root': os.path.join("data", opts.city),
        'eid_cate': opts.eid_cate,
        'inferred_seg_path': opts.inferred_seg_path,  # 推断段路径
        'planner': opts.planner,
        'direction_flag': opts.direction_flag,
        'attn_flag': opts.attn_flag,
        'candi_size': opts.candi_size,
        'only_direction': opts.only_direction
    }
    args.update(args_dict)

    mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
    args.grid_num = gps2grid(SPoint(args.max_lat, args.max_lng), mbr, args.grid_size)
    args.grid_num = (args.grid_num[0] + 1, args.grid_num[1] + 1)

    print(args)
    logging.info(args_dict)

    dam = DAPlanner(args.dam_root, args.id_size - 1, args.utc)
    rid_features_dict = torch.from_numpy(rn.get_rid_rnfea_dict(dam, ts)).to(device)

    traj_root = os.path.join("data", args.city)

    train_dataset = E2ETrajData(rn, traj_root, mbr, args, 'train')
    valid_dataset = E2ETrajData(rn, traj_root, mbr, args, 'valid')
    print('training dataset shape: ' + str(len(train_dataset)))
    print('validation dataset shape: ' + str(len(valid_dataset)))
    logging.info('Finish data preparing.')
    logging.info('training dataset shape: ' + str(len(train_dataset)))
    logging.info('validation dataset shape: ' + str(len(valid_dataset)))

    train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=opts.num_worker, pin_memory=False)
    valid_iterator = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=opts.num_worker, pin_memory=False)

    model = End2EndModel(args).to(device)

    print('model', str(model))
    logging.info('model' + str(model))

    ls_train_loss, ls_train_mma_loss, ls_train_id_loss, ls_train_rate_loss = [], [], [], []
    ls_valid_loss, ls_valid_mma_loss, ls_valid_id_loss, ls_valid_rate_loss = [], [], [], []

    best_valid_loss = float('inf')
    best_epoch = 0
    
    tb_writer = SummaryWriter(log_dir=os.path.join(model_save_path, 'tensorboard'))

    lr = args.learning_rate
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_step, factor=args.lr_decay, threshold=1e-3)
    criterion = E2ELoss(lambda_selector=opts.lambda_selector, lambda_id=args.lambda1, lambda_rate=args.lambda2)

    stopping_count = 0
    train_times = []
    for epoch in tqdm(range(args.n_epochs), desc='epoch num'):
        start_time = time.time()

        print("==> training keep_ratio={}, tf_ratio={}, lr={}...".format(args.keep_ratio, args.tf_ratio, lr))
        t_train = time.time()
        train_loss, train_selector, train_id, train_rate = train(model, train_iterator, optimizer, criterion, rid_features_dict, args, device)
        end_train = time.time()
        print("training: {}".format(end_train - t_train))

        ls_train_loss.append(train_loss)
        ls_train_mma_loss.append(train_selector)
        ls_train_id_loss.append(train_id)
        ls_train_rate_loss.append(train_rate)

        print("==> validating...")
        t_valid = time.time()
        valid_loss, valid_selector, valid_id, valid_rate = evaluate(model, valid_iterator, criterion, rid_features_dict, args, device)
        print("validating: {}".format(time.time() - t_valid))

        ls_valid_loss.append(valid_loss)
        ls_valid_mma_loss.append(valid_selector)
        ls_valid_id_loss.append(valid_id)
        ls_valid_rate_loss.append(valid_rate)

        end_time = time.time()
        epoch_secs = end_time - start_time
        train_times.append(end_train - t_train)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, os.path.join(model_save_path, 'val-best-model.pt'))
            best_epoch = epoch
            stopping_count = 0
        else:
            stopping_count += 1

        tb_writer.add_scalars('Train_loss', {'total': train_loss, 'RID': train_id, 'Rate': train_rate, 'Selector': train_selector}, epoch)
        tb_writer.add_scalars('Valid_loss', {'total': valid_loss, 'RID': valid_id, 'Rate': valid_rate, 'Selector': valid_selector}, epoch)
        tb_writer.add_scalar('learning_rate', lr, epoch)
        tb_writer.add_scalars('TTL_loss', {'Train': train_loss, 'Valid': valid_loss}, epoch)
        tb_writer.add_scalars('Seg_loss', {'Train': train_id, 'Valid': valid_id}, epoch)
        tb_writer.add_scalars('Rate_loss', {'Train': train_rate, 'Valid': valid_rate}, epoch)
        tb_writer.add_scalars('Selector_loss', {'Train': train_selector, 'Valid': valid_selector}, epoch)

        if (epoch % args.log_step == 0) or (epoch == args.n_epochs - 1):
            logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_secs) + 's')
            logging.info('Epoch: ' + str(epoch + 1) + ' TF Ratio: ' + str(args.tf_ratio) + ' Keep Ratio: ' + str(args.keep_ratio))
            logging.info('\tTrain Total:' + str(train_loss) + '\tSelector:' + str(train_selector) + '\tSeg:' + str(train_id) + '\tRate:' + str(train_rate))
            logging.info('\tValid Total:' + str(valid_loss) + '\tSelector:' + str(valid_selector) + '\tSeg:' + str(valid_id) + '\tRate:' + str(valid_rate))
            torch.save(model, os.path.join(model_save_path, 'train-mid-model.pt'))
        if args.decay_flag:
            args.tf_ratio = args.tf_ratio * args.decay_ratio

        scheduler.step(valid_loss)
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']

        if lr <= 0.9 * 1e-5:
            print("==> [Info] Early Stop since lr is too small After Epoch {}.".format(epoch))
            break
        if stopping_count >= 5:
            print("==> [Info] Early Stop After Epoch {}.".format(epoch))
            break
    tb_writer.close()
    logging.info('Best Epoch: {}, {}'.format(best_epoch, best_valid_loss))
    print('==> Best Epoch: {}, {}'.format(best_epoch, best_valid_loss))
    logging.info('==> Training Time: {}, {}, {}, {}'.format(np.sum(train_times) / 3600, np.mean(train_times), np.min(train_times), np.max(train_times)))
    print('==> Training Time: {}, {}, {}, {}'.format(np.sum(train_times) / 3600, np.mean(train_times), np.min(train_times), np.max(train_times)))

    test_dataset = E2ETrajTestData(rn, traj_root, mbr, args)
    print('testing dataset shape: ' + str(len(test_dataset)))
    logging.info('testing dataset shape: ' + str(len(test_dataset)))

    test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_test, num_workers=opts.num_worker, pin_memory=True)

    model = torch.load(os.path.join(model_save_path, 'val-best-model.pt'), map_location=device)
    print('==> Model Loaded')

    print('==> Starting Prediction...')
    start_time = time.time()
    data = infer(model, test_iterator, rid_features_dict, device)
    end_time = time.time()
    epoch_secs = end_time - start_time
    print('Time: ' + str(epoch_secs) + 's')
    logging.info('Inference Time: {}, {}, {}'.format(end_time - start_time, (end_time - start_time) / max(1, len(test_dataset)) * 1000, len(test_dataset) / max(1e-9, (end_time - start_time))))
    print('Inference Time: {}, {}, {}'.format(end_time - start_time, (end_time - start_time) / max(1, len(test_dataset)) * 1000, len(test_dataset) / max(1e-9, (end_time - start_time))))
    pickle.dump(data, open(os.path.join(model_save_path, 'infer_output_e2e_{}_{}.pkl'.format(opts.planner, opts.eid_cate)), "wb"))

    outputs = []
    for pred_seg, pred_rate, trg_id, trg_rate, trg_gps, route in data:
        pred_gps = toseq(rn, pred_seg, pred_rate, route, dam.seg_info)
        outputs.append([pred_gps, pred_seg, trg_gps, trg_id])

    test_trajs = pickle.load(open(os.path.join(traj_root, 'test_output.pkl'), "rb"))
    groups = Counter(test_dataset.groups)
    nums = []
    for i in range(len(test_trajs)):
        nums.append(groups[i])
    # results = []
    # for traj, num, src_mm in zip(test_trajs, nums, test_dataset.src_mms):
    #     tmp_all = outputs[:num]
    #     low_idx = traj.low_idx
    #     gps, segs, _ = zip(*src_mm)
    #     predict_ids = [segs[0]]
    #     predict_gps = [gps[0]]
    #     # predict_ids = []
    #     # predict_gps = []
    #     pointer = -1
    #     for p1_idx, p2_idx, seg, latlng in zip(low_idx[:-1], low_idx[1:], segs[1:], gps[1:]):
    #         if (p1_idx + 1) < p2_idx:
    #             pointer += 1
    #             tmp = tmp_all[pointer]
    #             predict_gps.extend(tmp[0])
    #             predict_ids.extend(tmp[1])
    #         predict_ids.append(seg)
    #         predict_gps.append(latlng)
    #     outputs = outputs[num:]

    #     mm_gps_seq = []
    #     mm_eids = []
    #     for i, pt in enumerate(traj.pt_list):
    #         candi_pt = pt.data['candi_pt']
    #         mm_eids.append(candi_pt.eid)
    #         # if i not in low_idx:
    #         mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
    #     assert len(predict_gps) == len(mm_gps_seq) == len(predict_ids) == len(mm_eids)
    #     results.append([predict_gps, predict_ids, mm_gps_seq, mm_eids])
    # pickle.dump(results, open(os.path.join(model_save_path, 'recovery_output_e2e_{}_{}.pkl'.format(opts.planner, opts.eid_cate)), "wb"))
    
    print('==> Starting Evaluation...')

    epoch_id1_loss = []
    epoch_recall_loss = []
    epoch_precision_loss = []
    epoch_f1_loss = []
    epoch_mae_loss = []
    epoch_rmse_loss = []
    for pred_gps, pred_seg, trg_gps, trg_id in outputs:
        recall, precision, f1, loss_ids1, loss_mae, loss_rmse = calc_metrics(pred_seg, pred_gps, trg_id, trg_gps)
        epoch_id1_loss.append(loss_ids1)
        epoch_recall_loss.append(recall)
        epoch_precision_loss.append(precision)
        epoch_f1_loss.append(f1)
        epoch_mae_loss.append(loss_mae)
        epoch_rmse_loss.append(loss_rmse)

    test_id_recall, test_id_precision, test_id_f1, test_id_acc, test_mae, test_rmse = np.mean(epoch_recall_loss), np.mean(epoch_precision_loss), np.mean(epoch_f1_loss), np.mean(epoch_id1_loss), np.mean(epoch_mae_loss), np.mean(epoch_rmse_loss)
    print(test_id_recall, test_id_precision, test_id_f1, test_id_acc, test_mae, test_rmse)

    logging.info('Time: ' + str(epoch_secs) + 's')
    logging.info('\tTest RID Acc:' + str(test_id_acc) +
                 '\tTest RID Recall:' + str(test_id_recall) +
                 '\tTest RID Precision:' + str(test_id_precision) +
                 '\tTest RID F1 Score:' + str(test_id_f1) +
                 '\tTest MAE Loss:' + str(test_mae) +
                 '\tTest RMSE Loss:' + str(test_rmse))


if __name__ == '__main__':
    main()




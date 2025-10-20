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
from models.utt import UTTData, UTTTestData, UTT
from models.trmma import DAPlanner
from utils.model_utils import AttrDict, gps2grid
from tqdm import tqdm
import numpy as np
from collections import Counter


def collate_fn(data0):
    """训练/验证数据整理函数"""
    data = []
    for item in data0:
        data.extend(item)
    
    (src_grid_seqs, src_pro_feas, src_seg_seqs,
     trg_gps_seqs, trg_rids, trg_rates,
     candi_labels, candi_ids, candi_feats, candi_masks,
     paths, d_rids, d_rates) = zip(*data)
    
    # 源序列处理
    src_lengths = [len(seq) for seq in src_grid_seqs]
    src_grid_seqs = rnn_utils.pad_sequence(src_grid_seqs, batch_first=True, padding_value=0)
    src_pro_feas = torch.vstack(src_pro_feas).squeeze(-1)
    src_seg_seqs = rnn_utils.pad_sequence(src_seg_seqs, batch_first=True, padding_value=0)
    
    # 候选路段处理
    candi_labels = rnn_utils.pad_sequence(candi_labels, batch_first=True, padding_value=0)
    candi_ids = rnn_utils.pad_sequence(candi_ids, batch_first=True, padding_value=0)
    candi_feats = rnn_utils.pad_sequence(candi_feats, batch_first=True, padding_value=0)
    candi_masks = rnn_utils.pad_sequence(candi_masks, batch_first=True, padding_value=0)
    
    # 目标序列处理
    trg_lengths = [len(seq) for seq in trg_rids]
    trg_gps_seqs = rnn_utils.pad_sequence(trg_gps_seqs, batch_first=True, padding_value=0)
    trg_rids = rnn_utils.pad_sequence(trg_rids, batch_first=True, padding_value=0)
    trg_rates = rnn_utils.pad_sequence(trg_rates, batch_first=True, padding_value=0)
    
    # 路径处理
    path_lengths = [len(seq) for seq in paths]
    paths = rnn_utils.pad_sequence(paths, batch_first=True, padding_value=0)
    
    d_rids = torch.vstack(d_rids).squeeze(-1)
    d_rates = torch.vstack(d_rates)
    
    return (src_grid_seqs, src_pro_feas, src_seg_seqs, src_lengths,
            candi_labels, candi_ids, candi_feats, candi_masks,
            trg_gps_seqs, trg_rids, trg_rates, trg_lengths,
            paths, path_lengths, d_rids, d_rates)


def collate_fn_test(data):
    """测试数据整理函数"""
    (src_grid_seqs, src_pro_feas, src_seg_seqs,
     trg_gps_seqs, trg_rids, trg_rates,
     paths, d_rids, d_rates) = zip(*data)
    
    # 源序列处理
    src_lengths = [len(seq) for seq in src_grid_seqs]
    src_grid_seqs = rnn_utils.pad_sequence(src_grid_seqs, batch_first=True, padding_value=0)
    src_pro_feas = torch.vstack(src_pro_feas).squeeze(-1)
    src_seg_seqs = rnn_utils.pad_sequence(src_seg_seqs, batch_first=True, padding_value=0)
    
    # 目标序列处理
    trg_lengths = [len(seq) for seq in trg_rids]
    trg_gps_seqs = rnn_utils.pad_sequence(trg_gps_seqs, batch_first=True, padding_value=0)
    trg_rids = rnn_utils.pad_sequence(trg_rids, batch_first=True, padding_value=0)
    trg_rates = rnn_utils.pad_sequence(trg_rates, batch_first=True, padding_value=0)
    
    # 路径处理
    path_lengths = [len(seq) for seq in paths]
    paths = rnn_utils.pad_sequence(paths, batch_first=True, padding_value=0)
    
    d_rids = torch.vstack(d_rids).squeeze(-1)
    d_rates = torch.vstack(d_rates)
    
    # 为测试创建dummy的候选路段数据
    max_src_len = max(src_lengths)
    candi_size = 10  # 默认候选数量
    candi_ids = torch.zeros((len(src_grid_seqs), max_src_len, candi_size), dtype=torch.long)
    candi_feats = torch.zeros((len(src_grid_seqs), max_src_len, candi_size, 9))
    candi_masks = torch.zeros((len(src_grid_seqs), max_src_len, candi_size))
    
    return (src_grid_seqs, src_pro_feas, src_seg_seqs, src_lengths,
            candi_ids, candi_feats, candi_masks,
            trg_gps_seqs, trg_rids, trg_rates, trg_lengths,
            paths, path_lengths, d_rids, d_rates)


def train(model, iterator, optimizer, rid_features_dict, parameters, device):
    """训练函数"""
    criterion_reg = nn.L1Loss(reduction='mean')
    criterion_bce = nn.BCELoss(reduction='mean')
    criterion_ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
    
    epoch_ttl_loss = 0
    epoch_mm_loss = 0
    epoch_traj_loss = 0
    epoch_rate_loss = 0
    
    model.train()
    for i, batch in enumerate(iterator):
        (src_seqs, src_pro_feas, src_seg_seqs, src_lengths,
         candi_labels, candi_ids, candi_feats, candi_masks,
         trg_gps_seqs, trg_rids, trg_rates, trg_lengths,
         paths, path_lengths, d_rids, d_rates) = batch
        
        # 转移到设备
        src_seqs = src_seqs.permute(1, 0, 2).to(device, non_blocking=True)
        src_pro_feas = src_pro_feas.to(device, non_blocking=True)
        src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
        
        candi_labels = candi_labels.to(device, non_blocking=True)
        candi_ids = candi_ids.to(device, non_blocking=True)
        candi_feats = candi_feats.to(device, non_blocking=True)
        candi_masks = candi_masks.to(device, non_blocking=True)
        
        trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device, non_blocking=True)
        trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
        trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)
        
        paths = paths.permute(1, 0).to(device, non_blocking=True)
        d_rids = d_rids.to(device, non_blocking=True)
        d_rates = d_rates.to(device, non_blocking=True)
        
        # 前向传播
        mm_probs, output_ids, output_rates = model(
            src_seqs, src_lengths, src_seg_seqs, candi_ids, candi_feats, candi_masks,
            trg_rids, trg_rates, trg_lengths,
            src_pro_feas, rid_features_dict, paths, path_lengths,
            d_rids, d_rates, teacher_forcing_ratio=parameters.tf_ratio)
        
        # 损失计算
        trg_lengths_sub = [length - 2 for length in trg_lengths]
        
        # 轨迹恢复损失
        path_labels = torch.full((output_ids.size(0), output_ids.size(1)), 
                                -100, dtype=torch.long, device=device)
        
        for t in range(output_ids.size(0)):
            for b in range(output_ids.size(1)):
                if t + 1 < trg_lengths[b] - 1:
                    trg_rid = trg_rids[t + 1, b]
                    path_b = paths[:path_lengths[b], b]
                    matches = (path_b == trg_rid).nonzero(as_tuple=True)[0]
                    if len(matches) > 0:
                        path_labels[t, b] = matches[0]
        
        loss_traj = criterion_ce(
            output_ids.permute(1, 2, 0),
            path_labels.permute(1, 0)
        ) * parameters.lambda1
        
        # 地图匹配损失（辅助任务）
        loss_mm = criterion_bce(mm_probs, candi_labels.float()) * 1.0
        
        # 通行率损失
        if parameters.rate_flag:
            loss_rate = criterion_reg(output_rates, trg_rates[1:-1]) * parameters.lambda2
        else:
            loss_rate = torch.tensor(0.0, device=device)
        
        ttl_loss = loss_traj + loss_mm + loss_rate
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        ttl_loss.backward()
        optimizer.step()
        
        # 记录损失
        epoch_ttl_loss += ttl_loss.item()
        epoch_mm_loss += loss_mm.item()
        epoch_traj_loss += loss_traj.item()
        epoch_rate_loss += loss_rate.item()
        
        if len(iterator) >= 10 and (i + 1) % (len(iterator) // 10) == 0:
            print("==>{}: ttl={:.4f}, mm={:.4f}, traj={:.4f}, rate={:.4f}".format(
                (i + 1) // (len(iterator) // 10), 
                epoch_ttl_loss / (i + 1),
                epoch_mm_loss / (i + 1),
                epoch_traj_loss / (i + 1),
                epoch_rate_loss / (i + 1)))
    
    return (epoch_ttl_loss / len(iterator),
            epoch_mm_loss / len(iterator),
            epoch_traj_loss / len(iterator),
            epoch_rate_loss / len(iterator))


def evaluate(model, iterator, rid_features_dict, parameters, device):
    """验证函数"""
    criterion_reg = nn.L1Loss(reduction='mean')
    criterion_bce = nn.BCELoss(reduction='mean')
    criterion_ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
    
    epoch_ttl_loss = 0
    epoch_mm_loss = 0
    epoch_traj_loss = 0
    epoch_rate_loss = 0
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            (src_seqs, src_pro_feas, src_seg_seqs, src_lengths,
             candi_labels, candi_ids, candi_feats, candi_masks,
             trg_gps_seqs, trg_rids, trg_rates, trg_lengths,
             paths, path_lengths, d_rids, d_rates) = batch
            
            # 转移到设备
            src_seqs = src_seqs.permute(1, 0, 2).to(device, non_blocking=True)
            src_pro_feas = src_pro_feas.to(device, non_blocking=True)
            src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
            
            candi_labels = candi_labels.to(device, non_blocking=True)
            candi_ids = candi_ids.to(device, non_blocking=True)
            candi_feats = candi_feats.to(device, non_blocking=True)
            candi_masks = candi_masks.to(device, non_blocking=True)
            
            trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
            trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)
            
            paths = paths.permute(1, 0).to(device, non_blocking=True)
            d_rids = d_rids.to(device, non_blocking=True)
            d_rates = d_rates.to(device, non_blocking=True)
            
            # 前向传播
            mm_probs, output_ids, output_rates = model(
                src_seqs, src_lengths, src_seg_seqs, candi_ids, candi_feats, candi_masks,
                trg_rids, trg_rates, trg_lengths,
                src_pro_feas, rid_features_dict, paths, path_lengths,
                d_rids, d_rates, teacher_forcing_ratio=0)
            
            # 损失计算
            trg_lengths_sub = [length - 2 for length in trg_lengths]
            
            path_labels = torch.full((output_ids.size(0), output_ids.size(1)), 
                                    -100, dtype=torch.long, device=device)
            for t in range(output_ids.size(0)):
                for b in range(output_ids.size(1)):
                    if t + 1 < trg_lengths[b] - 1:
                        trg_rid = trg_rids[t + 1, b]
                        path_b = paths[:path_lengths[b], b]
                        matches = (path_b == trg_rid).nonzero(as_tuple=True)[0]
                        if len(matches) > 0:
                            path_labels[t, b] = matches[0]
            
            loss_traj = criterion_ce(
                output_ids.permute(1, 2, 0),
                path_labels.permute(1, 0)
            ) * parameters.lambda1
            
            loss_mm = criterion_bce(mm_probs, candi_labels.float()) * 1.0
            
            if parameters.rate_flag:
                loss_rate = criterion_reg(output_rates, trg_rates[1:-1]) * parameters.lambda2
            else:
                loss_rate = torch.tensor(0.0, device=device)
            
            ttl_loss = loss_traj + loss_mm + loss_rate
            
            epoch_ttl_loss += ttl_loss.item()
            epoch_mm_loss += loss_mm.item()
            epoch_traj_loss += loss_traj.item()
            epoch_rate_loss += loss_rate.item()
    
    print("==> Valid: ttl={:.4f}, mm={:.4f}, traj={:.4f}, rate={:.4f}".format(
        epoch_ttl_loss / len(iterator),
        epoch_mm_loss / len(iterator),
        epoch_traj_loss / len(iterator),
        epoch_rate_loss / len(iterator)))
    
    return (epoch_ttl_loss / len(iterator),
            epoch_mm_loss / len(iterator),
            epoch_traj_loss / len(iterator),
            epoch_rate_loss / len(iterator))


def get_results(predict_id, predict_rate, target_id, target_rate, target_gps, 
                trg_len, paths, path_lengths, inverse_flag=True):
    """整理推理结果"""
    if inverse_flag:
        predict_id = predict_id - 1
        target_id = target_id - 1
        paths = paths - 1
    
    predict_id = predict_id.permute(1, 0).detach().cpu().tolist()
    predict_rate = predict_rate.permute(1, 0).detach().cpu().tolist()
    target_gps = target_gps.permute(1, 0, 2).detach().cpu().tolist()
    target_id = target_id.permute(1, 0).detach().cpu().tolist()
    target_rate = target_rate.permute(1, 0).detach().cpu().tolist()
    paths = paths.permute(1, 0).detach().cpu().tolist()
    
    results = []
    for pred_seg, pred_rate, trg_id, trg_rate, trg_gps, length, path, path_len in zip(
            predict_id, predict_rate, target_id, target_rate, target_gps, trg_len, paths, path_lengths):
        results.append([pred_seg[:length], pred_rate[:length], 
                       trg_id[:length], trg_rate[:length], 
                       trg_gps[:length], path[:path_len]])
    return results


def infer(model, iterator, rid_features_dict, device):
    """推理函数"""
    data = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            (src_seqs, src_pro_feas, src_seg_seqs, src_lengths,
             candi_ids, candi_feats, candi_masks,
             trg_gps_seqs, trg_rids, trg_rates, trg_lengths,
             paths, path_lengths, d_rids, d_rates) = batch
            
            # 转移到设备
            src_seqs = src_seqs.permute(1, 0, 2).to(device, non_blocking=True)
            src_pro_feas = src_pro_feas.to(device, non_blocking=True)
            src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
            
            candi_ids = candi_ids.to(device, non_blocking=True)
            candi_feats = candi_feats.to(device, non_blocking=True)
            candi_masks = candi_masks.to(device, non_blocking=True)
            
            trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
            trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)
            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device, non_blocking=True)
            
            paths = paths.permute(1, 0).to(device, non_blocking=True)
            d_rids = d_rids.to(device, non_blocking=True)
            d_rates = d_rates.to(device, non_blocking=True)
            
            # 前向传播
            mm_probs, output_ids, output_rates = model(
                src_seqs, src_lengths, src_seg_seqs, candi_ids, candi_feats, candi_masks,
                trg_rids, trg_rates, trg_lengths,
                src_pro_feas, rid_features_dict, paths, path_lengths,
                d_rids, d_rates, teacher_forcing_ratio=-1)
            
            # 从路径中选择预测的路段
            output_tmp = (F.one_hot(output_ids.argmax(-1), paths.shape[0]) * 
                         paths.permute(1, 0).unsqueeze(1).repeat(1, output_ids.shape[0], 1).permute(1, 0, 2)).sum(dim=-1)
            
            output_rates = output_rates.squeeze(2)
            trg_rates = trg_rates.squeeze(2)
            trg_lengths_sub = [length - 2 for length in trg_lengths]
            
            results = get_results(output_tmp, output_rates, trg_rids[1:-1], 
                                trg_rates[1:-1], trg_gps_seqs[1:-1], 
                                trg_lengths_sub, paths, path_lengths)
            data.extend(results)
            
            if len(iterator) >= 10 and (i + 1) % (len(iterator) // 10) == 0:
                print("==> Test: {}".format((i + 1) // (len(iterator) // 10)))
    
    return data


def main():
    parser = argparse.ArgumentParser(description='UTT')
    parser.add_argument('--city', type=str, default='porto')
    parser.add_argument('--keep_ratio', type=float, default=0.125)
    parser.add_argument('--tf_ratio', type=float, default=1)
    parser.add_argument('--lambda1', type=int, default=10)
    parser.add_argument('--lambda2', type=float, default=5)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--transformer_layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument('--model_old_path', type=str, default='')
    parser.add_argument('--train_flag', action='store_true')
    parser.add_argument('--test_flag', action='store_true')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--gps_flag', action='store_true')
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--candi_size', type=int, default=10)
    
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
    
    load_pretrained_flag = False
    if opts.model_old_path != '':
        model_save_path = opts.model_old_path
        load_pretrained_flag = True
    else:
        model_save_root = f'./model/TRMMA/{opts.city}/'
        model_save_path = model_save_root + 'UTT_' + opts.city + '_' + 'keep-ratio_' + str(opts.keep_ratio) + '_' + time.strftime("%Y%m%d_%H%M%S") + '/'
        
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
        'pro_features_flag': True,
        'rate_flag': True,
        'dest_type': 2,
        'gps_flag': opts.gps_flag,
        'rid_feats_flag': True,
        'candi_size': opts.candi_size,
        
        # extra info module
        'rid_fea_dim': 18,
        'pro_input_dim': 48,
        'pro_output_dim': 8,
        
        # MBR
        'min_lat': zone_range[0],
        'min_lng': zone_range[1],
        'max_lat': zone_range[2],
        'max_lng': zone_range[3],
        
        # input data params
        'city': opts.city,
        'keep_ratio': opts.keep_ratio,
        'grid_size': 50,
        'time_span': ts,
        
        # model params
        'hid_dim': opts.hid_dim,
        'id_emb_dim': opts.hid_dim,
        'dropout': 0.1,
        'id_size': rn.valid_edge_cnt_one,
        
        'lambda1': opts.lambda1,
        'lambda2': opts.lambda2,
        'n_epochs': opts.epochs,
        'batch_size': opts.batch_size,
        'learning_rate': opts.lr,
        "lr_step": 2,
        "lr_decay": 0.8,
        'tf_ratio': opts.tf_ratio,
        'decay_flag': True,
        'decay_ratio': 0.9,
        'clip': 1,
        'log_step': 1,
        
        'utc': utc,
        'small': opts.small,
        'dam_root': os.path.join("data", opts.city),
        'planner': 'da'
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
    
    if opts.train_flag:
        # 加载数据集
        train_dataset = UTTData(rn, traj_root, mbr, args, 'train')
        valid_dataset = UTTData(rn, traj_root, mbr, args, 'valid')
        print('training dataset shape: ' + str(len(train_dataset)))
        print('validation dataset shape: ' + str(len(valid_dataset)))
        logging.info('Finish data preparing.')
        logging.info('training dataset shape: ' + str(len(train_dataset)))
        logging.info('validation dataset shape: ' + str(len(valid_dataset)))
        
        train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, 
                                   shuffle=True, collate_fn=lambda x: collate_fn(x), 
                                   num_workers=opts.num_worker, pin_memory=False)
        valid_iterator = DataLoader(valid_dataset, batch_size=args.batch_size, 
                                   shuffle=False, collate_fn=lambda x: collate_fn(x), 
                                   num_workers=opts.num_worker, pin_memory=False)
        
        model = UTT(args, rn).to(device)
        
        if load_pretrained_flag:
            model = torch.load(os.path.join(model_save_path, 'val-best-model.pt'), 
                             map_location=device)
        
        print('model', str(model))
        logging.info('model' + str(model))
        
        ls_train_loss, ls_train_mm_loss, ls_train_traj_loss, ls_train_rate_loss = [], [], [], []
        ls_valid_loss, ls_valid_mm_loss, ls_valid_traj_loss, ls_valid_rate_loss = [], [], [], []
        
        best_valid_loss = float('inf')
        best_epoch = 0
        
        tb_writer = SummaryWriter(log_dir=os.path.join(model_save_path, 'tensorboard'))
        
        lr = args.learning_rate
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                         patience=args.lr_step,
                                                         factor=args.lr_decay, 
                                                         threshold=1e-3)
        stopping_count = 0
        train_times = []
        
        for epoch in tqdm(range(args.n_epochs), desc='epoch num'):
            start_time = time.time()
            
            print("==> training {}, {}...".format(args.tf_ratio, lr))
            train_loss, train_mm_loss, train_traj_loss, train_rate_loss = train(
                model, train_iterator, optimizer, rid_features_dict, args, device)
            end_train = time.time()
            
            ls_train_loss.append(train_loss)
            ls_train_mm_loss.append(train_mm_loss)
            ls_train_traj_loss.append(train_traj_loss)
            ls_train_rate_loss.append(train_rate_loss)
            
            print("==> validating...")
            valid_loss, valid_mm_loss, valid_traj_loss, valid_rate_loss = evaluate(
                model, valid_iterator, rid_features_dict, args, device)
            
            ls_valid_loss.append(valid_loss)
            ls_valid_mm_loss.append(valid_mm_loss)
            ls_valid_traj_loss.append(valid_traj_loss)
            ls_valid_rate_loss.append(valid_rate_loss)
            
            end_time = time.time()
            epoch_secs = end_time - start_time
            train_times.append(end_train - start_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model, os.path.join(model_save_path, 'val-best-model.pt'))
                best_epoch = epoch
                stopping_count = 0
            else:
                stopping_count += 1
            
            tb_writer.add_scalars('Train_loss', {
                'total': train_loss, 
                'MM': train_mm_loss,
                'Traj': train_traj_loss, 
                'Rate': train_rate_loss
            }, epoch)
            tb_writer.add_scalars('Valid_loss', {
                'total': valid_loss,
                'MM': valid_mm_loss,
                'Traj': valid_traj_loss, 
                'Rate': valid_rate_loss
            }, epoch)
            tb_writer.add_scalar('learning_rate', lr, epoch)
            
            if (epoch % args.log_step == 0) or (epoch == args.n_epochs - 1):
                logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_secs) + 's')
                logging.info('Epoch: ' + str(epoch + 1) + ' TF Ratio: ' + str(args.tf_ratio))
                logging.info('\tTrain Loss:' + str(train_loss) +
                           '\tTrain MM Loss:' + str(train_mm_loss) +
                           '\tTrain Traj Loss:' + str(train_traj_loss) +
                           '\tTrain Rate Loss:' + str(train_rate_loss))
                logging.info('\tValid Loss:' + str(valid_loss) +
                           '\tValid MM Loss:' + str(valid_mm_loss) +
                           '\tValid Traj Loss:' + str(valid_traj_loss) +
                           '\tValid Rate Loss:' + str(valid_rate_loss))
                
                torch.save(model, os.path.join(model_save_path, 'train-mid-model.pt'))
            
            if args.decay_flag:
                args.tf_ratio = args.tf_ratio * args.decay_ratio
            
            scheduler.step(valid_traj_loss)
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
        logging.info('==> Training Time: {}, {}, {}, {}'.format(
            np.sum(train_times) / 3600, np.mean(train_times), 
            np.min(train_times), np.max(train_times)))
        print('==> Training Time: {}, {}, {}, {}'.format(
            np.sum(train_times) / 3600, np.mean(train_times), 
            np.min(train_times), np.max(train_times)))
    
    if opts.test_flag:
        test_dataset = UTTTestData(rn, traj_root, mbr, args, 'test')
        print('testing dataset shape: ' + str(len(test_dataset)))
        logging.info('testing dataset shape: ' + str(len(test_dataset)))
        
        test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, 
                                  shuffle=False, 
                                  collate_fn=lambda x: collate_fn_test(x), 
                                  num_workers=opts.num_worker, pin_memory=True)
        
        model = torch.load(os.path.join(model_save_path, 'val-best-model.pt'), 
                         map_location=device)
        print('==> Model Loaded')
        
        print("==> Starting Prediction...")
        start_time = time.time()
        data = infer(model, test_iterator, rid_features_dict, device)
        end_time = time.time()
        epoch_secs = end_time - start_time
        print('Time: ' + str(epoch_secs) + 's')
        logging.info('Inference Time: {}, {}, {}'.format(
            end_time - start_time, 
            (end_time - start_time) / len(test_dataset) * 1000, 
            len(test_dataset) / (end_time - start_time)))
        print('Inference Time: {}, {}, {}'.format(
            end_time - start_time, 
            (end_time - start_time) / len(test_dataset) * 1000, 
            len(test_dataset) / (end_time - start_time)))
        
        pickle.dump(data, open(os.path.join(model_save_path, 'infer_output_utt.pkl'), "wb"))
        
        outputs = []
        for pred_seg, pred_rate, trg_id, trg_rate, trg_gps, route in data:
            pred_gps = toseq(rn, pred_seg, pred_rate, route, dam.seg_info)
            outputs.append([pred_gps, pred_seg, trg_gps, trg_id])
        
        test_trajs = pickle.load(open(os.path.join(traj_root, 'test_output.pkl'), "rb"))
        groups = Counter(test_dataset.groups)
        nums = []
        for i in range(len(test_trajs)):
            nums.append(groups[i])
        
        results = []
        for traj, num, src_mm in zip(test_trajs, nums, test_dataset.src_mms):
            tmp_all = outputs[:num]
            low_idx = traj.low_idx
            gps, segs, _ = zip(*src_mm)
            predict_ids = [segs[0]]
            predict_gps = [gps[0]]
            pointer = -1
            
            for p1_idx, p2_idx, seg, latlng in zip(low_idx[:-1], low_idx[1:], segs[1:], gps[1:]):
                if (p1_idx + 1) < p2_idx:
                    pointer += 1
                    tmp = tmp_all[pointer]
                    predict_gps.extend(tmp[0])
                    predict_ids.extend(tmp[1])
                predict_ids.append(seg)
                predict_gps.append(latlng)
            outputs = outputs[num:]
            
            mm_gps_seq = []
            mm_eids = []
            for i, pt in enumerate(traj.pt_list):
                candi_pt = pt.data['candi_pt']
                mm_eids.append(candi_pt.eid)
                mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
            
            assert len(predict_gps) == len(mm_gps_seq) == len(predict_ids) == len(mm_eids)
            results.append([predict_gps, predict_ids, mm_gps_seq, mm_eids])
        
        pickle.dump(results, open(os.path.join(model_save_path, 'recovery_output_utt.pkl'), "wb"))
        
        print("==> Starting Evaluation...")
        epoch_id1_loss = []
        epoch_recall_loss = []
        epoch_precision_loss = []
        epoch_f1_loss = []
        epoch_mae_loss = []
        epoch_rmse_loss = []
        
        for pred_gps, pred_seg, trg_gps, trg_id in results:
            recall, precision, f1, loss_ids1, loss_mae, loss_rmse = calc_metrics(
                pred_seg, pred_gps, trg_id, trg_gps)
            epoch_id1_loss.append(loss_ids1)
            epoch_recall_loss.append(recall)
            epoch_precision_loss.append(precision)
            epoch_f1_loss.append(f1)
            epoch_mae_loss.append(loss_mae)
            epoch_rmse_loss.append(loss_rmse)
        
        test_id_recall, test_id_precision, test_id_f1, test_id_acc, test_mae, test_rmse = (
            np.mean(epoch_recall_loss), np.mean(epoch_precision_loss), 
            np.mean(epoch_f1_loss), np.mean(epoch_id1_loss), 
            np.mean(epoch_mae_loss), np.mean(epoch_rmse_loss))
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


import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.demo1 import GPS2SegData, GPS2Seg, TrajRecData, TrajRecovery, DAPlanner, TrajRecTestData, IterativeModel, compute_iterative_loss
from utils.evaluation_utils import cal_id_acc, calc_metrics, toseq
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
from utils.model_utils import gps2grid, AttrDict
from tqdm import tqdm
import numpy as np
from collections import Counter


def collate_fn_mma(data):
    """MMA模型的数据整理函数"""
    src_seqs, trg_rids, candi_onehots, candi_ids, candi_feats, candi_masks = zip(*data)

    lengths = [len(seq) for seq in src_seqs]
    src_seqs = rnn_utils.pad_sequence(src_seqs, batch_first=True, padding_value=0)

    candi_onehots = rnn_utils.pad_sequence(candi_onehots, batch_first=True, padding_value=0)
    candi_ids = rnn_utils.pad_sequence(candi_ids, batch_first=True, padding_value=0)
    candi_feats = rnn_utils.pad_sequence(candi_feats, batch_first=True, padding_value=0)
    candi_masks = rnn_utils.pad_sequence(candi_masks, batch_first=True, padding_value=0)

    return src_seqs, lengths, trg_rids, candi_onehots, candi_ids, candi_feats, candi_masks


def collate_fn_trmma(data0):
    """TRMMA模型的数据整理函数"""
    data = []
    for item in data0:
        data.extend(item)

    da_routes, src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, trg_rids, trg_rates, \
    trg_rid_labels, d_rids, d_rates = zip(*data)

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

    return src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates


def collate_fn_trmma_test(data):
    """TRMMA测试模型的数据整理函数"""
    da_routes, src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, trg_gps_seqs, trg_rids, trg_rates, \
    trg_rid_labels, d_rids, d_rates = zip(*data)

    src_lengths = [len(seq) for seq in src_seqs]
    src_seqs = rnn_utils.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    src_pro_feas = torch.vstack(src_pro_feas).squeeze(-1)
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
    d_rids = torch.vstack(d_rids).squeeze(-1)
    d_rates = torch.vstack(d_rates)
    max_da = max(da_lengths)
    trg_rid_labels = list(trg_rid_labels)
    for i in range(len(trg_rid_labels)):
        if trg_rid_labels[i].shape[1] < max_da:
            tmp = torch.zeros(trg_rid_labels[i].shape[0], max_da - trg_rid_labels[i].shape[1])
            trg_rid_labels[i] = torch.cat([trg_rid_labels[i], tmp], dim=-1)
    trg_rid_labels = rnn_utils.pad_sequence(trg_rid_labels, batch_first=True, padding_value=0)

    return src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates




def train_iterative(model, mma_iterator, trmma_iterator, optimizer, rid_features_dict, args, device):
    """
    端到端迭代训练函数
    """
    epoch_total_loss = 0
    epoch_mma_loss = 0
    epoch_trmma_loss = 0
    epoch_consistency_loss = 0
    epoch_refined_mma_loss = 0

    model.train()
    
    # 简化处理：使用较短的迭代器长度
    min_len = min(len(mma_iterator), len(trmma_iterator))
    mma_iter = iter(mma_iterator)
    trmma_iter = iter(trmma_iterator)
    
    for i in range(min_len):
        try:
            # 获取MMA数据
            mma_batch = next(mma_iter)
            src_seqs_mma, src_lengths_mma, _, candi_labels, candi_ids, candi_feats, candi_masks = mma_batch
            
            # 获取TRMMA数据
            trmma_batch = next(trmma_iter)
            src_seqs_trmma, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths_trmma, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates = trmma_batch
            
            # 移动MMA数据到设备
            src_seqs_mma = src_seqs_mma.to(device, non_blocking=True)
            candi_labels = candi_labels.to(device, non_blocking=True)
            candi_ids = candi_ids.to(device, non_blocking=True)
            candi_feats = candi_feats.to(device, non_blocking=True)
            candi_masks = candi_masks.to(device, non_blocking=True)
            
            # 移动TRMMA数据到设备
            src_pro_feas = src_pro_feas.to(device, non_blocking=True)
            trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
            src_seqs_trmma = src_seqs_trmma.permute(1, 0, 2).to(device, non_blocking=True)
            src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
            src_seg_feats = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)
            trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
            trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)
            da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
            da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
            d_rids = d_rids.to(device, non_blocking=True)
            d_rates = d_rates.to(device, non_blocking=True)
            
            # 端到端前向传播
            outputs = model(
                # MMA输入
                src_seqs_mma, src_lengths_mma, candi_ids, candi_feats, candi_masks,
                # TRMMA输入
                src_seqs_trmma, src_lengths_trmma, trg_rids, trg_rates, trg_lengths,
                src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos, 
                src_seg_seqs, src_seg_feats, d_rids, d_rates, args.tf_ratio,
                # 训练标志
                training=True
            )
            
            # 计算损失
            loss_dict = compute_iterative_loss(
                outputs, candi_labels, trg_rid_labels, trg_rates[1:-1],
                mma_weight=1.0, trmma_weight=1.0, consistency_weight=args.consistency_weight
            )
            
            total_loss = loss_dict['total_loss']
            
            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            
            # 累计损失
            epoch_total_loss += total_loss.item()
            epoch_mma_loss += loss_dict['initial_mma_loss'].item()
            epoch_trmma_loss += loss_dict['trmma_loss'].item()
            epoch_consistency_loss += loss_dict['consistency_loss'].item()
            epoch_refined_mma_loss += loss_dict['refined_mma_loss'].item()
            
            if min_len >= 10 and (i + 1) % (min_len // 10) == 0:
                print("==>{}: {}, {}, {}, {}, {}".format((i + 1) // (min_len // 10), 
                      epoch_total_loss / (i + 1), epoch_mma_loss / (i + 1), epoch_trmma_loss / (i + 1),
                      epoch_consistency_loss / (i + 1), epoch_refined_mma_loss / (i + 1)))
        
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue
    
    return (epoch_total_loss/min_len, epoch_mma_loss/min_len, 
            epoch_trmma_loss/min_len, epoch_consistency_loss/min_len, epoch_refined_mma_loss/min_len)


def evaluate_iterative(model, mma_iterator, trmma_iterator, rid_features_dict, args, device):
    """
    端到端迭代评估函数
    """
    epoch_total_loss = 0
    epoch_mma_loss = 0
    epoch_trmma_loss = 0
    epoch_consistency_loss = 0

    model.eval()
    
    min_len = min(len(mma_iterator), len(trmma_iterator))
    mma_iter = iter(mma_iterator)
    trmma_iter = iter(trmma_iterator)
    
    with torch.no_grad():
        for i in range(min_len):
            try:
                # 获取数据
                mma_batch = next(mma_iter)
                src_seqs_mma, src_lengths_mma, _, candi_labels, candi_ids, candi_feats, candi_masks = mma_batch
                
                trmma_batch = next(trmma_iter)
                src_seqs_trmma, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths_trmma, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates = trmma_batch
                
                # 移动数据到设备
                src_seqs_mma = src_seqs_mma.to(device, non_blocking=True)
                candi_labels = candi_labels.to(device, non_blocking=True)
                candi_ids = candi_ids.to(device, non_blocking=True)
                candi_feats = candi_feats.to(device, non_blocking=True)
                candi_masks = candi_masks.to(device, non_blocking=True)
                
                src_pro_feas = src_pro_feas.to(device, non_blocking=True)
                trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
                src_seqs_trmma = src_seqs_trmma.permute(1, 0, 2).to(device, non_blocking=True)
                src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
                src_seg_feats = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)
                trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
                trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)
                da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
                da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
                d_rids = d_rids.to(device, non_blocking=True)
                d_rates = d_rates.to(device, non_blocking=True)
                
                # 前向传播
                outputs = model(
                    src_seqs_mma, src_lengths_mma, candi_ids, candi_feats, candi_masks,
                    src_seqs_trmma, src_lengths_trmma, trg_rids, trg_rates, trg_lengths,
                    src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos, 
                    src_seg_seqs, src_seg_feats, d_rids, d_rates, 0,  # teacher_forcing_ratio=0 for evaluation
                    training=True  # 仍然使用迭代，但不更新参数
                )
                
                # 计算损失
                loss_dict = compute_iterative_loss(
                    outputs, candi_labels, trg_rid_labels, trg_rates[1:-1],
                    mma_weight=1.0, trmma_weight=1.0, consistency_weight=args.consistency_weight
                )
                
                epoch_total_loss += loss_dict['total_loss'].item()
                epoch_mma_loss += loss_dict['initial_mma_loss'].item()
                epoch_trmma_loss += loss_dict['trmma_loss'].item()
                epoch_consistency_loss += loss_dict['consistency_loss'].item()
                
            except Exception as e:
                print(f"Error in validation batch {i}: {e}")
                continue
    
    print("==> Valid: {}, {}, {}, {}".format(epoch_total_loss / min_len, epoch_mma_loss / min_len, 
                                           epoch_trmma_loss / min_len, epoch_consistency_loss / min_len))
    
    return epoch_total_loss/min_len, epoch_mma_loss/min_len, epoch_trmma_loss/min_len, epoch_consistency_loss/min_len


def main():
    parser = argparse.ArgumentParser(description='Demo1 Combined Training')
    
    # 通用参数
    parser.add_argument('--city', type=str, default='porto')
    parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')
    parser.add_argument('--hid_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument("--gpu_id", type=str, default="0")
    # parser.add_argument('--model_old_path', type=str, default='', help='old model path')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--num_worker', type=int, default=8)
    # parser.add_argument('--init_ratio', type=float, default=0.5)

    # 迭代参数
    parser.add_argument('--max_iterations', type=int, default=2, help='最大迭代次数')
    parser.add_argument('--consistency_weight', type=float, default=0.5, help='一致性损失权重')
    
    # MMA特定参数
    parser.add_argument('--attn_flag', action='store_true', help='flag of using attention')
    parser.add_argument('--direction_flag', action='store_true')
    parser.add_argument("--candi_size", type=int, default=10)
    # parser.add_argument('--only_direction', action='store_true')
    
    # TRMMA特定参数
    parser.add_argument('--tf_ratio', type=float, default=1, help='teaching ratio in float')
    parser.add_argument('--lambda1', type=int, default=10, help='weight for multi task id')
    parser.add_argument('--lambda2', type=float, default=5, help='weight for multi task rate')
    parser.add_argument('--heads', type=int, default=4)
    # parser.add_argument('--eid_cate', type=str, default='gps2seg')
    # parser.add_argument('--inferred_seg_path', type=str, default='')
    # parser.add_argument('--da_route_flag', action='store_true')
    # parser.add_argument('--srcseg_flag', action='store_true')
    # parser.add_argument('--gps_flag', action='store_true')
    # parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--planner', type=str, default='da')
    
    # # 训练和测试标志
    # parser.add_argument('--train_flag', action='store_true', help='flag of training')
    # parser.add_argument('--test_flag', action='store_true', help='flag of testing')
    
    # 训练标志（默认开启）
    parser.add_argument('--train_flag', action='store_true', default=True, help='训练标志')

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
    print('multi_task device', device)

    load_pretrained_flag = False
    # if opts.model_old_path != '':
    #     model_save_path = opts.model_old_path
    #     load_pretrained_flag = True
    # else:
    model_save_root = f'./model/IterativeDemo/{opts.city}/'
    model_save_path = model_save_root + f'Iterative_{opts.city}_keep-ratio_{str(opts.keep_ratio)}_{time.strftime("%Y%m%d_%H%M%S")}/'

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

    # MMA参数配置
    mma_args = AttrDict()
    mma_args_dict = {
        'device': device,
        'transformer_layers': opts.transformer_layers,
        'candi_size': opts.candi_size,
        'attn_flag': opts.attn_flag,
        'direction_flag': opts.direction_flag,
        'gps_flag': False,
        'search_dist': 50,
        'beta': 15,
        'gamma': 30,
        'min_lat': zone_range[0],
        'min_lng': zone_range[1],
        'max_lat': zone_range[2],
        'max_lng': zone_range[3],
        'city': opts.city,
        'keep_ratio': opts.keep_ratio,
        'grid_size': 50,
        'time_span': ts,
        'hid_dim': opts.hid_dim,
        'id_emb_dim': opts.hid_dim,
        'dropout': 0.1,
        'id_size': rn.valid_edge_cnt_one,
        'n_epochs': opts.epochs,
        'batch_size': opts.batch_size,
        'learning_rate': opts.lr,
        'decay_flag': True,
        'decay_ratio': 0.9,
        'clip': 1,
        'log_step': 1,
        'utc': utc,
        'small': opts.small,
        'init_ratio': 0.5,
        'only_direction': False,
        'cate': "g2s",
        'threshold': 1
    }
    mma_args.update(mma_args_dict)
    
    # TRMMA参数配置
    trmma_args = AttrDict()
    trmma_args_dict = {
        'device': device,
        'transformer_layers': opts.transformer_layers,
        'heads': opts.heads,
        'tandem_fea_flag': True,
        'pro_features_flag': True,
        'srcseg_flag': True,
        'da_route_flag': True,
        'rate_flag': True,
        'prog_flag': False,
        'dest_type': 2,
        'gps_flag': False,
        'rid_feats_flag': True,
        'learn_pos': True,
        'search_dist': 50,
        'beta': 15,
        'gamma': 30,
        'rid_fea_dim': 18,
        'pro_input_dim': 48,
        'pro_output_dim': 8,
        'min_lat': zone_range[0],
        'min_lng': zone_range[1],
        'max_lat': zone_range[2],
        'max_lng': zone_range[3],
        'city': opts.city,
        'keep_ratio': opts.keep_ratio,
        'grid_size': 50,
        'time_span': ts,
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
        'eid_cate': 'gps2seg',
        'inferred_seg_path': '',
        'planner': 'da',
        'debug': False,
        'consistency_weight': opts.consistency_weight,
    }
    trmma_args.update(trmma_args_dict)
    
    # 使用TRMMA参数作为主参数
    args = trmma_args

    mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
    args.grid_num = gps2grid(SPoint(args.max_lat, args.max_lng), mbr, args.grid_size)
    args.grid_num = (args.grid_num[0] + 1, args.grid_num[1] + 1)

    print(args)
    logging.info(trmma_args_dict)

    # 准备DAM和路段特征
    dam = DAPlanner(args.dam_root, args.id_size - 1, args.utc)
    rid_features_dict = torch.from_numpy(rn.get_rid_rnfea_dict(dam, ts)).to(device)

    traj_root = os.path.join("data", args.city)

    # 端到端迭代训练逻辑
    if opts.train_flag:
        # 准备MMA数据集
        mma_train_dataset = GPS2SegData(rn, traj_root, mbr, mma_args, 'train')
        mma_valid_dataset = GPS2SegData(rn, traj_root, mbr, mma_args, 'valid')
        print('MMA training dataset shape: ' + str(len(mma_train_dataset)))
        print('MMA validation dataset shape: ' + str(len(mma_valid_dataset)))
        
        # 准备TRMMA数据集
        trmma_train_dataset = TrajRecData(rn, traj_root, mbr, args, 'train')
        trmma_valid_dataset = TrajRecData(rn, traj_root, mbr, args, 'valid')
        print('TRMMA training dataset shape: ' + str(len(trmma_train_dataset)))
        print('TRMMA validation dataset shape: ' + str(len(trmma_valid_dataset)))

        logging.info('Finish data preparing.')
        logging.info('MMA training dataset shape: ' + str(len(mma_train_dataset)))
        logging.info('MMA validation dataset shape: ' + str(len(mma_valid_dataset)))
        logging.info('TRMMA training dataset shape: ' + str(len(trmma_train_dataset)))
        logging.info('TRMMA validation dataset shape: ' + str(len(trmma_valid_dataset)))
        
        # 创建数据加载器
        mma_train_iterator = DataLoader(mma_train_dataset, batch_size=args.batch_size, shuffle=True, 
                                      collate_fn=collate_fn_mma, num_workers=opts.num_worker, pin_memory=False)
        mma_valid_iterator = DataLoader(mma_valid_dataset, batch_size=args.batch_size, shuffle=False, 
                                      collate_fn=collate_fn_mma, num_workers=8, pin_memory=False)
        
        trmma_train_iterator = DataLoader(trmma_train_dataset, batch_size=args.batch_size, shuffle=True, 
                                        collate_fn=lambda x: collate_fn_trmma(x), num_workers=opts.num_worker, pin_memory=False)
        trmma_valid_iterator = DataLoader(trmma_valid_dataset, batch_size=args.batch_size, shuffle=False, 
                                        collate_fn=lambda x: collate_fn_trmma(x), num_workers=8, pin_memory=False)
        
        # 创建迭代模型
        model = IterativeModel(mma_args, args, max_iterations=opts.max_iterations).to(device)
        model.set_road_network(rn)
        model.consistency_weight = opts.consistency_weight
        
        print('model', str(model))
        logging.info('model' + str(model))
        
        # 损失历史记录
        ls_train_total_loss, ls_train_mma_loss, ls_train_trmma_loss, ls_train_consistency_loss, ls_train_refined_mma_loss = [], [], [], [], []
        ls_valid_total_loss, ls_valid_mma_loss, ls_valid_trmma_loss, ls_valid_consistency_loss = [], [], [], []

        best_valid_loss = float('inf')
        best_epoch = 0

        # TensorBoard
        tb_writer = SummaryWriter(log_dir=os.path.join(model_save_path, 'tensorboard'))

        # 创建优化器
        lr = args.learning_rate
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_step,
                                                       factor=args.lr_decay, threshold=1e-3)

        # 训练历史
        stopping_count = 0
        train_times = []
        
        print("Starting iterative end-to-end training...")
        for epoch in tqdm(range(args.n_epochs), desc='epoch num'):
            start_time = time.time()
            
            print("==> training {}, {}...".format(args.tf_ratio, lr))
            
            t_train = time.time()
            total_loss, mma_loss, trmma_loss, consistency_loss, refined_mma_loss = train_iterative(
                model, mma_train_iterator, trmma_train_iterator, optimizer, 
                rid_features_dict, args, device
            )
            end_train = time.time()
            print("training: {}".format(end_train - t_train))

            ls_train_total_loss.append(total_loss)
            ls_train_mma_loss.append(mma_loss)
            ls_train_trmma_loss.append(trmma_loss)
            ls_train_consistency_loss.append(consistency_loss)
            ls_train_refined_mma_loss.append(refined_mma_loss)
            
            print("==> validating...")
            
            t_valid = time.time()
            valid_total_loss, valid_mma_loss, valid_trmma_loss, valid_consistency_loss = evaluate_iterative(
                model, mma_valid_iterator, trmma_valid_iterator, rid_features_dict, args, device
            )
            print("validating: {}".format(time.time() - t_valid))
            
            ls_valid_total_loss.append(valid_total_loss)
            ls_valid_mma_loss.append(valid_mma_loss)
            ls_valid_trmma_loss.append(valid_trmma_loss)
            ls_valid_consistency_loss.append(valid_consistency_loss)

            end_time = time.time()
            epoch_secs = end_time - start_time
            train_times.append(end_train - t_train)

            # 保存最佳模型
            if valid_total_loss < best_valid_loss:
                best_valid_loss = valid_total_loss
                torch.save(model, os.path.join(model_save_path, 'val-best-model.pt'))
                best_epoch = epoch
                stopping_count = 0
            else:
                stopping_count += 1

            # 日志记录
            if (epoch % args.log_step == 0) or (epoch == args.n_epochs - 1):
                logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_secs) + 's')
                logging.info('Epoch: ' + str(epoch + 1) + ' TF Ratio: ' + str(args.tf_ratio))
                logging.info('\tTrain Loss:' + str(total_loss) +
                             '\tTrain MMA Loss:' + str(mma_loss) +
                             '\tTrain TRMMA Loss:' + str(trmma_loss) +
                             '\tTrain Consistency Loss:' + str(consistency_loss) +
                             '\tTrain Refined MMA Loss:' + str(refined_mma_loss))
                logging.info('\tValid Loss:' + str(valid_total_loss) +
                             '\tValid MMA Loss:' + str(valid_mma_loss) +
                             '\tValid TRMMA Loss:' + str(valid_trmma_loss) +
                             '\tValid Consistency Loss:' + str(valid_consistency_loss))
                
                torch.save(model, os.path.join(model_save_path, 'train-mid-model.pt'))
            
            # 学习率调整
            if args.decay_flag:
                args.tf_ratio = args.tf_ratio * args.decay_ratio
            
            scheduler.step(valid_total_loss)
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


if __name__ == '__main__':
    main()

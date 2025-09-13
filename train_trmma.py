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
import torch.profiler
from utils.map import RoadNetworkMapFull
from utils.spatial_func import SPoint
from utils.mbr import MBR
from models.trmma import DAPlanner, TrajRecData, TrajRecTestData, TrajRecovery
from utils.model_utils import AttrDict, gps2grid
from tqdm import tqdm
import numpy as np
from collections import Counter


def collate_fn(data0):
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


def collate_fn_test(data):
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


def train(model, iterator, optimizer, rid_features_dict, parameters, device):
    criterion_reg = nn.L1Loss(reduction='sum')
    criterion_bce = nn.BCELoss(reduction='sum')

    epoch_ttl_loss = 0
    epoch_train_id_loss = 0
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
        # 解包batch中的各个字段，含义如下：
        # src_seqs:         源轨迹的GPS序列（张量，形状为[batch, seq, 3]）
        # src_pro_feas:     源轨迹的辅助特征（如时间、小时等，张量）
        # src_seg_seqs:     源轨迹对应的路段序列（张量，形状为[batch, seq]）
        # src_seg_feats:    源轨迹路段的特征（如通行率等，张量）
        # src_lengths:      源轨迹每条序列的长度（列表）
        # trg_rids:         目标轨迹的路段ID序列（张量，形状为[batch, seq]）
        # trg_rates:        目标轨迹的通行率序列（张量，形状为[batch, seq, 1]）
        # trg_lengths:      目标轨迹每条序列的长度（列表）
        # trg_rid_labels:   目标轨迹路段ID的标签（多分类one-hot，张量，形状为[batch, seq, da_len]）
        # da_routes:        动态规划得到的候选路径（张量，形状为[batch, da_len]）
        # da_lengths:       每条候选路径的长度（列表）
        # da_pos:           候选路径的位置信息（张量，形状为[batch, da_len]）
        # d_rids:           目标轨迹最后一个点的路段ID（张量，形状为[batch, 1]）
        # d_rates:          目标轨迹最后一个点的通行率（张量，形状为[batch, 1]）
        src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates = batch

        src_pro_feas = src_pro_feas.to(device, non_blocking=True)
        trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
        src_seqs = src_seqs.permute(1, 0, 2).to(device, non_blocking=True)
        src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
        src_seg_feats = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)
        trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
        trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)

        da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
        da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
        d_rids = d_rids.to(device, non_blocking=True)
        d_rates = d_rates.to(device, non_blocking=True)

        time_move += time.time() - t1
        t2 = time.time()

        output_ids, output_rates = model(src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                                        src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos, None, None, d_rids, d_rates, teacher_forcing_ratio=parameters.tf_ratio)

        time_forward += time.time() - t2
        t3 = time.time()

        trg_lengths_sub = [length - 2 for length in trg_lengths]
        loss_train_ids = criterion_bce(output_ids, trg_rid_labels) * parameters.lambda1 / np.sum(np.array(trg_lengths_sub) * np.array(da_lengths))
        epoch_train_id_loss += loss_train_ids.item()
        ttl_loss = loss_train_ids
        if parameters.rate_flag:
            loss_rates = criterion_reg(output_rates, trg_rates[1:-1]) * parameters.lambda2 / sum(trg_lengths_sub)

            epoch_rate_loss += loss_rates.item()
            ttl_loss += loss_rates

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

        if len(iterator) >= 10 and (i + 1) % (len(iterator) // 10) == 0:
            print("==>{}: {}, {}, {}".format((i + 1) // (len(iterator) // 10), epoch_ttl_loss / (i + 1), epoch_train_id_loss / (i + 1), epoch_rate_loss / (i + 1)))
        time_ttl2 += time.time() - t1
    time_ttl += time.time() - t0
    # print(time_ttl, time_ttl - time_ttl2, time_move, time_forward, time_loss, time_zero, time_gradient, time_update)
    # print(np.sum(model.timer6), np.sum(model.timer1), np.sum(model.timer2), np.sum(model.timer3), np.sum(model.timer4), np.sum(model.timer5))

    return epoch_ttl_loss / len(iterator), epoch_train_id_loss / len(iterator), epoch_rate_loss / len(iterator)


def evaluate(model, iterator, rid_features_dict, parameters, device):
    criterion_reg = nn.L1Loss(reduction='sum')
    criterion_bce = nn.BCELoss(reduction='sum')

    epoch_train_id_loss = 0
    epoch_rate_loss = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates = batch

            src_pro_feas = src_pro_feas.to(device, non_blocking=True)
            trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
            src_seqs = src_seqs.permute(1, 0, 2).to(device, non_blocking=True)
            src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
            src_seg_feats = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)
            trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
            trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)

            da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
            da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
            d_rids = d_rids.to(device, non_blocking=True)
            d_rates = d_rates.to(device, non_blocking=True)

            output_ids, output_rates = model(src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                                          src_pro_feas, rid_features_dict,
                                                          da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates,
                                                          teacher_forcing_ratio=0)

            trg_lengths_sub = [length - 2 for length in trg_lengths]
            loss_train_ids = criterion_bce(output_ids, trg_rid_labels) * parameters.lambda1 / np.sum(np.array(trg_lengths_sub) * np.array(da_lengths))
            if parameters.rate_flag:
                loss_rates = criterion_reg(output_rates, trg_rates[1:-1]) * parameters.lambda2 / sum(trg_lengths_sub)

                epoch_rate_loss += loss_rates.item()

            epoch_train_id_loss += loss_train_ids.item()

            # if (i + 1) % (len(iterator) // 10) == 0:
            #     print("==> Valid: {}".format((i + 1) // (len(iterator) // 10)))
        print((epoch_train_id_loss + epoch_rate_loss) / (i + 1), epoch_train_id_loss / (i + 1), epoch_rate_loss / (i + 1))

        return (epoch_train_id_loss + epoch_rate_loss) / len(iterator), epoch_train_id_loss / len(iterator), epoch_rate_loss / len(iterator)


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
    for pred_seg, pred_rate, trg_id, trg_rate, trg_gps, length, route, route_len in zip(predict_id, predict_rate, target_id, target_rate, target_gps, trg_len, routes, route_lengths):
        results.append([pred_seg[:length], pred_rate[:length], trg_id[:length], trg_rate[:length], trg_gps[:length], route[:route_len]])
    return results


def infer(model, iterator, rid_features_dict, device):
    data = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates = batch

            src_pro_feas = src_pro_feas.to(device, non_blocking=True)
            trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
            src_seqs = src_seqs.permute(1, 0, 2).to(device, non_blocking=True)
            src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
            src_seg_feats = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)
            trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
            trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)

            da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
            da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
            d_rids = d_rids.to(device, non_blocking=True)
            d_rates = d_rates.to(device, non_blocking=True)

            output_ids, output_rates = model(src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                                          src_pro_feas, rid_features_dict,
                                                          da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates,
                                                          teacher_forcing_ratio=-1)

            output_tmp = (F.one_hot(output_ids.argmax(-1), da_routes.shape[0]) * da_routes.permute(1, 0).unsqueeze(1).repeat(1, trg_rid_labels.shape[0], 1).permute(1, 0, 2)).sum(dim=-1)
            output_rates = output_rates.squeeze(2)
            trg_rates = trg_rates.squeeze(2)
            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2)
            trg_lengths_sub = [length - 2 for length in trg_lengths]

            results = get_results(output_tmp, output_rates, trg_rids[1:-1], trg_rates[1:-1], trg_gps_seqs[1:-1], trg_lengths_sub, da_routes, da_lengths)
            data.extend(results)

            if (i + 1) % (len(iterator) // 10) == 0:
                print("==> Test: {}".format((i + 1) // (len(iterator) // 10)))

    return data


def main():
    parser = argparse.ArgumentParser(description='TRMMA')  # 创建命令行参数解析器，描述为TRMMA
    parser.add_argument('--city', type=str, default='porto')  # 城市名称，字符串类型，默认porto
    parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')  # 保留比例，浮点型，默认0.125
    parser.add_argument('--tf_ratio', type=float, default=1, help='teaching ratio in float')  # 教师强制比率，浮点型，默认1
    parser.add_argument('--lambda1', type=int, default=10, help='weight for multi task id')  # 多任务ID损失权重，整型，默认10
    parser.add_argument('--lambda2', type=float, default=5, help='weight for multi task rate')  # 多任务rate损失权重，浮点型，默认5
    parser.add_argument('--hid_dim', type=int, default=256, help='hidden dimension')  # 隐藏层维度，整型，默认256
    parser.add_argument('--epochs', type=int, default=30, help='epochs')  # 训练轮数，整型，默认30
    parser.add_argument('--batch_size', type=int, default=4)  # 批次大小，整型，默认4
    parser.add_argument('--lr', type=float, default=1e-3)  # 学习率，浮点型，默认1e-3
    parser.add_argument('--transformer_layers', type=int, default=2)  # transformer层数，整型，默认2
    parser.add_argument('--heads', type=int, default=4)  # 多头注意力头数，整型，默认4
    parser.add_argument("--gpu_id", type=str, default="0")  # GPU编号，字符串类型，默认0
    parser.add_argument('--model_old_path', type=str, default='', help='old model path')  # 旧模型路径，字符串类型，默认空
    parser.add_argument('--train_flag', action='store_true', help='flag of training')  # 训练标志，布尔型，出现则为True
    parser.add_argument('--test_flag', action='store_true', help='flag of testing')  # 测试标志，布尔型，出现则为True
    parser.add_argument('--small', action='store_true')  # 是否使用小数据集，布尔型，出现则为True
    parser.add_argument('--eid_cate', type=str, default='gps2seg')  # eid类别，字符串类型，默认gps2seg
    parser.add_argument('--inferred_seg_path', type=str, default='')  # 推断分段路径，字符串类型，默认空
    parser.add_argument('--da_route_flag', action='store_true')  # 是否使用DA路线，布尔型，出现则为True
    parser.add_argument('--srcseg_flag', action='store_true')  # 是否使用源分段，布尔型，出现则为True
    parser.add_argument('--gps_flag', action='store_true')  # 是否使用GPS，布尔型，出现则为True
    parser.add_argument('--debug', action='store_true')  # 调试模式，布尔型，出现则为True
    parser.add_argument('--planner', type=str, default='da')  # 路径规划器类型，字符串类型，默认da
    parser.add_argument('--num_worker', type=int, default=8)  # 数据加载线程数，整型，默认8

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
    if opts.model_old_path != '':
        model_save_path = opts.model_old_path
        load_pretrained_flag = True
    else:
        model_save_root = f'./model/TRMMA/{opts.city}/'
        model_save_path = model_save_root + 'TR_' + opts.city + '_' + 'keep-ratio_' + str(opts.keep_ratio) + '_' + time.strftime("%Y%m%d_%H%M%S") + '/'

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

    args = AttrDict()  # 创建参数字典对象
    args_dict = {  # 定义模型参数字典
        'device': device,  # 设备类型（CPU或GPU）
        'transformer_layers': opts.transformer_layers,  # Transformer层数
        'heads': opts.heads,  # 注意力头数
        'tandem_fea_flag': True,  # 串联特征标志
        'pro_features_flag': True,  # 专业特征标志
        'srcseg_flag': opts.srcseg_flag,  # 源段标志
        'da_route_flag': opts.da_route_flag,  # DA路由标志
        'rate_flag': True,  # 速率标志
        'prog_flag': False,  # 进度标志
        'dest_type': 2,  # 目标类型
        'gps_flag': opts.gps_flag,  # GPS标志
        'rid_feats_flag': True,  # 路段ID特征标志
        'learn_pos': True,  # 学习位置编码标志

        # constraint  # 约束参数
        'search_dist': 50,  # 搜索距离
        'beta': 15,  # Beta参数
        'gamma': 30,  # Gamma参数

        # extra info module  # 额外信息模块参数
        'rid_fea_dim': 18,  # 路段特征维度：1[标准化长度] + 8[道路类型] + 1[入度] + 1[出度]
        'pro_input_dim': 48,  # 专业输入维度：24[小时] + 1[节假日]
        'pro_output_dim': 8,  # 专业输出维度

        # MBR  # 最小边界矩形参数
        'min_lat': zone_range[0],  # 最小纬度
        'min_lng': zone_range[1],  # 最小经度
        'max_lat': zone_range[2],  # 最大纬度
        'max_lng': zone_range[3],  # 最大经度

        # input data params  # 输入数据参数
        'city': opts.city,  # 城市名称
        'keep_ratio': opts.keep_ratio,  # 保留比例
        'grid_size': 50,  # 网格大小
        'time_span': ts,  # 时间跨度

        # model params  # 模型参数
        'hid_dim': opts.hid_dim,  # 隐藏层维度
        'id_emb_dim': opts.hid_dim,  # ID嵌入维度
        'dropout': 0.1,  # Dropout比例
        'id_size': rn.valid_edge_cnt_one,  # ID大小

        'lambda1': opts.lambda1,  # Lambda1参数
        'lambda2': opts.lambda2,  # Lambda2参数
        'n_epochs': opts.epochs,  # 训练轮数
        'batch_size': opts.batch_size,  # 批次大小
        'learning_rate': opts.lr,  # 学习率
        "lr_step": 2,  # 学习率调整步长
        "lr_decay": 0.8,  # 学习率衰减率
        'tf_ratio': opts.tf_ratio,  # Teacher forcing比例
        'decay_flag': True,  # 衰减标志
        'decay_ratio': 0.9,  # 衰减比例
        'clip': 1,  # 梯度裁剪阈值
        'log_step': 1,  # 日志记录步长

        'utc': utc,  # UTC时区偏移
        'small': opts.small,  # 小数据集标志
        'dam_root': os.path.join("data", opts.city),  # DAM数据根目录
        'eid_cate': opts.eid_cate,  # 边ID类别
        'inferred_seg_path': opts.inferred_seg_path,  # 推断段路径
        'planner': opts.planner,  # 规划器类型
        'debug': opts.debug,  # 调试模式标志
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
        # load dataset
        train_dataset = TrajRecData(rn, traj_root, mbr, args, 'train')
        valid_dataset = TrajRecData(rn, traj_root, mbr, args, 'valid')
        print('training dataset shape: ' + str(len(train_dataset)))
        print('validation dataset shape: ' + str(len(valid_dataset)))
        logging.info('Finish data preparing.')
        logging.info('training dataset shape: ' + str(len(train_dataset)))
        logging.info('validation dataset shape: ' + str(len(valid_dataset)))

        train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x), num_workers=opts.num_worker, pin_memory=False)
        valid_iterator = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x), num_workers=8, pin_memory=False)

        model = TrajRecovery(args).to(device)

        if load_pretrained_flag:
            model = torch.load(os.path.join(model_save_path, 'val-best-model.pt'), map_location=device)

        print('model', str(model))
        logging.info('model' + str(model))

        num_params = 0
        seg_params = []
        rate_params = []
        for name, param in model.named_parameters():
            # print(num_params, name, param.shape)
            num_params += 1
            if 'fc_rate_out' not in name:
                seg_params.append(param)
            else:
                rate_params.append(param)
        print(num_params)

        ls_train_loss, ls_train_id_acc1, ls_train_id_recall, ls_train_id_precision, \
            ls_train_rate_loss, ls_train_id_loss, ls_train_mae, ls_train_rmse = [], [], [], [], [], [], [], []
        ls_valid_loss, ls_valid_id_acc1, ls_valid_id_recall, ls_valid_id_precision, \
            ls_valid_rate_loss, ls_valid_id_loss, ls_valid_mae, ls_valid_rmse = [], [], [], [], [], [], [], []

        best_valid_loss = float('inf')  # compare id loss
        best_epoch = 0

        tb_writer = SummaryWriter(log_dir=os.path.join(model_save_path, 'tensorboard'))

        # get all parameters (model parameters + task dependent log variances)
        lr = args.learning_rate
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_step,
                                                         factor=args.lr_decay, threshold=1e-3)
        stopping_count = 0
        train_times = []
        for epoch in tqdm(range(args.n_epochs), desc='epoch num'):
            start_time = time.time()

            print("==> training {}, {}...".format(args.tf_ratio, lr))
            t_train = time.time()
            train_loss, train_id_loss, train_rate_loss = train(model, train_iterator, optimizer, rid_features_dict, args, device)
            end_train = time.time()
            print("training: {}".format(end_train - t_train))

            ls_train_loss.append(train_loss)
            ls_train_id_loss.append(train_id_loss)
            ls_train_rate_loss.append(train_rate_loss)

            print("==> validating...")
            t_valid = time.time()
            valid_loss, valid_id_loss, valid_rate_loss = evaluate(model, valid_iterator, rid_features_dict, args, device)
            print("validating: {}".format(time.time() - t_valid))

            ls_valid_id_loss.append(valid_id_loss)
            ls_valid_rate_loss.append(valid_rate_loss)
            ls_valid_loss.append(valid_loss)

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

            tb_writer.add_scalars('Train_loss', {'total': train_loss, 'RID': train_id_loss, 'Rate': train_rate_loss}, epoch)
            tb_writer.add_scalars('Valid_loss', {'total': valid_loss, 'RID': valid_id_loss, 'Rate': valid_rate_loss}, epoch)
            tb_writer.add_scalar('learning_rate', lr, epoch)
            tb_writer.add_scalars('TTL_loss', {'Train': train_loss, 'Valid': valid_loss}, epoch)
            tb_writer.add_scalars('Seg_loss', {'Train': train_id_loss, 'Valid': valid_id_loss}, epoch)
            tb_writer.add_scalars('Rate_loss', {'Train': train_rate_loss, 'Valid': valid_rate_loss}, epoch)

            if (epoch % args.log_step == 0) or (epoch == args.n_epochs - 1):
                logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_secs) + 's')
                logging.info('Epoch: ' + str(epoch + 1) + ' TF Ratio: ' + str(args.tf_ratio))
                logging.info('\tTrain Loss:' + str(train_loss) +
                             '\tTrain RID Loss:' + str(train_id_loss) +
                             '\tTrain Rate Loss:' + str(train_rate_loss))
                logging.info('\tValid Loss:' + str(valid_loss) +
                             '\tValid RID Loss:' + str(valid_id_loss) +
                             '\tValid Rate Loss:' + str(valid_rate_loss))

                torch.save(model, os.path.join(model_save_path, 'train-mid-model.pt'))
            if args.decay_flag:
                args.tf_ratio = args.tf_ratio * args.decay_ratio

            scheduler.step(valid_id_loss)
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

    if opts.test_flag:
        test_dataset = TrajRecTestData(rn, traj_root, mbr, args, 'test')
        print('testing dataset shape: ' + str(len(test_dataset)))
        logging.info('testing dataset shape: ' + str(len(test_dataset)))

        test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn_test(x), num_workers=8, pin_memory=True)

        model = torch.load(os.path.join(model_save_path, 'val-best-model.pt'), map_location=device)
        print('==> Model Loaded')

        print("==> Starting Prediction...")
        start_time = time.time()
        data = infer(model, test_iterator, rid_features_dict, device)
        end_time = time.time()
        epoch_secs = end_time - start_time
        print('Time: ' + str(epoch_secs) + 's')
        logging.info('Inference Time: {}, {}, {}'.format(end_time - start_time, (end_time - start_time) / len(test_dataset) * 1000, len(test_dataset) / (end_time - start_time)))
        print('Inference Time: {}, {}, {}'.format(end_time - start_time, (end_time - start_time) / len(test_dataset) * 1000, len(test_dataset) / (end_time - start_time)))
        pickle.dump(data, open(os.path.join(model_save_path, 'infer_output_{}_{}.pkl'.format(opts.planner, opts.eid_cate)), "wb"))

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
            # predict_ids = []
            # predict_gps = []
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
                # if i not in low_idx:
                mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
            assert len(predict_gps) == len(mm_gps_seq) == len(predict_ids) == len(mm_eids)
            results.append([predict_gps, predict_ids, mm_gps_seq, mm_eids])
        pickle.dump(results, open(os.path.join(model_save_path, 'recovery_output_{}_{}.pkl'.format(opts.planner, opts.eid_cate)), "wb"))

        print("==> Starting Evaluation...")
        epoch_id1_loss = []
        epoch_recall_loss = []
        epoch_precision_loss = []
        epoch_f1_loss = []
        epoch_mae_loss = []
        epoch_rmse_loss = []
        for pred_gps, pred_seg, trg_gps, trg_id in results:
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

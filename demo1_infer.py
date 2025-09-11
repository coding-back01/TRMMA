from models.demo1 import GPS2SegData, TrajRecTestData, DAPlanner
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
from utils.map import RoadNetworkMapFull
from utils.spatial_func import SPoint
from utils.mbr import MBR
from utils.model_utils import gps2grid, AttrDict
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


def get_results_mma(predict_id, target_id, lengths):
    """MMA结果处理函数"""
    predict_id = predict_id.detach().cpu().tolist()

    results = []
    for pred, trg, length in zip(predict_id, target_id, lengths):
        results.append([pred[:length], trg])
    return results


def get_results_trmma(predict_id, predict_rate, target_id, target_rate, target_gps, trg_len, routes, route_lengths, inverse_flag=True):
    """TRMMA结果处理函数"""
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


def infer_mma(model, iterator, device):
    """MMA推理函数"""
    data = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_seqs, src_lengths, trg_rids, _, candi_ids, candi_feats, candi_masks = batch

            src_seqs = src_seqs.to(device, non_blocking=True)
            candi_ids = candi_ids.to(device, non_blocking=True)
            candi_feats = candi_feats.to(device, non_blocking=True)
            candi_masks = candi_masks.to(device, non_blocking=True)

            output_ids = model(src_seqs, src_lengths, candi_ids, candi_feats, candi_masks)

            candi_size = candi_ids.shape[-1]
            output_tmp = (F.one_hot(output_ids.argmax(-1), candi_size) * candi_ids).sum(dim=-1) - 1

            results = get_results_mma(output_tmp, trg_rids, src_lengths)
            data.extend(results)

            if (i + 1) % (len(iterator) // 10) == 0:
                print("==> Test: {}".format((i + 1) // (len(iterator) // 10)))

    return data


def infer_trmma(model, iterator, rid_features_dict, device):
    """TRMMA推理函数"""
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

            results = get_results_trmma(output_tmp, output_rates, trg_rids[1:-1], trg_rates[1:-1], trg_gps_seqs[1:-1], trg_lengths_sub, da_routes, da_lengths)
            data.extend(results)

            if (i + 1) % (len(iterator) // 10) == 0:
                print("==> Test: {}".format((i + 1) // (len(iterator) // 10)))

    return data


def main():
    parser = argparse.ArgumentParser(description='Demo1 Combined Inference')
    
    # 通用参数
    parser.add_argument('--city', type=str, default='porto')
    parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')
    parser.add_argument('--hid_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument('--model_old_path', type=str, default='', help='old model path')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--init_ratio', type=float, default=0.5)
    
    # MMA特定参数
    parser.add_argument('--attn_flag', action='store_true', help='flag of using attention')
    parser.add_argument('--direction_flag', action='store_true')
    parser.add_argument("--candi_size", type=int, default=10)
    parser.add_argument('--only_direction', action='store_true')
    
    # TRMMA特定参数
    parser.add_argument('--tf_ratio', type=float, default=1, help='teaching ratio in float')
    parser.add_argument('--lambda1', type=int, default=10, help='weight for multi task id')
    parser.add_argument('--lambda2', type=float, default=5, help='weight for multi task rate')
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--eid_cate', type=str, default='gps2seg')
    parser.add_argument('--inferred_seg_path', type=str, default='')
    parser.add_argument('--da_route_flag', action='store_true')
    parser.add_argument('--srcseg_flag', action='store_true')
    parser.add_argument('--gps_flag', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--planner', type=str, default='da')
    
    # 模型选择
    parser.add_argument('--model_type', type=str, default='mma', choices=['mma', 'trmma'], help='选择推理哪个模型')

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

    if opts.model_old_path == '':
        raise ValueError("model path error - must provide model_old_path for inference")
    
    model_save_path = opts.model_old_path

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
    # 修复 porto 数据集路径映射
    if opts.city == "porto":
        map_root = os.path.join("data", "porto", "roadnet")
    else:
        map_root = os.path.join("data", opts.city, "roadnet")
    rn = RoadNetworkMapFull(map_root, zone_range=zone_range, unit_length=50)

    args = AttrDict()
    
    if opts.model_type == 'mma':
        # MMA参数配置
        args_dict = {
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
            'init_ratio': opts.init_ratio,
            'only_direction': opts.only_direction,
            'cate': "g2s",
            'threshold': 1
        }
    else:  # trmma
        # TRMMA参数配置
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
            'eid_cate': opts.eid_cate,
            'inferred_seg_path': opts.inferred_seg_path,
            'planner': opts.planner,
            'debug': opts.debug,
        }
    
    args.update(args_dict)

    mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
    args.grid_num = gps2grid(SPoint(args.max_lat, args.max_lng), mbr, args.grid_size)
    args.grid_num = (args.grid_num[0] + 1, args.grid_num[1] + 1)

    print(args)
    logging.info(args_dict)

    # 修复 porto 数据集轨迹路径映射
    if args.city == "porto":
        traj_root = os.path.join("data", "porto")
    else:
        traj_root = os.path.join("data", args.city)

    if opts.model_type == 'mma':
        # MMA推理逻辑
        test_dataset = GPS2SegData(rn, traj_root, mbr, args, 'test')
        print('testing dataset shape: ' + str(len(test_dataset)))
        logging.info('testing dataset shape: ' + str(len(test_dataset)))

        test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                 collate_fn=lambda x: collate_fn_mma(x), num_workers=8, pin_memory=True)

        model = torch.load(os.path.join(model_save_path, 'val-best-model.pt'), map_location=device)
        print('==> Model Loaded')

        print("==> Predicting...")
        start_time = time.time()
        pred_data = infer_mma(model, test_iterator, device)
        end_time = time.time()
        epoch_secs = end_time - start_time
        print('Time: ' + str(epoch_secs) + 's')
        logging.info('Inference Time: {}, {}, {}'.format(end_time - start_time, (end_time - start_time) / len(test_dataset) * 1000, len(test_dataset) / (end_time - start_time)))
        print('Inference Time: {}, {}, {}'.format(end_time - start_time, (end_time - start_time) / len(test_dataset) * 1000, len(test_dataset) / (end_time - start_time)))

        print("==> Starting Evaluation...")
        epoch_id1_loss = []
        epoch_recall_loss = []
        epoch_precision_loss = []
        epoch_f1_loss = []
        for tmp_predict, tmp_target in pred_data:
            rid_acc, rid_recall, rid_precision, rid_f1 = cal_id_acc(tmp_predict, tmp_target)
            epoch_id1_loss.append(rid_acc)
            epoch_recall_loss.append(rid_recall)
            epoch_precision_loss.append(rid_precision)
            epoch_f1_loss.append(rid_f1)

        pickle.dump(pred_data, open(os.path.join(model_save_path, 'infer_output.pkl'), "wb"))

        test_id_acc, test_id_recall, test_id_precision, test_id_f1 = np.mean(epoch_id1_loss), np.mean(
            epoch_recall_loss), np.mean(epoch_precision_loss), np.mean(epoch_f1_loss)
        print(test_id_recall, test_id_precision, test_id_f1, test_id_acc)

        logging.info('Time: ' + str(epoch_secs) + 's')
        logging.info('\tTest RID Acc:' + str(test_id_acc) +
                     '\tTest RID Recall:' + str(test_id_recall) +
                     '\tTest RID Precision:' + str(test_id_precision) +
                     '\tTest RID F1 Score:' + str(test_id_f1))

    else:  # trmma
        # TRMMA推理逻辑
        dam = DAPlanner(args.dam_root, args.id_size - 1, args.utc)
        rid_features_dict = torch.from_numpy(rn.get_rid_rnfea_dict(dam, ts)).to(device)

        test_dataset = TrajRecTestData(rn, traj_root, mbr, args, 'test')
        print('testing dataset shape: ' + str(len(test_dataset)))
        logging.info('testing dataset shape: ' + str(len(test_dataset)))

        test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                 collate_fn=lambda x: collate_fn_trmma_test(x), num_workers=8, pin_memory=True)

        model = torch.load(os.path.join(model_save_path, 'val-best-model.pt'), map_location=device)
        print('==> Model Loaded')

        print("==> Starting Prediction...")
        start_time = time.time()
        data = infer_trmma(model, test_iterator, rid_features_dict, device)
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

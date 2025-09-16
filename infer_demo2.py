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
from utils.map import RoadNetworkMapFull
from utils.spatial_func import SPoint
from utils.mbr import MBR
from models.trmma import DAPlanner, TrajRecTestData
from models.demo2 import E2ETrajData
from utils.model_utils import AttrDict, gps2grid
import numpy as np
from collections import Counter

from train_demo2 import collate_fn, infer


def main():
    parser = argparse.ArgumentParser(description='infer_E2E')
    parser.add_argument('--city', type=str, default='porto')
    parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')
    parser.add_argument('--tf_ratio', type=float, default=1, help='teaching ratio in float')
    parser.add_argument('--lambda_mma', type=float, default=1.0, help='weight for mma bce')
    parser.add_argument('--lambda1', type=float, default=10, help='weight for seg bce')
    parser.add_argument('--lambda2', type=float, default=5, help='weight for rate l1')
    parser.add_argument('--hid_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument('--model_old_path', type=str, default='', help='old model path')
    parser.add_argument('--test_flag', action='store_true', help='flag of testing')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--direction_flag', action='store_true')
    parser.add_argument('--attn_flag', action='store_true')
    parser.add_argument("--candi_size", type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--init_ratio', type=float, default=0.5)
    parser.add_argument('--only_direction', action='store_true')
    parser.add_argument('--da_route_flag', action='store_true')
    parser.add_argument('--srcseg_flag', action='store_true')
    parser.add_argument('--gps_flag', action='store_true')
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
    map_root = os.path.join("data", opts.city, "roadnet")
    rn = RoadNetworkMapFull(map_root, zone_range=zone_range, unit_length=50)

    args = AttrDict()
    args_dict = {
        'device': device,
        'transformer_layers': opts.transformer_layers,
        'heads': opts.heads,
        'direction_flag': opts.direction_flag,
        'attn_flag': opts.attn_flag,
        'candi_size': opts.candi_size,
        'init_ratio': opts.init_ratio,
        'only_direction': opts.only_direction,
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

        # train (for compatibility)
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
        'planner': opts.planner,
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

    if opts.test_flag:
        test_dataset = E2ETrajData(rn, traj_root, mbr, args, 'test')
        print('testing dataset shape: ' + str(len(test_dataset)))
        logging.info('testing dataset shape: ' + str(len(test_dataset)))

        test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x), num_workers=opts.num_worker, pin_memory=True)

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
        pickle.dump(data, open(os.path.join(model_save_path, 'infer_output_e2e.pkl'), "wb"))

        outputs = []
        for pred_seg, pred_rate, trg_id, trg_rate, route in data:
            pred_gps = toseq(rn, pred_seg, pred_rate, route, dam.seg_info)
            trg_gps = toseq(rn, trg_id, trg_rate, route, dam.seg_info)
            outputs.append([pred_gps, pred_seg, trg_gps, trg_id])

        print("==> Starting Evaluation...")
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

        test_id_recall = np.mean(epoch_recall_loss) if len(epoch_recall_loss) > 0 else 0
        test_id_precision = np.mean(epoch_precision_loss) if len(epoch_precision_loss) > 0 else 0
        test_id_f1 = np.mean(epoch_f1_loss) if len(epoch_f1_loss) > 0 else 0
        test_id_acc = np.mean(epoch_id1_loss) if len(epoch_id1_loss) > 0 else 0
        test_mae = np.mean(epoch_mae_loss) if len(epoch_mae_loss) > 0 else 0
        test_rmse = np.mean(epoch_rmse_loss) if len(epoch_rmse_loss) > 0 else 0
        print(test_id_recall, test_id_precision, test_id_f1, test_id_acc, test_mae, test_rmse)

        logging.info('Time: ' + str(epoch_secs) + 's')
        logging.info('\tTest RID Acc:' + str(test_id_acc) +
                     '\tTest RID Recall:' + str(test_id_recall) +
                     '\tTest RID Precision:' + str(test_id_precision) +
                     '\tTest RID F1 Score:' + str(test_id_f1) +
                     '\tTest MAE Loss:' + str(test_mae) +
                     '\tTest RMSE Loss:' + str(test_rmse))

        test_trajs = test_dataset.trajs
        groups = Counter(test_dataset.groups)
        nums = []
        for i in range(len(test_trajs)):
            nums.append(groups[i])

        outputs2 = outputs.copy()
        results = []
        for traj, num, src_mm in zip(test_trajs, nums, test_dataset.src_mms):
            tmp_all = outputs2[:num]
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
            outputs2 = outputs2[num:]

            mm_gps_seq = []
            mm_eids = []
            for pt in traj.pt_list:
                candi_pt = pt.data['candi_pt']
                mm_eids.append(candi_pt.eid)
                mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
            assert len(predict_gps) == len(mm_gps_seq) == len(predict_ids) == len(mm_eids)
            results.append([predict_gps, predict_ids, mm_gps_seq, mm_eids])

        pickle.dump(results, open(os.path.join(model_save_path, 'recovery_output_e2e.pkl'), "wb"))

if __name__ == '__main__':
    main()




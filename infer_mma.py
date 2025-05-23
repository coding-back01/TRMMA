from models.mma import GPS2SegData
from utils.evaluation_utils import cal_id_acc
import random
import time
import logging

import os
import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from utils.map import RoadNetworkMapFull
from utils.spatial_func import SPoint
from utils.mbr import MBR
from utils.model_utils import gps2grid, AttrDict
import numpy as np

from train_mma import collate_fn, infer


def main():
    parser = argparse.ArgumentParser(description='infer_MMA')
    parser.add_argument('--city', type=str, default='porto')
    parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')
    parser.add_argument('--hid_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--attn_flag', action='store_true', help='flag of using attention')
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument("--gpu_id", type=str, default="1")
    parser.add_argument('--model_old_path', type=str, default='', help='old model path')
    parser.add_argument('--train_flag', action='store_true', help='flag of training')
    parser.add_argument('--test_flag', action='store_true', help='flag of testing')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--direction_flag', action='store_true')
    parser.add_argument("--candi_size", type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--init_ratio', type=float, default=0.5)
    parser.add_argument('--only_direction', action='store_true')

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
        raise "model path error"

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
        'candi_size': opts.candi_size,
        # attention
        'attn_flag': opts.attn_flag,
        'direction_flag': opts.direction_flag,
        'gps_flag': False,

        # constraint
        'search_dist': 50,
        'beta': 15,
        'gamma': 30,

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
    args.update(args_dict)

    mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
    args.grid_num = gps2grid(SPoint(args.max_lat, args.max_lng), mbr, args.grid_size)
    args.grid_num = (args.grid_num[0] + 1, args.grid_num[1] + 1)

    print(args)
    logging.info(args_dict)

    traj_root = os.path.join("data", args.city)
    if opts.test_flag:
        test_dataset = GPS2SegData(rn, traj_root, mbr, args, 'test')
        print('testing dataset shape: ' + str(len(test_dataset)))
        logging.info('testing dataset shape: ' + str(len(test_dataset)))

        test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x), num_workers=8, pin_memory=True)

        model = torch.load(os.path.join(model_save_path, 'val-best-model.pt'), map_location=device)
        print('==> Model Loaded')

        print("==> Predicting...")
        start_time = time.time()
        pred_data = infer(model, test_iterator, device)
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


if __name__ == '__main__':
    main()

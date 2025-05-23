import torch.nn as nn

from models.mma import GPS2SegData, GPS2Seg
from utils.evaluation_utils import cal_id_acc
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


def collate_fn(data):
    src_seqs, trg_rids, candi_onehots, candi_ids, candi_feats, candi_masks = zip(*data)

    lengths = [len(seq) for seq in src_seqs]
    src_seqs = rnn_utils.pad_sequence(src_seqs, batch_first=True, padding_value=0)

    candi_onehots = rnn_utils.pad_sequence(candi_onehots, batch_first=True, padding_value=0)
    candi_ids = rnn_utils.pad_sequence(candi_ids, batch_first=True, padding_value=0)
    candi_feats = rnn_utils.pad_sequence(candi_feats, batch_first=True, padding_value=0)
    candi_masks = rnn_utils.pad_sequence(candi_masks, batch_first=True, padding_value=0)

    return src_seqs, lengths, trg_rids, candi_onehots, candi_ids, candi_feats, candi_masks


def train(model, iterator, optimizer, device):
    criterion_bce = nn.BCELoss(reduction='mean')

    epoch_train_id_loss = 0
    model.train()
    for i, batch in enumerate(iterator):
        src_seqs, src_lengths, _, candi_labels, candi_ids, candi_feats, candi_masks = batch

        src_seqs = src_seqs.to(device, non_blocking=True)
        candi_labels = candi_labels.float().to(device, non_blocking=True)
        candi_ids = candi_ids.to(device, non_blocking=True)
        candi_feats = candi_feats.to(device, non_blocking=True)
        candi_masks = candi_masks.to(device, non_blocking=True)

        output_ids = model(src_seqs, src_lengths, candi_ids, candi_feats, candi_masks)

        # for bbp
        bce_loss = criterion_bce(output_ids, candi_labels) * candi_ids.shape[-1]

        optimizer.zero_grad(set_to_none=True)
        bce_loss.backward()
        optimizer.step()

        epoch_train_id_loss += bce_loss.item()

        if len(iterator) >= 10 and (i + 1) % (len(iterator) // 10) == 0:
            print("==>{}: {}".format((i + 1) // (len(iterator) // 10), epoch_train_id_loss / (i + 1)))

    return epoch_train_id_loss / len(iterator)


def evaluate(model, iterator, device):
    model.eval()

    epoch_train_id_loss = 0
    criterion_bce = nn.BCELoss(reduction='mean')

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_seqs, src_lengths, _, candi_labels, candi_ids, candi_feats, candi_masks = batch

            src_seqs = src_seqs.to(device, non_blocking=True)
            candi_labels = candi_labels.float().to(device, non_blocking=True)
            candi_ids = candi_ids.to(device, non_blocking=True)
            candi_feats = candi_feats.to(device, non_blocking=True)
            candi_masks = candi_masks.to(device, non_blocking=True)

            output_ids = model(src_seqs, src_lengths, candi_ids, candi_feats, candi_masks)

            bce_loss = criterion_bce(output_ids, candi_labels) * candi_ids.shape[-1]

            epoch_train_id_loss += bce_loss.item()
        print("==> Valid: {}".format(epoch_train_id_loss / (i + 1)))

        return epoch_train_id_loss / len(iterator)


def get_results(predict_id, target_id, lengths):
    predict_id = predict_id.detach().cpu().tolist()

    results = []
    for pred, trg, length in zip(predict_id, target_id, lengths):
        results.append([pred[:length], trg])
    return results


def infer(model, iterator, device):
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

            results = get_results(output_tmp, trg_rids, src_lengths)
            data.extend(results)

            if (i + 1) % (len(iterator) // 10) == 0:
                print("==> Test: {}".format((i + 1) // (len(iterator) // 10)))

    return data


def main():
    parser = argparse.ArgumentParser(description='MMA')
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
        model_save_root = f'./model/TRMMA/{opts.city}/'
        model_save_path = model_save_root + 'MMA_' + opts.city + '_' + 'keep-ratio_' + str(opts.keep_ratio) + '_' + time.strftime("%Y%m%d_%H%M%S") + '/'

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
    if opts.train_flag:
        train_dataset = GPS2SegData(rn, traj_root, mbr, args, 'train')
        valid_dataset = GPS2SegData(rn, traj_root, mbr, args, 'valid')
        print('training dataset shape: ' + str(len(train_dataset)))
        print('validation dataset shape: ' + str(len(valid_dataset)))
        logging.info('Finish data preparing.')
        logging.info('training dataset shape: ' + str(len(train_dataset)))
        logging.info('validation dataset shape: ' + str(len(valid_dataset)))

        train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x), num_workers=opts.num_worker, pin_memory=False)
        valid_iterator = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x), num_workers=8, pin_memory=False)

        model = GPS2Seg(args).to(device)

        if load_pretrained_flag:
            model = torch.load(os.path.join(model_save_path, 'val-best-model.pt'))

        print('model', str(model))
        logging.info('model' + str(model))

        ls_train_id_loss = []
        ls_valid_id_loss = []

        best_valid_loss = float('inf')
        best_epoch = 0

        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        stopping_count = 0
        train_times = []
        for epoch in tqdm(range(args.n_epochs), desc='epoch num'):
            print("==> training {} ...".format(train_iterator.dataset.keep_ratio))
            start_time = time.time()
            train_id_loss = train(model, train_iterator, optimizer, device)
            end_time = time.time()
            epoch_secs = end_time - start_time
            train_times.append(end_time - start_time)

            ls_train_id_loss.append(train_id_loss)

            print("==> validating...")
            valid_id_loss = evaluate(model, valid_iterator, device)

            ls_valid_id_loss.append(valid_id_loss)

            if valid_id_loss < best_valid_loss:
                best_valid_loss = valid_id_loss
                torch.save(model, os.path.join(model_save_path, 'val-best-model.pt'))
                best_epoch = epoch
                stopping_count = 0
            else:
                stopping_count += 1

            if (epoch % args.log_step == 0) or (epoch == args.n_epochs - 1):
                logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_secs) + 's')
                logging.info('\tTrain RID Loss:' + str(train_id_loss))
                logging.info('\tValid RID Loss:' + str(valid_id_loss))

                torch.save(model, os.path.join(model_save_path, 'train-mid-model.pt'))
            if args.decay_flag:
                train_iterator.dataset.keep_ratio = max(args.keep_ratio, train_iterator.dataset.keep_ratio * args.decay_ratio)

            if stopping_count >= 5:
                print("==> [Info] Early Stop After Epoch {}.".format(epoch))
                break
        logging.info('Best Epoch: {}, {}'.format(best_epoch, best_valid_loss))
        print('==> Best Epoch: {}, {}'.format(best_epoch, best_valid_loss))
        logging.info('==> Training Time: {}, {}, {}'.format(np.mean(train_times), np.min(train_times), np.max(train_times)))
        print('==> Training Time: {}, {}, {}'.format(np.mean(train_times), np.min(train_times), np.max(train_times)))

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

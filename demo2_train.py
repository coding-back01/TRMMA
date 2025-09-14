import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import random
import time
import logging

import os
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from utils.map import RoadNetworkMapFull
from utils.spatial_func import SPoint
from utils.mbr import MBR
from models.demo2 import E2ETrajData, End2EndModel, E2ELoss, collate_fn_e2e
from models.trmma import DAPlanner
from utils.model_utils import AttrDict, gps2grid
from tqdm import tqdm
import numpy as np


def train(model, iterator, optimizer, criterion, rid_features_dict, parameters, device):
    epoch_ttl_loss = 0
    epoch_mma_loss = 0
    epoch_id_loss = 0
    epoch_rate_loss = 0

    model.train()
    for i, batch in enumerate(iterator):
        # 解包 batch
        src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, \
        trg_rids, trg_rates, trg_lengths, trg_rid_labels, \
        da_routes, da_lengths, da_pos, d_rids, d_rates, \
        candi_onehots, candi_ids, candi_feats, candi_masks = batch

        # 保存 MMA 输入（batch-first）
        mma_src_seqs = src_seqs.to(device, non_blocking=True)
        mma_src_lens = src_lengths
        mma_candi_ids = candi_ids.to(device, non_blocking=True)
        mma_candi_feats = candi_feats.to(device, non_blocking=True)
        mma_candi_masks = candi_masks.to(device, non_blocking=True)

        # TRMMA 输入（与现有训练脚本保持一致的维度）
        src_pro_feas = src_pro_feas.to(device, non_blocking=True)
        trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
        src_seqs_tr = mma_src_seqs.permute(1, 0, 2)  # [src_len, bs, 3]
        trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
        trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)

        da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
        da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
        d_rids = d_rids.to(device, non_blocking=True)
        d_rates = d_rates.to(device, non_blocking=True)

        # 源路段信息（若启用 srcseg_flag）
        src_seg_seqs_dev = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
        src_seg_feats_dev = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)

        outputs = model(
            # MMA
            mma_src_seqs, mma_src_lens, mma_candi_ids, mma_candi_feats, mma_candi_masks,
            # TRMMA
            src_seqs_tr, src_lengths, trg_rids, trg_rates, trg_lengths,
            src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos,
            src_seg_seqs_dev, src_seg_feats_dev, d_rids, d_rates, teacher_forcing_ratio=parameters.tf_ratio
        )

        labels = {
            'mma_onehot': candi_onehots.to(device, non_blocking=True),
            'trg_labels': trg_rid_labels,
            'trg_rates': trg_rates
        }
        lengths = {
            'trg_lengths': trg_lengths,
            'da_lengths': da_lengths
        }

        ttl_loss, loss_dict = criterion(outputs, labels, lengths)

        optimizer.zero_grad(set_to_none=True)
        ttl_loss.backward()
        optimizer.step()

        epoch_ttl_loss += ttl_loss.item()
        epoch_mma_loss += loss_dict['mma']
        epoch_id_loss += loss_dict['id']
        epoch_rate_loss += loss_dict['rate']

        if len(iterator) >= 10 and (i + 1) % (len(iterator) // 10) == 0:
            print("==>{}: {}, {}, {}, {}".format((i + 1) // (len(iterator) // 10), epoch_ttl_loss / (i + 1), epoch_mma_loss / (i + 1), epoch_id_loss / (i + 1), epoch_rate_loss / (i + 1)))

    return epoch_ttl_loss / len(iterator), epoch_mma_loss / len(iterator), epoch_id_loss / len(iterator), epoch_rate_loss / len(iterator)


def evaluate(model, iterator, criterion, rid_features_dict, parameters, device):
    epoch_ttl_loss = 0
    epoch_mma_loss = 0
    epoch_id_loss = 0
    epoch_rate_loss = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, \
            trg_rids, trg_rates, trg_lengths, trg_rid_labels, \
            da_routes, da_lengths, da_pos, d_rids, d_rates, \
            candi_onehots, candi_ids, candi_feats, candi_masks = batch

            mma_src_seqs = src_seqs.to(device, non_blocking=True)
            mma_src_lens = src_lengths
            mma_candi_ids = candi_ids.to(device, non_blocking=True)
            mma_candi_feats = candi_feats.to(device, non_blocking=True)
            mma_candi_masks = candi_masks.to(device, non_blocking=True)

            src_pro_feas = src_pro_feas.to(device, non_blocking=True)
            trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
            src_seqs_tr = mma_src_seqs.permute(1, 0, 2)
            trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
            trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)

            da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
            da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
            d_rids = d_rids.to(device, non_blocking=True)
            d_rates = d_rates.to(device, non_blocking=True)

            src_seg_seqs_dev = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
            src_seg_feats_dev = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)

            outputs = model(
                mma_src_seqs, mma_src_lens, mma_candi_ids, mma_candi_feats, mma_candi_masks,
                src_seqs_tr, src_lengths, trg_rids, trg_rates, trg_lengths,
                src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos,
                src_seg_seqs_dev, src_seg_feats_dev, d_rids, d_rates, teacher_forcing_ratio=0
            )

            labels = {
                'mma_onehot': candi_onehots.to(device, non_blocking=True),
                'trg_labels': trg_rid_labels,
                'trg_rates': trg_rates
            }
            lengths = {
                'trg_lengths': trg_lengths,
                'da_lengths': da_lengths
            }

            ttl_loss, loss_dict = criterion(outputs, labels, lengths)

            epoch_ttl_loss += ttl_loss.item()
            epoch_mma_loss += loss_dict['mma']
            epoch_id_loss += loss_dict['id']
            epoch_rate_loss += loss_dict['rate']

        print((epoch_ttl_loss) / (i + 1), epoch_mma_loss / (i + 1), epoch_id_loss / (i + 1), epoch_rate_loss / (i + 1))

    return epoch_ttl_loss / len(iterator), epoch_mma_loss / len(iterator), epoch_id_loss / len(iterator), epoch_rate_loss / len(iterator)


def main():
    parser = argparse.ArgumentParser(description='E2E MMA+TRMMA')
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
    parser.add_argument('--train_flag', action='store_true', help='flag of training')
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
    print('e2e device', device)

    model_save_root = f'./model/TRMMA/{opts.city}/'
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

    # 参数（与两侧模型兼容）
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
        'tf_ratio': opts.tf_ratio,
        'decay_flag': True,
        'decay_ratio': 0.9,
        'clip': 1,
        'log_step': 1,

        'utc': utc,
        'small': opts.small,
        'dam_root': os.path.join("data", opts.city),
        'planner': opts.planner,
        'direction_flag': opts.direction_flag,
        'attn_flag': opts.attn_flag,
        'candi_size': opts.candi_size,
        'init_ratio': opts.init_ratio,
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

    if opts.train_flag:
        train_dataset = E2ETrajData(rn, traj_root, mbr, args, 'train')
        valid_dataset = E2ETrajData(rn, traj_root, mbr, args, 'valid')
        print('training dataset shape: ' + str(len(train_dataset)))
        print('validation dataset shape: ' + str(len(valid_dataset)))
        logging.info('Finish data preparing.')
        logging.info('training dataset shape: ' + str(len(train_dataset)))
        logging.info('validation dataset shape: ' + str(len(valid_dataset)))

        train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn_e2e(x), num_workers=opts.num_worker, pin_memory=False)
        valid_iterator = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn_e2e(x), num_workers=8, pin_memory=False)

        # 将同一份参数用于两侧子模型，确保嵌入维度、id_size 等一致
        mma_args = args
        trmma_args = args
        model = End2EndModel(mma_args, trmma_args).to(device)

        print('model', str(model))
        logging.info('model' + str(model))

        tb_writer = SummaryWriter(log_dir=os.path.join(model_save_path, 'tensorboard'))

        lr = args.learning_rate
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.8, threshold=1e-3)
        criterion = E2ELoss(lambda_mma=opts.lambda_mma, lambda_id=args.lambda1, lambda_rate=args.lambda2)

        best_valid_loss = float('inf')
        best_epoch = 0
        stopping_count = 0
        train_times = []

        for epoch in tqdm(range(args.n_epochs), desc='epoch num'):
            start_time = time.time()

            print("==> training {}, {}...".format(args.tf_ratio, lr))
            t_train = time.time()
            train_loss, train_mma, train_id, train_rate = train(model, train_iterator, optimizer, criterion, rid_features_dict, args, device)
            end_train = time.time()
            print("training: {}".format(end_train - t_train))

            print("==> validating...")
            t_valid = time.time()
            valid_loss, valid_mma, valid_id, valid_rate = evaluate(model, valid_iterator, criterion, rid_features_dict, args, device)
            print("validating: {}".format(time.time() - t_valid))

            tb_writer.add_scalars('Loss_total', {'Train': train_loss, 'Valid': valid_loss}, epoch)
            tb_writer.add_scalars('Loss_mma', {'Train': train_mma, 'Valid': valid_mma}, epoch)
            tb_writer.add_scalars('Loss_id', {'Train': train_id, 'Valid': valid_id}, epoch)
            tb_writer.add_scalars('Loss_rate', {'Train': train_rate, 'Valid': valid_rate}, epoch)
            tb_writer.add_scalar('learning_rate', lr, epoch)

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

            logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_secs) + 's')
            logging.info('Epoch: ' + str(epoch + 1) + ' TF Ratio: ' + str(args.tf_ratio))
            logging.info('\tTrain Total:' + str(train_loss) + '\tMMA:' + str(train_mma) + '\tSeg:' + str(train_id) + '\tRate:' + str(train_rate))
            logging.info('\tValid Total:' + str(valid_loss) + '\tMMA:' + str(valid_mma) + '\tSeg:' + str(valid_id) + '\tRate:' + str(valid_rate))

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


if __name__ == '__main__':
    main()



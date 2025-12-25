import time
from tqdm import tqdm
import logging
import sys
import argparse
import pandas as pd
import os
import numpy as np
import warnings
import json
import pickle
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Multi-task Traj Interp')
    parser.add_argument('--dataset', type=str, default='Porto', help='data set')
    parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--diff_T', type=int, default=500, help='diffusion step')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='min beta')
    parser.add_argument('--beta_end', type=float, default=0.02, help='max beta')
    parser.add_argument('--pre_trained_dim', type=int, default=128, help='pre-trained dim of the road segment')
    parser.add_argument('--rdcl', type=int, default=10, help='stack layers on the denoise network')
    parser.add_argument('--gpu_id', type=str, default='0')
    
    
    opts = parser.parse_args()
    # 必须在所有 torch 相关导入之前设置 CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    
    # 现在才导入 torch 相关模块
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    from utils.utils import save_json_data, create_dir, load_pkl_data
    from common.mbr import MBR
    from common.spatial_func import SPoint, distance
    from build_graph import load_graph_adj_mtx, load_graph_node_features
    from models.model_utils import epoch_time, AttrDict
    from models.multi_train import init_weights, train
    from models.model import Diff_RNTraj
    from models.diff_module import diff_CSDI

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = AttrDict()

    if opts.dataset == 'Porto':
        args_dict = {
            'dataset': opts.dataset,
            # MBR
            'min_lat':41.142,
            'min_lng':-8.652,
            'max_lat':41.174,
            'max_lng':-8.578,
            'grid_size': 50, 

            # model params
            'hid_dim':opts.hid_dim,
            'id_size':None,  # 训练时根据 road_embed.txt 自动设置
            'n_epochs':opts.epochs,
            'batch_size':opts.batch_size,
            'learning_rate':opts.lr,
            'tf_ratio':0.5,
            'clip':1,
            'log_step':1,

            'diff_T': opts.diff_T,
            'beta_start': opts.beta_start,
            'beta_end': opts.beta_end,
            'pre_trained_dim': opts.pre_trained_dim,
            'rdcl': opts.rdcl,
            'id_loss_weight': 1  # id_loss的权重
        }
    elif opts.dataset == 'Chengdu':
        args_dict = {
            'dataset': opts.dataset,

            # MBR
            'min_lat':30.655,
            'min_lng':104.043,
            'max_lat':30.727,
            'max_lng':104.129,
            'grid_size': 50, 

            # model params
            'hid_dim':opts.hid_dim,
            'id_size':None,  # 训练时根据 road_embed.txt 自动设置
            'n_epochs':opts.epochs,
            'batch_size':opts.batch_size,
            'learning_rate':opts.lr,
            'tf_ratio':0.5,
            'clip':1,
            'log_step':1,

            'diff_T': opts.diff_T,
            'beta_start': opts.beta_start,
            'beta_end': opts.beta_end,
            'pre_trained_dim': opts.pre_trained_dim,
            'rdcl': opts.rdcl,
            'id_loss_weight': 1  # id_loss的权重
        }
    
    assert opts.dataset in ['Porto', 'Chengdu'], 'Check dataset name if in [Porto, Chengdu]'

    args.update(args_dict)

    print('Preparing data...')
    
    beta = np.linspace(opts.beta_start ** 0.5, opts.beta_end ** 0.5, opts.diff_T) ** 2
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    alpha = torch.tensor(alpha).float().to(device)
    alpha_bar = torch.tensor(alpha_bar).float().to(device)

    diffusion_hyperparams = {}
    diffusion_hyperparams['T'], diffusion_hyperparams['alpha_bar'], diffusion_hyperparams['alpha'] = opts.diff_T,  alpha_bar, alpha
    diffusion_hyperparams['beta'] = beta
    
    test_flag = True
    # test_flag = False

    if opts.dataset == 'Porto':
        path_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'porto') + '/'
    elif opts.dataset == 'Chengdu':
        path_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'chengdu') + '/'


    extra_info_dir = path_dir + "roadnet/"  # 与 TRMMA 工作空间保持一致
    rn_dir = path_dir + "roadnet/"
    UTG_file = path_dir + 'graph/graph_A.csv'
    pre_trained_road = path_dir + 'graph/road_embed.txt'

                
                
    model_save_path = './results/'+opts.dataset + '/'
    create_dir(model_save_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=model_save_path + 'log.txt',
                        filemode='a')
    # spatial embedding
    spatial_A = load_graph_adj_mtx(UTG_file)
    spatial_A_trans = np.zeros((spatial_A.shape[0]+1, spatial_A.shape[1]+1)) + 1e-10
    spatial_A_trans[1:,1:] = spatial_A

    f = open(pre_trained_road, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0])+1, int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index+1] = temp[1 :]
    
    SE = torch.from_numpy(SE)
    # 根据 embedding 行数自动更新 id_size，确保与图/映射一致
    args.id_size = N
    args_dict['id_size'] = N
    # 根据 embedding 维度自动对齐预训练维度，避免通道数不匹配
    args.pre_trained_dim = dims
    opts.pre_trained_dim = dims
    args_dict['pre_trained_dim'] = dims

    # 下方这些原本用于构建路网特征和在线特征的变量，
    # 在当前条件扩散训练流程中暂未使用，为避免对 networkx.read_shp/osgeo 的依赖，这里先注释掉。
    # 如后续需要使用路网相关的 loss 或可视化，再单独处理。
    # rn = load_rn_shp(rn_dir, is_directed=True)
    # raw_rn_dict = load_rn_dict(extra_info_dir, file_name='raw_rn_dict.json')
    # new2raw_rid_dict = load_rid_freqs(extra_info_dir, file_name='new2raw_rid.json')
    # raw2new_rid_dict = load_rid_freqs(extra_info_dir, file_name='raw2new_rid.json')
    # rn_dict = load_rn_dict(extra_info_dir, file_name='rn_dict.json')
    #
    # mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
    # grid_rn_dict, max_xid, max_yid = get_rid_grid(mbr, args.grid_size, rn_dict)
    # args_dict['max_xid'] = max_xid
    # args_dict['max_yid'] = max_yid
    args.update(args_dict)
    print(args)
    logging.info(args_dict)
    with open(model_save_path+'logging.txt', 'w') as f:
        f.write(str(args_dict))
        f.write('\n')
        

    # load dataset: 使用预处理好的条件序列 cond_seqs_{city}_train.bin 和 valid.bin
    city_lower = opts.dataset.lower()
    cond_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', city_lower, 'cond_data')
    
    # 加载训练集
    train_cond_path = os.path.join(cond_dir, f'cond_seqs_{city_lower}_train.bin')
    print(f'Loading training conditional sequences from: {train_cond_path}')
    with open(train_cond_path, 'rb') as f:
        all_cond_dict_train = pickle.load(f)

    # 加载 road embed 时生成的节点索引映射，确保 dense/sparse id 与 SE 对齐
    mapping_path = os.path.join(path_dir, 'graph', 'graph_node_id2idx.txt')
    print(f'Loading node id mapping from: {mapping_path}')
    eid2idx = {}
    max_idx = 0
    with open(mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            eid, idx = int(parts[0]), int(parts[1])
            eid2idx[eid] = idx
            max_idx = max(max_idx, idx)

    def remap_seq(seq):
        remapped = []
        for eid in seq:
            idx = eid2idx.get(int(eid), -1)
            if idx == -1:
                remapped.append(0)
            else:
                remapped.append(idx + 1)  # +1 因为 SE[0] 预留为 padding
        return remapped

    # 重映射训练集
    remapped_cond_train = {}
    for L, samples in all_cond_dict_train.items():
        remapped = []
        for sample in samples:
            dense = remap_seq(sample["dense"])
            sparse = remap_seq(sample["sparse"])
            mask = [int(m) for m in sample["mask"]]
            remapped.append({"dense": dense, "sparse": sparse, "mask": mask})
        remapped_cond_train[L] = remapped
    all_cond_dict = remapped_cond_train

    diff_model = diff_CSDI(args.hid_dim, args.hid_dim, opts.diff_T, args.hid_dim, args.pre_trained_dim, args.rdcl)
    model = Diff_RNTraj(diff_model, diffusion_hyperparams).to(device)
    model.apply(init_weights)  # learn how to init weights
    
    print('model', str(model))
    logging.info('model' + str(model))
    with open(model_save_path+'logging.txt', 'a+') as f:
        f.write('model' + str(model) + '\n')
        
    # 训练损失记录
    ls_train_loss, ls_train_const_loss, ls_train_diff_loss, ls_train_x0_loss, ls_train_sparse_loss, ls_train_id_loss = [], [], [], [], [], []
    
    dict_train_loss = {}

    # get all parameters (model parameters + task dependent log variances)
    log_vars = [torch.zeros((1,), requires_grad=True, device=device)] * 2  # use for auto-tune multi-task param
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    scheduler = StepLR(optimizer, 
                step_size = 3, # Period of learning rate decay
                gamma = 0.5)
    for epoch in tqdm(range(args.n_epochs)):
        start_time = time.time()

        new_log_vars, train_loss, train_const_loss, train_diff_loss, train_x0_loss, train_sparse_loss, train_id_loss = \
                train(model, spatial_A_trans, SE, all_cond_dict, optimizer, log_vars, args, diffusion_hyperparams, device)
        scheduler.step()

        ls_train_loss.append(train_loss)
        ls_train_const_loss.append(train_const_loss)
        ls_train_diff_loss.append(train_diff_loss)
        ls_train_x0_loss.append(train_x0_loss)
        ls_train_sparse_loss.append(train_sparse_loss)
        ls_train_id_loss.append(train_id_loss)

        dict_train_loss['train_ttl_loss'] = ls_train_loss
        dict_train_loss['train_const_loss'] = ls_train_const_loss
        dict_train_loss['train_diff_loss'] = ls_train_diff_loss
        dict_train_loss['train_x0_loss'] = ls_train_x0_loss
        dict_train_loss['train_sparse_loss'] = ls_train_sparse_loss
        dict_train_loss['train_id_loss'] = ls_train_id_loss

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 打印信息
        print(f'Epoch: {epoch + 1}/{args.n_epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
              f'Const: {train_const_loss:.4f} | '
              f'Diff: {train_diff_loss:.4f} | '
              f'X0: {train_x0_loss:.4f} | '
              f'Sparse: {train_sparse_loss:.4f} | '
              f'ID: {train_id_loss:.4f}')
        
        logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
        weights = [torch.exp(weight) ** 0.5 for weight in new_log_vars]
        logging.info('log_vars:' + str(weights))
        
        log_str = (f'\tTrain Loss: {train_loss:.4f}'
                  f'\tConst Loss: {train_const_loss:.4f}'
                  f'\tDiff Loss: {train_diff_loss:.4f}'
                  f'\tX0 Loss: {train_x0_loss:.4f}'
                  f'\tSparse Loss: {train_sparse_loss:.4f}'
                  f'\tID Loss: {train_id_loss:.4f}')
        logging.info(log_str)
        with open(model_save_path+'logging.txt', 'a+') as f:
            f.write('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's' + '\n')
            f.write(log_str + '\n')
                
        torch.save(model.state_dict(), model_save_path + 'train-mid-model.pt')
        save_json_data(dict_train_loss, model_save_path, "train_loss.json")
                

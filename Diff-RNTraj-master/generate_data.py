import time
from tqdm import tqdm
import logging
import sys
import argparse
import os
import torch
import numpy as np
from utils.utils import create_dir
from models.model_utils import AttrDict
from models.model import Diff_RNTraj
from models.diff_module import diff_CSDI
from build_graph import load_graph_adj_mtx
import pickle

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Conditional Diff-RNTraj evaluation (RID accuracy)')
    parser.add_argument('--dataset', type=str, default='Porto', help='data set')
    parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--diff_T', type=int, default=500, help='diffusion step')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='min beta')
    parser.add_argument('--beta_end', type=float, default=0.02, help='max beta')
    parser.add_argument('--pre_trained_dim', type=int, default=64, help='pre-trained dim of the road segment')
    parser.add_argument('--rdcl', type=int, default=10, help='stack layers on the denoise network')
    parser.add_argument('--gpu_id', type=str, default='0')
    opts = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = AttrDict()

    assert opts.dataset in ['Porto', 'Chengdu', 'Xian', 'Beijing'], "dataset must be one of [Porto, Chengdu, Xian, Beijing]"
    args_dict = {
        'dataset': opts.dataset,
        # model params（id_size 将在读取 embedding 后自动更新）
        'hid_dim': opts.hid_dim,
        'id_size': None,
        'n_epochs': opts.epochs,
        'batch_size': opts.batch_size,
        'learning_rate': opts.lr,
        'tf_ratio': 0.5,
        'clip': 1,
        'log_step': 1,

        'diff_T': opts.diff_T,
        'beta_start': opts.beta_start,
        'beta_end': opts.beta_end,
        'pre_trained_dim': opts.pre_trained_dim,
        'rdcl': opts.rdcl
    }
    args.update(args_dict)

    print('Preparing data...')

    beta = np.linspace(opts.beta_start ** 0.5, opts.beta_end ** 0.5, opts.diff_T) ** 2
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    alpha = torch.tensor(alpha).float().to(device)
    alpha_bar = torch.tensor(alpha_bar).float().to(device)

    diffusion_hyperparams = {}
    diffusion_hyperparams['T'], diffusion_hyperparams['alpha_bar'], diffusion_hyperparams['alpha'] = opts.diff_T, alpha_bar, alpha
    diffusion_hyperparams['beta'] = beta

    # 本项目内的数据路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    city_lower = opts.dataset.lower()
    path_dir = os.path.join(script_dir, 'data', city_lower) + '/'

    UTG_file = path_dir + 'graph/graph_A.csv'
    pre_trained_road = path_dir + 'graph/road_embed.txt'

    model_save_path = './results/' + opts.dataset + '/'
    create_dir(model_save_path)

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
    
    SE = torch.from_numpy(SE).to(device)
    # 自动更新 id_size，确保与训练一致
    args.id_size = N
    
    diff_model = diff_CSDI(args.hid_dim, args.hid_dim, opts.diff_T, args.hid_dim, args.pre_trained_dim, args.rdcl)
    model = Diff_RNTraj(diff_model, diffusion_hyperparams).to(device)

    print('model', str(model))

    # 加载训练好的权重
    model_path = './results/{}/'.format(args.dataset)
    state_dict = torch.load(model_path + 'val-best-model.pt', map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 加载 cond test 数据
    cond_dir = os.path.join(script_dir, 'data', 'porto', 'cond_data')
    cond_path = os.path.join(cond_dir, 'cond_seqs_porto_test.bin')
    print(f'Loading conditional test sequences from: {cond_path}')
    with open(cond_path, 'rb') as f:
        all_cond_dict = pickle.load(f)

    # 加载 eid -> idx 映射，确保与 SE 对齐
    mapping_path = os.path.join(path_dir, 'graph', 'graph_node_id2idx.txt')
    print(f'Loading node id mapping from: {mapping_path}')
    eid2idx = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            eid, idx = int(parts[0]), int(parts[1])
            # +1 保留 0 给 padding
            eid2idx[eid] = idx + 1

    def remap_seq(seq):
        return [eid2idx.get(int(eid), 0) for eid in seq]

    # 评估：按长度分组+批量处理，加速推理
    total_correct = 0
    total_tokens = 0
    total_samples = sum(len(samples) for _, samples in all_cond_dict.items())
    pbar = tqdm(total=total_samples, desc="Processing")

    from models.diff_util import cal_x0_conditional_ddpm

    # 预计算 SE 的范数，避免重复计算
    SE_norm = SE.norm(dim=1, keepdim=True)  # N,1

    with torch.no_grad():
        for L, samples in all_cond_dict.items():
            for batch_idx in range(0, len(samples), opts.batch_size):
                batch_samples = samples[batch_idx: batch_idx + opts.batch_size]
                B = len(batch_samples)
                # 准备批量张量
                dense_ids = torch.zeros(B, L, dtype=torch.long, device=device)
                sparse_ids = torch.zeros(B, L, dtype=torch.long, device=device)
                mask = torch.zeros(B, L, dtype=torch.float32, device=device)
                for i, sample in enumerate(batch_samples):
                    dense_ids[i] = torch.tensor(remap_seq(sample["dense"]), device=device)
                    sparse_ids[i] = torch.tensor(remap_seq(sample["sparse"]), device=device)
                    mask[i] = torch.tensor(sample["mask"], device=device, dtype=torch.float32)

                # 条件嵌入
                sparse_embed = SE[sparse_ids]  # B, L, D

                # 条件采样（扩散去噪）
                x0_hat = cal_x0_conditional_ddpm(
                    model.diff_model,
                    sparse_embed,
                    mask,
                    diffusion_hyperparams
                )  # B, L, D

                # 最近邻映射回路段ID
                B, L_, D = x0_hat.shape
                x0_hat_flat = x0_hat.reshape(B * L_, D)  # BL, D
                x0_hat_norm = x0_hat_flat.norm(dim=1, keepdim=True)  # BL,1
                sim_matrix = torch.mm(x0_hat_flat, SE.t()) / (torch.mm(x0_hat_norm, SE_norm.t()) + 1e-6)  # BL, N
                pred_ids = sim_matrix.argmax(dim=1).reshape(B, L_)

                # token-level 准确率
                correct = (pred_ids == dense_ids).sum().item()
                total_correct += correct
                total_tokens += B * L_

                pbar.update(B)

    pbar.close()
    rid_accuracy = total_correct / max(total_tokens, 1)
    print(f"RID accuracy: {rid_accuracy:.4f} ({total_correct}/{total_tokens})")
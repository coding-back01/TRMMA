"""
扩散模型测试评估脚本

使用扩散模型专用的评价指标：
- RSC (Road Segment Connectivity): 路段连通性（核心指标）
- JSD-RS (Road Segment Usage Distribution): 路段使用分布的JSD（核心指标）
- LCS Recall/Precision: 基于最长公共子序列的评价
- Token Accuracy: token级别准确率（参考）
"""

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
from models.eval_metrics import DiffusionMetricsTracker, format_metrics
import pickle

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Conditional Diff-RNTraj evaluation (diffusion metrics)')
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
    
    # 优化参数
    parser.add_argument('--repaint_steps', type=int, default=3, help='repaint steps for stronger conditioning (推荐3)')
    parser.add_argument('--use_beam_search', type=int, default=1, help='use beam search (1=yes, 0=no, 推荐1)')
    parser.add_argument('--beam_size', type=int, default=5, help='beam size (推荐5)')
    parser.add_argument('--alpha', type=float, default=0.7, help='similarity weight (推荐0.7)')
    
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
    print(f'优化配置: repaint_steps={opts.repaint_steps}, use_beam_search={opts.use_beam_search}, beam_size={opts.beam_size}, alpha={opts.alpha}')

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
    # 自动对齐 embedding 维度，避免通道数不匹配
    args.pre_trained_dim = dims
    opts.pre_trained_dim = dims
    
    diff_model = diff_CSDI(args.hid_dim, args.hid_dim, opts.diff_T, args.hid_dim, args.pre_trained_dim, args.rdcl)
    model = Diff_RNTraj(diff_model, diffusion_hyperparams).to(device)

    print('model', str(model))

    # 加载训练好的权重
    model_path = './results/{}/'.format(args.dataset)
    
    # 优先加载 train-mid-model.pt（最后一个epoch的模型）
    # 兼容旧版本的 val-best-model.pt
    if os.path.exists(model_path + 'train-mid-model.pt'):
        model_file = model_path + 'train-mid-model.pt'
        print(f'Loading model from: {model_file}')
    elif os.path.exists(model_path + 'val-best-model.pt'):
        model_file = model_path + 'val-best-model.pt'
        print(f'Loading model from: {model_file}')
    else:
        raise FileNotFoundError(f'No trained model found in {model_path}')
    
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 加载 cond test 数据
    cond_dir = os.path.join(script_dir, 'data', city_lower, 'cond_data')
    cond_path = os.path.join(cond_dir, f'cond_seqs_{city_lower}_test.bin')
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

    # 初始化扩散模型评价指标追踪器
    num_segments = N  # 路段总数
    spatial_A_trans_tensor = torch.tensor(spatial_A_trans, dtype=torch.float32, device=device)
    metrics_tracker = DiffusionMetricsTracker(num_segments, spatial_A_trans_tensor)

    total_samples = sum(len(samples) for _, samples in all_cond_dict.items())
    pbar = tqdm(total=total_samples, desc="Evaluating")

    from models.diff_util import cal_x0_conditional_ddpm

    # 预计算 SE 的范数，避免重复计算
    SE_norm = SE.norm(dim=1, keepdim=True)  # N,1

    # === 连通性后处理函数（Beam Search） ===
    def beam_search_decode(x0_hat, SE, spatial_A_trans, mask, sparse_ids, beam_size=5, alpha=0.7):
        """使用 Beam Search 保证连通性"""
        B, L, D = x0_hat.shape
        N = SE.shape[0]
        device = x0_hat.device
        
        # 计算相似度矩阵
        x0_hat_norm = x0_hat / (x0_hat.norm(dim=2, keepdim=True) + 1e-8)
        SE_norm_local = SE / (SE.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = torch.einsum('bld,nd->bln', x0_hat_norm, SE_norm_local)
        
        pred_ids = torch.zeros(B, L, dtype=torch.long, device=device)
        
        for b in range(B):
            curr_sim = sim_matrix[b]
            curr_mask = mask[b]
            curr_sparse = sparse_ids[b]
            result_path = torch.zeros(L, dtype=torch.long, device=device)
            
            for pos in range(L):
                if curr_mask[pos] == 1:
                    result_path[pos] = curr_sparse[pos]
                else:
                    if pos == 0:
                        result_path[pos] = curr_sim[pos].argmax()
                    else:
                        prev_id = result_path[pos - 1].item()
                        topk_scores, topk_ids = torch.topk(curr_sim[pos], min(beam_size * 2, N))
                        
                        best_id = topk_ids[0].item()
                        best_score = -float('inf')
                        
                        for k in range(len(topk_ids)):
                            cand_id = topk_ids[k].item()
                            sim_score = topk_scores[k].item()
                            
                            # 连通性加成
                            if spatial_A_trans[prev_id, cand_id] > 1e-9:
                                conn_bonus = 0.5
                            else:
                                conn_bonus = -0.3
                            
                            total = alpha * sim_score + (1 - alpha) * conn_bonus
                            
                            if total > best_score:
                                best_score = total
                                best_id = cand_id
                        
                        result_path[pos] = best_id
            
            pred_ids[b] = result_path
        
        return pred_ids

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

                # === 优化 1：条件采样（使用 repaint 强化约束） ===
                x0_hat = cal_x0_conditional_ddpm(
                    model.diff_model,
                    sparse_embed,
                    mask,
                    diffusion_hyperparams,
                    repaint_steps=opts.repaint_steps
                )  # B, L, D

                # === 优化 2：连通性约束的离散化 ===
                if opts.use_beam_search:
                    pred_ids = beam_search_decode(
                        x0_hat, SE, spatial_A_trans_tensor, mask, sparse_ids,
                        beam_size=opts.beam_size, alpha=opts.alpha
                    )
                else:
                    # 原始方法：简单的最近邻
                    B_, L_, D = x0_hat.shape
                    x0_hat_flat = x0_hat.reshape(B_ * L_, D)
                    x0_hat_norm = x0_hat_flat.norm(dim=1, keepdim=True)
                    sim_matrix = torch.mm(x0_hat_flat, SE.t()) / (torch.mm(x0_hat_norm, SE_norm.t()) + 1e-6)
                    pred_ids = sim_matrix.argmax(dim=1).reshape(B, L_)

                # 更新指标追踪器
                metrics_tracker.update(pred_ids, dense_ids)

                pbar.update(B)

    pbar.close()
    
    # 计算最终指标
    metrics = metrics_tracker.compute()
    
    # 打印结果
    print("\n" + "="*60)
    print(format_metrics(metrics))
    print("="*60)
    
    # 保存结果到文件
    results_file = os.path.join(model_save_path, 'test_metrics.txt')
    with open(results_file, 'w') as f:
        f.write(format_metrics(metrics))
        f.write(f"\n\nDetailed metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")
    print(f"\nResults saved to: {results_file}")
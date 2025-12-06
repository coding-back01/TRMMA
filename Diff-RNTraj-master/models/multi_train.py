import numpy as np
import random

import torch
import torch.nn as nn
from tqdm import tqdm
from models.model_utils import toseq, get_constraint_mask
from models.loss_fn import cal_id_acc, check_rn_dis_loss, cal_id_acc_train
from models.trajectory_graph import build_graph,search_road_index


def init_weights(self):
    """
    Here we reproduce Keras default initialization weights for consistency with Keras version
    Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
    """
    ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
    hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
    b = (param.data for name, param in self.named_parameters() if 'bias' in name)

    for t in ih:
        nn.init.xavier_uniform_(t)
    for t in hh:
        nn.init.orthogonal_(t)
    for t in b:
        nn.init.constant_(t, 0)

def next_batch(dense_list, sparse_list, mask_list):
    length = len(dense_list)
    for i in range(length):
        yield dense_list[i], sparse_list[i], mask_list[i]


def train(model, spatial_A_trans, SE, all_cond_dict, optimizer, log_vars, parameters, diffusion_hyperparams, device):
    model.train()  # not necessary to have this line but it's safe to use model.train() to train model

    criterion_reg = nn.MSELoss()
    criterion_ce = nn.NLLLoss()
    criterion_ce1 = nn.NLLLoss()
    epoch_ttl_loss = 0
    epoch_const_loss = 0
    epoch_diff_loss = 0
    epoch_x0_loss = 0
    epoch_sparse_loss = 0
    epoch_id_loss = 0
    
    # id_loss的权重（可调整，建议0.1-0.5）
    id_loss_weight = getattr(parameters, 'id_loss_weight', 0.1)

    all_batch_dense, all_batch_sparse, all_batch_mask = [], [], []
    batch_size = parameters.batch_size

    # all_cond_dict: {length: [ {dense,sparse,mask}, ... ]}
    for L, samples in all_cond_dict.items():
        n = len(samples)
        idx = 0
        while idx < n:
            batch_samples = samples[idx: idx + batch_size]
            dense_batch = [s["dense"] for s in batch_samples]
            sparse_batch = [s["sparse"] for s in batch_samples]
            mask_batch = [s["mask"] for s in batch_samples]
            all_batch_dense.append(dense_batch)
            all_batch_sparse.append(sparse_batch)
            all_batch_mask.append(mask_batch)
            idx += batch_size

    # 打乱数据
    zipped_list = list(zip(all_batch_dense, all_batch_sparse, all_batch_mask))
    random.shuffle(zipped_list)
    all_batch_dense, all_batch_sparse, all_batch_mask = zip(*zipped_list)

    SE = SE.to(device)
    spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)

    cnt = 0
    for dense_ids, sparse_ids, masks in next_batch(all_batch_dense, all_batch_sparse, all_batch_mask):
        cnt += 1
        src_dense_ids = torch.tensor(dense_ids).long().to(device)
        src_sparse_ids = torch.tensor(sparse_ids).long().to(device)
        src_masks = torch.tensor(masks).float().to(device)

        optimizer.zero_grad()
        diff_loss, const_loss, x0_loss, sparse_loss, id_loss = model(
            spatial_A_trans, SE, src_dense_ids, src_sparse_ids, src_masks
        )

        # 总损失：id_loss使用较小权重，避免与x0_loss冲突
        ttl_loss = diff_loss + const_loss + x0_loss + sparse_loss + id_loss_weight * id_loss
        ttl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)  # log_vars are not necessary to clip
        optimizer.step()
        
        epoch_ttl_loss += ttl_loss.item()
        epoch_const_loss += const_loss.item()
        epoch_diff_loss += diff_loss.item()
        epoch_x0_loss += x0_loss.item()
        epoch_sparse_loss += sparse_loss.item()
        epoch_id_loss += id_loss.item()

    return log_vars, epoch_ttl_loss / cnt, epoch_const_loss / cnt, epoch_diff_loss / cnt, epoch_x0_loss / cnt, epoch_sparse_loss / cnt, epoch_id_loss / cnt


def validate(model, spatial_A_trans, SE, all_cond_dict, diffusion_hyperparams, device, batch_size=256):
    """
    在验证集上评估模型，计算 RID 准确率
    
    Args:
        model: 训练好的模型
        spatial_A_trans: 空间邻接矩阵
        SE: 路段嵌入矩阵
        all_cond_dict: 验证集条件数据 {length: [{dense, sparse, mask}, ...]}
        diffusion_hyperparams: 扩散超参数
        device: 设备
        batch_size: 批处理大小，默认256（与训练保持一致）
    
    Returns:
        rid_accuracy: RID 准确率
    """
    model.eval()
    from models.diff_util import cal_x0_conditional_ddpm
    
    total_correct = 0
    total_tokens = 0
    
    SE = SE.to(device)
    spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)
    
    # 预计算 SE 的归一化，避免重复计算
    SE_norm = SE.norm(dim=1, keepdim=True)  # N, 1
    
    # 收集所有样本（验证全部10000个样本）
    all_samples = []
    for L, samples in all_cond_dict.items():
        for sample in samples:
            all_samples.append((L, sample))
    
    # 按长度分组，便于批处理
    samples_by_length = {}
    for L, sample in all_samples:
        if L not in samples_by_length:
            samples_by_length[L] = []
        samples_by_length[L].append(sample)
    
    with torch.no_grad():
        for L, samples in samples_by_length.items():
            # 批量处理同一长度的样本
            for batch_idx in range(0, len(samples), batch_size):
                batch_samples = samples[batch_idx:batch_idx + batch_size]
                B_batch = len(batch_samples)
                
                # 准备批量数据
                dense_ids_batch = torch.zeros(B_batch, L, dtype=torch.long, device=device)
                sparse_ids_batch = torch.zeros(B_batch, L, dtype=torch.long, device=device)
                mask_batch = torch.zeros(B_batch, L, dtype=torch.float32, device=device)
                
                for i, sample in enumerate(batch_samples):
                    dense_ids_batch[i] = torch.tensor(sample["dense"], device=device)
                    sparse_ids_batch[i] = torch.tensor(sample["sparse"], device=device)
                    mask_batch[i] = torch.tensor(sample["mask"], device=device, dtype=torch.float32)
                
                # 获取 sparse 嵌入作为条件
                sparse_embed = SE[sparse_ids_batch]  # B, L, D
                
                # 条件采样：从噪声开始，使用 sparse 和 mask 作为条件进行去噪
                x0_hat = cal_x0_conditional_ddpm(
                    model.diff_model,
                    sparse_embed,
                    mask_batch,
                    diffusion_hyperparams
                )  # B, L, D
                
                # 将生成的嵌入映射回 EID（通过最近邻搜索）
                x0_hat_flat = x0_hat.reshape(B_batch * L, -1)  # BL, D
                
                # 计算余弦相似度（批量计算）
                x0_hat_norm = x0_hat_flat.norm(dim=1, keepdim=True)  # BL, 1
                sim_matrix = torch.mm(x0_hat_flat, SE.t()) / (torch.mm(x0_hat_norm, SE_norm.t()) + 1e-6)  # BL, N
                
                # 找到最相似的路段 ID
                pred_ids = sim_matrix.argmax(dim=1).reshape(B_batch, L)  # B, L
                
                # 计算准确率（token-level）
                correct = (pred_ids == dense_ids_batch).sum().item()
                total_correct += correct
                total_tokens += B_batch * L
    
    rid_accuracy = total_correct / max(total_tokens, 1)
    return rid_accuracy


def generate_data(model, spatial_A_trans, rn_dict, parameters, SE):
    model.eval()  
    SE = SE.to(device)
    spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)
    if parameters.dataset == "Chengdu":
        length2num = np.load("/data/WeiTongLong/data/traj_gen/A_new_dataset/Chengdu/gen_all/length_distri.npy")
        start_length, end_length = 20, 60
    if parameters.dataset == "Porto":
        length2num = np.load("/data/WeiTongLong/data/traj_gen/A_new_dataset/Porto/gen_all/length_distri.npy")
        start_length, end_length = 20, 100
    # batchsize=1
    traj_all_num = int(length2num.sum())
    # batchsize=1
    with torch.no_grad():  # this line can help speed up evaluation
        for i in tqdm(range(start_length, end_length+1)):
            curr_length =i
            curr_batch = int(length2num[i] / length2num.sum() * traj_all_num)

            while True:
                if curr_batch > 256: 
                    _curr_batch = 256
                else:
                    _curr_batch = curr_batch
                ids, rates = model.generate_data(spatial_A_trans, SE, _curr_batch, curr_length, parameters.pre_trained_dim)
                rates[rates>1]=1
                rates[rates<-1]=-1
                rates = (rates + 1) / 2
                output_seqs = toseq(rn_dict, ids,rates, parameters, save_txt_num = i)
                curr_batch = curr_batch - _curr_batch
                if curr_batch <= 0: break
                exit()
            # print(output_seqs)
        # exit()


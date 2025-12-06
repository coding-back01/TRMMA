import os
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

def std_normal(size, device=None):
    """
    Generate the standard Gaussian variable of a certain size
    """
    if device is None:
        # 尝试自动检测设备
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    return torch.normal(0, 1, size=size).to(device)

def diff_forward_x0_constraint(net, x_dense, x_sparse, mask, diffusion_hyperparams, SE, spatial_A_trans, dense_ids=None, compute_id_loss=True):
    """
    x_dense: B, T, D   加噪 / 去噪的 dense 路段嵌入
    x_sparse: B, T, D  干净的 sparse 路段嵌入（条件）
    mask: B, T         稀疏位置掩码（1=锚点，0=缺失）
    dense_ids: B, T    真实的路段ID序列（用于ID分类损失，可选）
    compute_id_loss: bool  是否计算ID分类损失（训练时=True，验证时=False）
    """
    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha = _dh["T"], _dh["alpha_bar"], _dh["alpha"]

    B, L, D = x_dense.shape  # B: batch, L: length, D: embed dim
    device = x_dense.device
    # 避免每步重复拷贝，必要时再迁移
    if Alpha_bar.device != device:
        Alpha_bar = Alpha_bar.to(device)
    if Alpha.device != device:
        Alpha = Alpha.to(device)

    diffusion_steps = torch.randint(T, size=(B, 1, 1), device=device)  # [B,1,1]

    z = std_normal(x_dense.shape, device=device)  # 噪声 ~ N(0, I)

    # 只对 dense 部分加噪
    xt_dense = torch.sqrt(Alpha_bar[diffusion_steps]) * x_dense + \
               torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z  # q(x_t|x_0)

    # 构造去噪网络的输入：concat(xt_dense, x_sparse, mask)
    mask_feat = mask.unsqueeze(-1).to(device)  # B, L, 1
    net_input = torch.cat([xt_dense, x_sparse.to(device), mask_feat], dim=-1)  # B, L, 2D+1

    #这里得到的是噪声, loss函数之一
    pred_noise = net(net_input, diffusion_steps.view(B, 1))  # predict \epsilon according to \epsilon_\theta

    noise_func = nn.MSELoss()
    diff_loss = noise_func(pred_noise, z)  # 噪声预测 MSE

    # 反推出 x0_hat（dense 部分）
    x0_hat = (xt_dense - torch.sqrt(1 - Alpha_bar[diffusion_steps]) * pred_noise) / \
             torch.sqrt(Alpha_bar[diffusion_steps])  # B, L, D

    x0_loss = noise_func(x0_hat, x_dense)

    # 基于 x0_hat 的结构约束（还原为 argmax 版本）
    # 计算 x0_hat 与所有路段嵌入的相似度
    if SE.device != device:
        SE = SE.to(device)
    id_sim_logits = torch.einsum('bld,nf->bln', x0_hat, SE)  # B, L, N
    # 取 argmax 得到最可能的路段 ID 序列
    id_sim = id_sim_logits.argmax(-1)  # B, L

    # 确保 spatial_A_trans 是 torch tensor 并在正确的设备上
    if not isinstance(spatial_A_trans, torch.Tensor):
        spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float32, device=device)
    elif spatial_A_trans.device != device:
        spatial_A_trans = spatial_A_trans.to(device)

    # 路网连通性约束：相邻路段不连通则计为惩罚
    rn_mask = spatial_A_trans[id_sim[:, :-1], id_sim[:, 1:]] == 1e-10
    # 归一化，避免量级过大主导总损失
    const_loss = rn_mask.float().mean()

    # 稀疏约束：在 mask=1 的位置，x0_hat 贴近 x_sparse
    # L2 in embedding space, mask 加权
    sparse_l2 = ((x0_hat - x_sparse.to(device)) ** 2).sum(dim=-1)  # B, L
    sparse_mask = mask.to(device)
    valid_cnt = sparse_mask.sum()
    if valid_cnt > 0:
        sparse_loss = (sparse_l2 * sparse_mask).sum() / valid_cnt
    else:
        sparse_loss = torch.tensor(0.0, device=device)

    # 路段ID分类损失：直接优化路段ID预测准确率（只在训练时计算，不影响验证速度）
    # 注意：这个loss与x0_loss有重叠但不完全等价
    # - x0_loss: 在嵌入空间，比较x0_hat和x_dense（更"软"）
    # - id_loss: 在ID空间，直接优化路段ID预测（更"硬"，更直接）
    # 建议：给id_loss一个较小的权重（如0.1-0.5），避免与x0_loss冲突
    id_loss = torch.tensor(0.0, device=device)
    if compute_id_loss and dense_ids is not None:
        # id_sim_logits 已经计算过了（用于 const_loss），这里只是加一个 CrossEntropyLoss
        # CrossEntropyLoss 计算很快，不会显著增加训练时间
        criterion_ce = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding (ID=0)
        dense_ids_tensor = dense_ids.to(device).long()
        # 重塑为 (B*L, N) 和 (B*L,)
        id_logits_flat = id_sim_logits.reshape(-1, id_sim_logits.shape[-1])  # B*L, N
        dense_ids_flat = dense_ids_tensor.reshape(-1)  # B*L
        id_loss = criterion_ce(id_logits_flat, dense_ids_flat)

    return diff_loss, const_loss, x0_loss, sparse_loss, id_loss

def cal_x0_from_noise_ddpm(net, diffusion_hyperparams, batchsize, length,  feature, device=None):
    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha = _dh["T"], _dh["alpha_bar"], _dh["alpha"]
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    diff_input = std_normal((batchsize, length, feature), device=device)
    
    with torch.no_grad():
        for t in range(T-1, -1, -1):
            
            predict_noise = net(diff_input, t)  
            
            coeff1 = 1 / (Alpha[t] ** 0.5)
            coeff2 = (1 - Alpha[t]) / ((1 - Alpha_bar[t]) ** 0.5)
            diff_input = coeff1 * (diff_input - coeff2 * predict_noise)
            
            if t>0:
                noise = std_normal(diff_input.shape, device=device)
                sigma = ( (1 - Alpha_bar[t-1]) / (1 - Alpha_bar[t]) * (1 - Alpha[t]) ) ** 0.5
                diff_input += sigma * noise
                
    return diff_input

def cal_x0_conditional_ddpm(net, x_sparse, mask, diffusion_hyperparams):
    """
    条件采样：从噪声开始，使用 sparse 和 mask 作为条件进行去噪
    
    Args:
        net: 去噪网络
        x_sparse: B, L, D  稀疏路段嵌入（条件，保持不变）
        mask: B, L         稀疏位置掩码（1=锚点，0=缺失）
        diffusion_hyperparams: 扩散超参数
    
    Returns:
        x0_hat: B, L, D    生成的 dense 路段嵌入
    """
    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha = _dh["T"], _dh["alpha_bar"], _dh["alpha"]
    
    B, L, D = x_sparse.shape
    device = x_sparse.device
    Alpha_bar = Alpha_bar.to(device)
    Alpha = Alpha.to(device)
    mask_expanded = mask.unsqueeze(-1).to(device)  # B, L, 1
    
    # 从纯噪声开始（对于 mask=0 的位置）
    # 对于 mask=1 的位置，我们会在每一步强制使用 sparse 的值
    xt_dense = std_normal((B, L, D), device=device)
    
    with torch.no_grad():
        for t in range(T-1, -1, -1):
            t_tensor = torch.full((B, 1), t, device=device, dtype=torch.long)
            
            # 在 mask=1 的位置，强制使用 sparse 的值（作为锚点）
            # 这样去噪网络在预测时就能看到正确的锚点信息
            xt_dense = xt_dense * (1 - mask_expanded) + x_sparse * mask_expanded
            
            # 构造网络输入：concat(xt_dense, x_sparse, mask)
            mask_feat = mask.unsqueeze(-1).to(device)  # B, L, 1
            net_input = torch.cat([xt_dense, x_sparse.to(device), mask_feat], dim=-1)  # B, L, 2D+1
            
            # 预测噪声
            predict_noise = net(net_input, t_tensor)  # B, L, D
            
            # DDPM 去噪步骤
            coeff1 = 1 / (Alpha[t] ** 0.5)
            coeff2 = (1 - Alpha[t]) / ((1 - Alpha_bar[t]) ** 0.5)
            xt_dense = coeff1 * (xt_dense - coeff2 * predict_noise)
            
            # 添加噪声（除了最后一步）
            if t > 0:
                noise = std_normal(xt_dense.shape, device=device)
                sigma = ((1 - Alpha_bar[t-1]) / (1 - Alpha_bar[t]) * (1 - Alpha[t])) ** 0.5
                xt_dense = xt_dense + sigma * noise
        
        # 最后一步，确保 mask=1 的位置使用 sparse 的值
        xt_dense = xt_dense * (1 - mask_expanded) + x_sparse * mask_expanded
                
    return xt_dense

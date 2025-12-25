"""
扩散模型评价指标模块（基于路段ID的路径规划器）

核心评价维度：
1. 准确性：LCS F1-Score - 恢复路径与真实路径的重合程度
2. 有效性：RSC (Road Segment Connectivity) - 生成路径的拓扑连通性
3. 真实感：JSD-RS - 路段使用频率分布的相似性
4. 真实感：JSD-Length - 序列长度分布的相似性

参考指标：
- Token Accuracy: token级别准确率（仅作额外参考）
"""

import numpy as np
import torch
from collections import Counter
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Optional


def compute_rsc(pred_ids: torch.Tensor, spatial_A_trans: torch.Tensor) -> float:
    """
    计算路段连通性 (Road Segment Connectivity)
    
    RSC = 连通的相邻路段对数 / 总相邻路段对数
    
    注意：跳过相同ID的连续转移（如 A->A），因为这是正常现象（GPS点密集）
    
    Args:
        pred_ids: 预测的路段ID序列，shape (B, L)
        spatial_A_trans: 路网邻接矩阵，A[i,j]=1 表示连通，A[i,j]=1e-10 表示不连通
    
    Returns:
        rsc: 路段连通性比例 (0-1)
    """
    if pred_ids.dim() == 1:
        pred_ids = pred_ids.unsqueeze(0)
    
    B, L = pred_ids.shape
    
    if L < 2:
        return 1.0  # 长度小于2无法计算连通性
    
    device = pred_ids.device
    if not isinstance(spatial_A_trans, torch.Tensor):
        spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float32, device=device)
    elif spatial_A_trans.device != device:
        spatial_A_trans = spatial_A_trans.to(device)
    
    # 获取相邻路段对
    src_ids = pred_ids[:, :-1]  # B, L-1
    dst_ids = pred_ids[:, 1:]   # B, L-1
    
    # 跳过相同ID的转移（A->A 不需要检查连通性）
    different_mask = (src_ids != dst_ids).float()  # 1表示不同，0表示相同
    
    # 检查连通性：A[src, dst] != 1e-10 表示连通
    max_id = spatial_A_trans.shape[0] - 1
    src_ids_clamped = src_ids.clamp(0, max_id)
    dst_ids_clamped = dst_ids.clamp(0, max_id)
    
    # 获取邻接矩阵中的值
    connectivity = spatial_A_trans[src_ids_clamped, dst_ids_clamped]
    
    # 连通的定义：值不是 1e-10（即值为 1）
    connected = (connectivity > 1e-9).float()
    
    # 只统计不同ID之间的转移
    valid_pairs = different_mask.sum().item()
    if valid_pairs == 0:
        return 1.0  # 如果全是相同ID的转移，认为完全连通
    
    connected_pairs = (connected * different_mask).sum().item()
    
    rsc = connected_pairs / valid_pairs
    return rsc


def compute_jsd_rs(pred_ids: torch.Tensor, 
                   target_ids: torch.Tensor, 
                   num_segments: int,
                   eps: float = 1e-10) -> float:
    """
    计算路段使用分布的 Jensen-Shannon 散度 (JSD-RS)
    
    JSD 衡量生成数据分布与真实数据分布的相似性，值越小越好。
    
    Args:
        pred_ids: 预测的路段ID序列，shape (B, L) 或展平的 (N,)
        target_ids: 真实的路段ID序列，shape (B, L) 或展平的 (N,)
        num_segments: 路段总数（用于构建完整的分布）
        eps: 平滑项，避免零概率
    
    Returns:
        jsd: Jensen-Shannon 散度 (0-1 之间，0表示完全相同)
    """
    # 展平
    pred_flat = pred_ids.reshape(-1).cpu().numpy()
    target_flat = target_ids.reshape(-1).cpu().numpy()
    
    # 统计频率
    pred_counts = Counter(pred_flat)
    target_counts = Counter(target_flat)
    
    # 构建概率分布（包含所有路段）
    pred_dist = np.zeros(num_segments) + eps
    target_dist = np.zeros(num_segments) + eps
    
    for seg_id, count in pred_counts.items():
        if 0 <= seg_id < num_segments:
            pred_dist[seg_id] += count
    
    for seg_id, count in target_counts.items():
        if 0 <= seg_id < num_segments:
            target_dist[seg_id] += count
    
    # 归一化为概率分布
    pred_dist = pred_dist / pred_dist.sum()
    target_dist = target_dist / target_dist.sum()
    
    # 计算 Jensen-Shannon 散度
    jsd = jensenshannon(pred_dist, target_dist, base=2) ** 2  # scipy返回的是sqrt(JSD)
    
    return float(jsd)


def compute_jsd_length(pred_ids: torch.Tensor,
                       target_ids: torch.Tensor,
                       max_length: int = 200,
                       eps: float = 1e-10) -> float:
    """
    计算序列长度分布的 Jensen-Shannon 散度 (JSD-Length)
    
    评估生成路径的长度（路段数量）分布是否符合真实数据。
    
    Args:
        pred_ids: 预测的路段ID序列，shape (B, L)
        target_ids: 真实的路段ID序列，shape (B, L)
        max_length: 最大序列长度（用于构建分布）
        eps: 平滑项，避免零概率
    
    Returns:
        jsd: Jensen-Shannon 散度 (0-1 之间，0表示完全相同)
    """
    # 计算每条序列的实际长度（非padding部分）
    # 假设0是padding ID
    pred_lengths = []
    target_lengths = []
    
    if pred_ids.dim() == 1:
        pred_ids = pred_ids.unsqueeze(0)
        target_ids = target_ids.unsqueeze(0)
    
    pred_np = pred_ids.cpu().numpy()
    target_np = target_ids.cpu().numpy()
    
    for seq in pred_np:
        # 计算非0元素的数量作为长度
        length = np.count_nonzero(seq)
        if length == 0:  # 全是0的情况，取实际序列长度
            length = len(seq)
        pred_lengths.append(length)
    
    for seq in target_np:
        length = np.count_nonzero(seq)
        if length == 0:
            length = len(seq)
        target_lengths.append(length)
    
    # 统计长度分布
    pred_length_counts = Counter(pred_lengths)
    target_length_counts = Counter(target_lengths)
    
    # 构建概率分布
    pred_dist = np.zeros(max_length) + eps
    target_dist = np.zeros(max_length) + eps
    
    for length, count in pred_length_counts.items():
        if 0 <= length < max_length:
            pred_dist[length] += count
    
    for length, count in target_length_counts.items():
        if 0 <= length < max_length:
            target_dist[length] += count
    
    # 归一化
    pred_dist = pred_dist / pred_dist.sum()
    target_dist = target_dist / target_dist.sum()
    
    # 计算 Jensen-Shannon 散度
    jsd = jensenshannon(pred_dist, target_dist, base=2) ** 2
    
    return float(jsd)


def shrink_seq(seq: List) -> List:
    """移除连续重复的路段ID"""
    if len(seq) == 0:
        return []
    s0 = seq[0]
    new_seq = [s0]
    for s in seq[1:]:
        if s != s0:
            new_seq.append(s)
        s0 = s
    return new_seq


def lcs(xs: List, ys: List) -> List:
    """
    计算最长公共子序列 (Longest Common Subsequence)
    使用动态规划实现
    """
    m, n = len(xs), len(ys)
    if m == 0 or n == 0:
        return []
    
    # DP表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if xs[i-1] == ys[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # 回溯找到LCS
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if xs[i-1] == ys[j-1]:
            result.append(xs[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return result[::-1]


def compute_lcs_metrics(pred_ids: torch.Tensor, 
                        target_ids: torch.Tensor,
                        shrink: bool = True) -> Tuple[float, float, float]:
    """
    计算基于 LCS 的 F1-Score（主要指标）以及召回率和精确率
    
    Args:
        pred_ids: 预测的路段ID序列，shape (B, L)
        target_ids: 真实的路段ID序列，shape (B, L)
        shrink: 是否移除连续重复的ID
    
    Returns:
        f1_score: F1分数（召回率和精确率的调和平均）
        recall: LCS长度 / 真实序列长度
        precision: LCS长度 / 预测序列长度
    """
    if pred_ids.dim() == 1:
        pred_ids = pred_ids.unsqueeze(0)
        target_ids = target_ids.unsqueeze(0)
    
    B = pred_ids.shape[0]
    total_lcs_len = 0
    total_pred_len = 0
    total_target_len = 0
    
    pred_np = pred_ids.cpu().numpy()
    target_np = target_ids.cpu().numpy()
    
    for b in range(B):
        pred_seq = pred_np[b].tolist()
        target_seq = target_np[b].tolist()
        
        if shrink:
            pred_seq = shrink_seq(pred_seq)
            target_seq = shrink_seq(target_seq)
        
        lcs_seq = lcs(pred_seq, target_seq)
        
        total_lcs_len += len(lcs_seq)
        total_pred_len += len(pred_seq)
        total_target_len += len(target_seq)
    
    recall = total_lcs_len / max(total_target_len, 1)
    precision = total_lcs_len / max(total_pred_len, 1)
    
    # 计算 F1-Score（调和平均）
    if recall + precision > 0:
        f1_score = 2 * (recall * precision) / (recall + precision)
    else:
        f1_score = 0.0
    
    return f1_score, recall, precision


def compute_token_accuracy(pred_ids: torch.Tensor, 
                           target_ids: torch.Tensor,
                           ignore_index: int = 0) -> float:
    """
    计算 token-level 准确率（作为参考指标）
    
    Args:
        pred_ids: 预测的路段ID序列，shape (B, L)
        target_ids: 真实的路段ID序列，shape (B, L)
        ignore_index: 忽略的ID（如padding）
    
    Returns:
        accuracy: token级别准确率
    """
    mask = target_ids != ignore_index
    correct = (pred_ids == target_ids) & mask
    
    total = mask.sum().item()
    correct_count = correct.sum().item()
    
    return correct_count / max(total, 1)


def compute_all_metrics(pred_ids: torch.Tensor,
                        target_ids: torch.Tensor,
                        spatial_A_trans: torch.Tensor,
                        num_segments: int) -> Dict[str, float]:
    """
    计算所有评价指标（基于路段ID的路径规划器）
    
    核心指标：
    - LCS F1-Score: 准确性（路径恢复的重合程度）
    - RSC: 有效性（拓扑连通性）
    - JSD-RS: 真实感（路段使用分布）
    - JSD-Length: 真实感（序列长度分布）
    
    Args:
        pred_ids: 预测的路段ID序列，shape (B, L)
        target_ids: 真实的路段ID序列，shape (B, L)
        spatial_A_trans: 路网邻接矩阵
        num_segments: 路段总数
    
    Returns:
        metrics: 包含所有指标的字典
    """
    metrics = {}
    
    # 1. LCS F1-Score - 准确性（核心指标）
    lcs_f1, lcs_recall, lcs_precision = compute_lcs_metrics(pred_ids, target_ids, shrink=True)
    metrics['lcs_f1'] = lcs_f1
    metrics['lcs_recall'] = lcs_recall  # 保留以便分析
    metrics['lcs_precision'] = lcs_precision  # 保留以便分析
    
    # 2. RSC - 有效性（核心指标）
    metrics['rsc'] = compute_rsc(pred_ids, spatial_A_trans)
    
    # 3. JSD-RS - 真实感：路段使用分布（核心指标）
    metrics['jsd_rs'] = compute_jsd_rs(pred_ids, target_ids, num_segments)
    
    # 4. JSD-Length - 真实感：序列长度分布（核心指标）
    metrics['jsd_length'] = compute_jsd_length(pred_ids, target_ids)
    
    # 5. Token-level 准确率（参考指标）
    metrics['token_acc'] = compute_token_accuracy(pred_ids, target_ids)
    
    return metrics


class DiffusionMetricsTracker:
    """
    扩散模型评价指标追踪器
    
    用于在验证过程中累积计算指标
    """
    
    def __init__(self, num_segments: int, spatial_A_trans: torch.Tensor):
        self.num_segments = num_segments
        self.spatial_A_trans = spatial_A_trans
        self.reset()
    
    def reset(self):
        """重置所有计数器"""
        self.total_pairs = 0
        self.connected_pairs = 0
        self.all_pred_ids = []
        self.all_target_ids = []
        self.total_lcs_len = 0
        self.total_pred_len = 0
        self.total_target_len = 0
        self.correct_tokens = 0
        self.total_tokens = 0
    
    def update(self, pred_ids: torch.Tensor, target_ids: torch.Tensor):
        """
        更新指标计算
        
        Args:
            pred_ids: 预测的路段ID序列，shape (B, L)
            target_ids: 真实的路段ID序列，shape (B, L)
        """
        if pred_ids.dim() == 1:
            pred_ids = pred_ids.unsqueeze(0)
            target_ids = target_ids.unsqueeze(0)
        
        B, L = pred_ids.shape
        device = pred_ids.device
        
        # 更新 RSC 计数（跳过相同ID的转移）
        if L >= 2:
            spatial_A = self.spatial_A_trans
            if not isinstance(spatial_A, torch.Tensor):
                spatial_A = torch.tensor(spatial_A, dtype=torch.float32, device=device)
            elif spatial_A.device != device:
                spatial_A = spatial_A.to(device)
            
            max_id = spatial_A.shape[0] - 1
            src_ids = pred_ids[:, :-1]
            dst_ids = pred_ids[:, 1:]
            
            # 跳过相同ID的转移
            different_mask = (src_ids != dst_ids).float()
            
            src_ids_clamped = src_ids.clamp(0, max_id)
            dst_ids_clamped = dst_ids.clamp(0, max_id)
            
            connectivity = spatial_A[src_ids_clamped, dst_ids_clamped]
            connected = (connectivity > 1e-9).float()
            
            # 只统计不同ID之间的转移
            valid_pairs = different_mask.sum().item()
            self.total_pairs += valid_pairs
            self.connected_pairs += (connected * different_mask).sum().item()
        
        # 累积所有ID用于JSD计算
        self.all_pred_ids.append(pred_ids.cpu())
        self.all_target_ids.append(target_ids.cpu())
        
        # 更新 LCS 指标
        pred_np = pred_ids.cpu().numpy()
        target_np = target_ids.cpu().numpy()
        
        for b in range(B):
            pred_seq = shrink_seq(pred_np[b].tolist())
            target_seq = shrink_seq(target_np[b].tolist())
            
            lcs_seq = lcs(pred_seq, target_seq)
            
            self.total_lcs_len += len(lcs_seq)
            self.total_pred_len += len(pred_seq)
            self.total_target_len += len(target_seq)
        
        # 更新 token 准确率
        mask = target_ids != 0
        correct = (pred_ids == target_ids) & mask
        self.correct_tokens += correct.sum().item()
        self.total_tokens += mask.sum().item()
    
    def compute(self) -> Dict[str, float]:
        """
        计算最终指标
        
        Returns:
            metrics: 包含所有指标的字典
        """
        metrics = {}
        
        # 1. LCS F1-Score - 准确性（核心指标）
        lcs_recall = self.total_lcs_len / max(self.total_target_len, 1)
        lcs_precision = self.total_lcs_len / max(self.total_pred_len, 1)
        if lcs_recall + lcs_precision > 0:
            lcs_f1 = 2 * (lcs_recall * lcs_precision) / (lcs_recall + lcs_precision)
        else:
            lcs_f1 = 0.0
        metrics['lcs_f1'] = lcs_f1
        metrics['lcs_recall'] = lcs_recall  # 保留以便分析
        metrics['lcs_precision'] = lcs_precision  # 保留以便分析
        
        # 2. RSC - 有效性（核心指标）
        metrics['rsc'] = self.connected_pairs / max(self.total_pairs, 1)
        
        # 3. JSD-RS - 真实感：路段使用分布（核心指标）
        if self.all_pred_ids and self.all_target_ids:
            # 先展平所有张量（因为不同batch可能有不同长度），再拼接
            all_pred_flat = [tensor.reshape(-1) for tensor in self.all_pred_ids]
            all_target_flat = [tensor.reshape(-1) for tensor in self.all_target_ids]
            all_pred = torch.cat(all_pred_flat, dim=0)
            all_target = torch.cat(all_target_flat, dim=0)
            metrics['jsd_rs'] = compute_jsd_rs(all_pred, all_target, self.num_segments)
        else:
            metrics['jsd_rs'] = 1.0
        
        # 4. JSD-Length - 真实感：序列长度分布（核心指标）
        if self.all_pred_ids and self.all_target_ids:
            # 分别处理每个批次，计算长度，避免拼接不同长度的张量
            pred_lengths = []
            target_lengths = []
            for pred_batch in self.all_pred_ids:
                for seq in pred_batch.cpu().numpy():
                    length = np.count_nonzero(seq)
                    if length == 0:
                        length = len(seq)
                    pred_lengths.append(length)
            
            for target_batch in self.all_target_ids:
                for seq in target_batch.cpu().numpy():
                    length = np.count_nonzero(seq)
                    if length == 0:
                        length = len(seq)
                    target_lengths.append(length)
            
            # 使用长度列表计算JSD
            from collections import Counter
            max_length = max(max(pred_lengths, default=0), max(target_lengths, default=0)) + 1
            
            pred_length_counts = Counter(pred_lengths)
            target_length_counts = Counter(target_lengths)
            
            pred_dist = np.zeros(max_length) + 1e-10
            target_dist = np.zeros(max_length) + 1e-10
            
            for length, count in pred_length_counts.items():
                if 0 <= length < max_length:
                    pred_dist[length] += count
            
            for length, count in target_length_counts.items():
                if 0 <= length < max_length:
                    target_dist[length] += count
            
            pred_dist = pred_dist / pred_dist.sum()
            target_dist = target_dist / target_dist.sum()
            
            from scipy.spatial.distance import jensenshannon
            metrics['jsd_length'] = float(jensenshannon(pred_dist, target_dist, base=2) ** 2)
        else:
            metrics['jsd_length'] = 1.0
        
        # 5. Token 准确率（参考指标）
        metrics['token_acc'] = self.correct_tokens / max(self.total_tokens, 1)
        
        return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    格式化指标输出（基于路段ID的路径规划器）
    
    Args:
        metrics: 指标字典
    
    Returns:
        formatted: 格式化的字符串
    """
    lines = []
    lines.append(f"=== 路径规划器评价指标 ===")
    lines.append(f"[核心指标]")
    lines.append(f"  准确性 - LCS F1-Score: {metrics.get('lcs_f1', 0):.4f}  ↑越高越好")
    lines.append(f"  有效性 - RSC (路段连通性): {metrics.get('rsc', 0):.4f}  ↑越高越好 (目标≈1.0)")
    lines.append(f"  真实感 - JSD-RS (路段分布): {metrics.get('jsd_rs', 1):.4f}  ↓越低越好")
    # 注意：JSD-Length 对固定长度的条件生成任务无意义（始终为0），因此不显示
    lines.append(f"[参考指标]")
    lines.append(f"  LCS Recall: {metrics.get('lcs_recall', 0):.4f}")
    lines.append(f"  LCS Precision: {metrics.get('lcs_precision', 0):.4f}")
    lines.append(f"  Token Accuracy: {metrics.get('token_acc', 0):.4f}")
    return "\n".join(lines)


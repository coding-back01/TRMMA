#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DA 算法基线评估脚本 (使用MMA代码逻辑)

参考 mma.py 中 GPS2SegData 的逻辑:
1. 稀疏点索引: keep_index = traj.low_idx
2. gt_segs 来源: ds_pt.data['candi_pt'].eid (按 low_idx 提取的稀疏点)
3. 真值稠密路段: traj.cpath
"""

import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm


def shrink_seq(seq):
    """移除序列中连续重复的元素"""
    if len(seq) == 0:
        return []
    result = [seq[0]]
    for s in seq[1:]:
        if s != result[-1]:
            result.append(s)
    return result


def lcs_length(xs, ys):
    """计算两个序列的最长公共子序列(LCS)长度"""
    m, n = len(xs), len(ys)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if xs[i-1] == ys[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def calc_accuracy(predict, target, exclude_endpoints=False):
    """计算预测序列与真值序列的匹配指标"""
    pred_shrink = shrink_seq(predict)
    target_shrink = shrink_seq(target)
    
    if exclude_endpoints:
        if len(pred_shrink) > 2:
            pred_shrink = pred_shrink[1:-1]
        else:
            pred_shrink = []
        if len(target_shrink) > 2:
            target_shrink = target_shrink[1:-1]
        else:
            target_shrink = []
    
    if len(target_shrink) == 0 or len(pred_shrink) == 0:
        return 0.0, 0.0, 0.0
    
    lcs_len = lcs_length(pred_shrink, target_shrink)
    
    recall = lcs_len / len(target_shrink)
    precision = lcs_len / len(pred_shrink)
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    
    return recall, precision, f1


def main():
    parser = argparse.ArgumentParser(description='DA算法基线评估')
    parser.add_argument('--mma_output', type=str, required=True, help='MMA推理结果pkl文件')
    parser.add_argument('--city', type=str, default='porto', help='城市名称')
    parser.add_argument('--data_root', type=str, default='data', help='数据根目录')
    parser.add_argument('--max_trajs', type=int, default=-1, help='最大轨迹数(-1表示全部)')
    args = parser.parse_args()
    
    city_configs = {
        'porto': {'zone_range': [41.1395, -8.6911, 41.1864, -8.5521], 'utc': 1},
        'beijing': {'zone_range': [39.7547, 116.1994, 40.0244, 116.5452], 'utc': 0},
        'chengdu': {'zone_range': [30.6443, 104.0288, 30.7416, 104.1375], 'utc': 8},
        'xian': {'zone_range': [34.2060, 108.9058, 34.2825, 109.0049], 'utc': 8},
    }
    
    config = None
    for key in city_configs:
        if args.city.startswith(key):
            config = city_configs[key]
            break
    if config is None:
        raise ValueError(f"未知城市: {args.city}")
    
    print("=" * 70)
    print("DA 算法基线评估 (使用MMA代码逻辑)")
    print("=" * 70)
    
    from models.trmma import DAPlanner
    from utils.map import RoadNetworkMapFull
    
    map_root = os.path.join(args.data_root, args.city, "roadnet")
    rn = RoadNetworkMapFull(map_root, zone_range=config['zone_range'], unit_length=50)
    dam = DAPlanner(os.path.join(args.data_root, args.city), rn.valid_edge_cnt_one - 1, config['utc'])
    
    # =========================================================================
    # 加载数据
    # =========================================================================
    print(f"\n【数据加载】")
    
    # 1. 加载 MMA 预测结果
    print(f"MMA预测结果: {args.mma_output}")
    mma_data = pickle.load(open(args.mma_output, 'rb'))
    print(f"  记录数: {len(mma_data)}")
    
    # 2. 加载测试轨迹 (参考 mma.py 第30行)
    test_path = os.path.join(args.data_root, args.city, 'test_output.pkl')
    print(f"测试轨迹: {test_path}")
    test_trajs = pickle.load(open(test_path, 'rb'))
    print(f"  轨迹数: {len(test_trajs)}")
    
    # 限制轨迹数
    if args.max_trajs > 0:
        test_trajs = test_trajs[:args.max_trajs]
        mma_data = mma_data[:args.max_trajs]
        print(f"  限制为前 {args.max_trajs} 条")
    
    # =========================================================================
    # 按照 mma.py 的逻辑处理数据
    # =========================================================================
    print(f"\n【数据处理 - 参考 mma.py GPS2SegData 逻辑】")
    print("  keep_index = traj.low_idx  # 稀疏点索引")
    print("  gt_segs = [pt.data['candi_pt'].eid for pt in sparse_pts]  # 真实稀疏路段")
    print("  gt_dense = traj.cpath  # 真实稠密路段")
    
    # =========================================================================
    # 评估
    # =========================================================================
    print(f"\n【开始评估】")
    
    metrics_with_ep = {'recall': [], 'precision': [], 'f1': []}
    metrics_without_ep = {'recall': [], 'precision': [], 'f1': []}
    
    segment_count = 0
    skipped_no_middle = 0
    mma_both_correct = 0
    mma_any_wrong = 0
    
    for traj_idx, (pred_segs, gt_segs) in enumerate(tqdm(zip(mma_data, test_trajs), 
                                                          total=len(test_trajs), 
                                                          desc='评估中')):
        # mma_data[i] = (pred_segs_list, gt_segs_list)
        # 但这里 gt_segs 变量实际上是 test_trajs[i] (traj 对象)
        # 需要重新从 mma_data 获取
        if traj_idx >= len(mma_data):
            break
        
        pred_segs_mma, gt_segs_mma = mma_data[traj_idx]
        traj = test_trajs[traj_idx]
        
        # 参考 mma.py 第55-56行: keep_index = traj.low_idx
        low_idx = traj.low_idx
        
        # 验证 gt_segs_mma 和从轨迹中提取的一致
        # 参考 mma.py 第113行: mm_eids.append(ds_pt.data['candi_pt'].eid)
        gt_segs_from_traj = [traj.pt_list[idx].data['candi_pt'].eid for idx in low_idx]
        
        if list(gt_segs_mma) != gt_segs_from_traj:
            # 如果不一致，跳过
            continue
        
        # 对每对相邻稀疏点进行评估
        for i in range(len(low_idx) - 1):
            # 稀疏点在 pt_list 中的索引
            idx1 = low_idx[i]
            idx2 = low_idx[i + 1]
            
            pt1 = traj.pt_list[idx1]
            pt2 = traj.pt_list[idx2]
            
            # 检查两点之间是否有中间路段需要补全
            # 参考原始代码判断逻辑
            if (idx1 + 1) >= idx2:
                # 两点相邻，无需补全
                continue
            
            segment_count += 1
            
            # 真值稠密路段：从 cpath 中截取
            # 使用 cpath_idx 来定位
            gt_dense = traj.cpath[pt1.cpath_idx : pt2.cpath_idx + 1]
            
            if len(shrink_seq(gt_dense)) == 0:
                continue
            
            # MMA 预测的起终点
            s1_pred = pred_segs_mma[i]
            s2_pred = pred_segs_mma[i + 1]
            
            # 真实的起终点
            s1_gt = gt_segs_mma[i]
            s2_gt = gt_segs_mma[i + 1]
            
            # 统计 MMA 预测情况
            if s1_pred == s1_gt and s2_pred == s2_gt:
                mma_both_correct += 1
            else:
                mma_any_wrong += 1
            
            # DA 规划
            try:
                da_output = dam.planning_multi([s1_pred, s2_pred], pt1.time, mode='da')
            except:
                continue
            
            if len(da_output) == 0:
                continue
            
            # 计算指标 - 包含端点
            r, p, f = calc_accuracy(da_output, gt_dense, exclude_endpoints=False)
            metrics_with_ep['recall'].append(r)
            metrics_with_ep['precision'].append(p)
            metrics_with_ep['f1'].append(f)
            
            # 计算指标 - 不包含端点
            gt_shrink = shrink_seq(gt_dense)
            if len(gt_shrink) > 2:
                r2, p2, f2 = calc_accuracy(da_output, gt_dense, exclude_endpoints=True)
                metrics_without_ep['recall'].append(r2)
                metrics_without_ep['precision'].append(p2)
                metrics_without_ep['f1'].append(f2)
            else:
                skipped_no_middle += 1
    
    # =========================================================================
    # 输出结果
    # =========================================================================
    print(f"\n" + "=" * 70)
    print("结果")
    print("=" * 70)
    
    print(f"\n【数据统计】")
    print(f"  评估轨迹数: {len(test_trajs)}")
    print(f"  评估片段总数: {segment_count}")
    print(f"  无中间路段片段: {skipped_no_middle}")
    
    print(f"\n【MMA预测分析】")
    total_mma = mma_both_correct + mma_any_wrong
    if total_mma > 0:
        print(f"  起终点都正确: {mma_both_correct}/{total_mma} ({mma_both_correct/total_mma*100:.2f}%)")
        print(f"  起点或终点错误: {mma_any_wrong}/{total_mma} ({mma_any_wrong/total_mma*100:.2f}%)")
    
    print(f"\n【包含端点】评估完整路径 A→B→C→D→E")
    if len(metrics_with_ep['f1']) > 0:
        print(f"  有效片段数: {len(metrics_with_ep['f1'])}")
        print(f"  召回率: {np.mean(metrics_with_ep['recall'])*100:.2f}%")
        print(f"  精确率: {np.mean(metrics_with_ep['precision'])*100:.2f}%")
        print(f"  F1分数: {np.mean(metrics_with_ep['f1'])*100:.2f}%")
    
    print(f"\n【不含端点】只评估中间补全部分 B→C→D（更严格）")
    if len(metrics_without_ep['f1']) > 0:
        print(f"  有效片段数: {len(metrics_without_ep['f1'])}")
        print(f"  召回率: {np.mean(metrics_without_ep['recall'])*100:.2f}%")
        print(f"  精确率: {np.mean(metrics_without_ep['precision'])*100:.2f}%")
        print(f"  F1分数: {np.mean(metrics_without_ep['f1'])*100:.2f}%")
    
    print(f"\n" + "=" * 70)
    print("【结论】")
    if len(metrics_with_ep['f1']) > 0:
        print(f"  包含端点的基线F1: {np.mean(metrics_with_ep['f1'])*100:.2f}%")
    if len(metrics_without_ep['f1']) > 0:
        print(f"  不含端点的基线F1: {np.mean(metrics_without_ep['f1'])*100:.2f}% ← 推荐使用")
    print("=" * 70)


if __name__ == '__main__':
    main()

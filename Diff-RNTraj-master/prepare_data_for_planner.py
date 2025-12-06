#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 TRMMA 的多城市数据 (train.pkl / valid.pkl / test_output.pkl)
构造用于「条件 Diff-RNTraj」的 (dense_eids, sparse_eids, mask) 三元组，
并按轨迹长度分桶，保存为一个 bin 文件（pickle 序列化）。

使用说明
--------
在 Diff-RNTraj-master/ 目录下运行，例如：

1）Porto 训练集：
    python prepare_data_for_planner.py \
        --city porto \
        --data_dir ../data/porto \
        --subset train \
        --keep_ratio 0.1 \
        --min_len 20 \
        --max_len 100

2）Chengdu 验证集（假设数据在 ../data/chengdu）：
    python prepare_data_for_planner.py \
        --city chengdu \
        --data_dir ../data/chengdu \
        --subset valid \
        --keep_ratio 0.1 \
        --min_len 20 \
        --max_len 100

3）如果你不指定 --output_path，脚本会自动在
   data/{city}/cond_data/cond_seqs_{city}_{subset}.bin
   下生成对应的 bin 文件，例如：
   - city=porto, subset=train -> data/porto/cond_data/cond_seqs_porto_train.bin
   - city=chengdu, subset=valid -> data/chengdu/cond_data/cond_seqs_chengdu_valid.bin

输出格式
--------
输出是一个 dict，经 pickle.dump 到 output_path：

    {
        L1: [
            {"dense": [eid1, eid2, ...], "sparse": [...], "mask": [...]},
            ...
        ],
        L2: [...],
        ...
    }

其中：
    - key: 轨迹长度 L（int）
    - dense: 稠密路段 ID 序列，长度为 L
    - sparse: 稀疏路段 ID 序列，同长 L，未观测位置为 0
    - mask: 稀疏掩码，同长 L，观测位置为 1，未观测位置为 0

后续在训练 Diff-RNTraj 条件扩散版本时，可以直接按长度 L 取 batch，
避免 padding，节省显存并加快训练。
"""

import os
import sys
import pickle
import random
import argparse
from typing import Dict, List, Any


def load_trajs(data_dir: str, subset: str):
    """
    从 TRMMA 的预处理结果中加载轨迹列表。

    subset:
        - 'train' -> data_dir/train.pkl
        - 'valid' -> data_dir/valid.pkl
        - 'test'  -> data_dir/test_output.pkl  （TRMMA 的高采样测试轨迹）
    """
    # 为了让 pickle 能正确反序列化 utils.trajectory_func.Trajectory，
    # 在运行时手动把 TRMMA 根目录加入 sys.path，并动态注册模块。
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    # 如果 utils.trajectory_func 还没被加载，则从源码加载并注册到 sys.modules
    if "utils.trajectory_func" not in sys.modules:
        traj_mod_path = os.path.join(root_dir, "utils", "trajectory_func.py")
        if os.path.exists(traj_mod_path):
            import importlib.util

            spec = importlib.util.spec_from_file_location("utils.trajectory_func", traj_mod_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                sys.modules["utils.trajectory_func"] = module
                spec.loader.exec_module(module)

    if subset == "train":
        fname = "train.pkl"
    elif subset == "valid":
        fname = "valid.pkl"
    elif subset == "test":
        fname = "test_output.pkl"
    else:
        raise ValueError(f"Unknown subset: {subset}")

    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}")

    print(f"[Info] Loading trajectories from: {path}")
    with open(path, "rb") as f:
        trajs = pickle.load(f)
    print(f"[Info] Loaded {len(trajs)} trajectories")
    return trajs


def extract_dense_eids_from_traj(traj) -> List[int]:
    """
    从 TRMMA 的 Trajectory 对象中抽取"稠密路段 ID 序列"。

    根据当前数据结构：
        - traj.pt_list: list[STPoint]
        - 每个 STPoint.data 是 dict，包含:
              data['candi_pt'] 是 utils.candidate_point.CandidatePoint 对象
          该对象有 eid 属性，直接访问 candi.eid 即可。
    """
    dense_eids: List[int] = []

    pt_list = getattr(traj, "pt_list", None)
    if pt_list is None:
        return dense_eids

    for pt in pt_list:
        data = getattr(pt, "data", None)
        if not isinstance(data, dict):
            continue
        candi = data.get("candi_pt", None)
        if candi is None:
            # 理论上 TRMMA 预处理后应该都有匹配结果，这里做个兜底
            continue

        # CandidatePoint 对象，直接访问 eid 属性
        if hasattr(candi, "eid"):
            try:
                eid = int(candi.eid)
                dense_eids.append(eid)
            except (ValueError, TypeError, AttributeError):
                continue
        # 兜底：如果是字符串格式（虽然 TRMMA 数据应该不是）
        elif isinstance(candi, str):
            parts = candi.split(",")
            if parts:
                try:
                    eid = int(float(parts[0]))
                    dense_eids.append(eid)
                except ValueError:
                    continue
        # 兜底：如果是 list/tuple
        elif isinstance(candi, (list, tuple)) and len(candi) > 0:
            try:
                eid = int(candi[0])
                dense_eids.append(eid)
            except (ValueError, TypeError):
                continue

    return dense_eids


def build_conditional_sequences(
    trajs: List[Any],
    keep_ratio: float = 0.1,
    min_len: int = 20,
    max_len: int = 100,
) -> Dict[int, List[Dict[str, List[int]]]]:
    """
    从一批轨迹构造：
      - dense_eids: 稠密路段 ID 序列 [L]
      - sparse_eids: 稀疏路段 ID 序列 [L]（未观测位置为 0）
      - mask: 稀疏位置掩码 [L]（观测为 1，未观测为 0）

    返回：
      cond_dict: { length(int) : [ { 'dense': [...], 'sparse': [...], 'mask': [...] }, ... ] }
    """
    cond_dict: Dict[int, List[Dict[str, List[int]]]] = {}
    dropped_too_short = 0
    dropped_no_candi = 0

    for idx, traj in enumerate(trajs):
        dense = extract_dense_eids_from_traj(traj)

        if len(dense) == 0:
            dropped_no_candi += 1
            continue

        L = len(dense)
        if L < min_len or L > max_len:
            dropped_too_short += 1
            continue

        # 生成稀疏观测位置
        if keep_ratio <= 0 or keep_ratio >= 1:
            keep_indices = list(range(L))
        else:
            k = max(1, int(L * keep_ratio))
            # 避免极端情况下 k > L
            k = min(k, L)
            keep_indices = sorted(random.sample(range(L), k))

        sparse = [0] * L
        mask = [0] * L
        for i in keep_indices:
            sparse[i] = dense[i]
            mask[i] = 1

        entry = {
            "dense": dense,
            "sparse": sparse,
            "mask": mask,
        }
        cond_dict.setdefault(L, []).append(entry)

    print(f"[Info] Total trajs: {len(trajs)}")
    print(f"[Info] Dropped (no matched candi_pt): {dropped_no_candi}")
    print(f"[Info] Dropped (length not in [{min_len}, {max_len}]): {dropped_too_short}")
    if cond_dict:
        print(f"[Info] Kept lengths (sample): {sorted(cond_dict.keys())[:10]} ...")
    else:
        print("[Warn] No trajectories kept after filtering.")
    return cond_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess TRMMA data into conditional sequences for Diff-RNTraj planner"
    )
    parser.add_argument(
        "--city",
        type=str,
        default="porto",
        choices=["porto", "xian", "beijing", "chengdu"],
        help="城市名称，用于输出路径命名（不影响 data_dir 的实际位置）",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/porto",
        help="该城市的数据目录（相对于 Diff-RNTraj-master/），包含 train.pkl / valid.pkl / test_output.pkl",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "valid", "test", "all"],
        help="处理哪个子集：train / valid / test / all(三个一起)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="输出 bin 文件路径（pickle 序列化的 dict）。"
             "如果不指定，则自动生成到 data/{city}/cond_data/cond_seqs_{city}_{subset}.bin",
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.1,
        help="稀疏采样比例，例如 0.1 表示保留约 10%% 的点作为稀疏观测",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=1,
        help="保留的最小轨迹长度（时间步），默认1表示保留所有轨迹",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=200,
        help="保留的最大轨迹长度（时间步），默认200足够覆盖所有轨迹",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证稀疏采样可复现",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[Info] Args: {args}")
    subsets = ["train", "valid", "test"] if args.subset == "all" else [args.subset]

    if args.output_path is not None and len(subsets) > 1:
        raise ValueError("当 --subset=all 时请不要同时指定 --output_path。")

    for sub in subsets:
        print(f"\n===== Processing subset: {sub} =====")
        random.seed(args.seed)  # 每个子集都使用同一个种子，保证可复现

        trajs = load_trajs(args.data_dir, sub)
        cond_dict = build_conditional_sequences(
            trajs,
            keep_ratio=args.keep_ratio,
            min_len=args.min_len,
            max_len=args.max_len,
        )

        if args.output_path is None:
            fname = f"cond_seqs_{args.city}_{sub}.bin"
            script_dir = os.path.dirname(os.path.abspath(__file__))
            out_dir = os.path.join(script_dir, "data", args.city, "cond_data")
            output_path = os.path.join(out_dir, fname)
        else:
            output_path = args.output_path
            out_dir = os.path.dirname(output_path)

        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(cond_dict, f)
        print(f"[Info] Saved conditional sequences to: {output_path}")


if __name__ == "__main__":
    main()


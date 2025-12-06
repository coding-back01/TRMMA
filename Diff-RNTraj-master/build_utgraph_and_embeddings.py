#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 TRMMA 预处理后的轨迹（train.pkl / valid.pkl / test_output.pkl）
为单个城市构建 UTGraph（路段级图）并训练 node2vec 路段嵌入。

输出：
- Diff-RNTraj-master/data/{city}/graph/graph_A.csv       邻接矩阵（权重为共现次数）
- Diff-RNTraj-master/data/{city}/graph/graph_node_id2idx.txt  路段ID到节点索引的映射
- Diff-RNTraj-master/data/{city}/graph/graph_edge.edgelist    边列表（用于 node2vec）
- Diff-RNTraj-master/data/{city}/graph/road_embed.txt    路段嵌入（与 Diff-RNTraj 原格式兼容）

使用示例（在项目根目录 TRMMA 下运行）：

1）仅使用训练集构图（推荐做法）：
    python Diff-RNTraj-master/build_utgraph_and_embeddings.py \
        --city porto \
        --data_dir data/porto

2）使用 train + valid 构图（可选）：
    python Diff-RNTraj-master/build_utgraph_and_embeddings.py \
        --city porto \
        --data_dir data/porto \
        --use_valid_in_graph

3）使用 train + valid + test 构图（不推荐，仅供实验）：
    python Diff-RNTraj-master/build_utgraph_and_embeddings.py \
        --city porto \
        --data_dir data/porto \
        --use_all_splits
"""

import os
import sys
import argparse
import pickle
from collections import defaultdict

import numpy as np
import torch

try:
    from gensim.models import Word2Vec
    _HAS_GENSIM = True
except ImportError:
    _HAS_GENSIM = False


def load_trajs_for_graph(data_dir: str, use_valid: bool, use_all: bool):
    """
    从指定目录加载用于构图的轨迹列表。
    - 始终加载 train.pkl
    - 如果 use_valid=True，则额外加载 valid.pkl
    - 如果 use_all=True，则再加载 test_output.pkl
    """
    # 让 pickle 能找到 TRMMA 的 utils.trajectory_func / utils.candidate_point 等类
    root_dir = os.path.dirname(os.path.abspath(__file__))  # Diff-RNTraj-master/
    root_dir = os.path.dirname(root_dir)                   # TRMMA 根目录
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    # 触发相关模块加载（主要是为了类型反序列化）
    # 不需要显式 import 类，只需保证模块在 sys.path 中即可

    def _load_pkl(name: str):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find file: {path}")
        print(f"[Info] Loading {name} from: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    trajs = []
    train_trajs = _load_pkl("train.pkl")
    print(f"[Info] train.pkl trajectories: {len(train_trajs)}")
    trajs.extend(train_trajs)

    if use_valid or use_all:
        valid_trajs = _load_pkl("valid.pkl")
        print(f"[Info] valid.pkl trajectories: {len(valid_trajs)}")
        trajs.extend(valid_trajs)

    if use_all:
        test_trajs = _load_pkl("test_output.pkl")
        print(f"[Info] test_output.pkl trajectories: {len(test_trajs)}")
        trajs.extend(test_trajs)

    print(f"[Info] Total trajectories used for graph: {len(trajs)}")
    return trajs


def build_utgraph_from_trajs(trajs):
    """
    根据 Trajectory 列表构建路段级 UTGraph：
    - 节点：路段 ID (candi_pt.eid)
    - 边权：在任意轨迹中连续出现的次数
    返回：
    - eid_list: 按索引排序的唯一路段ID列表
    - edges_count: dict[(eid_u, eid_v)] -> weight
    """
    edges_count = defaultdict(int)
    eid_set = set()

    for traj in trajs:
        pt_list = getattr(traj, "pt_list", None)
        if not pt_list:
            continue
        prev_eid = None
        for pt in pt_list:
            data = getattr(pt, "data", None)
            if not isinstance(data, dict):
                continue
            candi = data.get("candi_pt", None)
            if candi is None or not hasattr(candi, "eid"):
                continue
            try:
                eid = int(candi.eid)
            except Exception:
                continue

            eid_set.add(eid)
            if prev_eid is not None and prev_eid != eid:
                edges_count[(prev_eid, eid)] += 1
            prev_eid = eid

    eid_list = sorted(eid_set)
    print(f"[Info] Num of unique road IDs (nodes): {len(eid_list)}")
    print(f"[Info] Num of edges (u,v) with weight>0: {len(edges_count)}")
    return eid_list, edges_count


def save_graph_and_edge_index(eid_list, edges_count, graph_dir):
    """
    根据 eid_list 和 edges_count 保存：
    - graph_A.csv：邻接矩阵（权重为共现次数）
    - graph_node_id2idx.txt：路段ID到节点索引映射
    - graph_edge.edgelist：边列表（u_idx v_idx weight），用于 node2vec
    并返回 edge_index（2, E）和节点总数。
    """
    os.makedirs(graph_dir, exist_ok=True)

    eid2idx = {eid: idx for idx, eid in enumerate(eid_list)}
    num_nodes = len(eid_list)

    # 保存 node_id2idx 映射
    node_map_path = os.path.join(graph_dir, "graph_node_id2idx.txt")
    with open(node_map_path, "w") as f:
        for idx, eid in enumerate(eid_list):
            f.write(f"{eid} {idx}\n")
    print(f"[Info] Saved node_id2idx mapping to: {node_map_path}")

    # 构建稀疏邻接矩阵（先以 dict 形式存，后面构 edge_index；dense 写 csv）
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    edge_src_idx = []
    edge_dst_idx = []

    edgelist_path = os.path.join(graph_dir, "graph_edge.edgelist")
    with open(edgelist_path, "w") as f_edge:
        for (u_eid, v_eid), w in edges_count.items():
            iu = eid2idx[u_eid]
            iv = eid2idx[v_eid]
            A[iu, iv] = float(w)
            edge_src_idx.append(iu)
            edge_dst_idx.append(iv)
            f_edge.write(f"{iu} {iv} {w}\n")

    print(f"[Info] Built adjacency matrix with shape: {A.shape}")
    adj_path = os.path.join(graph_dir, "graph_A.csv")
    np.savetxt(adj_path, A, delimiter=",")
    print(f"[Info] Saved adjacency matrix to: {adj_path}")

    edge_index = torch.tensor([edge_src_idx, edge_dst_idx], dtype=torch.long)
    print(f"[Info] edge_index shape: {edge_index.shape}")
    return edge_index, num_nodes


def train_and_save_node2vec(edge_index, num_nodes, embed_dim, graph_dir, city):
    """
    在给定的 edge_index 上训练“类 node2vec”嵌入，并将结果保存为 road_embed.txt。
    这里使用 gensim.Word2Vec 在随机游走序列上训练，避免对 torch_geometric/pyg-lib 的依赖。
    """
    if not _HAS_GENSIM:
        raise ImportError(
            "需要 gensim 才能训练 node2vec 嵌入，请先安装：\n"
            "  pip install gensim\n"
        )

    # 从 edge_index 构建邻接表（有向图）
    edge_index = edge_index.cpu().numpy()
    src, dst = edge_index
    neighbors = defaultdict(list)
    for u, v in zip(src, dst):
        neighbors[int(u)].append(int(v))

    # 生成随机游走序列（DeepWalk 风格，等价于 p=q=1 的 node2vec）
    walk_length = 50
    walks_per_node = 10
    all_walks = []
    rng = np.random.default_rng(42)

    for node in range(num_nodes):
        for _ in range(walks_per_node):
            walk = [node]
            current = node
            for _ in range(walk_length - 1):
                nbrs = neighbors.get(current, [])
                if not nbrs:
                    break
                current = int(rng.choice(nbrs))
                walk.append(current)
            # gensim 需要字符串 token
            all_walks.append([str(n) for n in walk])

    print(f"[Info] Generated {len(all_walks)} random walks for node2vec training")

    # 使用 gensim.Word2Vec 训练嵌入
    model = Word2Vec(
        sentences=all_walks,
        vector_size=embed_dim,
        window=10,
        min_count=0,
        sg=1,              # skip-gram
        hs=0,
        negative=5,
        workers=4,
        epochs=5,
    )

    embs = np.zeros((num_nodes, embed_dim), dtype=np.float32)
    for idx in range(num_nodes):
        token = str(idx)
        if token in model.wv:
            embs[idx] = model.wv[token]
        else:
            embs[idx] = rng.normal(scale=0.01, size=(embed_dim,))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Trained gensim-based node2vec embeddings on device: {device} (Word2Vec CPU)")

    assert embs.shape == (num_nodes, embed_dim)

    out_path = os.path.join(graph_dir, "road_embed.txt")
    with open(out_path, "w") as f:
        # 第一行：节点数 和 维度
        f.write(f"{num_nodes} {embed_dim}\n")
        for idx in range(num_nodes):
            vec = embs[idx]
            vec_str = " ".join(f"{v:.6f}" for v in vec)
            # 注意：这里的“索引”对应 graph_node_id2idx.txt 中的 idx
            f.write(f"{idx} {vec_str}\n")
    print(f"[Info] Saved node2vec embeddings to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build UTGraph and train node2vec embeddings for a single city"
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        choices=["porto", "xian", "beijing", "chengdu"],
        help="城市名称",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="该城市的数据目录（包含 train.pkl / valid.pkl / test_output.pkl）",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=64,
        help="node2vec 嵌入维度，需与 Diff-RNTraj 的 pre_trained_dim 一致（默认64）",
    )
    parser.add_argument(
        "--use_valid_in_graph",
        action="store_true",
        help="是否在构图时加入 valid.pkl（默认只用 train）",
    )
    parser.add_argument(
        "--use_all_splits",
        action="store_true",
        help="是否在构图时加入 train+valid+test（不推荐，仅供实验）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.use_all_splits and args.use_valid_in_graph:
        raise ValueError("不能同时指定 --use_valid_in_graph 和 --use_all_splits。")

    print(f"[Info] Args: {args}")

    use_valid = args.use_valid_in_graph or args.use_all_splits
    use_all = args.use_all_splits

    trajs = load_trajs_for_graph(args.data_dir, use_valid=use_valid, use_all=use_all)
    eid_list, edges_count = build_utgraph_from_trajs(trajs)

    # 输出目录：Diff-RNTraj-master/data/{city}/graph/
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Diff-RNTraj-master/
    graph_dir = os.path.join(script_dir, "data", args.city, "graph")

    edge_index, num_nodes = save_graph_and_edge_index(eid_list, edges_count, graph_dir)
    train_and_save_node2vec(edge_index, num_nodes, args.embed_dim, graph_dir, args.city)

    print("[Info] Done.")


if __name__ == "__main__":
    main()



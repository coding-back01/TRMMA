import os.path
from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()

    parser.add_argument('--workspace', type=str, default="data/porto_large")
    parser.add_argument('--scale', type=str, default="100")
    parser.add_argument('--map_dir', type=str, default="data/porto/map")
    parser.add_argument('-left', type=int, default=5)
    parser.add_argument('-right', type=int, default=300)

    parser.add_argument('-utc', type=int, default=8)
    parser.add_argument("-neg", action="store_true", default=False)

    parser.add_argument('--ori_dir', type=str, default='data/porto')
    parser.add_argument('-low_cate', type=int, default=0)

    args = parser.parse_args()
    if args.neg:
        args.utc = 0 - args.utc
    args.edges_shp = os.path.join(args.map_dir, "edges.shp")
    args.train_file = os.path.join(args.ori_dir, "traj_train.csv")
    args.map_out_dir = os.path.join(args.workspace, "roadnet")
    if not os.path.exists(args.map_out_dir):
        os.makedirs(args.map_out_dir)

    return args

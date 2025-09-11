import os.path
from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()  # 创建参数解析器

    parser.add_argument('--workspace', type=str, default="data/porto")  # 工作空间目录
    parser.add_argument('--scale', type=str, default="100")  # 缩放比例
    parser.add_argument('--map_dir', type=str, default="data/porto/map")  # 地图目录
    parser.add_argument('-left', type=int, default=5)  # 左边界参数
    parser.add_argument('-right', type=int, default=300)  # 右边界参数

    parser.add_argument('-utc', type=int, default=8)  # 时区参数
    parser.add_argument("-neg", action="store_true", default=False)  # 是否取负时区

    parser.add_argument('--ori_dir', type=str, default='data/porto')  # 原始数据目录
    parser.add_argument('-low_cate', type=int, default=0)  # 低类别参数

    args = parser.parse_args()  # 解析命令行参数
    if args.neg:  # 如果设置了-neg参数
        args.utc = 0 - args.utc  # 将时区取负
    args.edges_shp = os.path.join(args.map_dir, "edges.shp")  # 边的shp文件路径
    args.train_file = os.path.join(args.ori_dir, "traj_train.csv")  # 训练文件路径
    args.map_out_dir = os.path.join(args.workspace, "roadnet")  # 输出地图目录
    if not os.path.exists(args.map_out_dir):  # 如果输出目录不存在
        os.makedirs(args.map_out_dir)  # 创建输出目录

    return args

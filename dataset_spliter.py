import argparse
import csv
import math
import os
from ast import literal_eval
import random
import numpy as np
import pandas as pd
import geopandas as gpd


def gen_path(filenames, zone, left=5, right=300, len_ratio=0.20):
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("[{}, {}), {}".format(left, right, len_ratio))
    data_rows = []
    for file in filenames:
        print("=====> {}".format(file))
        reader = csv.reader(open(file, "r"))
        cnt = 0
        cnt_filter = 0
        for item in reader:
            path = literal_eval(item[3])
            cnt += 1
            if float(item[8]) > len_ratio or len(path) < left or len(path) >= right:
                continue

            raw = literal_eval(item[4])
            a, b, _, c, d, *_ = zip(*raw)
            min_lon, max_lon, min_lat, max_lat = np.min(a + c), np.max(a + c), np.min(b + d), np.max(b + d)
            if min_lon <= zone[0] or max_lon >= zone[1] or min_lat <= zone[2] or max_lat >= zone[3]:
                continue

            idxs = []
            length = len(raw)
            for keep_ratio in ratios:
                low_sample = [0] + sorted(random.sample(range(1, length - 1), int((length - 2) * keep_ratio))) + [length - 1]
                idxs.append(low_sample)
            data_rows.append(item[:5] + [idxs])
            cnt_filter += 1
        print("# Input / Filtered Trajectories: {}, {}".format(cnt, cnt_filter))
    return data_rows


def get_zone(map_dir, dist):
    edges = gpd.read_file(os.path.join(map_dir, "edges.shp"))
    zone = [180, -180, 90, -90]
    for i in range(edges.shape[0]):
        tmp = edges.iloc[i]
        points = tmp['geometry'].coords
        zone[0] = min(zone[0], np.min(points.xy[0]))
        zone[1] = max(zone[1], np.max(points.xy[0]))
        zone[2] = min(zone[2], np.min(points.xy[1]))
        zone[3] = max(zone[3], np.max(points.xy[1]))
    print(zone)
    zone_s = [zone[0] + dist,
              zone[1] - dist,
              zone[2] + dist,
              zone[3] - dist]
    print(zone_s)
    return zone_s


def get_stats(filenames):
    data_rows = []
    cnt = 0
    for file in filenames:
        print("=====> {}".format(file))
        reader = csv.reader(open(file, "r"))
        for item in reader:
            cnt += 1
            raw = literal_eval(item[4])
            data_rows.append([item[6], item[7], item[8], item[9], item[10], item[11], item[5], len(raw)])
    print(cnt, len(data_rows))
    return data_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/chengdu_NMLR")
    parser.add_argument('--workspace', type=str, default="../data/len_test")
    parser.add_argument('--map_dir', type=str, default="sf")
    parser.add_argument('-ratio', type=str, default="0.6,0.2,0.2")
    parser.add_argument('-group_num', type=int, default=5)
    parser.add_argument('-quality', type=float, default=0.20)
    parser.add_argument('-mode', type=int, default=1)
    parser.add_argument('-left', type=int, default=5)
    parser.add_argument('-right', type=int, default=300)
    parser.add_argument('-small', type=int, default=-1)
    parser.add_argument('-time', action="store_true", default=False)
    parser.add_argument('-dist', type=float, default=0.0005)
    args = parser.parse_args()
    print(args)

    files = []
    if args.mode in [0, 1]:
        for item in os.listdir(args.data_dir):
            path = os.path.join(args.data_dir, item)
            if os.path.isfile(path):
                files.append(path)
    print("files number: {}".format(len(files)))

    if args.mode == 0:
        stats = get_stats(files)
        with open(os.path.join(args.workspace, "matching_ratio.csv"), 'w') as fp:
            fields_output_file = csv.writer(fp, delimiter=',')
            fields_output_file.writerows(stats)
    elif args.mode == 1:
        zone = get_zone(args.map_dir, args.dist)
        paths = gen_path(files, zone, len_ratio=args.quality, left=args.left, right=args.right)
        print("==> Number of Trajectories: {}".format(len(paths)))

        if args.small > 0:
            paths = random.sample(paths, args.small)

        # shuffle data
        ratio = args.ratio.split(",")
        ratio = [float(item) for item in ratio]
        print("Input ratio: {}".format(ratio))
        ratio = [r / sum(ratio) for r in ratio]
        print("Normalized ratio: {}".format(ratio))

        data_size = len(paths)
        print("==> Trajectories data size: {}".format(data_size))
        if args.time:
            time_order = []
            for i, item in enumerate(paths):
                tmp = literal_eval(item[2])
                time_order.append([i, tmp[0][1]])
            time_order.sort(key=lambda elem: elem[1])
            index, _ = zip(*time_order)
            index = list(index)
        else:
            index = [i for i in range(data_size)]
            random.shuffle(index)
        paths = np.array(paths, dtype=object)
        train = paths[index[0: int(data_size * ratio[0])]].tolist()
        valid = paths[index[int(data_size * ratio[0]): int(data_size * (ratio[0] + ratio[1]))]].tolist()
        test = paths[index[int(data_size * (ratio[0] + ratio[1])): data_size]].tolist()

        num = args.group_num
        scale = math.ceil(len(train) * 1.0 / num)
        label = int(100 / num)
        print("# Groups: {}, Scale: {}".format(num, scale))
        for i in range(num):
            with open(os.path.join(args.workspace, "traj_train_{}".format(label * (i+1))), 'w') as fp:
                fields_output_file = csv.writer(fp, delimiter=',')
                fields_output_file.writerows(train[:scale * (i+1)])

        with open(os.path.join(args.workspace, "traj_valid"), 'w') as fp:
            fields_output_file = csv.writer(fp, delimiter=',')
            fields_output_file.writerows(valid)
        with open(os.path.join(args.workspace, "traj_test"), 'w') as fp:
            fields_output_file = csv.writer(fp, delimiter=',')
            fields_output_file.writerows(test)
        print("Training: {}, Validation: {}, Test: {}".format(len(train), len(valid), len(test)))
    elif args.mode == 2:
        train = pd.read_csv(os.path.join(args.workspace, 'traj_train_20'), sep=",", header=None, names=['oid', 'tid', 'offsets', 'path', 'raw', 'low']).to_numpy()
        num = train.shape[0] // 20
        scale = 1
        groups = [1, 3, 5, 10]
        for item in groups:
            with open(os.path.join(args.workspace, "traj_train_{}".format(scale * item)), 'w') as fp:
                fields_output_file = csv.writer(fp, delimiter=',')
                fields_output_file.writerows(train[:num * item])

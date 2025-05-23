import argparse
import csv
import json
import os
import pickle
from ast import literal_eval

import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt
import geopandas as gpd
from haversine import haversine

from utils.candidate_point import CandidatePoint
from utils.trajectory_func import Trajectory, STPoint


rt_dict = {
    "motorway": 7,
    "motorway_link": 7,
    "trunk": 6,
    "trunk_link": 6,
    "primary": 5,
    "primary_link": 5,
    "secondary": 4,
    "secondary_link": 4,
    "tertiary": 3,
    "tertiary_link": 3,
    "unclassified": 2,
    "residential": 1,
    "living_street": 1,
}


def get_road_type(rt_str):
    avg = "secondary"

    if "[" in rt_str:
        rts = literal_eval(rt_str)
        new_rts = []
        for item in rts:
            if item in rt_dict:
                new_rts.append(item)
        if len(new_rts) == 0:
            new_rts.append(avg)
        codes = []
        for item in new_rts:
            codes.append(rt_dict[item])
        ans_code = np.max(codes)
        ans_desc = new_rts[np.argmax(codes)]
    else:
        if rt_str not in rt_dict:
            rt_str = avg
        ans_code = rt_dict[rt_str]
        ans_desc = rt_str
    return ans_desc, int(ans_code)


def gen_map(map_dir, out_dir):
    nodes = gpd.read_file(os.path.join(map_dir, "nodes.shp"))
    index = [i for i in range(nodes.shape[0])]
    nodes["fid"] = np.array(index, dtype=int)
    data = []
    nid_dict = {}
    for i in tqdm(range(nodes.shape[0]), desc="node num"):
        tmp = nodes.iloc[i]
        osmid = int(tmp['osmid'])
        fid = int(tmp['fid'])
        x = float(tmp['x'])
        y = float(tmp['y'])

        nid_dict[osmid] = fid
        data.append([fid, y, x])
    with open(os.path.join(out_dir, "nodeOSM.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(data)

    edges = gpd.read_file(os.path.join(map_dir, "edges.shp"))
    zone = [180, -180, 90, -90]
    rn_dict = {}
    data = []
    wayType = []
    for i in tqdm(range(edges.shape[0]), desc='edge num'):
        tmp = edges.iloc[i]
        eid = int(tmp['fid'])
        u = int(tmp['u'])
        v = int(tmp['v'])
        points = tmp['geometry'].coords
        zone[0] = min(zone[0], np.min(points.xy[0]))
        zone[1] = max(zone[1], np.max(points.xy[0]))
        zone[2] = min(zone[2], np.min(points.xy[1]))
        zone[3] = max(zone[3], np.max(points.xy[1]))

        desc, code = get_road_type(tmp['highway'])
        wayType.append([eid, desc, code])

        row = [eid, nid_dict[u], nid_dict[v]]
        row.append(len(points))
        pts = []
        for lon, lat in points:
            row += [float(lat), float(lon)]
            pts.append([float(lat), float(lon)])
        data.append(row)

        tmp_dict = {"coords": pts, "length": float(tmp['length']), "level": code}
        rn_dict[eid] = tmp_dict
    print(zone)
    with open(os.path.join(out_dir, "rn_dict.json"), 'w') as fp:
        json.dump(rn_dict, fp)
    with open(os.path.join(out_dir, "edgeOSM.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(data)
    with open(os.path.join(out_dir, "wayTypeOSM.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(wayType)


def parse_trajs(ori_dir, mode, low_cate, tz, scale):
    if mode == 'train':
        ori_file = os.path.join(ori_dir, "traj_train_{}".format(scale))
    elif mode == 'valid':
        ori_file = os.path.join(ori_dir, "traj_valid")
    elif mode == 'test':
        ori_file = os.path.join(ori_dir, "traj_test")
    else:
        raise NotImplementedError

    mm = pd.read_csv(ori_file, sep=",", header=None, names=['oid', 'tid', 'offsets', 'path', 'raw', 'low']).to_numpy()

    num = len(mm)
    if mode == 'valid':
        num = min(10000, num)
    elif mode == 'test':
        num = min(50000, num)

    trajs = []
    for oid, tid, offsets, path, raw, low in tqdm(mm[:num], desc='traj num'):
        rec = literal_eval(raw)
        pt_list = []
        for attrs in rec:
            lng = float(attrs[0])
            lat = float(attrs[1])
            timestamp = int(attrs[2])
            lng_p = float(attrs[3])
            lat_p = float(attrs[4])
            rid = int(attrs[5])
            rate = float(attrs[6])
            offset = float(attrs[7])
            speed = float(attrs[8])
            idx = int(attrs[9])
            dist = haversine((lat, lng), (lat_p, lng_p), unit='m')

            candi_pt = CandidatePoint(lat_p, lng_p, rid, dist, offset, round(rate, 4))
            pt = STPoint(lat, lng, timestamp, {'candi_pt': candi_pt})
            pt.time_arr = dt.datetime.fromtimestamp(timestamp, tz)
            pt.speed = round(speed, 2)
            pt.cpath_idx = idx
            pt_list.append(pt)
        if len(pt_list) > 2:
            traj = Trajectory(pt_list)
            cpath = literal_eval(path)
            cpath, *_ = zip(*cpath)
            cpath = list(cpath)
            traj.cpath = cpath
            traj.low_idx = literal_eval(low)[low_cate]
            traj.tid = tid
            trajs.append(traj)
    return trajs


def get_input(trajs):
    inputs = []
    for traj in trajs:
        pt_list = np.array(traj.pt_list, dtype=object)
        pt_list = pt_list[traj.low_idx].tolist()
        tmp = Trajectory(pt_list)
        tmp.cpath = traj.cpath
        tmp.tid = traj.tid
        inputs.append(tmp)
    return inputs


def get_graph(rn, trajs):
    freq = {}
    for i in range(len(rn.nodes)):
        nbrs = list(rn.neighbors(i))
        for item in nbrs:
            freq[(i, item)] = 1

    for traj in tqdm(trajs, desc='traj num'):
        for pt1, pt2 in zip(traj.pt_list[:-1], traj.pt_list[1:]):
            key = (pt1.data['candi_pt'].eid, pt2.data['candi_pt'].eid)
            if key in freq:
                freq[key] += 1
            else:
                freq[key] = 1

    data = []
    for k, v in freq.items():
        data.append([k[0], k[1], v])
    return data


def get_model_data(ori_dir, output_dir, utc, low_cate, scale):
    tz = dt.timezone(dt.timedelta(hours=utc))

    train_trajs = parse_trajs(ori_dir, "train", low_cate, tz, scale)
    pickle.dump(train_trajs, open(os.path.join(output_dir, "train.pkl"), 'wb'))

    roadnet = pickle.load(open(os.path.join(output_dir, "road_graph_wtime"), "rb"))
    g = get_graph(roadnet, train_trajs)
    with open(os.path.join(output_dir, "graph.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=',')
        fields_output_file.writerow(['src', 'dst', 'weight'])
        fields_output_file.writerows(g)

    valid_trajs = parse_trajs(ori_dir, "valid", low_cate, tz, scale)
    pickle.dump(valid_trajs, open(os.path.join(output_dir, "valid.pkl"), 'wb'))

    test_out_trajs = parse_trajs(ori_dir, "test", low_cate, tz, scale)
    pickle.dump(test_out_trajs, open(os.path.join(output_dir, "test_output.pkl"), 'wb'))

    data = []
    for tid, traj in enumerate(test_out_trajs):
        pts = []
        for item in traj.pt_list:
            pts.append([item.lat, item.lng, item.time_arr])
        data.append([tid, tid, pts, traj.low_idx])
    pickle.dump(data, open(os.path.join(output_dir, "test_linear.pkl"), "wb"))

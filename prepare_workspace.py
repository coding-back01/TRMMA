from args import make_args
from preprocess import gen_dam, txt2npy, get_road_graph
from preprocess.reformat import get_model_data, gen_map

# 设置 PROJ 数据目录
import pyproj
pyproj.datadir.set_data_dir(r"D:\Anaconda3\envs\TRMMA\Library\share\proj")


if __name__ == '__main__':
    opt = make_args()
    print(opt)

    gen_dam(opt)
    txt2npy(opt.workspace)
    get_road_graph(opt.workspace)

    gen_map(opt.map_dir, opt.map_out_dir)
    get_model_data(opt.ori_dir, opt.workspace, opt.utc, opt.low_cate, opt.scale)

    print("Prepare Workspace Finished!")

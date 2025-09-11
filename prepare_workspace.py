from args import make_args
from preprocess import gen_dam, txt2npy, get_road_graph
from preprocess.reformat import get_model_data, gen_map

# 设置 PROJ 数据目录
import pyproj
pyproj.datadir.set_data_dir(r"D:\Anaconda3\envs\TRMMA\Library\share\proj")


if __name__ == '__main__':
    opt = make_args()  # 解析命令行参数，获取配置选项
    print(opt)  # 打印参数信息，便于调试
    # gen_dam: 生成道路相关的统计信息和特征文件，输出如下文件：
    #   - csm_all.txt：道路分段之间的连接强度稀疏矩阵，表示各分段之间的转移概率或联系强度。
    #   - seg_info.csv：每条道路分段的详细属性，包括长度、起止点坐标、方位角、历史流量、平均速度等。
    #   - traffic_num.txt：每条道路分段在不同时间段的交通流量统计，反映道路的时变拥堵情况。
    #   - weighted_edges.txt：道路分段之间的加权边，权重为轨迹中出现的次数，用于建图和交通分析。
    gen_dam(opt)

    # txt2npy: 将上述文本格式的特征文件转换为npy格式，输出如下文件：
    #   - segs_geo.npy：每条道路分段的起止点经纬度坐标，便于空间特征处理。
    #   - vehicle_num_*.npy：每条道路分段在各时间片的车辆数矩阵，便于时序建模。
    #   - traffic_popularity.npy：归一化后的道路分段时序流量热度，用于交通流建模和特征归一化。
    txt2npy(opt.workspace)

    # get_road_graph: 生成道路网络的图结构数据，输出如下文件：
    #   - road_graph_wtime：以pickle格式保存的道路网络图结构，包含节点、边、邻接关系等信息，供图神经网络等模型使用。
    get_road_graph(opt.workspace)

    # gen_map: 生成地图基础信息文件，输出如下文件：
    #   - edgeOSM.txt：每条道路分段的基础信息（如节点编号、坐标序列等），用于地图重建和可视化。
    #   - nodeOSM.txt：所有节点的经纬度坐标信息，辅助空间索引和地图展示。
    gen_map(opt.map_dir, opt.map_out_dir)

    # get_model_data: 生成模型训练所需的轨迹数据文件，输出如下文件：
    #   - train.pkl：训练集轨迹，pickle格式保存，包含轨迹对象列表。
    #   - valid.pkl：验证集轨迹，pickle格式。
    #   - test_output.pkl：测试集轨迹，pickle格式。
    #   - test_linear.pkl：测试集线性轨迹，pickle格式。
    #   - graph.txt：道路网络的边权重信息，csv文本格式。
    #   这些文件用于模型的训练、验证和测试，主文件为pkl格式，内容为轨迹对象及分段ID序列。
    get_model_data(opt.ori_dir, opt.workspace, opt.utc, opt.low_cate, opt.scale)

    print("Prepare Workspace Finished!")

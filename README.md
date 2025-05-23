# TRMMA: [Efficient Methods for Accurate Sparse Trajectory Recovery and Map Matching]() (ICDE 2025)

## Environment and Package Dependencies
The code can be successfully run under the following environments:
- Python: 3.8
- Pytorch version: 1.13 with CUDA 11
- OS: Ubuntu 20.04

The project will also need following package dependencies:
- numpy
- pandas
- networkx


## Format of the Data

### 1. Data Download
The raw data and preprocessed data are provided in <https://1drv.ms/f/c/a9118c0e680825ca/EmJzZIir2HZFmY8Bz7JDKBcBbgHjwPJsWuJRTtjIixCZug?e=cJx8gP>

Download the preprocessed data and unzip the downloaded .zip file. For each city (Porto, Xi'an, Beijing, Chengdu), there are two types of data:

### 2. GPS Trajectory Data
The format of GPS trajectory data like as follows.

```angular2html
umvohug,1212302983:umvohug,"[[46.06236105759681, 1212302983.0, -122.41045451308185, 37.78421809482506], [111.45038193275329, 1212303103.0, -122.40535959553759, 37.78182564913661]]","[[7681, 1212302969.861, 3.506, 41.769], [6203, 1212302991.839, 3.506, 20.185], [4234, 1212303037.317, 6.415, 41.292], [6038, 1212303042.26, 6.415, 4.706], [12053, 1212303085.628, 6.415, 24.2]]","[[-122.41046, 37.78426, 1212302983.0, -122.41045451308185, 37.78421809482506, 7681.0, 0.31456270825289934, 46.06236105759681, 3.505806062610371, 0], [-122.40844, 37.78429, 1212303043.0, -122.40843417984907, 37.78429718884578, 4234.0, 0.0752032495335526, 19.922017630683893, 6.415449529500113, 2], [-122.40538, 37.7818, 1212303103.0, -122.40535959553759, 37.78182564913661, 12053.0, 0.7178537369666242, 111.45038193275329, 6.415449529500113, 4]]"
...
```
Each line records five fields to represent a GPS trajectory, i.e., moving object id, trajectory id, offsets, segment sequence and Map-matched point sequence. For example,

```angular2html
offsets: (137.96932188674592, 1212038019, -122.39606, 37.792731) means 
         (the distance between the start point of source segment and the source location,
          departure time,
          the longitude of source location,
          the latitude of source location,)
          The destination location has similar information.

segment sequence: [6557, 1212037972.113, 2.943, 71.205] means 
                  [segment id,
                   the timestamp when trajectory enter this segment,
                   the average speed when the trajectory go through this segment,
                   the duration for the trajectory through this segment]

Map-matched point sequence: [-122.41046, 37.78426, 1212302983.0, -122.41045451308185, 37.78421809482506, 7681.0, 0.31456270825289934, 46.06236105759681, 3.505806062610371, 0] means
                   [raw longitude,
                    raw latitude,
                    timestamp,
                    Map-matched longitude,
                    Map-matched latitude,
                    Map-matched segment id,
                    Map-matched position ratio,
                    moving distance according to the entry point of Map-matched segment,
                    travel speed,
                    index of Map-matched segment in segment sequence]
```

### 3. OSM map data
The maps used in TRMMA is extracted from [OpenStreetMap](http://www.openstreetmap.org/export#). In the `map` folder, there are the following files:

1. `nodes.shp` : Contains OSM node information with unique `node id`.
2. `edges.shp` : Contains network connectivity information with unique `edge id`.


## Directory Structure
The directory structure is as follows.
```
codespace
    ├ args.py: the parameters for preparing a workspace
    ├ prepare_workspace.py: generate the labeled historical trajectories for model training
    ├ train_mma.py: MMA model training process
    ├ infer_mma.py: detect Map-matched points for sparse trajectories
    ├ train_trmma.py: TRMMA model training process
    ├ infer_trmma.py: recover sparse trajectories to the high-sampling trajectories
    ├ models
    |   ├ layers.py: DualFormer layer and Attention layer
    |   ├ mma.py: MMA model
    |   └ trmma.py: TRMMA model
    ├ preprocess
    |   ├ dam.py: DA route generation
    |   └ seg_info.py: generate attributes for road segments
    └ utils
        ├ map.py: Map implentation with R-tree for top-k query
        └ evaluation_utils.py: evaluation metrics for trajectory recovery task
        
workspace (e.g., /data)
    └ dataset_name1 (e.g., porto_data)
        ├ map
        |   ├ nodes.shp
        |   └ edges.shp
        ├ traj_train.csv
        ├ traj_valid.csv
        └ traj_test.csv
```

## Usage and command examples
- Run `prepare_workspace.py` to generate the labeled historical trajectories for model training. You can also modify `args.py` for the corresponding parameters when preparing your data workspace.
- Run `train_mma.py` to train a MMA model and run `infer_mma.py` to detect Map-matched points for sparse trajectories.
- Run `train_trmma.py` to train a TRMMA model and run `infer_trmma.py` to recover high-sampling trajectories.

To get the details of the parameters when using TRMMA, please refer to `args.py` where each field is described in detail in comments.

### 1. MMA model training and inference examples
To train MMA model on Porto dataset, run the following command:
```bash
python -u train_mma.py --city porto1 --keep_ratio 0.1 --attn_flag --hid_dim 64 --transformer_layers 2 --direction_flag --candi_size 10 --gpu_id 1 --epochs 50 --batch_size 256 --train_flag
```
- city: the city name, choices are porto, xian, beijing and chengdu.
- keep_ratio: control the level of trajectory sparsity, from 0.1 to 0.5.
- attn_flag: flag of using attention to compute point embedding in MMA.
- hid_dim: the dimension of point embedding and candidate segment embedding in MMA.
- transformer_layers: the number of transformer encoder stacked in MMA.
- direction_flag: flag of using direction information for candidate segment embedding.
- candi_size: the number of candidate segments for GPS point.
- gpu_id: cpu/cuda(0,1,2..), where cuda(0,1,...) means using GPU for training the model.
- epochs: the maximum number of learning epochs, the default value is 50.
- batch_size: the batch size for training the model, the default value is 256.
- train_flag: indicate whether to execute the training process for MMA.

To execute the inference process for MMA model, run the following command:
```bash
python -u infer_mma.py --city porto1 --keep_ratio 0.1 --attn_flag --hid_dim 64 --transformer_layers 2 --direction_flag --candi_size 10 --gpu_id 1 --batch_size 256 --test_flag --model_old_path {saved_model_path}
```
- test_flag: indicate whether to execute the inference process for MMA.
- model_old_path: the saved path for well-trained MMA model.

### 2. TRMMA model training and inference examples
To train TRMMA model on Porto dataset, run the following command:
```bash
python -u train_trmma.py --city porto1 --keep_ratio 0.1 --da_route_flag --gps_flag --hid_dim 64 --transformer_layers 4 --lambda1 10 --lambda2 5 --tf_ratio 1 --gpu_id 0 --epochs 50 --batch_size 256 --train_flag
```
- city: the city name, choices are porto, xian, beijing and chengdu.
- keep_ratio: control the level of trajectory sparsity, from 0.1 to 0.5.
- da_route_flag: flag of using route in DualFormer of TRMMA.
- gps_flag: flag of using GPS point feature (i.e., normalized longitude and latitude).
- hid_dim: the hidden dimension in TRMMA.
- transformer_layers: the number of DualFormer encoder stacked in TRMMA.
- lambda1: weight for multi task segment id.
- lambda2: weight for multi task position ratio.
- tf_ratio: teaching ratio in the decoding process of TRMMA.
- gpu_id: cpu/cuda(0,1,2..), where cuda(0,1,...) means using GPU for training the model.
- epochs: the maximum number of learning epochs, the default value is 50.
- batch_size: the batch size for training the model, the default value is 256.
- train_flag: indicate whether to execute the training process for TRMMA.

To recover the sparse trajectories to high-sampling trajectories, run the following command:
```bash
python -u infer_trmma.py --city porto1 --keep_ratio 0.1 --da_route_flag --gps_flag --hid_dim 64 --transformer_layers 4 --lambda1 10 --lambda2 5 --tf_ratio 1 --gpu_id 0 --batch_size 256 --test_flag --eid_cate gps2seg --planner da --inferred_seg_path {output_of_MMA} --model_old_path {saved_model_path}
```
- test_flag: indicate whether to execute the inference process for TRMMA.
- eid_cate: inferred segments by different strategies, choices are MMA as `gps2seg`, Nearest as `nn` and HMM as `mm`.
- planner: routing algorithms, choices are DA route as `da`, fastest route as `time` and shortest route as `length`.
- inferred_seg_path: the output of MMA model.
- model_old_path: the saved path for well-trained TRMMA model.

## Citations
If you use the code or data in this repository, citing our paper as the following will be really appropriate.
```
@inproceedings{tian2025trmma,
  author       = {Wei Tian and
                  Jieming Shi and
                  Man Lung Yiu},
  title        = {Efficient Methods for Accurate Sparse Trajectory Recovery and Map Matching},
  booktitle    = {{ICDE}},
  pages        = {1--13},
  year         = {2025}
}
```

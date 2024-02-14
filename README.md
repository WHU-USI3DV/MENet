# MENet 
This is the official code for the paper [《MENet: Map-enhanced 3D object detection in bird’s-eye view for LiDAR point clouds》](https://www.sciencedirect.com/science/article/pii/S1569843223001590).

[![demo_nus](imgs/demo_nus.gif)](https://youtu.be/YEzCHtUVj8M "Click to youtube")

## News
- **2024.02.14**: The official code of MENet is published.
- **2023.05.19**: [《MENet: Map-enhanced 3D object detection in bird’s-eye view for LiDAR point clouds》](https://www.sciencedirect.com/science/article/pii/S1569843223001590) is published on JAG.


## Catalogue
- [MENet](#menet)
  - [News](#news)
  - [Catalogue](#catalogue)
  - [Performance](#performance)
  - [Installation](#installation)
    - [Tested Environment](#tested-environment)
    - [Steps](#steps)
  - [Dataset Preparation](#dataset-preparation)
  - [Train](#train)
    - [Train on multiple GPUs](#train-on-multiple-gpus)
    - [Train on single GPU](#train-on-single-gpu)
  - [Evaluate](#evaluate)
    - [Evaluate on multiple GPUs](#evaluate-on-multiple-gpus)
    - [Evaluate on single GPU](#evaluate-on-single-gpu)
  - [Related Projects](#related-projects)


## Performance

Performance without CBGS:
|                      Method                       | Modality |  mAP  |  NDS  |                                                Weights                                                 |
| :-----------------------------------------------: | :------: | :---: | :---: | :---------------------------------------------------------------------------------------------------: |
|      [CenterPoint](./configs/centerpoint.py)      |    L     | 52.7  | 61.2  | [Google Drive](https://drive.google.com/file/d/13nqSh3hfjdYKecchp_0JutKJK_Y-0hKE/view?usp=share_link) |
|            [MENet](./configs/menet.py)            |   L+M    | 56.9  | 63.4  | [Google Drive](https://drive.google.com/file/d/10Fex_w3k8KASrJ0x3zNqZTVrZmU40wgF/view?usp=share_link) |
|    [SECOND](./configs/hv_second_secfpn_nus.py)    |    L     | 35.7  | 51.3  | [Google Drive](https://drive.google.com/file/d/1l0MDs8h88ymvDE023SXZRyOg4xe8igw4/view?usp=share_link) |
| [ME SECOND](./configs/me_hv_second_secfpn_nus.py) |   L+M    | 43.1  | 55.5  | [Google Drive](https://drive.google.com/file/d/1KpYIGs248xoMo6oYiUEtDSe4d0MH1UzO/view?usp=share_link) |

Performance with CBGS:
|                Method            | Modality |  mAP  |  NDS  |         Weights      |
| :------------------------------------: | :------: | :---: | :---: | :-----------: |
|      CenterPoint   |   L    | 56.2 | 64.7  | TODO |
|      MENet   |   L+M    | 56.7  | 65.5  | TODO |
|    SECOND  |    L     | 47.6  | 59.2  | TODO |
| ME SECOND |   L+M    | 50.9  | 61.4  | TODO |


## Installation

### Tested Environment
- pyTorch
- [mmcv-full==1.4.1](https://mmcv.readthedocs.io/en/v1.4.1/get_started/installation.html)
- [mmsegmentation==0.14.1](https://github.com/open-mmlab/mmsegmentation/blob/v0.14.1/docs/get_started.md#installation)
- [mmdet==2.14.0](https://github.com/open-mmlab/mmdetection/blob/v2.14.0/docs/get_started.md)
- [mmdet3d==0.17.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1)
- Pillow==8.4.0

### Steps
1. **Install the package listed above**. We recommend that you create a new conda environment. To install the *`mm`xxx* series packages, you can click the hyperlinks and follow the instructions in the official documentations.
2. **Install cuda extension**. 
```bash
python setup.py develop
```


## Dataset Preparation
You can download, organize and prepare the dataset according to the documentory of *mmdetection3d*([nuScenes](https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html#nuscenes) | [Lyft](https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html#lyft)).


## Train
### Train on multiple GPUs
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 4 GPUs for example
export GPU_NUM=4
export CONFIG_FILE="configs/menet.py" # the config of any model
export WORK_DIR="./work_dirs/menet" # the output directory
./tools/dist_train.sh ${CONFIG_FILE}  ${GPU_NUM} --work-dir ${WORK_DIR} --deterministic
```

### Train on single GPU
```bash
python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --deterministic
```

## Evaluate
### Evaluate on multiple GPUs
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 4 GPUs for example
export GPU_NUM=4
export EVAL_METRICS=bbox

export CONFIG_FILE=./configs/menet.py
export CHECKPOINT_FILE="path of the weight of the model"
export RESULT_FILE=./work_dirs/menet/results.pkl

./tools/dist_test.sh ${CONFIG_FILE}  ${CHECKPOINT_FILE}  ${GPU_NUM} --out ${RESULT_FILE} --eval ${EVAL_METRICS}
```
### Evaluate on single GPU
```bash
python tools/test.py ${CONFIG_FILE}  ${CHECKPOINT_FILE} --eval ${EVAL_METRICS}
```

## Related Projects
- [mmdetection](https://github.com/open-mmlab/mmdetection3d)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
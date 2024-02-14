import os
import argparse
from typing import List, Optional, Tuple

import cv2
import torch
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.core.bbox import LiDARInstance3DBoxes


from tqdm import tqdm
from torchpack import distributed as dist

import method

# OBJECT_PALETTE = {
#     "car": (255, 158, 0),
#     "truck": (255, 99, 71),
#     "construction_vehicle": (233, 150, 70),
#     "bus": (255, 69, 0),
#     "trailer": (255, 140, 0),
#     "barrier": (112, 128, 144),
#     "motorcycle": (255, 61, 99),
#     "bicycle": (220, 20, 60),
#     "pedestrian": (0, 0, 230),
#     "traffic_cone": (47, 79, 79),
# }

OBJECT_PALETTE = {
    "car": (106, 157, 83),
    "truck": (92, 98, 220),
    "construction_vehicle": (220, 180, 0),
    "bus": (220, 100, 0),
    "trailer": (220, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

OBJECTS_IDX = {cat: idx for idx, cat in enumerate(OBJECT_PALETTE.keys())}

MAP_PALETTE = {
    "drivable_area": (249, 249, 249),
    "ped_crossing": (236, 236, 236),
    "walkway": (236, 236, 236),
    "stop_line": (236, 236, 236),
    "carpark_area": (236, 236, 236), 
    "divider": (240, 240, 240),
}

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize detection result')
    parser.add_argument("config", help='test config file path')
    parser.add_argument("checkpoint", help='checkpoint file')
    parser.add_argument('--dynamic-only', action='store_true', 
                        help="only show dynamic objects if show")
    parser.add_argument('--save-res', action='store_true', help="save result in pkl files")
    parser.add_argument("--out-dir", type=str, default="work_dirs/viz")
    parser.add_argument("--bbox-score", type=float, default=0.5)
    parser.add_argument("--collect3d", action='store_true', help="whether use Collect3D")
    parser.add_argument("--combine", action='store_true', help="print point cloud and map in one image.")
    args = parser.parse_args()
    return args

def visualize_lidar(
        fpath: str,
        lidar: Optional[np.ndarray] = None,
        bboxes: Optional[LiDARInstance3DBoxes] = None,
        labels: Optional[np.ndarray] = None,
        box_color: Optional[Tuple[int, int, int]] = None,
        gt_bboxes: Optional[LiDARInstance3DBoxes] = None,
        gt_labels: Optional[np.ndarray] = None,
        gt_color: Optional[Tuple[int, int, int]] = None,
        classes: Optional[List[str]] = None,
        xlim: Tuple[float, float] = (-50, 50),
        ylim: Tuple[float, float] = (-50, 50),
        radius: float = 30,
        thickness: float = 25,
        facecolor = "white",
        point_color = "black",
        downsample_rate = 7
    ) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    num_pts = lidar.shape[0]
    lidar = lidar[:int(num_pts//downsample_rate), :]
    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c=point_color,
        )

    if gt_bboxes is not None and len(gt_bboxes) > 0 and gt_color is not None:
        coords = gt_bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[gt_labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=int(2*thickness),
                color=np.array(gt_color) / 255,
            )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(box_color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor=facecolor,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

def visualize_map(
        fpath: str,
        masks: np.ndarray,
        classes: List[str],
        background: Tuple[int, int, int] = (240, 240, 240),
    ) -> None:
    masks = masks.astype(np.bool)
    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    # canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            # canvas[masks[k], :] = MAP_PALETTE[name]
            canvas[masks[k]] = 255
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)

def visualize_lidar_and_map(
        fpath: str,
        lidar: Optional[np.ndarray],
        bboxes: Optional[LiDARInstance3DBoxes],
        labels: Optional[np.ndarray],
        map_masks: np.ndarray,
        classes: Optional[List[str]],
        map_classes: List[str],
        box_color: Optional[Tuple[int, int, int]] = None,
        gt_bboxes: Optional[LiDARInstance3DBoxes] = None,
        gt_labels: Optional[np.ndarray] = None,
        gt_color: Optional[Tuple[int, int, int]] = None,
        xbound: List[float] = [-51.2, 51.2, 0.2],
        ybound: List[float] = [-51.2, 51.2, 0.2],
        radius: float = 30,
        thickness: float = 25,
        # facecolor = "white",
        point_color = "black",
        background: Tuple[int, int, int] = (252, 252, 252),
        downsample_rate = 6
    ):

    fig = plt.figure(figsize=(100, 100))

    ax = plt.gca()
    ax.set_xlim(0, int((xbound[1] - xbound[0]) / xbound[2]))
    ax.set_ylim(0, int((ybound[1] - ybound[0]) / ybound[2]))
    ax.set_aspect(1)
    ax.set_axis_off()

    # draw map
    map_masks = map_masks.astype(np.bool)
    canvas = np.zeros((*map_masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background
    for k, name in enumerate(map_classes):
        if name in MAP_PALETTE:
            canvas[map_masks[k], :] = MAP_PALETTE[name]
            # canvas[map_masks[k]] = 255
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    # plt.imshow(canvas)
    plt.imshow(np.transpose(canvas, (1, 0, 2)))

    # 点要转换到像素坐标系
    num_pts = lidar.shape[0]
    lidar = lidar[:int(num_pts//downsample_rate), :]
    lidar[:, 0] = (lidar[:, 0] - xbound[0]) / xbound[2]
    lidar[:, 1] = (lidar[:, 1] - ybound[0]) / ybound[2]
    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c=point_color,
        )

    # box 要转换到像素坐标系
    if gt_bboxes is not None and len(gt_bboxes) > 0 and gt_color is not None:
        coords = gt_bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            coords[index, :, 0] = (coords[index, :, 0] - xbound[0]) / xbound[2]
            coords[index, :, 1] = (coords[index, :, 1] - ybound[0]) / ybound[2]
            name = classes[gt_labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=int(2*thickness),
                color=np.array(gt_color) / 255,
            )
    # box 要转换到像素坐标系
    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            coords[index, :, 0] = (coords[index, :, 0] - xbound[0]) / xbound[2]
            coords[index, :, 1] = (coords[index, :, 1] - ybound[0]) / ybound[2]
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(box_color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        # facecolor=facecolor,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

def save_res(path, data):
    mmcv.mkdir_or_exist(os.path.dirname(path))
    mmcv.dump(data, path)

if __name__ == '__main__':
    dist.init()
    torch.cuda.set_device(dist.local_rank())
    
    # parse arguments and config
    args = parse_args()
    cfg = Config.fromfile(args.config)

    category_of_interst = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer']
    # category_of_interst = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
    #                         'motorcycle', 'bicycle', 'pedestrian']
    # category_of_interst = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
                    # 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    # build the dataloader
    print("Building dataset...")
    dataset = build_dataset(cfg.data["viz"])
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build model and load weights
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )
    model.eval()

    for data in tqdm(data_loader):
        if not args.collect3d:
            metas = data["metas"].data[0][0]
            points = data["input_data"]["points"].data[0][0].numpy()
            map_mask = data["input_data"]["map_mask"].data[0][0].numpy()
            name = "{}-{}".format(metas["timestamp"], metas["token"])
        else:
            metas = data["img_metas"][0].data[0][0]
            points = data["points"][0].data[0][0].numpy()
            map_mask = metas["map_mask"].data.numpy()
            name = "{}-{}".format(metas["timestamp"], metas["token"])

        # ground truth
        gt_bboxes = metas["gt_bboxes_3d"].data
        gt_labels = metas["gt_labels_3d"].data.numpy()

        with torch.inference_mode():
            outputs = model(return_loss=False, **data)
        
        # predicted box
        if cfg.model.type == "CenterPoint":
            bboxes = outputs[0]["pts_bbox"]["boxes_3d"]
            scores = outputs[0]["pts_bbox"]["scores_3d"].numpy()
            labels = outputs[0]["pts_bbox"]["labels_3d"].numpy()
        else:
            bboxes = outputs[0]["boxes_3d"]
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

        # select certain catgories
        cat_indices = np.zeros(labels.shape).astype(np.bool)
        cat_indices_gt = np.zeros(gt_labels.shape).astype(np.bool)
        for cat in category_of_interst:
            cat_idx = OBJECTS_IDX[cat]
            cat_indices = np.logical_or(cat_indices, (labels == cat_idx))
            cat_indices_gt = np.logical_or(cat_indices_gt, (gt_labels == cat_idx))

        # pred boxes
        bboxes = bboxes[cat_indices]
        scores = scores[cat_indices]
        labels = labels[cat_indices]
        # gt boxes
        gt_bboxes = gt_bboxes[cat_indices_gt]
        gt_labels = gt_labels[cat_indices_gt]

        if args.bbox_score is not None:
            indices = scores >= args.bbox_score
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]

        if args.combine:
            visualize_lidar_and_map(
                os.path.join(args.out_dir, "result", f"{name}.png"),
                points,
                bboxes = bboxes,
                labels = labels,
                map_masks = map_mask,
                classes = cfg.class_names,
                map_classes = cfg.map_classes,
                box_color = None,
                gt_bboxes = None,
                gt_labels = None,
                gt_color = None,
                xbound = [-51.2, 51.2, 0.2],
                ybound = [-51.2, 51.2, 0.2],
                radius = 30,
                thickness = 25,
                point_color=np.array([146, 87, 53])/255,
            )
        else:
            visualize_lidar(
                os.path.join(args.out_dir, "points", f"{name}.png"),
                points,
                radius=100,
                thickness=25,
                bboxes=bboxes,
                labels=labels,
                box_color=[77, 153, 16], # pred: green
                xlim=[cfg.xbound[0], cfg.xbound[1]],
                ylim=[cfg.ybound[0], cfg.ybound[1]],
                classes=cfg.class_names,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_color=[55, 66, 215], # gt: blue
                point_color=np.array([120, 50, 5])/255,
            )

            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                map_mask,
                classes=cfg.map_classes,
            )

        if args.save_res:
            save_res(
                os.path.join(args.out_dir, "pkl", f"{name}.pkl"),
                dict(
                    bboxes=bboxes, scores=scores, labels=labels,
                    gt_bboxes=gt_bboxes, gt_labels=gt_labels, 
                    map_mask=map_mask, points=points)
            )

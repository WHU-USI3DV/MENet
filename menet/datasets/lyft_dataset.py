from os import path as osp

import numpy as np
import mmcv
from mmcv.utils import print_log
from mmdet3d.datasets import DATASETS, LyftDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from pyquaternion import Quaternion
from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from mmdet3d.core.evaluation.lyft_eval import load_lyft_gts, load_lyft_predictions, get_classwise_aps
from terminaltables import AsciiTable

def lyft_eval(lyft, data_root, res_path, eval_set, output_dir, class_names, logger=None):
    """Evaluation API for Lyft dataset.

    Args:
        lyft (:obj:`LyftDataset`): Lyft class in the sdk.
        data_root (str): Root of data for reading splits.
        res_path (str): Path of result json file recording detections.
        eval_set (str): Name of the split for evaluation.
        output_dir (str): Output directory for output json files.
        logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.

    Returns:
        dict[str, float]: The evaluation results.
    """
    # evaluate by lyft metrics
    gts = load_lyft_gts(lyft, data_root, eval_set, logger)
    predictions = load_lyft_predictions(res_path)

    print('Calculating mAP@0.5:0.95...')

    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    metrics = {}
    average_precisions = \
        get_classwise_aps(gts, predictions, class_names, iou_thresholds)
    APs_data = [['IOU', 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]]

    mAPs = np.mean(average_precisions, axis=0)
    mAPs_cate = np.mean(average_precisions, axis=1)
    final_mAP = np.mean(mAPs)

    metrics['average_precisions'] = average_precisions.tolist()
    metrics['mAPs'] = mAPs.tolist()
    metrics['Final mAP'] = float(final_mAP)
    metrics['class_names'] = class_names
    metrics['mAPs_cate'] = mAPs_cate.tolist()

    APs_data = [['class', 'mAP@0.5:0.95']]
    for i in range(len(class_names)):
        row = [class_names[i], round(mAPs_cate[i], 3)]
        APs_data.append(row)
    APs_data.append(['Overall', round(final_mAP, 3)])
    APs_table = AsciiTable(APs_data, title='mAPs@0.5:0.95')
    APs_table.inner_footing_row_border = True
    print_log(APs_table.table, logger=logger)

    res_path = osp.join(output_dir, 'lyft_metrics.json')
    mmcv.dump(metrics, res_path)
    return metrics


@DATASETS.register_module(name='LyftDataset', force=True)
class LyftDatasetCustom(LyftDataset):
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - sweeps (list[dict]): infos of sweeps
                - timestamp (float): sample timestamp
                - img_filename (str, optional): image filename
                - lidar2img (list[np.ndarray], optional): transformations
                    from lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]

        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        input_dict["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
    
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        gt_bboxes_3d = info['gt_boxes']
        gt_names_3d = info['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if 'gt_shape' in info:
            gt_shape = info['gt_shape']
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_shape], axis=-1)

        # the lyft box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d
        )
        return anns_results
    
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or ("gt_labels" in example.keys() and \
                    ~(example['gt_labels']['gt_labels_3d']._data != -1).any())):
            return None
        return example

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in Lyft protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """

        output_dir = osp.join(*osp.split(result_path)[:-1])
        lyft = Lyft(
            data_path=osp.join(self.data_root, self.version),
            json_path=osp.join(self.data_root, self.version, self.version),
            verbose=True)
        eval_set_map = {
            'v1.01-train': 'val',
        }
        metrics = lyft_eval(lyft, self.data_root, result_path,
                            eval_set_map[self.version], output_dir, 
                            self.CLASSES, logger)

        # record metrics
        detail = dict()
        metric_prefix = f'{result_name}_Lyft'

        for i, name in enumerate(metrics['class_names']):
            AP = float(metrics['mAPs_cate'][i])
            detail[f'{metric_prefix}/{name}_AP'] = AP

        detail[f'{metric_prefix}/mAP'] = metrics['Final mAP']
        return detail
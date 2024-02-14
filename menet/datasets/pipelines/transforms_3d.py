from typing import Any, Dict

import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from numpy import random

import mmcv
from ..builder import PIPELINES
from mmdet3d.datasets.pipelines import GlobalRotScaleTrans, RandomFlip3D, ObjectSample

@PIPELINES.register_module(name="ObjectSample", force=True)
class ObjectSampleCustom(ObjectSample):
    """
    Args: 
        - stop_epoch (int | None): The number of stop epoch. If None, will not stop. 
    """
    def __init__(self, db_sampler, stop_epoch=None, sample_2d=False):
        super().__init__(db_sampler, sample_2d=False)
        self.epoch = -1
        self.stop_epoch = stop_epoch
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, data):
        """Call function to sample ground truth objects to the data.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
            return data

        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]

        # change to float for blending operation
        points = data["points"]
        if self.sample_2d:
            img = data["img"]
            gt_bboxes_2d = data["gt_bboxes"]
            # Assume for now 3D & 2D bboxes are the same
            gt_info = {
                "gt_bboxes_3d": gt_bboxes_3d.tensor.numpy(), 
                "gt_labels_3d": gt_labels_3d,
                "gt_bboxes_2d": gt_bboxes_2d,
                "img": img,
            }
        else:
            gt_info = {
                "gt_bboxes_3d": gt_bboxes_3d.tensor.numpy(), 
                "gt_labels_3d": gt_labels_3d,
                "map_mask": data.get("map_mask", None),
            }
        sampled_dict = self.db_sampler.sample_all(gt_info)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict["gt_bboxes_3d"]
            sampled_points = sampled_dict["points"]
            sampled_gt_labels = sampled_dict["gt_labels_3d"]

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels], axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d])
            )

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict["gt_bboxes_2d"]
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]
                ).astype(np.float32)

                data["gt_bboxes"] = gt_bboxes_2d
                data["img"] = sampled_dict["img"]

        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d.astype(np.long)
        data["points"] = points

        return data

@PIPELINES.register_module()
class ImageAug3D:
    def __init__(
        self, 
        is_train,
        final_dim=[256, 704],
        resize_lim=(-0.06, 0.11), 
        bot_pct_lim=[0.0, 0.0], 
        rot_lim=(-5.4, 5.4),  # degree
        rand_flip=False,
        resize_test=0.04
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train
        self.resize_test = resize_test

    def sample_augmentation(self, results):
        H, W = results["ori_shape"]
        fH, fW = self.final_dim
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = float(fW) / float(W)
            resize += self.resize_test
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip, rotate
    ):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["img"]
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(new_img)
            transforms.append(transform.numpy())
        data["img"] = new_imgs
        # update the calibration matrices
        data["img_aug_matrix"] = np.array(transforms)
        return data

@PIPELINES.register_module(name="ImageNormalize", force=True)
class ImageNormalizeCustom:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["img"] = [self.compose(img) for img in data["img"]]
        data["img_norm_cfg"] = dict(mean=self.mean, std=self.std)
        return data

@PIPELINES.register_module(name="GlobalRotScaleTrans", force=True)
class GlobalRotScaleTransCustom:
    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False,
                 update_img2lidar=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height
        self.update_img2lidar = update_img2lidar

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor
        
    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # if no bbox in input_dict, only rotate points
        if "points" in input_dict.keys() and len(input_dict['bbox3d_fields']) == 0:
            rot_mat_T = input_dict['points'].rotate(noise_rotation)
            input_dict['pcd_rotation'] = rot_mat_T
            input_dict['pcd_rotation_angle'] = noise_rotation
            return

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                if "points" in input_dict.keys():
                    points, rot_mat_T = input_dict[key].rotate(
                        noise_rotation, input_dict['points'])
                    input_dict['points'] = points
                else:
                    pseudo_points = torch.zeros((1, 3), 
                            dtype=input_dict[key].tensor.dtype,
                            device=input_dict[key].tensor.device)
                    points, rot_mat_T = input_dict[key].rotate(
                        noise_rotation, pseudo_points)
            else:
                rot_mat_T = torch.eye(3,
                            dtype=input_dict[key].tensor.dtype,
                            device=input_dict[key].tensor.device)
                
            input_dict['pcd_rotation'] = rot_mat_T
            input_dict['pcd_rotation_angle'] = noise_rotation
    
    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        if "points" in input_dict.keys():
            points = input_dict['points']
            points.scale(scale)
            if self.shift_height:
                assert 'height' in points.attribute_dims.keys(), \
                    'setting shift_height=True but points have no height attribute'
                points.tensor[:, points.attribute_dims['height']] *= scale
            input_dict['points'] = points

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)
    
    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T
        input_dict['pcd_trans'] = trans_factor

        if "points" in input_dict.keys():
            input_dict['points'].translate(trans_factor)
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

    def _rot_scale_trans_map_mask(self, input_dict):
        map_mask = input_dict["map_mask"]
        _, H, W = map_mask.shape
        res_x = input_dict["xbound"][-1]
        res_y = input_dict["ybound"][-1]

        # get rotation matrix
        if "pcd_rotation" in input_dict.keys():
            rotation_matrix = input_dict["pcd_rotation"].numpy()
        else:
            rotation_matrix = np.identity(3, dtype=np.float32)

        # get scale matrix
        scaling_ratio = input_dict["pcd_scale_factor"]
        scaling_matrix = np.array(
            [[scaling_ratio, 0., 0.], [0., scaling_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)

        # get translation matrix
        trans_x = input_dict["pcd_trans"][0] / res_x
        trans_y = input_dict["pcd_trans"][1] / res_y
        translate_matrix = np.array([[1, 0., trans_y], [0., 1, trans_x], [0., 0., 1.]],
                                      dtype=np.float32)

        c_x = -W/2
        c_y = -H/2
        trans2center_matrix = np.array([[1, 0., c_y], [0., 1, c_x], [0., 0., 1.]],
                                      dtype=np.float32)

        # affine transformation
        warp_matrix = (
            np.linalg.inv(trans2center_matrix) @ translate_matrix @ 
            rotation_matrix @ scaling_matrix @ trans2center_matrix)

        map_mask_cv = np.transpose(map_mask, (1, 2, 0)).astype(np.float32)
        map_mask_warp = cv2.warpPerspective(
            map_mask_cv,
            warp_matrix,
            dsize=(W, H),
            borderValue=0).astype(np.uint8)
        input_dict["map_mask"] = np.transpose(map_mask_warp, (2, 0, 1))

    def _rot_scale_trans_vision_bev(self, input_dict):
        aug_transform = np.zeros((len(input_dict["img"]), 4, 4)).astype(np.float32)
        if self.update_img2lidar:
            aug_transform[:, :3, :3] = input_dict['pcd_rotation'].T * input_dict['pcd_scale_factor']
        else:
            aug_transform[:, :3, :3] = np.eye(3).astype(np.float32) * input_dict['pcd_scale_factor']
        aug_transform[:, :3, 3] = input_dict['pcd_trans'].reshape(1,3)
        aug_transform[:, -1, -1] = 1.0

        input_dict['camera2lidar'] = aug_transform @ input_dict['camera2lidar']

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        # update box & points
        self._rot_bbox_points(input_dict)
        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)
        self._trans_bbox_points(input_dict)

        # update map mask
        if "map_mask" in input_dict:
            self._rot_scale_trans_map_mask(input_dict)

        # update camera2lidar_matrix for vision transformer
        if self.update_img2lidar:
            self._rot_scale_trans_vision_bev(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict
    

@PIPELINES.register_module(name="RandomFlip3D", force=True)
class RandomFlip3DCustom(RandomFlip3D):
    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        super(RandomFlip3D, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        # flip 2D image and its annotations
        super(RandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        rotation = np.eye(3)
        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
            self.random_flip_map_mask(input_dict, 'horizontal') # flip map

        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
            self.random_flip_map_mask(input_dict, 'vertical') # flip map
        
        if "camera2lidar" in input_dict:
            self.update_transform(input_dict)
        return input_dict

    def random_flip_map_mask(self, input_dict, direction='horizontal'):
        assert direction in ['horizontal', 'vertical']
        if "map_mask" not in input_dict:
            return
        
        map_mask = input_dict["map_mask"]
        map_mask_cv = np.transpose(map_mask, (1, 2, 0)).astype(np.float32)
        if direction == 'horizontal':
            map_mask_flip = mmcv.imflip(map_mask_cv, "horizontal")
        else:
            map_mask_flip = mmcv.imflip(map_mask_cv, "vertical")
        map_mask_flip = np.transpose(map_mask_flip, (2, 0, 1))
        input_dict["map_mask"] = map_mask_flip

    def update_transform(self, input_dict):
        aug_transform = torch.eye(4).float()
        if input_dict['pcd_horizontal_flip']:
            aug_transform[1, 1] = -1
        if input_dict['pcd_vertical_flip']:
            aug_transform[0, 0] = -1
        aug_transform = aug_transform.view(1,4,4).numpy()
        input_dict['camera2lidar'] = aug_transform @ input_dict['camera2lidar']

@PIPELINES.register_module()
class GridMask:
    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results["img"]
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results.update(img=imgs)
        return results
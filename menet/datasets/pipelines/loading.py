import os.path as osp
from tkinter import N
from typing import Any, Dict, Tuple, Optional, Union
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from lyft_dataset_sdk.utils.map_mask import MapMask
from ..builder import PIPELINES
from PIL import Image
from pathlib import Path

@PIPELINES.register_module(name="LoadMultiViewImageFromFiles", force=True)
class LoadMultiViewImageFromFilesCustom:
    """Load multi channel images from a list of separate channel files.

    Expects results['image_paths'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):

        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        # modified for waymo
        images = []
        h, w = 0, 0
        for name in filename:
            images.append(Image.open(name))
        
        #TODO: consider image padding in waymo

        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = images
        # [900, 1600]
        results["img_shape"] = [images[0].height, images[0].width]
        results["ori_shape"] = results["img_shape"]
        # Set initial values for default meta_keys
        results["pad_shape"] = results["img_shape"]
        results["scale_factor"] = 1.0
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadMapMask:
    """Load BEV map mask

    Args:
        data_root (str): The path of the dateset root directory.
        xbound (Tuple(float, float, float)): xmin, xmax, map resolution in x direction
        ybound (Tuple(float, float, float)): ymin, ymax, map resolution in y direction
        classes (Tuple(str, ...)): Classes that the map contains.
        translate_noise (Union[int, float, list, None]): Translation noise(m).
        rotate_noise (Union[int, float, None]): Rotation noise(degree).
        drop_out (Union[float, None]): The probability of losing the map.
    """
    def __init__(
        self,
        data_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        dataset: str = "nuscenes",
        classes: Tuple[str, ...] = None,
        translate_noise: Union[int, float, list, None] = None,
        rotate_noise: Union[int ,float, None] = None,
        drop_out: Union[float, None] = None
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        self.canvas_h = int(patch_h / ybound[2])
        self.canvas_w = int(patch_w / xbound[2])
        self.xbound = xbound
        self.ybound = ybound
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (self.canvas_h, self.canvas_w)
        self.dataset = dataset
        
        if translate_noise is not None:
            if isinstance(translate_noise, float):
                self.translate_noise = [translate_noise, translate_noise]
            elif isinstance(translate_noise, list):
                self.translate_noise = translate_noise
            else:
                raise NotImplementedError("`translate_noise` must be float or list now.")
        else:
            self.translate_noise = None
        self.rotate_noise = rotate_noise # degree

        self.drop_out = drop_out

        if self.dataset.lower() == "nuscenes":
            self.classes = classes
            self.map_channels = len(self.classes)
            self.maps = {}
            for location in LOCATIONS:
                self.maps[location] = NuScenesMap(data_root, location)
            self.mappings = {}
            for name in self.classes:
                if name == "drivable_area*":
                    self.mappings[name] = ["road_segment", "lane"]
                elif name == "divider":
                    self.mappings[name] = ["road_divider", "lane_divider"]
                else:
                    self.mappings[name] = [name]
            self.layer_names = []
            for name in self.mappings:
                self.layer_names.extend(self.mappings[name])
            self.layer_names = list(set(self.layer_names))
        elif self.dataset.lower() == "lyft":
            self.map_channels = 3
            assert xbound[2] == ybound[2], \
                    "For Lyfy Dataset, resolution of dimension x and y must be same."
            map_path = osp.join(data_root, "v1.01-train/maps", "map_raster_palo_alto.png")
            self.map_mask = MapMask(Path(map_path), xbound[2])
            self.mask_raster = self.map_mask.mask()
        else:
            raise NotImplementedError("Only surpport NuScenes and Lyft dataset now.")
        # Lyft special

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["xbound"] = self.xbound
        data["ybound"] = self.ybound
        data["translate_std"] = self.translate_noise
        data["rotate_std"] = self.rotate_noise
        
        if self.drop_out is None or np.random.uniform() > self.drop_out:
            # compute map pose
            loc, angle, data = self._getMapPose(data)

            if self.dataset.lower() == "nuscenes":
                map_mask = self._getNusMapMask(loc, angle, data["location"])
            elif self.dataset.lower() == "lyft":
                map_mask = self._getLyftMapMask(loc, angle)
            else:
                raise NotImplementedError("Only surpport nuscenes and lyft dataset now.")
        else:
            map_mask = np.zeros((self.map_channels, *self.canvas_size), dtype=np.float32)
        data["map_mask"] = map_mask
        return data

    def _getMapPose(self, data):
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        lidar2global = ego2global @ lidar2ego
        
        map_pose = lidar2global[:2, 3]
        # add translate noise
        if self.translate_noise is not None:
            # map_pose[0] += np.random.normal(0, self.translate_noise[0])
            # map_pose[1] += np.random.normal(0, self.translate_noise[1])
            tran_noise0 =  np.random.normal(0, self.translate_noise[0])
            map_pose[0] += tran_noise0
            tran_noise1 =  np.random.normal(0, self.translate_noise[1])
            map_pose[1] += tran_noise1
            data["translate_noise"] = [tran_noise0, tran_noise1]
        loc = [map_pose[0], map_pose[1]]

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        angle = yaw / np.pi * 180

        # add rotate noise
        if self.rotate_noise is not None:
            # angle += np.random.normal(0, self.rotate_noise)
            rotate_noise = np.random.normal(0, self.rotate_noise)
            angle += rotate_noise
            data["rotate_noise"] = rotate_noise
            
        return loc, angle, data

    def _getNusMapMask(self, loc, angle, location):
        """Get map mask in nuscenes dataset
        
        Args:
            loc (list[float]): global location of the lidar. (x,y)
            angle (float): yaw angle (degree)
            location (str): The location of current sample.
        """
        map_mask = np.zeros((self.map_channels, *self.canvas_size), dtype=np.float32)
        patch_box = (loc[0], loc[1], self.patch_size[0], self.patch_size[1])
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=angle,
            layer_names=self.layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)
        masks = masks.astype(np.bool)

        for k, name in enumerate(self.classes):
            for layer_name in self.mappings[name]:
                index = self.layer_names.index(layer_name)
                map_mask[k, masks[index]] = 1
        return map_mask

    def _getLyftMapMask(self, loc, angle):
        """Get map mask in nuscenes dataset
        
        Args:
            loc (list[float]): global location of the lidar. (x,y)
            angle (float): yaw angle (degree)
        """
        def crop_image(image: np.array,
                       pixel_coords: list,
                       size: int) -> np.array:
            x, y = pixel_coords
            x_min = int(x - size//2)
            x_max = int(x + size//2)
            y_min = int(y - size//2)
            y_max = int(y + size//2)
            cropped_image = image[y_min:y_max, x_min:x_max]
            return cropped_image
        
        pixel_coords = self.map_mask.to_pixel_coords(loc[0], loc[1])
        cropped = crop_image(self.mask_raster, pixel_coords, int(self.canvas_size[0] * np.sqrt(2)))

        yaw_deg = -angle
        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
        center = [rotated_cropped.shape[1] / 2, rotated_cropped.shape[0] / 2]
        ego_centric_map = crop_image(rotated_cropped, center, self.canvas_size[0])[::-1]
        ego_centric_map = np.transpose(ego_centric_map, (2, 1, 0))
        ego_centric_map = ego_centric_map.astype(np.float32)/255
        return ego_centric_map

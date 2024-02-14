import torch
from ..builder import ENCODERS
from .base_encoder import BaseEncoder
from torch.nn import functional as F
from mmcv.ops import Voxelization
from mmcv.runner import force_fp32
from mmdet3d.models import builder

@ENCODERS.register_module()
class VoxelEncoder(BaseEncoder):
    def __init__(
        self,
        voxel_layer,
        voxel_encoder,
        middle_encoder,
        stream_name = "lidar",
        init_cfg=None
        ):
        super(VoxelEncoder, self).__init__(stream_name, init_cfg)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)

    def forward(self, data, metas=None):
        points = data["points"]

        if self.voxel_encoder.__class__.__name__ == "DynamicSimpleVFE":
            voxels, coors = self.voxelize_dv(points)
            voxel_features, feature_coors = self.voxel_encoder(voxels, coors)
            batch_size = coors[-1, 0].item() + 1
            x = self.middle_encoder(voxel_features, feature_coors, batch_size)
        else:
            voxels, num_points, coors = self.voxelize(points)
            voxel_features = self.voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0].item() + 1
            x = self.middle_encoder(voxel_features, coors, batch_size)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()
    @force_fp32()
    def voxelize_dv(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch
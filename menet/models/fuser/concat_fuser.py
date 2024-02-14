from ..builder import FUSERS
from mmcv.runner import BaseModule
import torch

@FUSERS.register_module()
class ConcatFuser(BaseModule):
    def __init__(self, init_cfg=None):
        super(ConcatFuser, self).__init__(init_cfg)

    def forward(self, features):
        """
            features (dict): Features output by encoders
                
                - "stream1": Torch.tensor, B,C,H,W
                - ...
        """
        feat_maps = list(features.values())
        
        # Check shape. B, H, W of each feature map should be same.
        shape_checker = True
        B0, _, H0, W0 = feat_maps[0].shape
        for fm in feat_maps:
            B, _, H, W = fm.shape
            shape_checker &= (B0 == B and H0 == H and W0 == W)
        assert shape_checker, "Shape of feature maps must be same."

        return torch.cat(feat_maps, dim=1)
import torch
from ..builder import DECODERS
from ..builder import build_backbone, build_neck
from mmcv.runner import BaseModule

@DECODERS.register_module()
class BaseDecoder(BaseModule):
    def __init__(
        self,
        backbone,
        neck = None,
        init_cfg=None
        ):
        super(BaseDecoder, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None
    
    @property
    def with_neck(self):
        return self.neck is not None

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Feature map output by Fuser. (B, C, H, W)
        """
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

@DECODERS.register_module()
class SECONDDecoder(BaseDecoder):
    def __init__(
        self,
        backbone,
        neck = None,
        init_cfg=None
        ):
        super(SECONDDecoder, self).__init__(backbone, neck, init_cfg)

@DECODERS.register_module()
class PointPillarsDecoder(BaseDecoder):
    def __init__(
        self,
        backbone,
        neck = None,
        init_cfg=None
        ):
        super(PointPillarsDecoder, self).__init__(backbone, neck, init_cfg=None)
from typing import Any, Dict, List, Optional
from mmdet3d.core import bbox3d2result
from .base import Base3DFusionModel
from ..builder import MODELS
from ..builder import build_encoder, build_fuser, build_decoder, build_head
from collections import Counter
import torch
from torch import nn
from mmcv.runner import ModuleList
from method.runner import ModuleDict

@MODELS.register_module()
class BEVFusion(Base3DFusionModel):
    """BEVFusion.

    Args:
        encoders (list): List of encoders' config.
        fuser (dict): Config of fuser.
        decoder (dict): Config of decoder.
        head (dict): Config of head.
        train_cfg (dict)
        test_cfg (dict)
    """

    def __init__(
        self,
        encoders: List[Dict],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        head: Dict[str, Any],
        train_cfg = None,
        test_cfg = None,
        init_cfg = None,
        ):
        # NOTE: Be careful about the init_cfg in sub module.
        if init_cfg is not None:
            for i in range(len(encoders)):
                encoders[i].init_cfg = None
            fuser.init_cfg = None
            decoder.init_cfg = None
            head.init_cfg = None
        super(BEVFusion, self).__init__(init_cfg)

        stream_names = Counter([ed.stream_name for ed in encoders])
        assert stream_names.most_common(1)[0][1] == 1, \
            "The names of encoders should be unique."

        self.encoders = {}
        for encoder in encoders:
            encoder_module = build_encoder(encoder)
            self.encoders[encoder_module.stream_name] = encoder_module
        self.encoders = ModuleDict(self.encoders)

        self.fuser = build_fuser(fuser)
        self.decoder = build_decoder(decoder)

        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)

    def extract_feature(self, input_data, metas):
        """
            input_data (dict): sensors' data
                
                - "imgs"
                - "map_mask"
        """
        features = {}
        # for encoder in self.encoders:
        for stream_name, encoder in self.encoders.items():
            features[stream_name] = encoder(input_data, metas)
        fuse = self.fuser(features)
        x = self.decoder(fuse)
        return x

    def forward_train(self, input_data, metas, gt_labels, gt_bboxes_ignore=None):
        # extract feature before head
        feat = self.extract_feature(input_data, metas)
        outs = self.head(feat)
        loss_bbox_head = self.head_loss(outs, metas, gt_labels, gt_bboxes_ignore)

        losses = dict()
        losses.update(loss_bbox_head)
        return losses

    def head_loss(self, outs, metas, gt_labels, gt_bboxes_ignore):
        """ 
            The loss() interfaces of the heads in the mmdetection3d are not unified.
            This is an adaptor.
        """
        if self.head.__class__.__name__ == "Anchor3DHead":
            loss_inputs = [*outs, gt_labels['gt_bboxes_3d'], gt_labels['gt_labels_3d'], metas]
        else:
            loss_inputs = [gt_labels['gt_bboxes_3d'], gt_labels['gt_labels_3d'], outs]
        loss_bbox_head = self.head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return loss_bbox_head

    def simple_test(self, input_data, metas, rescale=False):
        feat = self.extract_feature(input_data, metas)
        outs = self.head(feat)

        if self.head.__class__.__name__ == "Anchor3DHead":
            bbox_list = self.head.get_bboxes(*outs, metas, rescale=rescale)
        else:
            bbox_list = self.head.get_bboxes(outs, metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def forward_dummy(self, input_data=None, metas=None, rescale=False):
        from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
        metas=[dict(box_type_3d=LiDARInstance3DBoxes)]

        feat = self.extract_feature(input_data, metas)
        outs = self.head(feat)
        if self.head.__class__.__name__ == "Anchor3DHead":
            bbox_list = self.head.get_bboxes(*outs, metas, rescale=rescale)
        else:
            bbox_list = self.head.get_bboxes(outs, metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def forward_train_kd(self, input_data, metas, gt_labels, gt_bboxes_ignore=None, 
                        return_kd_only=False, mode="bev_response"):
        """Forward function for training when knowledge distillation.

        Returns:
            loss_dict (dict) [optional]: loss of original head loss
            kd_signal (dict):

                - "bev_response" (Torch.Tensor): (B, H, W)
        """
        # extract feature before head
        features = {}
        for stream_name, encoder in self.encoders.items():
            features[stream_name] = encoder(input_data, metas)
        fuse = self.fuser(features)

        # compute bev response by the fused bev feature
        if mode == "bev_response":
            bev_response = torch.mean(torch.abs(fuse), dim=1) # (B, H, W)
            kd_signal = {"bev_response": bev_response}
        elif mode == "feature_alignment":
            kd_signal = {
                "map_bev": features["map"],
                "fuse_bev": fuse,
            }
            if "multiview" in self.encoders.keys():
                kd_signal["multiview_bev"] = features["multiview"]
            if "lidar" in self.encoders.keys():
                kd_signal["lidar_bev"] = features["lidar"]

        if return_kd_only:
            return kd_signal
        else:
            feat = self.decoder(fuse)
            outs = self.head(feat)
            loss_bbox_head = self.head_loss(outs, metas, gt_labels, gt_bboxes_ignore)
            losses = dict()
            losses.update(loss_bbox_head)
            return losses, kd_signal
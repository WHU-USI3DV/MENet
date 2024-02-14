from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule, auto_fp16
from ..builder import NECKS

@NECKS.register_module()
class FPN_LSS(BaseModule):
    def __init__(self, in_channels, out_channels, scale_factor=4,
                 input_feature_index = (0,2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 sync_bn: bool = False,
                 init_cfg=None
                 ):
        super().__init__(init_cfg)
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor,
                      kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample , mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral=  lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral,
                          kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
        )

        if sync_bn:
            self.conv = nn.SyncBatchNorm.convert_sync_batchnorm(self.conv)
            if self.lateral:
                self.lateral = nn.SyncBatchNorm.convert_sync_batchnorm(self.lateral)
            if self.extra_upsample:
                self.up2 = nn.SyncBatchNorm.convert_sync_batchnorm(self.up2)

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        if self.extra_upsample:
            x = self.up2(x)
        return x

@NECKS.register_module()
class GeneralizedLSSFPN(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        start_level=0,
        end_level=-1,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN2d"),
        act_cfg=dict(type="ReLU"),
        upsample_cfg=dict(mode="bilinear", align_corners=True),
        init_cfg=None
    ) -> None:
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i]
                + (
                    in_channels[i + 1]
                    if i == self.backbone_end_level - 1
                    else out_channels
                ),
                out_channels,
                3,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        # upsample -> cat -> conv1x1 -> conv3x3
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)

from typing import List, Tuple

import torch
from torch import nn

import torch.utils.checkpoint as checkpoint
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock

from mmcv.runner import BaseModule
from ..builder import BACKBONES

@BACKBONES.register_module()
class ResNetForBEVDet(BaseModule):
    def __init__(self, 
                 numC_input, 
                 numC_middle=None,
                 num_layer=[2,2,2], 
                 num_channels=None, 
                 stride=[2,2,2],
                 backbone_output_ids=None, 
                 norm_cfg=dict(type='BN'),
                 with_cp=False, 
                 block_type='Basic',
                 init_cfg=None
                 ):
        super(ResNetForBEVDet, self).__init__(init_cfg)
        #build backbone
        # assert len(num_layer)>=3
        assert len(num_layer)==len(stride)
        if numC_middle is None:
            numC_middle = numC_input
        num_channels = [numC_middle*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[Bottleneck(curr_numC, num_channels[i]//4, stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([Bottleneck(curr_numC, curr_numC//4,
                                         norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[BasicBlock(curr_numC, num_channels[i], stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

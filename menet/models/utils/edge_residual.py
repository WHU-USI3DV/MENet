# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import ConvModule, DropPath, build_activation_layer
from mmcv.runner import BaseModule
from ..attentions.se_layer import SELayer
from ..builder import CONV_GROUP

class EdgeResidual(BaseModule):
    """Edge Residual Block.

    Args:
        in_channels (int): The input channels of this module.
        out_channels (int): The output channels of this module.
        mid_channels (int): The input channels of the second convolution.
        kernel_size (int): The kernel size of the first convolution.
            Defaults to 3.
        stride (int): The stride of the first convolution. Defaults to 1.
        se_cfg (dict, optional): Config dict for se layer. Defaults to None,
            which means no se layer.
        with_residual (bool): Use residual connection. Defaults to True.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 se_cfg=None,
                 with_residual=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 drop_path_rate=0.,
                 with_cp=False,
                 init_cfg=None):
        super(EdgeResidual, self).__init__(init_cfg=init_cfg)
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.with_se = se_cfg is not None
        self.with_residual = (
            stride == 1 and in_channels == out_channels and with_residual)

        if self.with_se:
            assert isinstance(se_cfg, dict)

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding="same",
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if self.with_se:
            self.se = SELayer(**se_cfg)

        # downsample in the seconda conv
        self.conv2 = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):

        def _inner_forward(x):
            out = x
            out = self.conv1(out)

            if self.with_se:
                out = self.se(out)

            out = self.conv2(out)
            if self.with_residual:
                return x + self.drop_path(out)
            else:
                return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.act(out)

        return out

@CONV_GROUP.register_module()
class EdgeResidualGroup(BaseModule):
    """Gather multi edge residual blocks as a group.
    """
    def __init__(
            self, 
            input_channels,
            out_channels=[32, 64],
            mid_channels=[16, 32],
            strides=[2, 2],
            kernel_size=3,
            dilation=1,
            with_se=False,
            with_residual=True,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            drop_path_rate=0.,
            with_cp=False,
            init_cfg=None,
            stream_name="map"):
        super(EdgeResidualGroup, self).__init__(init_cfg=init_cfg)

        num_layers = len(mid_channels)
        if with_se:
            se_cfgs = [ dict(in_channels=c) for c in mid_channels]
        else:
            se_cfgs = [None for _ in range(num_layers)]

        assert len(out_channels) == len(mid_channels) == len(strides)
        self.convs = nn.ModuleList()
        for i, (out, mid, stride, se_cfg) in enumerate(
                                        zip(out_channels, mid_channels, strides, se_cfgs)):
            self.convs.append(
                EdgeResidual(
                    input_channels,
                    out,
                    mid,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    se_cfg=se_cfg,
                    with_residual=with_residual,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    drop_path_rate=drop_path_rate,
                    with_cp=with_cp,
                    init_cfg=init_cfg))
            input_channels = out

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn.bricks import Linear, build_activation_layer
from ..builder import build_conv_group

class SpatialTransformer(BaseModule):
    """Spatial Transform. 
    
    NOTE: The output feature map has the same size with the input.

    Args:
        conv_group_cfg: Conv group in localization network.
    """
    def __init__(
            self,
            conv_group_cfg = dict(
                type="EdgeResidualGroup",
                input_channels=6,
                out_channels=[32],
                mid_channels=[16],
                strides=[1],
            ),
            theta_num = 6,
            trans_type = "affine2d",
            mlp_hiddens = [16],
            act_cfg = dict(type='ReLU'),
            init_cfg = None
            ):
        super(SpatialTransformer, self).__init__(init_cfg=init_cfg)
        assert trans_type == "affine2d", "ONLY surpport affine2d now."
        self.trans_type = trans_type
        self.theta_num = theta_num

        self.conv_group = build_conv_group(conv_group_cfg)
        self.act = build_activation_layer(act_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        mlp_inputs = [conv_group_cfg["out_channels"][-1]] + mlp_hiddens
        mlp_outputs = mlp_hiddens + [theta_num]
        mlps = []
        for i, (mlp_in, mlp_out) in enumerate(zip(mlp_inputs, mlp_outputs)):
            mlps.append(Linear(mlp_in, mlp_out))
            if not i == len(mlp_inputs) - 1:
                mlps.append(nn.BatchNorm1d(mlp_out))
                mlps.append(self.act)
        self.mlps = nn.Sequential(*mlps)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # localiztion net
        paras = self.conv_group(x)
        # print("x2.sum: ", torch.sum(x))
        # print("paras_conv: \n", torch.sum(paras))
        paras = self.global_avgpool(paras).view(batch_size, -1)
        # print("paras_gap: \n", torch.sum(paras))
        paras = self.mlps(paras)
        if self.theta_num == 6:
            theta = paras.view(batch_size, 2, 3)
        elif self.theta_num == 3:
            paras = paras.view(batch_size, 3) # (x, y ,theta)
            theta = torch.zeros((batch_size, 2, 3), dtype=paras.dtype, device=paras.device)
            c = torch.cos(paras[:, 2])
            s = torch.sin(paras[:, 2])
            theta[:, 0, 0] = c
            theta[:, 0, 1] = -s
            theta[:, 1, 0] = s
            theta[:, 1, 1] = c
            theta[:, 0, 2] = paras[:, 0]
            theta[:, 1, 2] = paras[:, 1]
        else:
            raise NotImplementedError("theta_num should be 6 or 3")

        # print("paras: ", paras)
        # print("theta: ", theta)
        grid = F.affine_grid(theta, x.size(), align_corners=False) # grid generator
        x = F.grid_sample(x, grid) # grid sampler
        return x,  paras

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn.bricks import ConvModule, build_norm_layer, build_activation_layer
from ..builder import ATTENTIONS

class Flatten(BaseModule):
    def forward(self, x):
        return x.view(x.size(0), -1)
        
class ChannelGate(BaseModule):
    def __init__(
                self, 
                gate_channel, 
                reduction_ratio=16, 
                num_layers=1,
                sync_bn=False):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() ) # (B,C,H,W) -> (B,C*H*W)
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        # enable synchronized batch normalization
        if sync_bn:
            self.gate_c = nn.SyncBatchNorm.convert_sync_batchnorm( self.gate_c )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor,(in_tensor.size(2),in_tensor.size(3)), 
                            stride=(in_tensor.size(2), in_tensor.size(3)))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(BaseModule):
    def __init__(
                self, 
                gate_channel, 
                reduction_ratio=16, 
                dilation_conv_num=2, 
                dilation_val=4,
                sync_bn=False
                ):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module('gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module('gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module('gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module('gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
        if sync_bn:
            self.gate_s = nn.SyncBatchNorm.convert_sync_batchnorm( self.gate_s )
    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)

class DepthwiseSeparableSpatialGate(BaseModule):
    def __init__(
                self,
                gate_channel, 
                reduction_ratio=16, 
                dilation_conv_num=2, 
                dilation_val=4,
                sync_bn=False):
        super(DepthwiseSeparableSpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        
        # interior-layer feature learning
        for i in range( dilation_conv_num ):
            self.gate_s.add_module('gate_s_conv_inter%d'%i, nn.Conv2d(gate_channel, gate_channel, \
                    kernel_size=3, padding=dilation_val, dilation=dilation_val, groups=gate_channel))
            self.gate_s.add_module('gate_s_bn_inter%d'%i, nn.BatchNorm2d(gate_channel))
            self.gate_s.add_module('gate_s_relu_inter%d'%i, nn.ReLU())
        
        # across-layers feature learning
        mid_channels = gate_channel//reduction_ratio
        self.gate_s.add_module('gate_s_conv_across', nn.Conv2d(gate_channel, \
                mid_channels, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_across', nn.BatchNorm2d(mid_channels))
        self.gate_s.add_module('gate_s_relu_across', nn.ReLU())

        # interior-layer feature learning of the layers fusion method
        self.gate_s.add_module('gate_s_conv_inter_reduce', nn.Conv2d(mid_channels, mid_channels, \
                kernel_size=3, padding=dilation_val, dilation=dilation_val, groups=mid_channels))
        self.gate_s.add_module('gate_s_bn_inter_reduce', nn.BatchNorm2d(mid_channels))
        self.gate_s.add_module('gate_s_relu_inter_reduce', nn.ReLU())

        # across-layers feature learning
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(mid_channels, 1, kernel_size=1) )

        if sync_bn:
            self.gate_s = nn.SyncBatchNorm.convert_sync_batchnorm( self.gate_s )

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)

@ATTENTIONS.register_module()
class BAM(BaseModule):
    def __init__(self, in_channels, depthwise_spatial=False, sync_bn=False):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(in_channels, sync_bn=sync_bn)
        if not depthwise_spatial:
            self.spatial_att = SpatialGate(in_channels, sync_bn=sync_bn)
        else:
            self.spatial_att = DepthwiseSeparableSpatialGate(in_channels, sync_bn=sync_bn)

    def forward(self,in_tensor):
        att = 1 + torch.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor

@ATTENTIONS.register_module()
class MultiplyBAM(BaseModule):
    def __init__(self, in_channels, sync_bn=False):
        super(MultiplyBAM, self).__init__()
        self.channel_att = ChannelGate(in_channels, sync_bn=sync_bn)
        self.spatial_att = SpatialGate(in_channels, sync_bn=sync_bn)
    def forward(self,in_tensor):
        att = torch.sigmoid(self.channel_att(in_tensor)) * torch.sigmoid(self.spatial_att(in_tensor))
        return in_tensor * att
import torch.nn as nn
from .weights_utils import fix_bn, load_module_from, freeze_module
from mmcv.cnn.utils.weight_init import INITIALIZERS, BaseInit, PretrainedInit
from mmdet3d.utils import get_root_logger

@INITIALIZERS.register_module(name="CustomBEVFusion")
class CustomBEVFusionInit(BaseInit):
    def __init__(
            self,
            load_multiview_encoder_from=None,
            load_lidar_encoder_from=None,
            load_map_encoder_from=None,
            load_fuser_from=None,
            load_decoder_from=None,
            load_head_from=None,
            if_freeze_multiview_encoders=False,
            if_freeze_lidar_encoders=False,
            if_freeze_map_encoders=False,
            if_freeze_fuser=False,
            if_freeze_decoder=False,
            if_freeze_head=False,
            if_load_partial_decoder=True,
            extra_prefix="",
            **kwargs):
        super().__init__(**kwargs)

        self.load_multiview_encoder_from = load_multiview_encoder_from
        self.load_lidar_encoder_from = load_lidar_encoder_from
        self.load_map_encoder_from = load_map_encoder_from
        self.load_fuser_from = load_fuser_from
        self.load_decoder_from = load_decoder_from
        self.load_head_from = load_head_from

        self.if_freeze_multiview_encoders = if_freeze_multiview_encoders
        self.if_freeze_lidar_encoders = if_freeze_lidar_encoders
        self.if_freeze_map_encoders = if_freeze_map_encoders
        self.if_freeze_fuser = if_freeze_fuser
        self.if_freeze_decoder = if_freeze_decoder
        self.if_freeze_head = if_freeze_head

        self.if_load_partial_decoder = if_load_partial_decoder
        self.extra_prefix = extra_prefix
        if not isinstance(self.extra_prefix, dict):
            self.extra_prefix = {}
            self.extra_prefix["multiview"] = extra_prefix
            self.extra_prefix["lidar"] = extra_prefix
            self.extra_prefix["map"] = extra_prefix
            self.extra_prefix["fuser"] = extra_prefix
            self.extra_prefix["decoder"] = extra_prefix
            self.extra_prefix["head"] = extra_prefix

    def __call__(self, module: nn.Module) -> None:
        logger = get_root_logger()
        # load weights
        if  self.load_multiview_encoder_from is not None:
            load_module_from(module, self.load_multiview_encoder_from, 
                            "encoders.multiview", self.extra_prefix["multiview"], logger)
        if  self.load_lidar_encoder_from is not None:
            load_module_from(module, self.load_lidar_encoder_from, 
                            "encoders.lidar", self.extra_prefix["lidar"], logger)
        if  self.load_map_encoder_from is not None:
            load_module_from(module, self.load_map_encoder_from, 
                            "encoders.map", self.extra_prefix["map"], logger)
        if  self.load_fuser_from is not None:
            load_module_from(module, self.load_fuser_from, 
                            "fuser", self.extra_prefix["fuser"], logger)
        if  self.load_decoder_from is not None:
            if self.if_load_partial_decoder:
                load_module_from(module, self.load_decoder_from, 
                            ["decoder.backbone.blocks.1", "decoder.neck.deblocks"], 
                            self.extra_prefix["decoder"], logger)
            else:
                load_module_from(module, self.load_decoder_from, 
                            "decoder", self.extra_prefix["decoder"], logger)
        if  self.load_head_from is not None:
            load_module_from(module, self.load_head_from, 
                        "head", self.extra_prefix["head"], logger)

        # freeze module
        if self.if_freeze_multiview_encoders:
            freeze_module(module.encoders["multiview"])
        if self.if_freeze_lidar_encoders:
            freeze_module(module.encoders["lidar"])
        if self.if_freeze_map_encoders:
            freeze_module(module.encoders["map"])
        if self.if_freeze_fuser:
            freeze_module(module.fuser)
        if self.if_freeze_decoder:
            freeze_module(module.decoder)
        if self.if_freeze_head:
            freeze_module(module.head)

@INITIALIZERS.register_module(name='Pretrained', force=True)
class CustomPretrainedInit(PretrainedInit):
    def __init__(
        self,
        if_freeze=False,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.if_freeze = if_freeze

    def __call__(self, module: nn.Module) -> None:
        super().__call__(module)

        # freeze module
        if self.if_freeze:
            freeze_module(module)

@INITIALIZERS.register_module(name='PretrainedHead')
class PretrainedHeadInit(BaseInit):
    def __init__(
        self,
        load_decoder_from,
        load_head_from,
        if_freeze_decoder=True,
        if_freeze_head=True,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.load_decoder_from = load_decoder_from
        self.load_head_from = load_head_from
        self.if_freeze_decoder = if_freeze_decoder
        self.if_freeze_head = if_freeze_head

    def __call__(self, module: nn.Module) -> None:
        logger = get_root_logger()
        load_module_from(module, self.load_head_from, "head", logger)

        # freeze module
        if self.if_freeze_head:
            freeze_module(module.head)

        if self.if_freeze_decoder:
            freeze_module(module.decoder)
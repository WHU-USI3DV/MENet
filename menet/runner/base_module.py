from typing import Optional
import torch.nn as nn
from mmcv.runner import BaseModule

class ModuleDict(BaseModule, nn.ModuleDict):
    """ModuleDict in openmmlab.

    Args:
        modules (dict, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module).
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 modules: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules)

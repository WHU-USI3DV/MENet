import torch
import torch.nn as nn
from collections import OrderedDict

# fix bn
def fix_bn(module):
    if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        module.track_running_stats = False

def freeze_module(module):
    for name, param in module.named_parameters():
        param.requires_grad = False
        module.apply(fix_bn)

def load_module_from(model, ckpt_path, prefix, extra_prefix=None, logger=None, map_location='cpu'):
    if isinstance(prefix, str):
        prefix = [prefix]
    
    if extra_prefix is None:
        extra_prefix = ""
    assert isinstance(extra_prefix, str)

    if logger is not None:
        logger.info(f'Load {prefix} from {ckpt_path}')
    checkpoint= torch.load(ckpt_path, map_location=map_location)
    ckpt = checkpoint['state_dict']
    target_ckpt = OrderedDict()
    for k, v in ckpt.items():
        for pre in prefix:
            if len(extra_prefix) == 0:
                src_pre = pre
            else:
                src_pre = extra_prefix + "." + pre
            if k.startswith(src_pre):
                target_v = v
                if len(extra_prefix) == 0:
                    target_k = k
                else:
                    target_k = k[len(extra_prefix)+1:]
                target_ckpt[target_k] = target_v
    model.load_state_dict(target_ckpt, strict=False)
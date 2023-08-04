from typing import Dict

import torch.nn as nn

from mycv.utils import TORCH_VERSION, build_from_cfg, digit_version
from .registry import ACTIVATION_LAYERS

def build_activation_layer(cfg: Dict) -> nn.Module:
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return build_from_cfg(cfg, ACTIVATION_LAYERS)
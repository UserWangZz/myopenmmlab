import warnings
from mycv.utils import Registry, build_from_cfg

POSTPROCESSOR = Registry('postprocessor')

from mycv.cnn import MODELS as MYCV_MODELS

BACKBONES = Registry('models', parent=MYCV_MODELS)
DETECTORS = BACKBONES
HEADS = BACKBONES
NECKS = BACKBONES
LOSSES = BACKBONES


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_postprocessor(cfg):
    """Build postprocessor for scene text detector."""
    return build_from_cfg(cfg, POSTPROCESSOR)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detectors."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

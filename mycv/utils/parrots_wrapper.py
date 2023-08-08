import torch

TORCH_VERSION = torch.__version__


def get_build_config():
    if TORCH_VERSION == 'parrots':
        from parrots.config import get_build_info
        return get_build_info()
    else:
        return torch.__config__.show()


def _get_norm():
    if TORCH_VERSION == 'parrots':
        from parrots.nn.modules.batchnorm import _BatchNorm, _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm2d
    else:
        from torch.nn.modules.batchnorm import _BatchNorm
        from torch.nn.modules.instancenorm import _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_


_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()

def is_rocm_pytorch() -> bool:
    is_rocm = False
    if TORCH_VERSION != 'parrots':
        try:
            from torch.utils.cpp_extension import ROCM_HOME
            is_rocm = True if ((torch.version.hip is not None) and
                               (ROCM_HOME is not None)) else False
        except ImportError:
            pass
    return is_rocm


def _get_cuda_home():
    if TORCH_VERSION == 'parrots':
        from parrots.utils.build_extension import CUDA_HOME
    else:
        if is_rocm_pytorch():
            from torch.utils.cpp_extension import ROCM_HOME
            CUDA_HOME = ROCM_HOME
        else:
            from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME

class SyncBatchNorm(SyncBatchNorm_):  # type: ignore

    def _check_input_dim(self, input):
        if TORCH_VERSION == 'parrots':
            if input.dim() < 2:
                raise ValueError(
                    f'expected at least 2D input (got {input.dim()}D input)')
        else:
            super()._check_input_dim(input)

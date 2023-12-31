import glob
import os

import torch

"""
用于获取编译器版本和编译的CUDA版本信息。
根据不同的PyTorch版本（普通版本或parrots版本），代码采用不同的方式来获取这些信息。
"""

if torch.__version__ == 'parrots':
    import parrots

    def get_compiler_version():
        return 'GCC ' + parrots.version.compiler

    def get_compiling_cuda_version():
        return parrots.version.cuda
else:
    from ..utils import ext_loader
    ext_module = ext_loader.load_ext(
        '_ext', ['get_compiler_version', 'get_compiling_cuda_version'])

    def get_compiler_version():
        return ext_module.get_compiler_version()

    def get_compiling_cuda_version():
        return ext_module.get_compiling_cuda_version()
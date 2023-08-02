import importlib
import os
import pkgutil
import warnings
from collections import namedtuple

import torch

"""
这个文件中的代码主要用于加载扩展模块，即加载一些由C或C++编写的底层函数，以加速一些计算密集型的操作，
例如非极大值抑制（NMS）、池化操作等。这些扩展模块可以通过Cython或其他方式编写，并在运行时动态地加载到Python中。

代码中的功能如下：

1. 首先检查是否使用的是 `parrots` 引擎，即判断是否使用了 `parrots` 这个特殊版本的PyTorch。
如果不是，则使用普通的方式加载扩展模块。

2. 如果使用了 `parrots` 引擎，则从 `parrots` 中加载扩展模块。如果在加载过程中遇到未注册的元素
（即缺少某个扩展模块），则在加载失败的情况下返回一个虚拟的假函数，并发出警告。

3. `load_ext` 函数负责加载指定扩展模块，并检查扩展模块中是否包含指定的函数。在加载过程中，
会将加载成功的函数按照给定的函数名构成一个命名元组 `ExtModule` 并返回。

4. `check_ops_exist` 函数用于检查是否存在扩展模块。它通过在 `mmcv._ext` 模块中查找扩展模块来
判断是否加载了扩展模块。如果存在扩展模块，则返回 `True`，否则返回 `False`。

总的来说，这些代码用于根据环境情况加载底层扩展模块，以提高计算性能。如果使用的是普通的PyTorch，
会通过 `importlib` 来加载扩展模块；如果使用的是 `parrots` 版本的PyTorch，会通过 `parrots` 来加载扩展模块。
"""

if torch.__version__ != 'parrots':

    def load_ext(name, funcs):
        ext = importlib.import_module('mmcv.' + name)
        for fun in funcs:
            assert hasattr(ext, fun), f'{fun} miss in module {name}'
        return ext
else:
    from parrots import extension
    from parrots.base import ParrotsException

    has_return_value_ops = [
        'nms',
        'softnms',
        'nms_match',
        'nms_rotated',
        'top_pool_forward',
        'top_pool_backward',
        'bottom_pool_forward',
        'bottom_pool_backward',
        'left_pool_forward',
        'left_pool_backward',
        'right_pool_forward',
        'right_pool_backward',
        'fused_bias_leakyrelu',
        'upfirdn2d',
        'ms_deform_attn_forward',
        'pixel_group',
        'contour_expand',
        'diff_iou_rotated_sort_vertices_forward',
    ]

    def get_fake_func(name, e):

        def fake_func(*args, **kwargs):
            warnings.warn(f'{name} is not supported in parrots now')
            raise e

        return fake_func

    def load_ext(name, funcs):
        ExtModule = namedtuple('ExtModule', funcs)
        ext_list = []
        lib_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        for fun in funcs:
            try:
                ext_fun = extension.load(fun, name, lib_dir=lib_root)
            except ParrotsException as e:
                if 'No element registered' not in e.message:
                    warnings.warn(e.message)
                ext_fun = get_fake_func(fun, e)
                ext_list.append(ext_fun)
            else:
                if fun in has_return_value_ops:
                    ext_list.append(ext_fun.op)
                else:
                    ext_list.append(ext_fun.op_)
        return ExtModule(*ext_list)


def check_ops_exist() -> bool:
    ext_loader = pkgutil.find_loader('mycv._ext')
    return ext_loader is not None
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import torch
import mycv

from mycv.utils import Config, DictAction, mkdir_or_exist
from myocr.myocr.utils import setup_multi_processes, get_root_logger, collect_env


def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description='Train a detector.')
    parser.add_argument('config', help='Train config file path.')
    parser.add_argument('--work-dir', help='The dir to save logs and models.')
    parser.add_argument(
        '--load-from', help='The checkpoint file to load from.')
    parser.add_argument(
        '--resume-from', help='The checkpoint file to resume from.')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Whether not to evaluate the checkpoint during training.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file (deprecate), '
             'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be of the form of either '
             'key="[a,b]" or key=a,b .The argument also allows nested list/tuple '
             'values, e.g. key="[(a,b),(c,d)]". Note that the quotation marks '
             'are necessary and that no white space is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Options for job launcher.')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args(arg_list)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def run_train_cmd(args):
    # 读取config文件，并对Config文件进行解析
    cfg = Config.fromfile(args.config)
    # 命令行中cfg_options不为空，则对config进行合并
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # setup_multi_processes(cfg)            # 暂时注释掉，看看对单卡训练有没有影响

    """
    用来提升卷积神经网络的训练速度和效率。它的作用是开启或关闭cudnn的自动调优功能，
    根据输入和硬件条件自动寻找最适合的卷积算法，从而提高训练速度。
    """
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True


    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:       # CLI(命令行参数)中work_dir不为空则对工作目录进行更新
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:         # 配置文件中没有指定则使用默认路径进行添加
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # 同上
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # create work_dir
    mycv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # 将配置文件写入工作目录
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # 初始化一个元字典，用于记录一些重要的信息，例如将被log的环境信息和随机种子
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)


def main():
    args = parse_args()
    run_train_cmd(args)
    print("over")


if __name__ == '__main__':
    main()

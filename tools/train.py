import argparse
import os
import os.path as osp
import sys
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend
from plugin.utils.legacy_config import adapt_legacy_runner_cfg
from plugin.core.hooks.save_latest_hook import SaveLatestCheckpointHook
from plugin.core.hooks.sync_iter_info_hook import SyncIterInfoHook


def _import_plugins(cfg: Config, config_path: str) -> None:
    """Import custom plugin packages declared in the config."""
    if not getattr(cfg, 'plugin', False):
        return

    import importlib
    import sys

    sys.path.append(os.path.abspath('.'))

    plugin_dirs: List[str]
    if hasattr(cfg, 'plugin_dir'):
        plugin_dirs = cfg.plugin_dir
        if not isinstance(plugin_dirs, list):
            plugin_dirs = [plugin_dirs]
    else:
        base_dir = os.path.dirname(config_path)
        parts = [p for p in base_dir.split('/') if p]
        if not parts:
            return
        module_path = parts[0]
        for part in parts[1:]:
            module_path = f'{module_path}.{part}'
        importlib.import_module(module_path)
        return

    for plugin_dir in plugin_dirs:
        module_dir = os.path.dirname(plugin_dir)
        parts = [p for p in module_dir.split('/') if p]
        if not parts:
            continue
        module_path = parts[0]
        for part in parts[1:]:
            module_path = f'{module_path}.{part}'
        importlib.import_module(module_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MapTracker with MMEngine')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='directory to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic mixed precision training')
    parser.add_argument(
        '--sync_bn',
        choices=['none', 'torch', 'mmcv'],
        default='none',
        help='convert BatchNorm layers to SyncBN or MMSyncBN')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='automatically scale learning rate by GPU count')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='resume training from the latest or a specified checkpoint')
    parser.add_argument(
        '--ceph',
        action='store_true',
        help='use ceph as the data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the config, e.g. key=value')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='disable validation hook')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    _import_plugins(cfg, args.config)

    adapt_legacy_runner_cfg(cfg)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.sync_bn != 'none':
        cfg.sync_bn = args.sync_bn

    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in config.',
                logger='current')
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '--amp requires optim_wrapper.type == "OptimWrapper", '
                f'got {optim_wrapper}')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
           'enable' in cfg.auto_scale_lr and \
           'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError(
                'Cannot enable auto_scale_lr because required fields are missing.')

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.register_hook(SyncIterInfoHook(), priority='VERY_HIGH')
    runner.register_hook(SaveLatestCheckpointHook(), priority='VERY_LOW')

    skip_legacy_eval = getattr(cfg, 'skip_legacy_eval', False)
    single_gpu_eval = getattr(cfg, 'single_gpu_eval', False)
    if (not args.no_validate) and (not skip_legacy_eval) \
            and hasattr(cfg, 'legacy_eval_hook_cfg'):
        from plugin.core.hooks.legacy_val_hook import LegacyValHook
        runner.register_hook(
            LegacyValHook(single_gpu=single_gpu_eval,
                          **cfg.legacy_eval_hook_cfg), priority='NORMAL')

    runner.train()


if __name__ == '__main__':
    main()

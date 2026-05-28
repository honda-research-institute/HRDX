import os.path as osp
from typing import Any, Dict

import torch
import torch.distributed as dist
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from mmdet3d.registry import DATASETS
from plugin.core.apis.test import custom_multi_gpu_test
from plugin.datasets.builder import build_dataloader


@HOOKS.register_module()
class LegacyValHook(Hook):
    """MMEngine hook that reuses the legacy evaluation pipeline."""

    priority = 'LOW'

    def __init__(self,
                 dataset_cfg: Dict[str, Any],
                 data_cfg: Dict[str, Any],
                 eval_cfg: Dict[str, Any],
                 interval: int,
                 single_gpu: bool = False) -> None:
        self.dataset_cfg = dataset_cfg
        self.data_cfg = data_cfg
        self.eval_cfg = eval_cfg
        self.interval = max(1, interval)
        self.single_gpu = single_gpu
        self._initialized = False

    def before_train(self, runner) -> None:
        self._lazy_init(runner)

    def _lazy_init(self, runner) -> None:
        if self._initialized:
            return

        if self.single_gpu and runner.rank != 0:
            self._initialized = True
            return

        self.dataset = DATASETS.build(self.dataset_cfg)
        if getattr(self.dataset, 'work_dir', None) in (None, ''):
            self.dataset.work_dir = osp.join(runner.work_dir, 'val_results')
        dist = runner.world_size > 1 and not self.single_gpu
        self.dataloader = build_dataloader(
            self.dataset,
            samples_per_gpu=self.data_cfg.get('samples_per_gpu', 1),
            workers_per_gpu=self.data_cfg.get('workers_per_gpu', 0),
            num_gpus=runner.world_size if dist else 1,
            dist=dist,
            shuffle=False,
            shuffler_sampler=self.data_cfg.get('shuffler_sampler', None),
            nonshuffler_sampler=self.data_cfg.get('nonshuffler_sampler', None),
            runner_type=dict(type='IterBasedRunner'))
        self._initialized = True

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        if (runner.iter + 1) % self.interval == 0:
            self._run_eval(runner)

    def after_train(self, runner) -> None:
        # Ensure the final checkpoint is evaluated if it was not covered.
        if (runner.iter + 1) % self.interval != 0:
            self._run_eval(runner)

    def _run_eval(self, runner) -> None:
        self._lazy_init(runner)

        if self.single_gpu and runner.rank != 0:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return

        was_training = runner.model.training
        runner.model.eval()
        tmpdir = osp.join(runner.work_dir, '.legacy_eval')
        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=False)
        if was_training:
            runner.model.train()

        if self.single_gpu and runner.world_size > 1:
            dist.barrier()

        # Only rank 0 logs metrics.
        if runner.rank != 0:
            return

        eval_results = self.dataset.evaluate(results, logger=runner.logger,
                                             **self.eval_cfg)
        if not isinstance(eval_results, dict):
            raise TypeError('`dataset.evaluate` must return a dict of metrics.')

        for key, value in eval_results.items():
            runner.logger.info(f'val/{key}: {value}')
            if getattr(runner, 'visualizer', None):
                runner.visualizer.add_scalar(
                    f'val/{key}', value, runner.iter + 1)

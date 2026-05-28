"""Helpers to adapt legacy MMDet-style configs to MMEngine runners."""
from copy import deepcopy
from typing import Any, Dict

from mmengine.config import Config, ConfigDict


def _to_config_dict(cfg: Any) -> Any:
    if isinstance(cfg, ConfigDict):
        return cfg
    if isinstance(cfg, dict):
        return ConfigDict(cfg)
    return cfg


def adapt_legacy_runner_cfg(cfg: Config) -> None:
    """Mutate ``cfg`` in-place to fill MMEngine runner fields."""
    if hasattr(cfg, 'train_dataloader'):
        # Already converted.
        return

    data_cfg = cfg.get('data', None)
    assert data_cfg is not None, 'Legacy config must contain a `data` section.'

    model_cfg = cfg.get('model', None)
    if model_cfg is not None and 'data_preprocessor' not in model_cfg:
        model_cfg.setdefault('data_preprocessor', dict(type='MapDataPreprocessor'))

    samples_per_gpu = data_cfg.get('samples_per_gpu', 1)
    workers_per_gpu = data_cfg.get('workers_per_gpu', 0)
    persistent_workers = workers_per_gpu > 0

    train_dataset = _to_config_dict(deepcopy(data_cfg.train))
    val_dataset = _to_config_dict(deepcopy(data_cfg.val))
    test_dataset = _to_config_dict(deepcopy(data_cfg.test))

    cfg.train_dataloader = ConfigDict(
        batch_size=samples_per_gpu,
        num_workers=workers_per_gpu,
        persistent_workers=persistent_workers,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=train_dataset,
    )
    cfg.test_dataloader = None

    evaluation = cfg.get('evaluation', {})
    eval_interval = evaluation.get('interval', 1)

    runner_cfg = cfg.get('runner', {})
    max_iters = runner_cfg.get('max_iters', None)
    if max_iters is None:
        max_epochs = runner_cfg.get('max_epochs', None)
        if max_epochs is None:
            raise ValueError('Legacy config missing runner.max_iters/max_epochs.')
        max_iters = max_epochs

    cfg.train_cfg = ConfigDict(
        type='IterBasedTrainLoop',
        max_iters=max_iters,
        val_interval=eval_interval if eval_interval > 0 else 0,
    )
    cfg.val_cfg = None
    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.test_cfg = None
    cfg.test_evaluator = None

    optimizer_cfg = _to_config_dict(deepcopy(cfg.optimizer))
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    clip_grad = cfg.get('optimizer_config', {}).get('grad_clip', None)
    cfg.optim_wrapper = ConfigDict(
        type='OptimWrapper',
        optimizer=optimizer_cfg,
    )
    if clip_grad is not None:
        cfg.optim_wrapper.clip_grad = clip_grad
    if paramwise_cfg is not None:
        cfg.optim_wrapper.paramwise_cfg = paramwise_cfg

    lr_config: Dict[str, Any] = cfg.get('lr_config', {})
    warmup_iters = lr_config.get('warmup_iters', 0)
    warmup_ratio = lr_config.get('warmup_ratio', 1.0)
    min_lr_ratio = lr_config.get('min_lr_ratio', None)

    schedulers = []
    if warmup_iters > 0:
        schedulers.append(
            ConfigDict(
                type='LinearLR',
                start_factor=warmup_ratio,
                by_epoch=False,
                begin=0,
                end=warmup_iters,
            ))

    eta_min = None
    base_lr = optimizer_cfg.get('lr', None)
    if base_lr is not None and min_lr_ratio is not None:
        eta_min = base_lr * min_lr_ratio

    schedulers.append(
        ConfigDict(
            type='CosineAnnealingLR',
            by_epoch=False,
            begin=warmup_iters,
            end=max_iters,
            T_max=max_iters - warmup_iters if warmup_iters < max_iters else max_iters,
            eta_min=eta_min if eta_min is not None else 0,
        ))
    cfg.param_scheduler = schedulers

    log_config = cfg.get('log_config', {})
    log_interval = log_config.get('interval', 50)
    checkpoint_cfg = cfg.get('checkpoint_config', {})
    checkpoint_interval = checkpoint_cfg.get('interval', max_iters)

    cfg.default_hooks = ConfigDict(
        timer=ConfigDict(type='IterTimerHook'),
        logger=ConfigDict(type='LoggerHook', interval=log_interval),
        param_scheduler=ConfigDict(type='ParamSchedulerHook'),
        checkpoint=ConfigDict(
            type='CheckpointHook',
            interval=checkpoint_interval,
            by_epoch=False),
        sampler_seed=ConfigDict(type='DistSamplerSeedHook'),
    )

    cfg.log_processor = ConfigDict(by_epoch=False)

    legacy_eval_cfg = ConfigDict(
        dataset_cfg=val_dataset,
        data_cfg=ConfigDict(
            samples_per_gpu=data_cfg.get('val_samples_per_gpu', 1),
            workers_per_gpu=workers_per_gpu,
            shuffler_sampler=data_cfg.get('shuffler_sampler', None),
            nonshuffler_sampler=data_cfg.get('nonshuffler_sampler', None),
        ),
        eval_cfg=evaluation,
        interval=eval_interval if eval_interval > 0 else max_iters,
    )
    if 'interval' in legacy_eval_cfg.eval_cfg:
        legacy_eval_cfg.eval_cfg.pop('interval')
    cfg.legacy_eval_hook_cfg = legacy_eval_cfg

    # Remove legacy keys that are no longer used to avoid confusion.
    for key in ['optimizer', 'optimizer_config', 'lr_config', 'runner']:
        if key in cfg:
            cfg.pop(key)

"""Compatibility helpers to replace deprecated mmdet.core utilities."""
from functools import partial
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from mmdet.registry import TASK_UTILS


def multi_apply(func: Callable, *args, **kwargs):
    """Apply ``func`` to a list of arguments.

    Mirrors the legacy ``mmdet.core.multi_apply`` helper.
    """
    if kwargs:
        func = partial(func, **kwargs)
    map_results = map(func, *args)
    return tuple(map(list, zip(*map_results)))


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Distributed mean reduction with fallback for single GPU."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor)
    tensor /= dist.get_world_size()
    return tensor


def build_assigner(cfg: Any):
    """Build an assigner via the TASK_UTILS registry."""
    return TASK_UTILS.build(cfg)


def build_sampler(cfg: Any, context: Any = None):
    """Build a sampler via the TASK_UTILS registry."""
    if context is not None:
        return TASK_UTILS.build(cfg, default_args=dict(context=context))
    return TASK_UTILS.build(cfg)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable inverse sigmoid."""
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class AssignResult:
    """Minimal stand-in for :class:`mmdet.models.task_modules.AssignResult`."""

    def __init__(self,
                 num_gts: int,
                 gt_inds: torch.Tensor,
                 extra: Any = None,
                 labels: Optional[torch.Tensor] = None) -> None:
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.extra = extra
        self.labels = labels

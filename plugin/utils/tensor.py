"""Utility helpers for converting data to torch tensors."""

from typing import Any

import numpy as np
import torch


def to_tensor(data: Any) -> torch.Tensor:
    """Convert various python types to :class:`torch.Tensor`.

    Mirrors the behavior of the legacy ``mmdet.datasets.pipelines.to_tensor``
    helper for the limited cases used in MapTracker pipelines.
    """

    if torch.is_tensor(data):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, (list, tuple)):
        return torch.tensor(data)
    if isinstance(data, (int, float)):
        return torch.tensor([data], dtype=torch.float32)
    raise TypeError(f'Cannot convert type {type(data)} to torch.Tensor')

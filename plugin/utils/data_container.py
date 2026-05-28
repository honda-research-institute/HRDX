# Copyright (c) OpenMMLab. All rights reserved.
"""Lightweight copies of mmcv's legacy DataContainer utilities.

The original MapTracker codebase relies on ``mmcv.parallel.DataContainer`` and
its custom collate logic. mmcv>=2.0 removed this module when migrating to
MMEngine. To avoid rewriting the entire data pipeline, we vendor the minimal
implementation required by the project.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Callable, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate


def _assert_tensor_type(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not isinstance(self.data, torch.Tensor):
            raise AttributeError(
                f'{self.__class__.__name__} has no attribute {func.__name__} '
                f'for type {self.datatype}')
        return func(self, *args, **kwargs)

    return wrapper


class DataContainer:
    """Container that carries stacking hints for dataloader collate."""

    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray, list, tuple, dict],
        stack: bool = False,
        padding_value: int = 0,
        cpu_only: bool = False,
        pad_dims: int | None = 2,
    ) -> None:
        self._data = data
        self._stack = stack
        self._padding_value = padding_value
        self._cpu_only = cpu_only
        if pad_dims is not None:
            assert pad_dims in (1, 2, 3)
        self._pad_dims = pad_dims

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.data)})'

    def __len__(self) -> int:
        return len(self._data)  # type: ignore[arg-type]

    @property
    def data(self):
        return self._data

    @property
    def cpu_only(self) -> bool:
        return self._cpu_only

    @property
    def stack(self) -> bool:
        return self._stack

    @property
    def padding_value(self) -> int:
        return self._padding_value

    @property
    def pad_dims(self) -> int | None:
        return self._pad_dims

    @property
    def datatype(self) -> Union[Type, str]:
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        return type(self.data)

    @_assert_tensor_type
    def size(self, *args, **kwargs) -> torch.Size:
        return self.data.size(*args, **kwargs)  # type: ignore[return-value]

    @_assert_tensor_type
    def dim(self) -> int:
        return self.data.dim()  # type: ignore[return-value]


def collate(batch: Sequence, samples_per_gpu: int = 1):
    """Extended default_collate that understands DataContainer instances."""

    if not isinstance(batch, Sequence):
        raise TypeError(f'{type(batch)} is not supported in collate.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        exemplar = batch[0]
        if exemplar.cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked,
                stack=exemplar.stack,
                padding_value=exemplar.padding_value,
                cpu_only=True)
        if exemplar.stack:
            for i in range(0, len(batch), samples_per_gpu):
                sample_data = batch[i].data
                assert isinstance(sample_data, torch.Tensor)
                if exemplar.pad_dims is not None:
                    ndim = exemplar.dim()
                    assert ndim > exemplar.pad_dims
                    max_shape = [0 for _ in range(exemplar.pad_dims)]
                    for dim in range(1, exemplar.pad_dims + 1):
                        max_shape[dim - 1] = exemplar.size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - exemplar.pad_dims):
                            assert exemplar.size(dim) == sample.size(dim)
                        for dim in range(1, exemplar.pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(exemplar.pad_dims * 2)]
                        for dim in range(1, exemplar.pad_dims + 1):
                            pad[2 * dim - 1] = max_shape[dim - 1] - \
                                sample.size(-dim)
                        padded.append(
                            F.pad(
                                sample.data, pad,
                                value=exemplar.padding_value))
                    stacked.append(default_collate(padded))
                else:
                    stacked.append(
                        default_collate(
                            [sample.data for sample in batch[
                                i:i + samples_per_gpu]]))
        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(
            stacked,
            stack=exemplar.stack,
            padding_value=exemplar.padding_value)

    if isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]

    if isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }

    return default_collate(batch)

"""Compatibility shims for deprecated mmcv.fp16 utilities."""
from __future__ import annotations

import functools
from typing import Iterable, Optional


def auto_fp16(apply_to: Optional[Iterable[str]] = None, out_fp32: bool = False):
    """Return a no-op decorator keeping the original API surface."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def force_fp32(apply_to: Optional[Iterable[str]] = None):
    """Return a no-op decorator for parity with mmcv.runner.force_fp32."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator

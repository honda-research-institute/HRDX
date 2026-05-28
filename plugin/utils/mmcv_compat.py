"""Compatibility helpers for deprecated mmcv APIs."""


def jit(*args, **kwargs):
    """Fallback no-op decorator for legacy ``mmcv.jit`` usage."""

    def decorator(func):
        return func

    return decorator


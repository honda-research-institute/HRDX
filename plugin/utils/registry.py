"""Custom registries bridging legacy MMCV APIs to MMEngine registries."""
import copy
from typing import Optional, Sequence

from mmengine.config import ConfigDict
from mmengine.registry import Registry
from mmdet.registry import MODELS as MMDET_MODELS
from mmcv.cnn.bricks.transformer import MODELS as MMCV_TRANSFORMER_MODELS

TRANSFORMER = Registry('maptracker_transformer', parent=MMDET_MODELS, scope='plugin_transformer')
ATTENTION = Registry('maptracker_attention', parent=MMCV_TRANSFORMER_MODELS, scope='plugin_attention')
FEEDFORWARD_NETWORK = Registry('maptracker_ffn', parent=MMCV_TRANSFORMER_MODELS, scope='plugin_ffn')
TRANSFORMER_LAYER = Registry('maptracker_transformer_layer', parent=MMCV_TRANSFORMER_MODELS, scope='plugin_transformer_layer')
TRANSFORMER_LAYER_SEQUENCE = Registry('maptracker_transformer_layer_sequence', parent=MMCV_TRANSFORMER_MODELS, scope='plugin_transformer_layer_sequence')
POSITIONAL_ENCODING = Registry('maptracker_positional_encoding', parent=MMCV_TRANSFORMER_MODELS, scope='plugin_positional_encoding')

PLUGIN_SCOPED_REGISTRIES = (
    (TRANSFORMER, 'plugin_transformer'),
    (ATTENTION, 'plugin_attention'),
    (FEEDFORWARD_NETWORK, 'plugin_ffn'),
    (TRANSFORMER_LAYER, 'plugin_transformer_layer'),
    (TRANSFORMER_LAYER_SEQUENCE, 'plugin_transformer_layer_sequence'),
    (POSITIONAL_ENCODING, 'plugin_positional_encoding'),
)


def _auto_scope_modules(cfg):
    """Recursively attach plugin scopes based on registered module names."""
    if cfg is None:
        return None

    # Handle ConfigDict and plain dict uniformly
    if isinstance(cfg, dict):
        cfg = ConfigDict(cfg)
        module_type = cfg.get('type')
        if module_type and '_scope_' not in cfg:
            for registry, scope in PLUGIN_SCOPED_REGISTRIES:
                if module_type in registry.module_dict:
                    cfg['_scope_'] = scope
                    break
        for key, value in list(cfg.items()):
            cfg[key] = _auto_scope_modules(value)
        return cfg

    if isinstance(cfg, list):
        return [_auto_scope_modules(item) for item in cfg]

    if isinstance(cfg, tuple):
        return tuple(_auto_scope_modules(item) for item in cfg)

    return cfg


def _clone_cfg(cfg):
    """Deep-copy a config object keeping its original type."""
    if cfg is None:
        return None
    return copy.deepcopy(cfg)


def _ensure_scope(cfg, scope: str):
    """Attach a default scope to config dicts if missing."""
    if cfg is None or scope is None:
        return cfg
    if isinstance(cfg, dict):
        cfg = ConfigDict(cfg)
        cfg.setdefault('_scope_', scope)
    else:
        # For configs that are dataclasses / custom objects
        if not hasattr(cfg, '_scope_') or getattr(cfg, '_scope_') is None:
            setattr(cfg, '_scope_', scope)
    return cfg


def build_with_scopes(registry: Registry,
                      cfg,
                      scope_candidates: Optional[Sequence[str]] = None):
    """Attempt to build a module while trying several scope fallbacks.

    Args:
        registry (Registry): Registry used for building the module.
        cfg (dict | ConfigDict): Configuration of the module.
        scope_candidates (Sequence[str], optional): Candidate scopes that will
            be applied (in order) when the default build fails with ``KeyError``.

    Returns:
        nn.Module | Any: The instantiated module.

    Raises:
        KeyError: If the module cannot be found in any registry scope.
    """
    if cfg is None:
        return None

    cfg_base = _auto_scope_modules(_clone_cfg(cfg))
    try:
        return registry.build(cfg_base)
    except KeyError as initial_error:
        last_error = initial_error

    if not scope_candidates:
        raise last_error

    for scope in scope_candidates:
        if scope is None:
            continue
        scoped_cfg = _clone_cfg(cfg)
        scoped_cfg = _ensure_scope(scoped_cfg, scope)
        scoped_cfg = _auto_scope_modules(scoped_cfg)
        try:
            return registry.build(scoped_cfg)
        except KeyError as candidate_error:
            last_error = candidate_error
            continue
    raise last_error


def build_mmdet_module(cfg, extra_scopes: Optional[Sequence[str]] = None):
    """Build modules registered under MMDetection with scope fallbacks."""
    scopes = list(extra_scopes) if extra_scopes else []
    scopes.extend(['mmdet', 'mmcv'])
    return build_with_scopes(MMDET_MODELS, cfg, scopes)

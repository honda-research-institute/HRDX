import numpy as np
import torch

from mmengine.registry import VISBACKENDS
from mmengine.visualization import WandbVisBackend
from mmengine.visualization.vis_backend import force_init_env


@VISBACKENDS.register_module()
class StepAwareWandbVisBackend(WandbVisBackend):
    """Wandb backend that respects the provided logging step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_step = -1

    def _to_scalar(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.cpu().numpy()
        if isinstance(value, np.ndarray) and value.size == 1:
            return value.item()
        return value

    def _normalize_step(self, step: int) -> int:
        if step <= self._last_step:
            step = self._last_step + 1
        self._last_step = step
        return step

    def _standardize_name(self, name: str) -> str:
        if name.startswith('val/') or name.startswith('train/'):
            return name
        if '/' in name:
            return name
        # treat bare metrics as training scalars by default
        return f'train/{name}'

    @force_init_env
    def add_scalar(self, name, value, step=0, **kwargs):
        name = self._standardize_name(name)
        raw_step = step
        step = self._normalize_step(step)
        value = self._to_scalar(value)
        log_data = {name: value}
        if name.startswith('train/') and name != 'train/iter':
            log_data.setdefault('train/iter', raw_step)
        if name.startswith('val/') and name != 'val/iter':
            log_data.setdefault('val/iter', raw_step)
        log_data.setdefault('iter', raw_step)
        self._wandb.log(log_data, step=step, commit=self._commit)

    @force_init_env
    def add_scalars(self, scalar_dict, step=0, file_path=None, **kwargs):
        raw_step = step
        step = self._normalize_step(step)
        scalar_dict = {self._standardize_name(key): self._to_scalar(val)
                       for key, val in scalar_dict.items()}
        if any(key.startswith('train/') for key in scalar_dict):
            scalar_dict.setdefault('train/iter', raw_step)
        if any(key.startswith('val/') for key in scalar_dict):
            scalar_dict.setdefault('val/iter', raw_step)
        scalar_dict.setdefault('iter', raw_step)
        self._wandb.log(scalar_dict, step=step, commit=self._commit)

    @force_init_env
    def add_image(self, name, image, step=0, **kwargs):
        step = self._normalize_step(step)
        image = self._wandb.Image(image)
        self._wandb.log({name: image}, step=step, commit=self._commit)

from mmengine.model import BaseDataPreprocessor
from mmdet3d.registry import MODELS as MMDET3D_MODELS
import torch


@MMDET3D_MODELS.register_module()
class MapDataPreprocessor(BaseDataPreprocessor):
    """Minimal data preprocessor that moves tensors to the target device."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, data, training: bool = False):  # type: ignore[override]
        data = self.cast_data(data)
        return self._to_device(data)

    def _to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, dict):
            return {k: self._to_device(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._to_device(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._to_device(v) for v in data)
        return data

"""Lightweight Transformer shim matching the legacy mmdet API."""

from typing import Optional

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmengine.model import BaseModule


class Transformer(BaseModule):
    """Minimal transformer wrapper compatible with older mmdet code."""

    def __init__(self,
                 encoder: Optional[dict] = None,
                 decoder: Optional[dict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg)
        self.encoder = (build_transformer_layer_sequence(encoder)
                        if encoder is not None else None)
        self.decoder = (build_transformer_layer_sequence(decoder)
                        if decoder is not None else None)

    def init_weights(self) -> None:
        if self.encoder is not None:
            self.encoder.init_weights()
        if self.decoder is not None:
            self.decoder.init_weights()


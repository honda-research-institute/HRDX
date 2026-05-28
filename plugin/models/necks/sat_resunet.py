import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

try:
    from mmdet.registry import MODELS as MMDET_MODELS
except Exception:
    # Allow import without registry in non-MMDet contexts
    NECKS = None


def _conv_relu(in_channels: int, out_channels: int, kernel: int, padding: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class _UpBlock(nn.Module):
    """Upsample + skip merge helper used inside :class:`ResNetUNet`."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DownsampleCNN(nn.Module):
    """Anti-aliased downsampler for aerial features.

    Applies one stride-2 conv to reach 1/2 resolution, uses a dilated conv to
    expand receptive field, then performs low-pass filtered stride-2 reduction.
    This keeps more fine-grained markings (e.g., lane dividers) intact compared
    to purely strided convolutions.
    """

    def __init__(self, in_channels=64, hidden_dim=128):
        super().__init__()
        self.conv_stride2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conv_dilated = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
        )
        # 3x3 average blur prior to stride-2 pooling to suppress aliasing.
        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
        self.register_buffer('blur_kernel', kernel.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stride2(x)
        x = self.conv_dilated(x)
        # Apply blur per-channel before reducing to 1/4 resolution.
        pad = (1, 1, 1, 1)
        blur_kernel = self.blur_kernel.to(dtype=x.dtype, device=x.device)
        x = torch.nn.functional.pad(x, pad, mode='reflect')
        kernel = blur_kernel.expand(x.shape[1], 1, 3, 3)
        x = torch.nn.functional.conv2d(x, kernel, groups=x.shape[1])
        x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResNetUNet(nn.Module):
    """ResNet50-based UNet head mirroring the AID4AD implementation.

    Compared to the previous lightweight version, this variant keeps the full
    decoder with all skip connections and loads ImageNet-pretrained weights to
    stabilise aerial branch optimisation.
    """

    def __init__(self, outC: int = 64, pretrained: bool = True):
        super().__init__()

        # TorchVision 0.13+ recommends using ``weights``. Fall back gracefully
        # when the cached checkpoint is unavailable (e.g. offline CI).
        def _create_resnet50(load_pretrained: bool):
            """Compatibility wrapper for TorchVision versions before/after 0.13."""
            if hasattr(tv_models, 'ResNet50_Weights'):
                weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if load_pretrained else None
                return tv_models.resnet50(weights=weights)
            return tv_models.resnet50(pretrained=load_pretrained)

        try:
            base_model = _create_resnet50(pretrained)
        except Exception as exc:  # pragma: no cover - rare offline path
            warnings.warn(
                f'Failed to load pretrained ResNet50 weights ({exc}). Falling back to random init.',
                RuntimeWarning)
            base_model = _create_resnet50(False)

        # Encoder blocks
        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
        )
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1  # 256 channels
        self.layer2 = base_model.layer2  # 512 channels
        self.layer3 = base_model.layer3  # 1024 channels
        self.layer4 = base_model.layer4  # 2048 channels

        # Low-level refinements matching AID4AD public release
        self.conv_original_size0 = _conv_relu(3, 64, 3, 1)
        self.conv_original_size1 = _conv_relu(64, 64, 3, 1)
        self.conv_original_size2 = _conv_relu(64 + 64, 64, 3, 1)

        # Decoder blocks with skip connections
        self.up4 = _UpBlock(in_channels=2048, skip_channels=1024, out_channels=512)
        self.up3 = _UpBlock(in_channels=512, skip_channels=512, out_channels=256)
        self.up2 = _UpBlock(in_channels=256, skip_channels=256, out_channels=128)
        self.up1 = _UpBlock(in_channels=128, skip_channels=64, out_channels=64)

        self.conv_last = nn.Conv2d(64, outC, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve original resolution features for the final refinement.
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        x0 = self.stem(x)                        # (B, 64, H/2,   W/2)
        x1 = self.layer1(self.maxpool(x0))       # (B,256, H/4,   W/4)
        x2 = self.layer2(x1)                     # (B,512, H/8,   W/8)
        x3 = self.layer3(x2)                     # (B,1024, H/16,  W/16)
        x4 = self.layer4(x3)                     # (B,2048, H/32,  W/32)

        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)

        # Restore input resolution and mix with shallow features.
        x = F.interpolate(x, size=x_original.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        return self.conv_last(x)

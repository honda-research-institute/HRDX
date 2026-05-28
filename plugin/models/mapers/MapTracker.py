"""
    MapTracker main module, adapted from StreamMapNet
"""
import csv
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner import load_checkpoint

from mmdet3d.registry import MODELS as MMDET3D_MODELS

from .base_mapper import BaseMapper, MAPPERS
from ..utils.query_update import MotionMLP
from copy import deepcopy
from plugin.utils.mmdet_compat import multi_apply
from plugin.utils.data_container import DataContainer
from plugin.utils.registry import build_mmdet_module

from einops import rearrange, repeat
from scipy.spatial.transform import Rotation as R

from .vector_memory import VectorInstanceMemory


@MAPPERS.register_module()
class MapTracker(BaseMapper):

    def __init__(self,
                 bev_h,
                 bev_w,
                 roi_size,
                 backbone_cfg=dict(),
                 head_cfg=dict(),
                 neck_cfg=None,
                 seg_cfg=None,
                 model_name=None, 
                 pretrained=None,
                 history_steps=None,
                 test_time_history_steps=None,
                 mem_select_dist_ranges=[0,0,0,0],
                 skip_vector_head=False,
                 freeze_bev=False,
                 freeze_bev_iters=None,
                 track_fp_aug=True,
                 use_memory=False,
                 mem_len=None,
                 mem_warmup_iters=-1,
                 lidar_bev_cfg=None,
                 fusion_bev_cfg=None,
                 sat_bev_cfg=None,
                 aerial_only=False,
                 data_preprocessor=None,
                 **kwargs):
        super().__init__()

        #Attribute
        self.model_name = model_name
        self.last_epoch = None
        self.num_iter = 0
        self.num_epoch = 0
        self.latest_bev_feats = None
        self.latest_seg_logits = None
        self.latest_head_preds = None
        self.latest_head_pos_inds = None
        self.latest_head_gt = None

        if data_preprocessor is None:
            self.data_preprocessor = MMDET3D_MODELS.build(
                dict(type='MapDataPreprocessor'))
        elif isinstance(data_preprocessor, dict):
            self.data_preprocessor = MMDET3D_MODELS.build(data_preprocessor)
        else:
            self.data_preprocessor = data_preprocessor
  
        self.backbone = MMDET3D_MODELS.build(backbone_cfg)

        if neck_cfg is not None:
            self.neck = build_mmdet_module(neck_cfg)
        else:
            self.neck = nn.Identity()

        self.head = build_mmdet_module(
            head_cfg,
            extra_scopes=(
                'plugin_transformer',
                'plugin_transformer_layer_sequence',
                'plugin_transformer_layer',
                'plugin_attention',
                'plugin_ffn',
                'plugin_positional_encoding',
            ))
        self.num_decoder_layers = self.head.transformer.decoder.num_layers
        self.skip_vector_head = skip_vector_head
        self.freeze_bev = freeze_bev # whether freeze bev related parameters
        self.freeze_bev_iters = freeze_bev_iters # whether freeze bev related parameters
        self.track_fp_aug = track_fp_aug
        self.use_memory = use_memory
        self.mem_warmup_iters = mem_warmup_iters

        # the track query propagation module, using relative pose
        c_dim = 7 # quaternion for rotation (4) + translation (3)
        self.query_propagate = MotionMLP(c_dim=c_dim, f_dim=self.head.embed_dims, identity=True)

        # Optional LiDAR BEV branch and fusion
        self.lidar_enabled = False
        self.fusion_enabled = False
        self.lidar_out_channels = None
        # Optional aerial (satellite) BEV branch
        self.sat_enabled = False
        self.sat_out_channels = None
        self._aerial_missing_warned = False
        self._latest_fusion_metrics = {}
        self.aerial_only = aerial_only
        self.aerial_patch_dropout_enabled = False
        self.aerial_patch_dropout_ratio_range = (0.0, 0.0)
        self.aerial_patch_dropout_num_patches_range = (1, 1)
        self.aerial_patch_dropout_patch_hw_ratio_range = (0.1, 0.3)
        self.aerial_patch_dropout_max_tries_per_patch = 4
        # Optional offline gate supervision from per-token targets.
        self.gate_sup_enabled = False
        self.gate_sup_targets = {}
        self.gate_sup_token_key = 'token'
        self.gate_sup_target_key = 'g_star_combo'
        self.gate_sup_weight_key = 'gate_target_weight'
        self.gate_sup_loss_type = 'bce'
        self.gate_sup_eps = 1e-6
        self.gate_sup_target_clip_min = 0.0
        self.gate_sup_target_clip_max = 1.0
        self.gate_sup_missing_policy = 'skip'
        self.gate_sup_min_weight = 0.0
        self.gate_sup_schedule_enabled = True
        self.gate_sup_schedule_start_iter = 8000
        self.gate_sup_schedule_end_iter = 25000
        self.gate_sup_lambda_target = 0.02
        self.gate_sup_csv_path = ''

        if lidar_bev_cfg is not None and lidar_bev_cfg.get('enabled', False):
            self.lidar_enabled = True
            _lidar_cfg = dict(lidar_bev_cfg)
            _lidar_cfg.pop('enabled')
            _lidar_cfg.setdefault('roi_size', roi_size)
            _lidar_cfg.setdefault('bev_h', bev_h)
            _lidar_cfg.setdefault('bev_w', bev_w)
            self.lidar_encoder = MMDET3D_MODELS.build(_lidar_cfg)
            self.lidar_out_channels = getattr(self.lidar_encoder, 'out_channels', _lidar_cfg.get('out_channels', None))

        # Optional aerial image encoder (AID4AD-style): ResUNet encoder + small downsampler
        if sat_bev_cfg is not None and sat_bev_cfg.get('enabled', False):
            self.sat_enabled = True
            from ..necks.sat_resunet import ResNetUNet, DownsampleCNN
            try:
                from mmcv.ops import ModulatedDeformConv2dPack
            except ImportError as exc:  # pragma: no cover - fallback when ops missing
                raise ImportError('mmcv.ops.ModulatedDeformConv2dPack is required for aerial fusion alignment.') from exc
            # ResUNet produces 64-ch feature at input resolution; then downsample to BEV size
            self.sat_encoder = ResNetUNet(outC=sat_bev_cfg.get('resunet_out_channels', 64))
            # Downsample to BEV grid spatially and project to bev_embed_dims channels
            # Default hidden_dim chosen to output 2*hidden_dim channels; set hidden_dim accordingly
            bev_embed_dims = head_cfg.get('in_channels', None)
            if bev_embed_dims is None:
                bev_embed_dims = getattr(self.backbone, 'embed_dims', None)
            assert bev_embed_dims is not None, 'Cannot infer BEV embed dims for satellite branch.'
            hidden_dim = sat_bev_cfg.get('down_hidden_dim', bev_embed_dims // 2)
            self.sat_downsampler = DownsampleCNN(in_channels=sat_bev_cfg.get('resunet_out_channels', 64),
                                                 hidden_dim=hidden_dim)
            # After two convs, output channels = hidden_dim*2
            self.sat_out_channels = hidden_dim * 2
            self.sat_alignment = ModulatedDeformConv2dPack(
                in_channels=self.sat_out_channels,
                out_channels=self.sat_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

            spatial_dropout_cfg = sat_bev_cfg.get('spatial_patch_dropout', None)
            if spatial_dropout_cfg is not None and spatial_dropout_cfg.get('enabled', False):
                self.aerial_patch_dropout_enabled = True
                mask_ratio_range = spatial_dropout_cfg.get('mask_ratio_range', (0.2, 0.3))
                num_patches_range = spatial_dropout_cfg.get('num_patches_range', (2, 6))
                patch_hw_ratio_range = spatial_dropout_cfg.get('patch_hw_ratio_range', (0.08, 0.3))
                max_tries_per_patch = spatial_dropout_cfg.get('max_tries_per_patch', 4)

                if len(mask_ratio_range) != 2:
                    raise ValueError('sat_bev_cfg.spatial_patch_dropout.mask_ratio_range must have length 2.')
                if len(num_patches_range) != 2:
                    raise ValueError('sat_bev_cfg.spatial_patch_dropout.num_patches_range must have length 2.')
                if len(patch_hw_ratio_range) != 2:
                    raise ValueError('sat_bev_cfg.spatial_patch_dropout.patch_hw_ratio_range must have length 2.')

                min_ratio, max_ratio = float(mask_ratio_range[0]), float(mask_ratio_range[1])
                min_patches, max_patches = int(num_patches_range[0]), int(num_patches_range[1])
                min_hw_ratio, max_hw_ratio = float(patch_hw_ratio_range[0]), float(patch_hw_ratio_range[1])

                if not (0.0 <= min_ratio <= max_ratio <= 1.0):
                    raise ValueError('mask_ratio_range must satisfy 0 <= min <= max <= 1.')
                if min_patches < 1 or min_patches > max_patches:
                    raise ValueError('num_patches_range must satisfy 1 <= min <= max.')
                if not (0.0 < min_hw_ratio <= max_hw_ratio <= 1.0):
                    raise ValueError('patch_hw_ratio_range must satisfy 0 < min <= max <= 1.')
                if int(max_tries_per_patch) < 1:
                    raise ValueError('max_tries_per_patch must be >= 1.')

                self.aerial_patch_dropout_ratio_range = (min_ratio, max_ratio)
                self.aerial_patch_dropout_num_patches_range = (min_patches, max_patches)
                self.aerial_patch_dropout_patch_hw_ratio_range = (min_hw_ratio, max_hw_ratio)
                self.aerial_patch_dropout_max_tries_per_patch = int(max_tries_per_patch)

        if fusion_bev_cfg is not None and fusion_bev_cfg.get('enabled', False):
            self.fusion_enabled = True
            _fusion_cfg = dict(fusion_bev_cfg)
            _fusion_cfg.pop('enabled')
            cam_bev_ch = head_cfg.get('in_channels', None)
            if cam_bev_ch is None:
                cam_bev_ch = getattr(self.backbone, 'embed_dims', None)
            assert cam_bev_ch is not None, 'Cannot infer camera BEV channels; set head_cfg.in_channels.'
            in_ch = cam_bev_ch
            if self.lidar_enabled:
                if self.lidar_out_channels is None:
                    self.lidar_out_channels = 64
                in_ch += self.lidar_out_channels
            if self.sat_enabled:
                assert self.sat_out_channels is not None, 'sat_out_channels must be set when sat branch is enabled'
                in_ch += self.sat_out_channels
            _fusion_cfg.setdefault('in_channels', in_ch)
            _fusion_cfg.setdefault('out_channels', cam_bev_ch)
            _fusion_cfg.setdefault('cam_channels', cam_bev_ch)
            self.fusion_bev = MMDET3D_MODELS.build(_fusion_cfg)

        self._init_gate_supervision(fusion_bev_cfg)

        # BEV semantic seg head
        if seg_cfg is not None:
            self.seg_decoder = build_mmdet_module(seg_cfg)
        else:
            self.seg_decoder = None
        
        # BEV 
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.roi_size = roi_size
        self.history_steps = history_steps

        self.mem_len = mem_len

        # Set up test time memory selection hyper-parameters
        if test_time_history_steps is None:
            self.test_time_history_steps = history_steps
        else:
            self.test_time_history_steps = test_time_history_steps
        self.mem_select_dist_ranges = mem_select_dist_ranges

        # vector instance memory module
        if self.use_memory:
            self.memory_bank = VectorInstanceMemory(
                dim_in=head_cfg.embed_dims,
                number_ins=head_cfg.num_queries,
                bank_size=mem_len,
                mem_len=mem_len,
                mem_select_dist_ranges=self.mem_select_dist_ranges,
            )

        xmin, xmax = -roi_size[0]/2, roi_size[0]/2
        ymin, ymax = -roi_size[1]/2, roi_size[1]/2
        x = torch.linspace(xmin, xmax, bev_w)
        y = torch.linspace(ymax, ymin, bev_h)
        y, x = torch.meshgrid(y, x)
        z = torch.zeros_like(x)
        ones = torch.ones_like(x)
        plane = torch.stack([x, y, z, ones], dim=-1)
        self.register_buffer('plane', plane.double())
        
        self.init_weights(pretrained)

    def _unwrap_dc(self, obj):
        """Recursively unwrap ``DataContainer`` instances."""
        if isinstance(obj, DataContainer):
            obj = obj.data
        if isinstance(obj, list):
            return [self._unwrap_dc(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._unwrap_dc(item) for item in obj)
        if isinstance(obj, dict):
            return {key: self._unwrap_dc(val) for key, val in obj.items()}
        return obj

    def _prepare_aerial_tensor(self, aerial_img, device):
        """Convert collated aerial inputs into a 4D tensor if available."""
        if aerial_img is None:
            return None, True

        # DataContainer with stack=True arrives as tensor, stack=False as list.
        if isinstance(aerial_img, torch.Tensor):
            return aerial_img.to(device), False

        if hasattr(aerial_img, 'data') and isinstance(aerial_img.data, torch.Tensor):
            return aerial_img.data.to(device), False

        if isinstance(aerial_img, (list, tuple)):
            if len(aerial_img) == 0:
                return None, True
            if any(elem is None or (hasattr(elem, 'data') and elem.data is None)
                   for elem in aerial_img):
                return None, True
            stacked_elems = []
            for elem in aerial_img:
                if isinstance(elem, torch.Tensor):
                    stacked_elems.append(elem.to(device))
                elif hasattr(elem, 'data') and isinstance(elem.data, torch.Tensor):
                    stacked_elems.append(elem.data.to(device))
                else:
                    return None, True
            stacked = torch.stack(stacked_elems, dim=0)
            return stacked, False

        # Gracefully handle unexpected container types by treating them as missing.
        return None, True

    def _warn_aerial_missing(self):
        if not self._aerial_missing_warned:
            warnings.warn('Aerial imagery missing for part of the batch; dropping satellite modality.',
                          RuntimeWarning)
            self._aerial_missing_warned = True

    def _store_fusion_metrics(self, metrics):
        if metrics is None:
            self._latest_fusion_metrics = {}
        else:
            self._latest_fusion_metrics = metrics

    def _consume_fusion_metrics(self):
        metrics = self._latest_fusion_metrics
        self._latest_fusion_metrics = {}
        return metrics

    def _init_gate_supervision(self, fusion_bev_cfg):
        if fusion_bev_cfg is None or not fusion_bev_cfg.get('enabled', False):
            return
        gate_cfg = fusion_bev_cfg.get('gate_cfg', None)
        if not isinstance(gate_cfg, dict):
            return
        supervision_cfg = gate_cfg.get('supervision', None)
        if not isinstance(supervision_cfg, dict) or not supervision_cfg.get('enabled', False):
            return
        if not self.fusion_enabled:
            raise ValueError('gate supervision requires fusion_bev_cfg.enabled=True.')

        token_key = str(supervision_cfg.get('token_key', 'token'))
        target_key = str(supervision_cfg.get('target_key', 'g_star_combo'))
        weight_key = str(supervision_cfg.get('weight_key', 'gate_target_weight'))
        loss_type = str(supervision_cfg.get('loss_type', 'bce')).lower()
        eps = float(supervision_cfg.get('eps', 1e-6))
        target_clip_min = float(supervision_cfg.get('target_clip_min', 0.0))
        target_clip_max = float(supervision_cfg.get('target_clip_max', 1.0))
        missing_policy = str(supervision_cfg.get('missing_policy', 'skip')).lower()
        min_weight = float(supervision_cfg.get('min_weight', 0.0))
        csv_path = str(supervision_cfg.get('csv_path', '')).strip()
        if not csv_path:
            raise ValueError('gate supervision requires non-empty supervision.csv_path.')
        csv_path = os.path.expanduser(csv_path)
        if not os.path.isfile(csv_path):
            raise ValueError(f'gate supervision csv_path not found: {csv_path}')
        if loss_type != 'bce':
            raise ValueError("gate supervision loss_type must be 'bce'.")
        if eps <= 0:
            raise ValueError('gate supervision eps must be > 0.')
        if not (0.0 <= target_clip_min <= target_clip_max <= 1.0):
            raise ValueError('gate supervision target clip bounds must satisfy 0 <= min <= max <= 1.')
        if missing_policy not in ('skip',):
            raise ValueError("gate supervision missing_policy must be 'skip'.")
        if min_weight < 0:
            raise ValueError('gate supervision min_weight must be >= 0.')

        schedule_cfg = supervision_cfg.get('lambda_schedule', {})
        if schedule_cfg is None:
            schedule_cfg = {}
        schedule_enabled = bool(schedule_cfg.get('enabled', True))
        schedule_start_iter = int(schedule_cfg.get('start_iter', 8000))
        schedule_end_iter = int(schedule_cfg.get('end_iter', 25000))
        lambda_target = float(schedule_cfg.get('lambda_target', 0.02))
        if schedule_enabled:
            if schedule_start_iter < 0:
                raise ValueError('gate supervision lambda_schedule.start_iter must be >= 0.')
            if schedule_end_iter <= schedule_start_iter:
                raise ValueError('gate supervision lambda_schedule.end_iter must be > start_iter.')
        if lambda_target < 0:
            raise ValueError('gate supervision lambda_schedule.lambda_target must be >= 0.')

        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f'gate supervision CSV has no header: {csv_path}')
            required_cols = (token_key, target_key, weight_key)
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]
            if missing_cols:
                raise ValueError(
                    f'gate supervision CSV missing required columns {missing_cols}; '
                    f'found={reader.fieldnames}')

            targets = {}
            for row in reader:
                token_raw = row.get(token_key, None)
                if token_raw is None:
                    continue
                token = str(token_raw).strip()
                if token == '':
                    continue
                try:
                    target_val = float(row.get(target_key, ''))
                    weight_val = float(row.get(weight_key, ''))
                except (TypeError, ValueError):
                    continue
                if not (target_val == target_val and weight_val == weight_val):
                    # Skip NaN rows.
                    continue
                target_val = max(target_clip_min, min(target_clip_max, target_val))
                weight_val = max(0.0, weight_val)
                targets[token] = (target_val, weight_val)

        if len(targets) == 0:
            raise ValueError(f'gate supervision loaded 0 valid rows from: {csv_path}')

        self.gate_sup_enabled = True
        self.gate_sup_targets = targets
        self.gate_sup_token_key = token_key
        self.gate_sup_target_key = target_key
        self.gate_sup_weight_key = weight_key
        self.gate_sup_loss_type = loss_type
        self.gate_sup_eps = eps
        self.gate_sup_target_clip_min = target_clip_min
        self.gate_sup_target_clip_max = target_clip_max
        self.gate_sup_missing_policy = missing_policy
        self.gate_sup_min_weight = min_weight
        self.gate_sup_schedule_enabled = schedule_enabled
        self.gate_sup_schedule_start_iter = schedule_start_iter
        self.gate_sup_schedule_end_iter = schedule_end_iter
        self.gate_sup_lambda_target = lambda_target
        self.gate_sup_csv_path = csv_path

    def _get_gate_sup_lambda(self, current_iter=None):
        if not self.gate_sup_enabled:
            return 0.0
        if not self.gate_sup_schedule_enabled:
            return self.gate_sup_lambda_target
        if current_iter is None:
            current_iter = self.num_iter
        curr_iter = int(current_iter)
        if curr_iter <= self.gate_sup_schedule_start_iter:
            return 0.0
        if curr_iter < self.gate_sup_schedule_end_iter:
            ratio = (
                (curr_iter - self.gate_sup_schedule_start_iter)
                / float(self.gate_sup_schedule_end_iter - self.gate_sup_schedule_start_iter)
            )
            return self.gate_sup_lambda_target * ratio
        return self.gate_sup_lambda_target

    def _compute_gate_supervision_loss(self, fusion_metrics, img_metas):
        if not self.gate_sup_enabled or not isinstance(fusion_metrics, dict):
            return None, {}
        gate_sample_mean_raw = fusion_metrics.get('gate_sample_mean_raw', None)
        if not torch.is_tensor(gate_sample_mean_raw):
            return None, {}
        if gate_sample_mean_raw.ndim != 1:
            gate_sample_mean_raw = gate_sample_mean_raw.flatten()
        if gate_sample_mean_raw.numel() == 0:
            return None, {}
        if not isinstance(img_metas, (list, tuple)):
            return None, {}
        if len(img_metas) != gate_sample_mean_raw.shape[0]:
            raise ValueError(
                f'gate supervision expected {gate_sample_mean_raw.shape[0]} img_metas, '
                f'got {len(img_metas)}.')
        if self.gate_sup_loss_type != 'bce':
            raise ValueError(f'Unsupported gate supervision loss_type: {self.gate_sup_loss_type}')

        num_samples = gate_sample_mean_raw.shape[0]
        device = gate_sample_mean_raw.device
        dtype = gate_sample_mean_raw.dtype
        targets = torch.zeros(num_samples, device=device, dtype=dtype)
        weights = torch.zeros(num_samples, device=device, dtype=dtype)
        valid = torch.zeros(num_samples, device=device, dtype=torch.bool)

        for idx, meta in enumerate(img_metas):
            token = None
            if isinstance(meta, dict):
                token = meta.get(self.gate_sup_token_key, None)
            if token is None:
                continue
            item = self.gate_sup_targets.get(str(token), None)
            if item is None:
                if self.gate_sup_missing_policy != 'skip':
                    raise ValueError(f'Unknown gate supervision missing_policy: {self.gate_sup_missing_policy}')
                continue
            target_val, weight_val = item
            if weight_val < self.gate_sup_min_weight:
                continue
            targets[idx] = target_val
            weights[idx] = weight_val
            valid[idx] = True

        zero = gate_sample_mean_raw.new_tensor(0.0)
        valid_float = valid.float()
        coverage = valid_float.mean()
        lambda_eff = gate_sample_mean_raw.new_tensor(self._get_gate_sup_lambda(self.num_iter))
        if not valid.any():
            stats = dict(
                loss=zero,
                lambda_eff=lambda_eff,
                coverage=coverage.detach(),
                weight_mean=zero,
                target_mean=zero,
                pred_mean=zero,
                mae=zero,
            )
            return None, stats

        eff_weights = weights * valid_float
        weight_sum = eff_weights.sum()
        if weight_sum.item() <= 0:
            stats = dict(
                loss=zero,
                lambda_eff=lambda_eff,
                coverage=coverage.detach(),
                weight_mean=zero,
                target_mean=zero,
                pred_mean=zero,
                mae=zero,
            )
            return None, stats

        pred = gate_sample_mean_raw.clamp(self.gate_sup_eps, 1.0 - self.gate_sup_eps)
        target = targets.clamp(self.gate_sup_eps, 1.0 - self.gate_sup_eps)
        per_sample = F.binary_cross_entropy(pred, target, reduction='none')
        loss_raw = (eff_weights * per_sample).sum() / (weight_sum + self.gate_sup_eps)
        loss_scaled = loss_raw * lambda_eff
        valid_count = valid_float.sum()
        weight_mean = eff_weights.sum() / (valid_count + self.gate_sup_eps)
        target_mean = (eff_weights * target).sum() / (weight_sum + self.gate_sup_eps)
        pred_mean = (eff_weights * pred).sum() / (weight_sum + self.gate_sup_eps)
        mae = (eff_weights * (pred - target).abs()).sum() / (weight_sum + self.gate_sup_eps)

        stats = dict(
            loss=loss_scaled.detach(),
            lambda_eff=lambda_eff.detach(),
            coverage=coverage.detach(),
            weight_mean=weight_mean.detach(),
            target_mean=target_mean.detach(),
            pred_mean=pred_mean.detach(),
            mae=mae.detach(),
        )
        if lambda_eff.item() <= 0:
            return None, stats
        return loss_scaled, stats

    def _apply_aerial_spatial_dropout(self, aerial_bev):
        """Randomly drop spatial patches in aerial BEV during training."""
        if (not self.training) or (not self.aerial_patch_dropout_enabled):
            return aerial_bev, None, None
        if aerial_bev is None:
            return aerial_bev, None, None

        batch_size, channels, height, width = aerial_bev.shape
        device = aerial_bev.device
        mask = torch.zeros((batch_size, 1, height, width), dtype=torch.bool, device=device)

        min_ratio, max_ratio = self.aerial_patch_dropout_ratio_range
        min_patches, max_patches = self.aerial_patch_dropout_num_patches_range
        min_hw_ratio, max_hw_ratio = self.aerial_patch_dropout_patch_hw_ratio_range

        for b_i in range(batch_size):
            target_ratio = float(torch.empty((), device=device).uniform_(min_ratio, max_ratio).item())
            if target_ratio <= 0:
                continue
            target_area = max(1, int(round(target_ratio * height * width)))
            num_patches = int(torch.randint(min_patches, max_patches + 1, (1,), device=device).item())
            max_attempts = max(num_patches * self.aerial_patch_dropout_max_tries_per_patch, num_patches)
            current_area = 0

            for _ in range(max_attempts):
                if current_area >= target_area:
                    break
                patch_h_ratio = float(torch.empty((), device=device).uniform_(min_hw_ratio, max_hw_ratio).item())
                patch_w_ratio = float(torch.empty((), device=device).uniform_(min_hw_ratio, max_hw_ratio).item())
                patch_h = max(1, min(height, int(round(patch_h_ratio * height))))
                patch_w = max(1, min(width, int(round(patch_w_ratio * width))))

                y0 = int(torch.randint(0, height - patch_h + 1, (1,), device=device).item())
                x0 = int(torch.randint(0, width - patch_w + 1, (1,), device=device).item())
                patch_mask = mask[b_i, 0, y0:y0 + patch_h, x0:x0 + patch_w]
                already_masked = int(patch_mask.sum().item())
                patch_mask.fill_(True)
                current_area += patch_h * patch_w - already_masked

        if mask.any():
            aerial_bev = aerial_bev.masked_fill(mask.expand(-1, channels, -1, -1), 0.0)

        masked_ratio = mask.float().mean()
        return aerial_bev, mask, masked_ratio

    def _fuse_modalities(self, bev_feats, img_metas, points=None, aerial_img=None, warn_missing=True):
        """Fuse camera BEV with optional LiDAR and aerial branches."""
        if not self.fusion_enabled:
            self._store_fusion_metrics({})
            return bev_feats

        cam_bev = bev_feats
        aux_features = []
        b, _, h, w = bev_feats.shape
        force_camera_only = False
        aerial_dropout_ratio = None
        aerial_dropout_mask = None

        if self.lidar_enabled and points is not None:
            lidar_bev = self.lidar_encoder(points, img_metas)
            if lidar_bev.shape[-2:] != bev_feats.shape[-2:]:
                raise RuntimeError('LiDAR BEV spatial size mismatch with camera BEV')
            aux_features.append(lidar_bev)
        elif self.lidar_enabled:
            if self.lidar_out_channels is None:
                raise RuntimeError('LiDAR branch enabled but lidar_out_channels is undefined.')
            lidar_bev = bev_feats.new_zeros((b, self.lidar_out_channels, h, w))
            aux_features.append(lidar_bev)

        if self.sat_enabled:
            aerial_tensor, aerial_missing = self._prepare_aerial_tensor(aerial_img, bev_feats.device)
            if aerial_missing:
                if warn_missing:
                    self._warn_aerial_missing()
                aid_bev = bev_feats.new_zeros((b, self.sat_out_channels, h, w))
                aux_features.append(aid_bev)
                force_camera_only = not self.lidar_enabled
            else:
                aid_bev = self.sat_encoder(aerial_tensor)
                aid_bev = self.sat_downsampler(aid_bev)
                aid_bev = self.sat_alignment(aid_bev)
                if aid_bev.shape[-2:] != bev_feats.shape[-2:]:
                    # final safeguard; expect alignment to preserve spatial dims
                    import torch.nn.functional as F
                    aid_bev = F.interpolate(aid_bev, size=bev_feats.shape[-2:], mode='bilinear', align_corners=False)
                aid_bev, aerial_dropout_mask, aerial_dropout_ratio = self._apply_aerial_spatial_dropout(aid_bev)
                aux_features.append(aid_bev)

        if not aux_features:
            self._store_fusion_metrics({})
            return cam_bev

        aux = torch.cat(aux_features, dim=1)
        expected_aux = getattr(self.fusion_bev, 'aux_channels', aux.shape[1])
        if aux.shape[1] != expected_aux:
            if aux.shape[1] < expected_aux:
                pad = expected_aux - aux.shape[1]
                aux = torch.cat([aux, aux.new_zeros((b, pad, h, w))], dim=1)
            else:
                aux = aux[:, :expected_aux]

        fused_bev, fusion_metrics = self.fusion_bev(
            cam_bev,
            aux,
            return_gate_info=True,
            force_camera_only=force_camera_only,
            aerial_drop_mask=aerial_dropout_mask,
            current_iter=self.num_iter,
        )
        if torch.is_tensor(aerial_dropout_ratio):
            fusion_metrics['aerial_patch_dropout_ratio'] = aerial_dropout_ratio.detach()
        gate_sample_mean = fusion_metrics.get('gate_sample_mean', None)
        if torch.is_tensor(gate_sample_mean):
            stale_flags = []
            for meta in img_metas:
                stale_flag = None
                if isinstance(meta, dict):
                    stale_flag = meta.get('aerial_is_stale', meta.get('aerial_stale', None))
                stale_flags.append(stale_flag)
            if any(flag is not None for flag in stale_flags):
                stale_tensor = gate_sample_mean.new_tensor(
                    [bool(flag) if flag is not None else False for flag in stale_flags],
                    dtype=torch.bool,
                )
                valid_tensor = gate_sample_mean.new_tensor(
                    [flag is not None for flag in stale_flags],
                    dtype=torch.bool,
                )
                stale_valid = stale_tensor & valid_tensor
                fresh_valid = (~stale_tensor) & valid_tensor
                if stale_valid.any():
                    fusion_metrics['gate_stale_mean'] = gate_sample_mean[stale_valid].mean().detach()
                if fresh_valid.any():
                    fusion_metrics['gate_fresh_mean'] = gate_sample_mean[fresh_valid].mean().detach()
        self._store_fusion_metrics(fusion_metrics)
        return fused_bev

    def _aerial_only_bev(self, aerial_img, device):
        """Produce BEV features from aerial imagery alone, bypassing the
        camera backbone and FusionBEV entirely.

        Returns a tensor of shape ``(B, bev_embed_dims, bev_h, bev_w)``
        suitable for direct consumption by the neck.
        """
        assert self.sat_enabled, 'aerial_only mode requires sat_bev_cfg to be enabled'
        aerial_tensor, aerial_missing = self._prepare_aerial_tensor(aerial_img, device)
        if aerial_missing:
            # Graceful fallback: return zeros so eval / rare missing samples
            # don't crash the entire run.  During training the dataloader
            # should always supply aerial images; at eval time a small number
            # of samples may lack aerial crops.
            self._warn_aerial_missing()
            embed_ch = self.sat_out_channels or 256
            return torch.zeros(1, embed_ch, self.bev_h, self.bev_w,
                               device=device)
        aid_bev = self.sat_encoder(aerial_tensor)
        aid_bev = self.sat_downsampler(aid_bev)
        aid_bev = self.sat_alignment(aid_bev)
        target_h, target_w = self.bev_h, self.bev_w
        if aid_bev.shape[-2:] != (target_h, target_w):
            import torch.nn.functional as Finterp
            aid_bev = Finterp.interpolate(aid_bev, size=(target_h, target_w),
                                          mode='bilinear', align_corners=False)
        return aid_bev

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        if pretrained:
            import logging
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            try:
                self.neck.init_weights()
            except AttributeError:
                pass
            # initialize optional LiDAR/fusion modules
            if getattr(self, 'lidar_enabled', False):
                try:
                    self.lidar_encoder.init_weights()
                except Exception:
                    pass
            if getattr(self, 'fusion_enabled', False):
                try:
                    self.fusion_bev.init_weights()
                except Exception:
                    pass
            if getattr(self, 'sat_enabled', False):
                try:
                    self.sat_alignment.init_weights()
                except Exception:
                    pass

    def temporal_propagate(self, curr_bev_feats, img_metas, all_history_curr2prev, all_history_prev2curr, use_memory,
                           track_query_info=None, timestep=None, get_trans_loss=False):
        '''
        Args:
            curr_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
            img_metas: current image metas (List of #bs samples)
            bev_memory: where to load and store (training and testing use different buffer)
            pose_memory: where to load and store (training and testing use different buffer)

        Out:
            fused_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
        '''

        bs = curr_bev_feats.size(0)

        if get_trans_loss: # init the trans_loss related variables here
            trans_reg_loss = curr_bev_feats.new_zeros((1,))
            trans_cls_loss = curr_bev_feats.new_zeros((1,))
            back_trans_reg_loss = curr_bev_feats.new_zeros((1,))
            back_trans_cls_loss = curr_bev_feats.new_zeros((1,))
            num_pos = 0
            num_tracks = 0

        if use_memory:
            self.memory_bank.clear_dict()
            
        for b_i in range(bs):
            curr_e2g_trans = self.plane.new_tensor(img_metas[b_i]['ego2global_translation'], dtype=torch.float64)
            curr_e2g_rot = self.plane.new_tensor(img_metas[b_i]['ego2global_rotation'], dtype=torch.float64)

            if use_memory:
                self.memory_bank.curr_rot[b_i] = curr_e2g_rot
                self.memory_bank.curr_trans[b_i] = curr_e2g_trans
                if self.memory_bank.curr_t > 0:
                    self.memory_bank.trans_memory_bank(self.query_propagate, b_i, img_metas[b_i])

            # transform the track queries
            if track_query_info is not None:
                history_curr2prev_matrix = all_history_curr2prev[b_i]
                history_prev2curr_matrix = all_history_prev2curr[b_i]

                track_pts = track_query_info[b_i]['track_query_boxes'].clone()
                track_pts = rearrange(track_pts, 'n (k c) -> n k c', c=2)
                # from (0, 1) to (-30, 30) or (-15, 15), prep for transform
                track_pts = self._denorm_lines(track_pts)

                # Transform the track ref-points using relative pose between prev and curr
                N, num_points = track_pts.shape[0], track_pts.shape[1]
                track_pts = torch.cat([
                    track_pts,
                    track_pts.new_zeros((N, num_points, 1)), # z-axis
                    track_pts.new_ones((N, num_points, 1)) # 4-th dim
                ], dim=-1) # (num_prop, num_pts, 4)

                pose_matrix = history_prev2curr_matrix[-1].float()[:3]
                rot_mat = pose_matrix[:, :3].cpu().numpy()
                rot = R.from_matrix(rot_mat)
                translation = pose_matrix[:, 3] 
                trans_matrix = history_prev2curr_matrix[-1].clone()

                # Add training-time perturbation here for the transformation matrix
                if self.training:
                    rot, translation = self.add_noise_to_pose(rot, translation)            
                    trans_matrix[:3, :3] = torch.tensor(rot.as_matrix()).to(trans_matrix.device)
                    trans_matrix[:3, 3] = torch.tensor(translation).to(trans_matrix.device)

                trans_track_pts = torch.einsum('lk,ijk->ijl', trans_matrix, track_pts.double()).float()
                trans_track_pts = trans_track_pts[..., :2]
                trans_track_pts = self._norm_lines(trans_track_pts)
                trans_track_pts = torch.clip(trans_track_pts, min=0., max=1.)
                trans_track_pts = rearrange(trans_track_pts, 'n k c -> n (k c)', c=2)
                track_query_info[b_i]['trans_track_query_boxes'] = trans_track_pts
                
                prop_q = track_query_info[b_i]['track_query_hs_embeds']

                rot_quat = torch.tensor(rot.as_quat()).float().to(pose_matrix.device)
                pose_info = torch.cat([rot_quat.view(-1), translation], dim=0)                

                track_query_updated = self.query_propagate(
                    prop_q, # (topk, embed_dims)
                    pose_info.repeat(len(prop_q), 1)
                )
                # Do not let future-frame loss backprop through the track queries
                track_query_info[b_i]['track_query_hs_embeds'] = track_query_updated.clone().detach()

                if get_trans_loss:
                    pred = self.head.reg_branches[-1](track_query_updated).sigmoid() # (num_prop, 2*num_pts)
                    pred_scores = self.head.cls_branches[-1](track_query_updated)
                    assert list(pred.shape) == [N, 2*num_points]

                    gt_pts = track_query_info[b_i]['track_query_gt_lines'].clone()
                    gt_labels = track_query_info[b_i]['track_query_gt_labels'].clone()
                    weights = gt_pts.new_ones((N, 2*num_points))
                    weights_labels = gt_labels.new_ones((N,))
                    bg_idx = gt_labels == 3
                    num_pos = num_pos + (N - bg_idx.sum())
                    num_tracks += len(gt_labels)
                    weights[bg_idx, :] = 0.0
                
                    gt_pts = rearrange(gt_pts, 'n (k c) -> n k c', c=2)
                    denormed_targets = self._denorm_lines(gt_pts)
                    denormed_targets = torch.cat([
                        denormed_targets,
                        denormed_targets.new_zeros((N, num_points, 1)), # z-axis
                        denormed_targets.new_ones((N, num_points, 1)) # 4-th dim
                    ], dim=-1) # (num_prop, num_pts, 4)
                    assert list(denormed_targets.shape) == [N, num_points, 4]

                    curr_targets = torch.einsum('lk,ijk->ijl', trans_matrix.float(), denormed_targets)
                    curr_targets = curr_targets[..., :2]
                    normed_targets = self._norm_lines(curr_targets)
                    normed_targets = rearrange(normed_targets, 'n k c -> n (k c)', c=2)
                    # set the weight of invalid normed targets to 0 (outside current bev frame)
                    invalid_bev_mask = (normed_targets <= 0) | (normed_targets>=1)
                    weights[invalid_bev_mask] = 0
                    # (num_prop, 2*num_pts)
                    trans_reg_loss += self.head.loss_reg(pred, normed_targets, weights, avg_factor=1.0)
                    if len(gt_labels) > 0:
                        trans_score = self.head.loss_cls(pred_scores, gt_labels, weights_labels, avg_factor=1.0)
                    else:
                        trans_score = 0.0
                    trans_cls_loss += trans_score

                    # backward trans loss
                    pose_matrix_inv = torch.inverse(trans_matrix).float()[:3]
                    rot_mat_inv = pose_matrix_inv[:, :3].cpu().numpy()

                    rot_inv = R.from_matrix(rot_mat_inv)
                    rot_quat_inv = torch.tensor(rot_inv.as_quat()).float().to(pose_matrix_inv.device)
                    translation_inv = pose_matrix_inv[:, 3]
                    pose_info_inv = torch.cat([rot_quat_inv.view(-1), translation_inv], dim=0)                
                    track_query_backtrans = self.query_propagate(
                        track_query_updated, # (topk, embed_dims)
                        pose_info_inv.repeat(len(prop_q), 1)
                    )
                    pred_backtrans = self.head.reg_branches[-1](track_query_backtrans).sigmoid() # (num_prop, 2*num_pts)
                    pred_scores_backtrans = self.head.cls_branches[-1](track_query_backtrans)
                    prev_gt_pts = track_query_info[b_i]['track_query_gt_lines']
                    back_trans_reg_loss += self.head.loss_reg(pred_backtrans, prev_gt_pts, weights, avg_factor=1.0)
                    if len(gt_labels) > 0:
                        trans_score_bak = self.head.loss_cls(pred_scores_backtrans, gt_labels, weights_labels, avg_factor=1.0)
                    else:
                        trans_score_bak = 0.0
                    back_trans_cls_loss += trans_score_bak

        if get_trans_loss:
            trans_loss = self.head.trans_loss_weight * (trans_reg_loss / (num_pos + 1e-10) + 
                            trans_cls_loss / (num_tracks + 1e-10))
            back_trans_loss = self.head.trans_loss_weight * (back_trans_reg_loss / (num_pos + 1e-10) +
                                    back_trans_cls_loss / (num_tracks + 1e-10))
            trans_loss_dict = {
                'f_trans': trans_loss,
                'b_trans': back_trans_loss,
            }
            return trans_loss_dict
    
    def add_noise_to_pose(self, rot, trans):
        rot_euler = rot.as_euler('zxy')
        # 0.08 mean is around 5-degree, 3-sigma is 15-degree
        noise_euler = np.random.randn(*list(rot_euler.shape)) * 0.08
        rot_euler += noise_euler
        noisy_rot = R.from_euler('zxy', rot_euler)

        # error within 0.25 meter
        noise_trans = torch.randn_like(trans) * 0.25
        noise_trans[2] = 0
        noisy_trans = trans + noise_trans

        return noisy_rot, noisy_trans

    def process_history_info(self, img_metas, history_img_metas):
        bs = len(img_metas)
        all_history_curr2prev = []
        all_history_prev2curr = []
        all_history_coord = []

        if len(history_img_metas) == 0:
            return all_history_curr2prev, all_history_prev2curr, all_history_coord

        for b_i in range(bs):
            history_e2g_trans = torch.stack([self.plane.new_tensor(prev[b_i]['ego2global_translation'], dtype=torch.float64) for prev in history_img_metas], dim=0)
            history_e2g_rot = torch.stack([self.plane.new_tensor(prev[b_i]['ego2global_rotation'], dtype=torch.float64) for prev in history_img_metas], dim=0)
            
            curr_e2g_trans = self.plane.new_tensor(img_metas[b_i]['ego2global_translation'], dtype=torch.float64)
            curr_e2g_rot = self.plane.new_tensor(img_metas[b_i]['ego2global_rotation'], dtype=torch.float64)

            # Do the coords transformation for all features in the history buffer
            ## Prepare the transformation matrix
            history_g2e_matrix = torch.stack([torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device),]*len(history_e2g_trans), dim=0)
            history_g2e_matrix[:, :3, :3] = torch.transpose(history_e2g_rot, 1, 2)
            history_g2e_matrix[:, :3, 3] = -torch.bmm(torch.transpose(history_e2g_rot, 1, 2), history_e2g_trans[..., None]).squeeze(-1)

            curr_g2e_matrix = torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device)
            curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
            curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

            curr_e2g_matrix = torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device)
            curr_e2g_matrix[:3, :3] = curr_e2g_rot
            curr_e2g_matrix[:3, 3] = curr_e2g_trans

            history_e2g_matrix = torch.stack([torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device),]*len(history_e2g_trans), dim=0)
            history_e2g_matrix[:, :3, :3] = history_e2g_rot
            history_e2g_matrix[:, :3, 3] = history_e2g_trans

            history_curr2prev_matrix = torch.bmm(history_g2e_matrix, repeat(curr_e2g_matrix,'n1 n2 -> r n1 n2', r=len(history_g2e_matrix)))
            history_prev2curr_matrix = torch.bmm(repeat(curr_g2e_matrix, 'n1 n2 -> r n1 n2', r=len(history_e2g_matrix)), history_e2g_matrix)

            history_coord = torch.einsum('nlk,ijk->nijl', history_curr2prev_matrix, self.plane).float()[..., :2]

            # from (-30, 30) or (-15, 15) to (-1, 1)
            history_coord[..., 0] = history_coord[..., 0] / (self.roi_size[0]/2)
            history_coord[..., 1] = -history_coord[..., 1] / (self.roi_size[1]/2)

            all_history_curr2prev.append(history_curr2prev_matrix)
            all_history_prev2curr.append(history_prev2curr_matrix)
            all_history_coord.append(history_coord)
        
        return all_history_curr2prev, all_history_prev2curr, all_history_coord
        

    def forward_train(self, img, vectors, semantic_mask, aerial_img=None, points=None, img_metas=None, all_prev_data=None,
                      all_local2global_info=None, **kwargs):
        '''
        Args:
            img: torch.Tensor of shape [B, N, 3, H, W]
                N: number of cams
            vectors: list[list[Tuple(lines, length, label)]]
                - lines: np.array of shape [num_points, 2]. 
                - length: int
                - label: int
                len(vectors) = batch_size
                len(vectors[_b]) = num of lines in sample _b
            img_metas: 
                img_metas['lidar2img']: [B, N, 4, 4]
        Out:
            loss, log_vars, num_sample
        '''
        # ensure images are stacked tensors
        target_device = self.plane.device

        def _stack_imgs(data):
            if isinstance(data, torch.Tensor):
                return data.to(target_device)
            if isinstance(data, DataContainer):
                return _stack_imgs(data.data)
            if isinstance(data, list):
                stacked = [_stack_imgs(item) for item in data]
                if stacked and isinstance(stacked[0], torch.Tensor):
                    return torch.stack(stacked, dim=0).to(target_device)
            return data

        def _unwrap(data):
            if isinstance(data, DataContainer):
                return _unwrap(data.data)
            if isinstance(data, list):
                return [_unwrap(item) for item in data]
            if isinstance(data, dict):
                return {k: _unwrap(v) for k, v in data.items()}
            return data

        vectors = _unwrap(vectors)
        semantic_mask = _unwrap(semantic_mask)
        points = _unwrap(points)
        img_metas = _unwrap(img_metas)
        if all_prev_data is not None:
            cleaned_prev = []
            for prev in all_prev_data:
                prev_clean = {}
                prev_clean['vectors'] = _unwrap(prev.get('vectors'))
                prev_clean['img'] = _stack_imgs(_unwrap(prev.get('img')))
                prev_clean['img_metas'] = _unwrap(prev.get('img_metas'))
                prev_clean['semantic_mask'] = _unwrap(prev.get('semantic_mask'))
                prev_clean['points'] = _unwrap(prev.get('points'))
                prev_clean['aerial_img'] = _stack_imgs(_unwrap(prev.get('aerial_img')))
                cleaned_prev.append(prev_clean)
            all_prev_data = cleaned_prev

        def _stack_semantics(data):
            if isinstance(data, torch.Tensor):
                return data.to(target_device)
            if isinstance(data, list):
                stacked = [_stack_semantics(item) for item in data]
                if stacked and isinstance(stacked[0], torch.Tensor):
                    return torch.stack(stacked, dim=0).to(target_device)
            return data

        img = _stack_imgs(img)
        if aerial_img is not None:
            aerial_img = _stack_imgs(aerial_img)

        semantic_mask = _stack_semantics(semantic_mask)

        self.latest_bev_feats = None
        self.latest_seg_logits = None
        self.latest_head_preds = None
        self.latest_head_pos_inds = None
        self.latest_head_gt = None

        #  prepare labels and images
        gts, img, img_metas, valid_idx, points = self.batch_data(
            vectors, img, img_metas, img.device, points)
        bs = img.shape[0]

        _use_memory = self.use_memory and self.num_iter > self.mem_warmup_iters
        
        if all_prev_data is not None:
            num_prev_frames = len(all_prev_data)        
            all_gts_prev, all_img_prev, all_img_metas_prev, all_semantic_mask_prev  = [], [], [], []
            for prev_data in all_prev_data:
                gts_prev, img_prev, img_metas_prev, valid_idx_prev, _ = self.batch_data(
                    prev_data['vectors'], prev_data['img'], prev_data['img_metas'], img.device      
                )
                all_gts_prev.append(gts_prev)
                all_img_prev.append(img_prev)
                all_img_metas_prev.append(img_metas_prev)
                all_semantic_mask_prev.append(_stack_semantics(prev_data['semantic_mask']))
        else:
            num_prev_frames = 0

        # points may be provided when LiDAR BEV branch is enabled

        if self.skip_vector_head:
            backprop_backbone_ids = [0, num_prev_frames] # first and last frame train the backbone (bev pretrain)
        else:
            backprop_backbone_ids = [num_prev_frames, ] # only the last frame trains the backbone (all other settings)

        track_query_info = None
        all_loss_dict_prev = []
        all_trans_loss = []
        all_outputs_prev = []
        fusion_gate_means = []
        fusion_gate_p90s = []
        fusion_gate_regs = []
        fusion_gate_entropies = []
        fusion_gate_entropy_regs = []
        fusion_gate_masked_means = []
        fusion_gate_unmasked_means = []
        fusion_gate_masked_regs = []
        fusion_gate_lambda_effs = []
        fusion_gate_entropy_lambda_effs = []
        fusion_gate_masked_lambda_effs = []
        fusion_gate_stale_means = []
        fusion_gate_fresh_means = []
        fusion_gate_sup_losses = []
        fusion_gate_sup_lambda_effs = []
        fusion_gate_sup_coverages = []
        fusion_gate_sup_weight_means = []
        fusion_gate_sup_target_means = []
        fusion_gate_sup_pred_means = []
        fusion_gate_sup_maes = []
        aerial_patch_dropout_ratios = []

        self.tracked_query_length = {}
        self._store_fusion_metrics({})

        if _use_memory:
            self.memory_bank.set_bank_size(self.mem_len)
            self.memory_bank.init_memory(bs=bs)

        # History records for bev features
        history_bev_feats = []
        history_img_metas = []
        
        gt_semantic = torch.flip(semantic_mask, [2,])

        # Iterate through all prev frames
        for t in range(num_prev_frames):
            # Backbone for prev
            img_backbone_gradient = (t in backprop_backbone_ids)
            fusion_metrics_prev = {}

            all_history_curr2prev, all_history_prev2curr, all_history_coord =  \
                    self.process_history_info(all_img_metas_prev[t], history_img_metas)

            if self.aerial_only:
                prev_aerial_img = None
                if all_prev_data is not None and isinstance(all_prev_data[t], dict):
                    prev_aerial_img = all_prev_data[t].get('aerial_img', None)
                _bev_feats = self._aerial_only_bev(prev_aerial_img, img.device)
                mlvl_feats = None
            else:
                _bev_feats, mlvl_feats = self.backbone(all_img_prev[t], all_img_metas_prev[t], t, history_bev_feats, 
                            history_img_metas, all_history_coord, points=None, 
                            img_backbone_gradient=img_backbone_gradient)

                prev_points = None
                prev_aerial_img = None
                if all_prev_data is not None and isinstance(all_prev_data[t], dict):
                    prev_points = all_prev_data[t].get('points', None)
                    prev_aerial_img = all_prev_data[t].get('aerial_img', None)

                _bev_feats = self._fuse_modalities(
                    _bev_feats,
                    all_img_metas_prev[t],
                    points=prev_points,
                    aerial_img=prev_aerial_img,
                )
                fusion_metrics_prev = self._consume_fusion_metrics()

            # Neck for prev
            bev_feats = self.neck(_bev_feats)

            if _use_memory:
                self.memory_bank.curr_t = t
            
            # Transform prev-frame feature & pts to curr frame
            if self.skip_vector_head or t == 0:
                self.temporal_propagate(bev_feats, all_img_metas_prev[t], all_history_curr2prev, 
                        all_history_prev2curr, _use_memory, track_query_info, timestep=t, get_trans_loss=False)
            else:
                trans_loss_dict = self.temporal_propagate(bev_feats, all_img_metas_prev[t], all_history_curr2prev, 
                        all_history_prev2curr, _use_memory, track_query_info, timestep=t, get_trans_loss=True)

                ########################################################
                # Debugging use: visualize the first-frame track query. and the corresponding 
                # ground-truth information     
                # Do this for every timestep > 0
                #self._viz_temporal_supervision(outputs_prev, track_query_info, gts_next[-1], gts_prev[-1], 
                #                gts_semantic_curr, gts_semantic_prev, img_metas_next, img_metas_prev, t)
                ########################################################
            
            img_metas_prev = all_img_metas_prev[t]
            img_metas_next = all_img_metas_prev[t+1] if t < num_prev_frames-1 else img_metas
            gts_prev = all_gts_prev[t]
            gts_next = all_gts_prev[t+1] if t!=num_prev_frames-1 else gts
            gts_semantic_prev = torch.flip(all_semantic_mask_prev[t], [2,])
            gts_semantic_curr = torch.flip(all_semantic_mask_prev[t+1], [2,]) if t!=num_prev_frames-1 else gt_semantic

            local2global_prev = all_local2global_info[t]
            local2global_next = all_local2global_info[t+1]

            # Compute the semantic segmentation loss
            seg_preds, seg_feats, seg_loss, seg_dice_loss = self.seg_decoder(bev_feats, gts_semantic_prev,
                    all_history_coord, return_loss=True)

            # Save the history 
            history_bev_feats.append(bev_feats)
            history_img_metas.append(all_img_metas_prev[t])
            if len(history_bev_feats) > self.history_steps:
                history_bev_feats.pop(0)
                history_img_metas.pop(0)
            
            if not self.skip_vector_head:
                # Prepare the two-frame instance matching info
                gt_cur2prev, gt_prev2cur = self.get_two_frame_matching(local2global_prev, local2global_next, 
                                                                       gts_prev, gts_next)
                if t == 0:
                    memory_bank = None
                else:
                    memory_bank = self.memory_bank if _use_memory else None
                # 1). Compute the loss for prev frame
                # 2). Get the matching results for computing the track query to next frame
                loss_dict_prev, outputs_prev, prev_inds_list, prev_gt_inds_list, prev_matched_reg_cost, \
                    prev_gt_list = self.head(
                                        bev_features=bev_feats, 
                                        img_metas=img_metas_prev, 
                                        gts=gts_prev,
                                        track_query_info=track_query_info,
                                        memory_bank=memory_bank,
                                        return_loss=True,
                                        return_matching=True)
                all_outputs_prev.append(outputs_prev)

                if t > 0:
                    all_trans_loss.append(trans_loss_dict)

                # Do the query prop and negative sampling, prepare the corrpespnding
                # updated G.T. labels. The prepared queries will be passed to the model,
                # and combind with the original queries inside the head model
                pos_th = 0.4
                track_query_info = self.prepare_track_queries_and_targets(gts_next, prev_inds_list, 
                    prev_gt_inds_list, prev_matched_reg_cost, prev_gt_list, outputs_prev, gt_cur2prev, gt_prev2cur, 
                    img_metas_prev, _use_memory, pos_th=pos_th, timestep=t)
            else:
                loss_dict_prev = {}

            loss_dict_prev['seg'] = seg_loss
            loss_dict_prev['seg_dice'] = seg_dice_loss
            gate_applied_prev = bool(fusion_metrics_prev.get('gate_applied', False))
            gate_reg_prev = fusion_metrics_prev.get('gate_reg', None)
            if torch.is_tensor(gate_reg_prev) and gate_reg_prev.requires_grad:
                loss_dict_prev['fusion_gate_reg'] = gate_reg_prev
            if gate_applied_prev and torch.is_tensor(gate_reg_prev):
                fusion_gate_regs.append(gate_reg_prev.detach())

            gate_mean_prev = fusion_metrics_prev.get('gate_mean', None)
            if gate_applied_prev and torch.is_tensor(gate_mean_prev):
                fusion_gate_means.append(gate_mean_prev)

            gate_p90_prev = fusion_metrics_prev.get('gate_p90', None)
            if gate_applied_prev and torch.is_tensor(gate_p90_prev):
                fusion_gate_p90s.append(gate_p90_prev)

            gate_entropy_prev = fusion_metrics_prev.get('gate_entropy', None)
            if gate_applied_prev and torch.is_tensor(gate_entropy_prev):
                fusion_gate_entropies.append(gate_entropy_prev)

            gate_entropy_reg_prev = fusion_metrics_prev.get('gate_entropy_reg', None)
            if gate_applied_prev and torch.is_tensor(gate_entropy_reg_prev):
                fusion_gate_entropy_regs.append(gate_entropy_reg_prev)

            gate_masked_mean_prev = fusion_metrics_prev.get('gate_masked_mean', None)
            if gate_applied_prev and torch.is_tensor(gate_masked_mean_prev):
                fusion_gate_masked_means.append(gate_masked_mean_prev)

            gate_unmasked_mean_prev = fusion_metrics_prev.get('gate_unmasked_mean', None)
            if gate_applied_prev and torch.is_tensor(gate_unmasked_mean_prev):
                fusion_gate_unmasked_means.append(gate_unmasked_mean_prev)

            gate_masked_reg_prev = fusion_metrics_prev.get('gate_masked_reg', None)
            if gate_applied_prev and torch.is_tensor(gate_masked_reg_prev):
                fusion_gate_masked_regs.append(gate_masked_reg_prev)

            gate_lambda_eff_prev = fusion_metrics_prev.get('gate_lambda_eff', None)
            if gate_applied_prev and torch.is_tensor(gate_lambda_eff_prev):
                fusion_gate_lambda_effs.append(gate_lambda_eff_prev)

            gate_entropy_lambda_eff_prev = fusion_metrics_prev.get('gate_entropy_lambda_eff', None)
            if gate_applied_prev and torch.is_tensor(gate_entropy_lambda_eff_prev):
                fusion_gate_entropy_lambda_effs.append(gate_entropy_lambda_eff_prev)

            gate_masked_lambda_eff_prev = fusion_metrics_prev.get('gate_masked_lambda_eff', None)
            if gate_applied_prev and torch.is_tensor(gate_masked_lambda_eff_prev):
                fusion_gate_masked_lambda_effs.append(gate_masked_lambda_eff_prev)

            gate_stale_mean_prev = fusion_metrics_prev.get('gate_stale_mean', None)
            if torch.is_tensor(gate_stale_mean_prev):
                fusion_gate_stale_means.append(gate_stale_mean_prev)

            gate_fresh_mean_prev = fusion_metrics_prev.get('gate_fresh_mean', None)
            if torch.is_tensor(gate_fresh_mean_prev):
                fusion_gate_fresh_means.append(gate_fresh_mean_prev)

            gate_sup_prev, gate_sup_stats_prev = self._compute_gate_supervision_loss(
                fusion_metrics=fusion_metrics_prev,
                img_metas=img_metas_prev,
            )
            if torch.is_tensor(gate_sup_prev) and gate_sup_prev.requires_grad:
                loss_dict_prev['fusion_gate_sup'] = gate_sup_prev
            if gate_sup_stats_prev:
                gate_sup_loss_prev = gate_sup_stats_prev.get('loss', None)
                if torch.is_tensor(gate_sup_loss_prev):
                    fusion_gate_sup_losses.append(gate_sup_loss_prev)
                gate_sup_lambda_eff_prev = gate_sup_stats_prev.get('lambda_eff', None)
                if torch.is_tensor(gate_sup_lambda_eff_prev):
                    fusion_gate_sup_lambda_effs.append(gate_sup_lambda_eff_prev)
                gate_sup_coverage_prev = gate_sup_stats_prev.get('coverage', None)
                if torch.is_tensor(gate_sup_coverage_prev):
                    fusion_gate_sup_coverages.append(gate_sup_coverage_prev)
                gate_sup_weight_mean_prev = gate_sup_stats_prev.get('weight_mean', None)
                if torch.is_tensor(gate_sup_weight_mean_prev):
                    fusion_gate_sup_weight_means.append(gate_sup_weight_mean_prev)
                gate_sup_target_mean_prev = gate_sup_stats_prev.get('target_mean', None)
                if torch.is_tensor(gate_sup_target_mean_prev):
                    fusion_gate_sup_target_means.append(gate_sup_target_mean_prev)
                gate_sup_pred_mean_prev = gate_sup_stats_prev.get('pred_mean', None)
                if torch.is_tensor(gate_sup_pred_mean_prev):
                    fusion_gate_sup_pred_means.append(gate_sup_pred_mean_prev)
                gate_sup_mae_prev = gate_sup_stats_prev.get('mae', None)
                if torch.is_tensor(gate_sup_mae_prev):
                    fusion_gate_sup_maes.append(gate_sup_mae_prev)

            patch_dropout_ratio_prev = fusion_metrics_prev.get('aerial_patch_dropout_ratio', None)
            if torch.is_tensor(patch_dropout_ratio_prev):
                aerial_patch_dropout_ratios.append(patch_dropout_ratio_prev)

            all_loss_dict_prev.append(loss_dict_prev)

        if _use_memory:
            self.memory_bank.curr_t = num_prev_frames

        # NOTE: we separate the last frame to be consistent with single-frame only setting)
        # Backbone for curr
        img_backbone_gradient = num_prev_frames in backprop_backbone_ids

        all_history_curr2prev, all_history_prev2curr, all_history_coord = self.process_history_info(img_metas, history_img_metas)

        fusion_metrics_curr = {}
        if self.aerial_only:
            _bev_feats = self._aerial_only_bev(aerial_img, img.device)
        else:
            _bev_feats, mlvl_feats = self.backbone(img, img_metas, num_prev_frames, history_bev_feats, history_img_metas, all_history_coord,
                        points=None, img_backbone_gradient=img_backbone_gradient)

            _bev_feats = self._fuse_modalities(
                _bev_feats,
                img_metas,
                points=points,
                aerial_img=aerial_img,
            )
            fusion_metrics_curr = self._consume_fusion_metrics()
        # Neck for curr
        bev_feats = self.neck(_bev_feats)
        self.latest_bev_feats = bev_feats

        if self.skip_vector_head or num_prev_frames == 0:
            # Transform prev-frame feature & pts to curr frame using the relative pose
            assert track_query_info is None
            self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev, 
                        all_history_prev2curr, _use_memory, track_query_info, timestep=num_prev_frames, get_trans_loss=False)
        else:
            trans_loss_dict = self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev, 
                        all_history_prev2curr, _use_memory, track_query_info, timestep=num_prev_frames, get_trans_loss=True)            
            all_trans_loss.append(trans_loss_dict)

            ########################################################
            # Debugging use: visualize the first-frame track query. and the corresponding 
            # ground-truth information     
            # Do this for every timestep > 0
            #assert num_prev_frames > 0
            #self._viz_temporal_supervision(outputs_prev, track_query_info, gts_next[-1], gts_prev[-1], gt_semantic,
            #        gts_semantic_prev, img_metas_next, img_metas_prev, timestep=num_prev_frames)
            ########################################################

        seg_preds, seg_feats, seg_loss, seg_dice_loss = self.seg_decoder(bev_feats, gt_semantic, 
                all_history_coord, return_loss=True)
        self.latest_seg_logits = seg_preds
        
        if not self.skip_vector_head:
            memory_bank = self.memory_bank if _use_memory else None
            # 3. run the head again and compute the loss for the second frame
            preds_list, loss_dict, det_match_idxs, det_match_gt_idxs, gt_list = self.head(
                bev_features=bev_feats, 
                img_metas=img_metas, 
                gts=gts,
                track_query_info=track_query_info,
                memory_bank=memory_bank,
                return_loss=True)
            self.latest_head_preds = preds_list[-1]
            self.latest_head_pos_inds = det_match_idxs[-1]
            self.latest_head_gt = gt_list[-1]
        else:
            loss_dict = {}
            self.latest_head_preds = None
            self.latest_head_pos_inds = None
            self.latest_head_gt = None
        
        loss_dict['seg'] = seg_loss
        loss_dict['seg_dice'] = seg_dice_loss
        gate_applied_curr = bool(fusion_metrics_curr.get('gate_applied', False))
        gate_reg_curr = fusion_metrics_curr.get('gate_reg', None)
        if torch.is_tensor(gate_reg_curr) and gate_reg_curr.requires_grad:
            loss_dict['fusion_gate_reg'] = gate_reg_curr
        if gate_applied_curr and torch.is_tensor(gate_reg_curr):
            fusion_gate_regs.append(gate_reg_curr.detach())

        gate_mean_curr = fusion_metrics_curr.get('gate_mean', None)
        if gate_applied_curr and torch.is_tensor(gate_mean_curr):
            fusion_gate_means.append(gate_mean_curr)

        gate_p90_curr = fusion_metrics_curr.get('gate_p90', None)
        if gate_applied_curr and torch.is_tensor(gate_p90_curr):
            fusion_gate_p90s.append(gate_p90_curr)

        gate_entropy_curr = fusion_metrics_curr.get('gate_entropy', None)
        if gate_applied_curr and torch.is_tensor(gate_entropy_curr):
            fusion_gate_entropies.append(gate_entropy_curr)

        gate_entropy_reg_curr = fusion_metrics_curr.get('gate_entropy_reg', None)
        if gate_applied_curr and torch.is_tensor(gate_entropy_reg_curr):
            fusion_gate_entropy_regs.append(gate_entropy_reg_curr)

        gate_masked_mean_curr = fusion_metrics_curr.get('gate_masked_mean', None)
        if gate_applied_curr and torch.is_tensor(gate_masked_mean_curr):
            fusion_gate_masked_means.append(gate_masked_mean_curr)

        gate_unmasked_mean_curr = fusion_metrics_curr.get('gate_unmasked_mean', None)
        if gate_applied_curr and torch.is_tensor(gate_unmasked_mean_curr):
            fusion_gate_unmasked_means.append(gate_unmasked_mean_curr)

        gate_masked_reg_curr = fusion_metrics_curr.get('gate_masked_reg', None)
        if gate_applied_curr and torch.is_tensor(gate_masked_reg_curr):
            fusion_gate_masked_regs.append(gate_masked_reg_curr)

        gate_lambda_eff_curr = fusion_metrics_curr.get('gate_lambda_eff', None)
        if gate_applied_curr and torch.is_tensor(gate_lambda_eff_curr):
            fusion_gate_lambda_effs.append(gate_lambda_eff_curr)

        gate_entropy_lambda_eff_curr = fusion_metrics_curr.get('gate_entropy_lambda_eff', None)
        if gate_applied_curr and torch.is_tensor(gate_entropy_lambda_eff_curr):
            fusion_gate_entropy_lambda_effs.append(gate_entropy_lambda_eff_curr)

        gate_masked_lambda_eff_curr = fusion_metrics_curr.get('gate_masked_lambda_eff', None)
        if gate_applied_curr and torch.is_tensor(gate_masked_lambda_eff_curr):
            fusion_gate_masked_lambda_effs.append(gate_masked_lambda_eff_curr)

        gate_stale_mean_curr = fusion_metrics_curr.get('gate_stale_mean', None)
        if torch.is_tensor(gate_stale_mean_curr):
            fusion_gate_stale_means.append(gate_stale_mean_curr)

        gate_fresh_mean_curr = fusion_metrics_curr.get('gate_fresh_mean', None)
        if torch.is_tensor(gate_fresh_mean_curr):
            fusion_gate_fresh_means.append(gate_fresh_mean_curr)

        gate_sup_curr, gate_sup_stats_curr = self._compute_gate_supervision_loss(
            fusion_metrics=fusion_metrics_curr,
            img_metas=img_metas,
        )
        if torch.is_tensor(gate_sup_curr) and gate_sup_curr.requires_grad:
            loss_dict['fusion_gate_sup'] = gate_sup_curr
        if gate_sup_stats_curr:
            gate_sup_loss_curr = gate_sup_stats_curr.get('loss', None)
            if torch.is_tensor(gate_sup_loss_curr):
                fusion_gate_sup_losses.append(gate_sup_loss_curr)
            gate_sup_lambda_eff_curr = gate_sup_stats_curr.get('lambda_eff', None)
            if torch.is_tensor(gate_sup_lambda_eff_curr):
                fusion_gate_sup_lambda_effs.append(gate_sup_lambda_eff_curr)
            gate_sup_coverage_curr = gate_sup_stats_curr.get('coverage', None)
            if torch.is_tensor(gate_sup_coverage_curr):
                fusion_gate_sup_coverages.append(gate_sup_coverage_curr)
            gate_sup_weight_mean_curr = gate_sup_stats_curr.get('weight_mean', None)
            if torch.is_tensor(gate_sup_weight_mean_curr):
                fusion_gate_sup_weight_means.append(gate_sup_weight_mean_curr)
            gate_sup_target_mean_curr = gate_sup_stats_curr.get('target_mean', None)
            if torch.is_tensor(gate_sup_target_mean_curr):
                fusion_gate_sup_target_means.append(gate_sup_target_mean_curr)
            gate_sup_pred_mean_curr = gate_sup_stats_curr.get('pred_mean', None)
            if torch.is_tensor(gate_sup_pred_mean_curr):
                fusion_gate_sup_pred_means.append(gate_sup_pred_mean_curr)
            gate_sup_mae_curr = gate_sup_stats_curr.get('mae', None)
            if torch.is_tensor(gate_sup_mae_curr):
                fusion_gate_sup_maes.append(gate_sup_mae_curr)

        patch_dropout_ratio_curr = fusion_metrics_curr.get('aerial_patch_dropout_ratio', None)
        if torch.is_tensor(patch_dropout_ratio_curr):
            aerial_patch_dropout_ratios.append(patch_dropout_ratio_curr)

        # format loss, average over all frames (2 frames for now)
        loss = 0
        losses_t = []
        for loss_dict_t in (all_loss_dict_prev + [loss_dict,]):
            loss_t = 0
            for name, var in loss_dict_t.items():
                loss_t = loss_t + var
            losses_t.append(loss_t)
            loss += loss_t
        
        for trans_loss_dict_t in all_trans_loss:
            trans_loss_t = trans_loss_dict_t['f_trans'] + trans_loss_dict_t['b_trans']
            loss += trans_loss_t
        
        # update the log
        log_vars = {k: v.item() for k, v in loss_dict.items()}

        for t, loss_dict_t in enumerate(all_loss_dict_prev):
            log_vars_t = {k+'_t{}'.format(t): v.item() for k, v in loss_dict_t.items()}
            log_vars.update(log_vars_t)
        
        for t, loss_t in enumerate(losses_t):
            log_vars.update({'total_t{}'.format(t): loss_t.item()})
        
        for t, trans_loss_dict_t in enumerate(all_trans_loss):
            log_vars_t = {k+'_t{}'.format(t): v.item() for k, v in trans_loss_dict_t.items()}
            log_vars.update(log_vars_t)

        if fusion_gate_means:
            fusion_gate_mean = torch.stack(fusion_gate_means).mean().item()
            log_vars['fusion_gate_mean'] = fusion_gate_mean
            log_vars['train/fusion_gate_mean'] = fusion_gate_mean
        if fusion_gate_p90s:
            fusion_gate_p90 = torch.stack(fusion_gate_p90s).mean().item()
            log_vars['fusion_gate_p90'] = fusion_gate_p90
            log_vars['train/fusion_gate_p90'] = fusion_gate_p90
        if fusion_gate_regs:
            fusion_gate_reg_mean = torch.stack(fusion_gate_regs).mean().item()
            log_vars['fusion_gate_reg_mean'] = fusion_gate_reg_mean
            log_vars['train/fusion_gate_reg_mean'] = fusion_gate_reg_mean
        if fusion_gate_entropies:
            fusion_gate_entropy_mean = torch.stack(fusion_gate_entropies).mean().item()
            log_vars['fusion_gate_entropy_mean'] = fusion_gate_entropy_mean
            log_vars['train/fusion_gate_entropy_mean'] = fusion_gate_entropy_mean
        if fusion_gate_entropy_regs:
            fusion_gate_entropy_reg_mean = torch.stack(fusion_gate_entropy_regs).mean().item()
            log_vars['fusion_gate_entropy_reg_mean'] = fusion_gate_entropy_reg_mean
            log_vars['train/fusion_gate_entropy_reg_mean'] = fusion_gate_entropy_reg_mean
        if fusion_gate_masked_means:
            fusion_gate_masked_mean = torch.stack(fusion_gate_masked_means).mean().item()
            log_vars['fusion_gate_masked_mean'] = fusion_gate_masked_mean
            log_vars['train/fusion_gate_masked_mean'] = fusion_gate_masked_mean
        if fusion_gate_unmasked_means:
            fusion_gate_unmasked_mean = torch.stack(fusion_gate_unmasked_means).mean().item()
            log_vars['fusion_gate_unmasked_mean'] = fusion_gate_unmasked_mean
            log_vars['train/fusion_gate_unmasked_mean'] = fusion_gate_unmasked_mean
        if fusion_gate_masked_regs:
            fusion_gate_masked_reg_mean = torch.stack(fusion_gate_masked_regs).mean().item()
            log_vars['fusion_gate_masked_reg_mean'] = fusion_gate_masked_reg_mean
            log_vars['train/fusion_gate_masked_reg_mean'] = fusion_gate_masked_reg_mean
        if fusion_gate_lambda_effs:
            fusion_gate_lambda_eff_mean = torch.stack(fusion_gate_lambda_effs).mean().item()
            log_vars['fusion_gate_lambda_eff_mean'] = fusion_gate_lambda_eff_mean
            log_vars['train/fusion_gate_lambda_eff_mean'] = fusion_gate_lambda_eff_mean
        if fusion_gate_entropy_lambda_effs:
            fusion_gate_entropy_lambda_eff_mean = torch.stack(fusion_gate_entropy_lambda_effs).mean().item()
            log_vars['fusion_gate_entropy_lambda_eff_mean'] = fusion_gate_entropy_lambda_eff_mean
            log_vars['train/fusion_gate_entropy_lambda_eff_mean'] = fusion_gate_entropy_lambda_eff_mean
        if fusion_gate_masked_lambda_effs:
            fusion_gate_masked_lambda_eff_mean = torch.stack(fusion_gate_masked_lambda_effs).mean().item()
            log_vars['fusion_gate_masked_lambda_eff_mean'] = fusion_gate_masked_lambda_eff_mean
            log_vars['train/fusion_gate_masked_lambda_eff_mean'] = fusion_gate_masked_lambda_eff_mean
        if fusion_gate_stale_means:
            fusion_gate_stale_mean = torch.stack(fusion_gate_stale_means).mean().item()
            log_vars['fusion_gate_stale_mean'] = fusion_gate_stale_mean
            log_vars['train/fusion_gate_stale_mean'] = fusion_gate_stale_mean
        if fusion_gate_fresh_means:
            fusion_gate_fresh_mean = torch.stack(fusion_gate_fresh_means).mean().item()
            log_vars['fusion_gate_fresh_mean'] = fusion_gate_fresh_mean
            log_vars['train/fusion_gate_fresh_mean'] = fusion_gate_fresh_mean
        if fusion_gate_sup_losses:
            fusion_gate_sup_loss_mean = torch.stack(fusion_gate_sup_losses).mean().item()
            log_vars['fusion_gate_sup_loss_mean'] = fusion_gate_sup_loss_mean
            log_vars['train/fusion_gate_sup_loss_mean'] = fusion_gate_sup_loss_mean
        if fusion_gate_sup_lambda_effs:
            fusion_gate_sup_lambda_eff_mean = torch.stack(fusion_gate_sup_lambda_effs).mean().item()
            log_vars['fusion_gate_sup_lambda_eff_mean'] = fusion_gate_sup_lambda_eff_mean
            log_vars['train/fusion_gate_sup_lambda_eff_mean'] = fusion_gate_sup_lambda_eff_mean
        if fusion_gate_sup_coverages:
            fusion_gate_sup_coverage_mean = torch.stack(fusion_gate_sup_coverages).mean().item()
            log_vars['fusion_gate_sup_coverage_mean'] = fusion_gate_sup_coverage_mean
            log_vars['train/fusion_gate_sup_coverage_mean'] = fusion_gate_sup_coverage_mean
        if fusion_gate_sup_weight_means:
            fusion_gate_sup_weight_mean = torch.stack(fusion_gate_sup_weight_means).mean().item()
            log_vars['fusion_gate_sup_weight_mean'] = fusion_gate_sup_weight_mean
            log_vars['train/fusion_gate_sup_weight_mean'] = fusion_gate_sup_weight_mean
        if fusion_gate_sup_target_means:
            fusion_gate_sup_target_mean = torch.stack(fusion_gate_sup_target_means).mean().item()
            log_vars['fusion_gate_sup_target_mean'] = fusion_gate_sup_target_mean
            log_vars['train/fusion_gate_sup_target_mean'] = fusion_gate_sup_target_mean
        if fusion_gate_sup_pred_means:
            fusion_gate_sup_pred_mean = torch.stack(fusion_gate_sup_pred_means).mean().item()
            log_vars['fusion_gate_sup_pred_mean'] = fusion_gate_sup_pred_mean
            log_vars['train/fusion_gate_sup_pred_mean'] = fusion_gate_sup_pred_mean
        if fusion_gate_sup_maes:
            fusion_gate_sup_mae_mean = torch.stack(fusion_gate_sup_maes).mean().item()
            log_vars['fusion_gate_sup_mae_mean'] = fusion_gate_sup_mae_mean
            log_vars['train/fusion_gate_sup_mae_mean'] = fusion_gate_sup_mae_mean
        if aerial_patch_dropout_ratios:
            aerial_patch_dropout_ratio_mean = torch.stack(aerial_patch_dropout_ratios).mean().item()
            log_vars['aerial_patch_dropout_ratio_mean'] = aerial_patch_dropout_ratio_mean
            log_vars['train/aerial_patch_dropout_ratio_mean'] = aerial_patch_dropout_ratio_mean
        
        log_vars.update({'total': loss.item()})
        num_sample = img.size(0)
        return loss, log_vars, num_sample

    @torch.no_grad()
    def forward_test(self, img, aerial_img=None, points=None, img_metas=None, seq_info=None, **kwargs):
        '''
            inference pipeline
        '''

        self.latest_bev_feats = None
        self.latest_seg_logits = None
        self.latest_head_preds = None
        self.latest_head_pos_inds = None
        self.latest_head_gt = None

        if isinstance(img, (list, tuple)):
            assert len(img) == 1, 'Only support bs=1 per-gpu for inference'
            img = img[0]

        assert img.shape[0] == 1, 'Only support bs=1 per-gpu for inference'

        tokens = []
        for img_meta in img_metas:
            tokens.append(img_meta['token'])

        # seq_info may arrive either as a tuple (scene_name, local_idx, seq_len)
        # or wrapped in a list by the collate_fn when cpu_only=True.
        if seq_info is None and img_metas is not None:
            try:
                candidate = img_metas[0].get('seq_info', None)  # type: ignore[index]
            except Exception:
                candidate = None
            if candidate is not None:
                seq_info = candidate

        if isinstance(seq_info, (list, tuple)):
            if len(seq_info) == 0:
                raise ValueError('Empty seq_info received during inference.')
            first_elem = seq_info[0]
            if isinstance(first_elem, (list, tuple)) and not isinstance(first_elem, (str, bytes)):
                scene_name, local_idx, seq_length = first_elem
            elif len(seq_info) >= 3 and not isinstance(first_elem, (list, tuple)):
                scene_name, local_idx, seq_length = seq_info[:3]
            else:
                raise TypeError(f'Unexpected seq_info structure: {seq_info!r}')
        else:
            raise TypeError(f'Unexpected seq_info type: {type(seq_info)!r}')

        first_frame = (local_idx == 0)
        img_metas[0]['local_idx'] = local_idx
    
        if first_frame:
            if self.use_memory:
                self.memory_bank.set_bank_size(self.test_time_history_steps)
                #self.memory_bank.set_bank_size(self.mem_len)
                self.memory_bank.init_memory(bs=1)
            self.history_bev_feats_all = []
            self.history_img_metas_all = []
        
        if self.use_memory:
            self.memory_bank.curr_t = local_idx
        
        selected_mem_ids = self.select_memory_entries(self.history_img_metas_all, img_metas)
        history_img_metas = [self.history_img_metas_all[idx] for idx in selected_mem_ids]
        history_bev_feats = [self.history_bev_feats_all[idx] for idx in selected_mem_ids]

        all_history_curr2prev, all_history_prev2curr, all_history_coord =  \
                    self.process_history_info(img_metas, history_img_metas)

        if self.aerial_only:
            _bev_feats = self._aerial_only_bev(aerial_img, img.device)
        else:
            _bev_feats, mlvl_feats = self.backbone(img, img_metas, local_idx, history_bev_feats, history_img_metas,
                            all_history_coord, points=points)

            # Optional fusion for current frame (test)
            _bev_feats = self._fuse_modalities(
                _bev_feats,
                img_metas,
                points=points,
                aerial_img=aerial_img,
            )

        img_shape = [_bev_feats.shape[2:] for i in range(_bev_feats.shape[0])]
        # Neck
        bev_feats = self.neck(_bev_feats)
        self.latest_bev_feats = bev_feats

        if self.skip_vector_head or first_frame:
            self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev, \
                    all_history_prev2curr, self.use_memory, track_query_info=None)
            seg_preds, seg_feats = self.seg_decoder(bev_features=bev_feats, return_loss=False)
            if not self.skip_vector_head:
                preds_list = self.head(bev_feats, img_metas=img_metas, return_loss=False)
            track_dict = None
        else:
            # Using the saved prev-frame output to prepare the track query inputs
            track_query_info = self.head.get_track_info(scene_name, local_idx)
            # Transform prev-frame feature & pts to curr frame using the relative pose
            self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev, 
                all_history_prev2curr, self.use_memory, track_query_info)
            seg_preds, seg_feats = self.seg_decoder(bev_features=bev_feats, return_loss=False)

            # Run the vector map decoder with instance-level memory
            memory_bank = self.memory_bank if self.use_memory else None
            preds_list = self.head(bev_feats, img_metas=img_metas, 
                        track_query_info=track_query_info, memory_bank=memory_bank,
                        return_loss=False)
            track_dict = self._process_track_query_info(track_query_info)

        self.latest_seg_logits = seg_preds
            
        if not self.skip_vector_head:
            # take predictions from the last layer
            preds_dict = preds_list[-1]
            self.latest_head_preds = preds_dict
        else:
            preds_dict = None
            self.latest_head_preds = None
        self.latest_head_pos_inds = None
        self.latest_head_gt = None

        # Save the BEV and meta-info history 
        self.history_bev_feats_all.append(bev_feats)
        self.history_img_metas_all.append(img_metas)

        if len(self.history_bev_feats_all) > self.test_time_history_steps:
            self.history_bev_feats_all.pop(0)
            self.history_img_metas_all.pop(0)
        
        if not self.skip_vector_head:
            memory_bank = self.memory_bank if self.use_memory else None
            thr_det = 0.4 if first_frame else 0.6
            pos_results = self.head.prepare_temporal_propagation(preds_dict, scene_name, local_idx, 
                                        memory_bank, thr_track=0.5, thr_det=thr_det)
    
        if not self.skip_vector_head:
            results_list = self.head.post_process(preds_dict, tokens, track_dict)
            results_list[0]['pos_results'] = pos_results
            results_list[0]['meta'] = img_metas[0]
        else:
            results_list = [{'vectors': [],
                'scores': [],
                'labels': [],
                'props': [],
                'token': token} for token in tokens]

        # Add the segmentation preds to the results to be saved
        for b_i in range(len(results_list)):
            tmp_scores, tmp_labels = seg_preds[b_i].max(0)
            tmp_scores = tmp_scores.sigmoid()
            preds_i = torch.zeros(tmp_labels.shape, dtype=torch.uint8).to(tmp_scores.device)
            pos_ids = tmp_scores >= 0.4
            preds_i[pos_ids] = tmp_labels[pos_ids].type(torch.uint8) + 1
            preds_i = preds_i.cpu().numpy()
            results_list[b_i]['semantic_mask'] = preds_i
            if 'token' not in results_list[b_i]:
                results_list[b_i]['token'] = tokens[b_i]

        return results_list

    def batch_data(self, vectors, imgs, img_metas, device, points=None):
        bs = len(vectors)
        # filter none vector's case
        num_gts = []
        for idx in range(bs):
            num_gts.append(sum([len(v) for k, v in vectors[idx].items()]))
        valid_idx = [i for i in range(bs) if num_gts[i] > 0]
        assert len(valid_idx) == bs,f'len(valid idx)={len(valid_idx)}, bs={bs}' # make sure every sample has gts

        attribute_specs = getattr(self.head, 'attribute_specs', {})
        attr_names = list(attribute_specs.keys())

        all_labels_list = []
        all_lines_list = []
        all_gt2local = []
        all_local2gt = []
        all_attr_labels = {name: [] for name in attr_names}
        all_attr_masks = {name: [] for name in attr_names}
        for idx in range(bs):
            labels = []
            lines = []
            gt2local = []
            local2gt = {}
            attr_labels_sample = {name: [] for name in attr_names}
            attr_masks_sample = {name: [] for name in attr_names}
            for label, _lines in vectors[idx].items():
                for _ins_id, _line in enumerate(_lines):
                    labels.append(label)
                    gt2local.append([label, _ins_id])
                    local2gt[(label, _ins_id)] = len(lines)
                    if attr_names:
                        attrs = getattr(_line, 'attrs', {}) if hasattr(_line, 'attrs') else {}
                        for attr_name in attr_names:
                            spec = attribute_specs[attr_name]
                            applies_to = spec['applies_to_set']
                            default_index = spec['default_index']
                            value_to_index = spec['value_to_index']
                            if label in applies_to:
                                raw_value = attrs.get(attr_name)
                                attr_idx = value_to_index.get(raw_value, default_index)
                                attr_labels_sample[attr_name].append(attr_idx)
                                attr_masks_sample[attr_name].append(1)
                            else:
                                attr_labels_sample[attr_name].append(default_index)
                                attr_masks_sample[attr_name].append(0)
                    if len(_line.shape) == 3: # permutation
                        num_permute, num_points, coords_dim = _line.shape
                        lines.append(torch.tensor(_line).reshape(num_permute, -1)) # (38, 40)
                    elif len(_line.shape) == 2:
                        lines.append(torch.tensor(_line).reshape(-1)) # (40, )
                    else:
                        assert False

            all_labels_list.append(torch.tensor(labels, dtype=torch.long).to(device))
            all_lines_list.append(torch.stack(lines).float().to(device))
            all_gt2local.append(gt2local)
            all_local2gt.append(local2gt)
            for attr_name in attr_names:
                labels_tensor = torch.tensor(attr_labels_sample[attr_name], dtype=torch.long, device=device)
                mask_tensor = torch.tensor(attr_masks_sample[attr_name], dtype=torch.bool, device=device)
                all_attr_labels[attr_name].append(labels_tensor)
                all_attr_masks[attr_name].append(mask_tensor)

        gts = {
            'labels': all_labels_list,
            'lines': all_lines_list,
            'gt2local': all_gt2local,
            'local2gt': all_local2gt,
        }
        if attr_names:
            gts['attrs'] = {
                attr_name: {
                    'labels': all_attr_labels[attr_name],
                    'mask': all_attr_masks[attr_name],
                } for attr_name in attr_names
            }

        gts = [deepcopy(gts) for _ in range(self.num_decoder_layers)]

        return gts, imgs, img_metas, valid_idx, points
    
    def get_two_frame_matching(self, local2global_prev, local2global_curr, gts_prev, gts):
        """
        Get the G.T. matching between the two frames
        Terminology: (1). local --> local idx inside each category;
                    (2). global --> global instance id inside category
                    (3). gt --> index in the flattened G.T. sequence
        Args:
            prev_ins_ids (_type_): global ids (pre-prepared) for prev frame
            curr_ins_ids (_type_): global ids (pre-prepared) for curr frame
            gts_prev (_type_): processed G.T. for prev frame
            gts (_type_): processed G.T. for curr frame
        """
        local2global_prev = self._unwrap_dc(local2global_prev)
        local2global_curr = self._unwrap_dc(local2global_curr)

        bs = len(local2global_prev)
        gt2local_curr = gts[-1]['gt2local'] # don't need the per-block supervision, just take one
        gt2local_prev = gts_prev[-1]['gt2local']
        local2gt_prev = gts_prev[-1]['local2gt']

        # the comma is to take the single-element output from multi_apply
        global2local_prev, = multi_apply(self._reverse_id_mapping, local2global_prev)

        all_gt_cur2prev, all_gt_prev2cur = multi_apply(self._compute_cur2prev, gt2local_curr, gt2local_prev, local2gt_prev, 
                                        local2global_curr, global2local_prev)
        
        return all_gt_cur2prev, all_gt_prev2cur
    
    def _compute_cur2prev(self, gt2local_curr, gt2local_prev, local2gt_prev, 
                          local2global_curr, global2local_prev):
        local2global_curr = self._unwrap_dc(local2global_curr)
        global2local_prev = self._unwrap_dc(global2local_prev)
        cur2prev = torch.zeros(len(gt2local_curr))
        prev2cur = torch.zeros(len(gt2local_prev))
        prev2cur[:] = -1
        for gt_idx_curr in range(len(gt2local_curr)):
            label = gt2local_curr[gt_idx_curr][0]
            local_idx = gt2local_curr[gt_idx_curr][1]

            label_curr_map = local2global_curr.get(label, {})
            if local_idx not in label_curr_map:
                gt_idx_prev = -1
                cur2prev[gt_idx_curr] = gt_idx_prev
                continue

            seq_id = label_curr_map[local_idx]

            label_prev_map = global2local_prev.get(label, {})
            if seq_id in label_prev_map:
                local_id_prev = label_prev_map[seq_id]
                gt_idx_prev = local2gt_prev.get((label, local_id_prev), -1)
            else:
                gt_idx_prev = -1
            cur2prev[gt_idx_curr] = gt_idx_prev
            if gt_idx_prev != -1: # there is a positive match in prev frame
                prev2cur[gt_idx_prev] = gt_idx_curr # update the information
            
        return cur2prev, prev2cur
                
    def _reverse_id_mapping(self, id_mapping):
        id_mapping = self._unwrap_dc(id_mapping)
        reversed_mapping = {}
        for label, mapping in id_mapping.items():
            r_map = {v:k for k,v in mapping.items()}
            reversed_mapping[label] = r_map
        return reversed_mapping,

    def prepare_track_queries_and_targets(self, gts, prev_inds_list, prev_gt_inds_list, prev_matched_reg_cost,
                     prev_gt_list, prev_out, gt_cur2prev, gt_prev2cur, metas_prev, use_memory, pos_th=0.4, timestep=None):
        bs = len(prev_inds_list)
        device = prev_out['lines'][0].device

        targets = []
        for b_i in range(bs):
            results = {}
            for key, val in gts[-1].items():
                if isinstance(val, list):
                    results[key] = val[b_i]
                elif isinstance(val, dict):
                    nested = {}
                    for sub_key, sub_val in val.items():
                        if isinstance(sub_val, list):
                            nested[sub_key] = sub_val[b_i]
                        else:
                            nested[sub_key] = sub_val
                    results[key] = nested
                else:
                    results[key] = val
            targets.append(results)
                
        # for each sample in the batch
        for b_i, (target, prev_out_ind, prev_target_ind) in enumerate(zip(targets, prev_inds_list, prev_gt_inds_list)):
            scene_seq_id = metas_prev[b_i]['local_idx']

            scores = prev_out['scores'][b_i].detach()
            scores, labels = scores.max(-1)
            scores = scores.sigmoid()

            match_cost = prev_matched_reg_cost[b_i]
            raw_prev2cur = gt_prev2cur[b_i]  # keep on cpu for safe indexing
            target['prev_target_ind'] = prev_target_ind  # record the matched g.t. index
            target['prev_out_ind'] = prev_out_ind
            target['gt_prev2cur'] = raw_prev2cur.to(device)

            # 1). filter the ones with low scores, create FN; 
            prev_out_ind = prev_out_ind.to(device)
            prev_target_ind = prev_target_ind.to(device)
            prev_pos_scores = scores[prev_out_ind]
            score_filter_mask = prev_pos_scores >= pos_th

            keep_mask = score_filter_mask
            prev_out_ind_filtered = prev_out_ind[keep_mask]
            prev_target_ind_filtered = prev_target_ind[keep_mask]

            indices_cpu = prev_target_ind_filtered.detach().cpu()
            mapping_cpu = raw_prev2cur.detach().cpu()
            valid_gt_idx = (indices_cpu >= 0) & (indices_cpu < mapping_cpu.shape[0])
            if not valid_gt_idx.all():
                if self.training and torch.cuda.current_device() == 0:
                    invalid = indices_cpu[~valid_gt_idx]
                    print('[prepare_track_queries] invalid prev_target indices:', invalid.tolist(),
                          'max_gt_idx:', int(mapping_cpu.shape[0]))
                valid_gt_idx_dev = valid_gt_idx.to(device)
                prev_out_ind_filtered = prev_out_ind_filtered[valid_gt_idx_dev]
                prev_target_ind_filtered = prev_target_ind_filtered[valid_gt_idx_dev]
                indices_cpu = indices_cpu[valid_gt_idx]
            target_prev2cur_cpu = mapping_cpu[indices_cpu]
            max_gt = target['lines'].shape[0]
            valid_range = (target_prev2cur_cpu >= 0) & (target_prev2cur_cpu < max_gt)
            target_prev2cur_cpu[~valid_range] = -1
            target_prev2cur = target_prev2cur_cpu.to(device)
            target['prev_target_ind'] = prev_target_ind_filtered
            target['prev_out_ind'] = prev_out_ind_filtered
            target['gt_prev2cur'] = target_prev2cur

            # Filter any queries whose indices fall outside the prediction tensor
            valid_query_mask = prev_out_ind_filtered < scores.shape[0]
            if not valid_query_mask.all():
                prev_out_ind_filtered = prev_out_ind_filtered[valid_query_mask]
                prev_target_ind_filtered = prev_target_ind_filtered[valid_query_mask]
                target_prev2cur = target_prev2cur[valid_query_mask.cpu()]

            # Clamp the mapped indices to valid GT range
            target_ind_matching = (target_prev2cur != -1) # -1 means no matching g.t. in curr frame
            # matched g.t. index in the current frame
            target_ind_matched_idx = target_prev2cur[target_prev2cur!=-1]

            target['track_query_match_ids'] = target_ind_matched_idx
            
            if timestep == 0:
                pad_bound = self.head.num_queries
            else:
                pad_bound = self.tracked_query_length[b_i] + self.head.num_queries
                
            all_indices = torch.arange(prev_out['lines'][b_i].shape[0], device=device)
            valid_mask = all_indices < pad_bound
            valid_mask[prev_out_ind] = False
            not_prev_out_ind = all_indices[valid_mask]

            # Get all non-matched pred with >0.5 conf score, serve as FP
            neg_scores = scores[not_prev_out_ind]
            neg_score_mask = neg_scores >= pos_th
            # Randomly pick 10% neg output instances and serve as FP
            _rand_insert = torch.rand([len(neg_scores)], device=device)

            if self.track_fp_aug:
                rand_insert_mask = _rand_insert >= 0.95
                fp_select_mask = neg_score_mask | rand_insert_mask
            else:
                fp_select_mask = neg_score_mask

            false_out_ind = not_prev_out_ind[fp_select_mask]

            prev_out_ind_final = torch.cat([prev_out_ind_filtered, false_out_ind]).long()
            target_ind_matching = torch.cat([
                target_ind_matching,
                torch.zeros(len(false_out_ind), dtype=torch.bool, device=device)
            ])

            target_prev2cur_aug = torch.cat([
                target_prev2cur,
                torch.full((len(false_out_ind),), -1, device=device, dtype=target_prev2cur.dtype)
            ])
            target['track_to_cur_gt_ids'] = target_prev2cur_aug

            # track query masks
            track_queries_mask = torch.ones_like(target_ind_matching).bool()
            track_queries_fal_pos_mask = torch.zeros_like(target_ind_matching).bool()
            track_queries_fal_pos_mask[~target_ind_matching] = True

            # set prev frame info
            target['track_query_hs_embeds'] = prev_out['hs_embeds'][b_i, prev_out_ind_final]
            target['track_query_boxes'] = prev_out['lines'][b_i][prev_out_ind_final].detach()
            tmp_labels = labels[prev_out_ind_final]
            tmp_scores = scores[prev_out_ind_final]
            target['track_query_labels'] = tmp_labels
            target['track_query_scores'] = tmp_scores

            # Prepare the G.T. line coords for the track queries, used in the transformation loss
            prev_gt_lines = prev_gt_list['lines'][b_i] 
            prev_gt_labels = prev_gt_list['labels'][b_i] 
            target['track_query_gt_lines'] = prev_gt_lines[prev_out_ind_final]
            target['track_query_gt_labels'] = prev_gt_labels[prev_out_ind_final]

            target['track_queries_mask'] = torch.cat([
                track_queries_mask,
                torch.tensor([False, ] * self.head.num_queries).to(device)
            ]).bool()

            target['track_queries_fal_pos_mask'] = torch.cat([
                track_queries_fal_pos_mask,
                torch.tensor([False, ] * self.head.num_queries).to(device)
            ]).bool()

            if use_memory:
                is_first_frame = (timestep == 0)
                num_tracks = 0 if timestep == 0 else self.tracked_query_length[b_i]
                self.memory_bank.update_memory(b_i, is_first_frame, prev_out_ind_final, prev_out, num_tracks, scene_seq_id, timestep)
        
        targets = self._batchify_tracks(targets)
        return targets
    
    def _batchify_tracks(self, targets):
        lengths = [len(t['track_queries_mask']) for t in targets]
        max_len = max(lengths)
        device = targets[0]['track_query_hs_embeds'].device
        for b_i in range(len(lengths)):
            target = targets[b_i]
            padding_len = max_len - lengths[b_i]
            pad_hs_embeds = torch.zeros([padding_len, target['track_query_hs_embeds'].shape[1]]).to(device)
            pad_query_boxes = torch.zeros([padding_len, target['track_query_boxes'].shape[1]]).to(device)
            query_padding_mask = torch.zeros([max_len]).bool().to(device)
            query_padding_mask[lengths[b_i]:] = True
            target['pad_hs_embeds'] = pad_hs_embeds
            target['pad_query_boxes'] = pad_query_boxes
            target['query_padding_mask'] = query_padding_mask
            self.tracked_query_length[b_i] = lengths[b_i] - self.head.num_queries
        return targets
        
    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if self.freeze_bev:
            self._freeze_bev()
        elif self.freeze_bev_iters is not None and self.num_iter < self.freeze_bev_iters:
            self._freeze_bev()
        else:
            self._unfreeze_bev()

    def eval(self):
        super().eval()
        
    def _freeze_bev(self,):
        """Freeze all bev-related backbone parameters, including the backbone and the seg head
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.seg_decoder.parameters():
            param.requires_grad = False
    
    def _unfreeze_bev(self,):
        """unfreeze all bev-related backbone parameters, including the backbone and the seg head
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.seg_decoder.parameters():
            param.requires_grad = True
    
    def _denorm_lines(self, line_pts):
        """from (0,1) to the BEV space in meters"""
        line_pts[..., 0] = line_pts[..., 0] * self.roi_size[0] \
                        - self.roi_size[0] / 2 
        line_pts[..., 1] = line_pts[..., 1] * self.roi_size[1] \
                        - self.roi_size[1] / 2 
        return line_pts

    def _norm_lines(self, line_pts):
        """from the BEV space in meters to (0,1) """
        line_pts[..., 0] = (line_pts[..., 0] + self.roi_size[0] / 2) \
                                        / self.roi_size[0] 
        line_pts[..., 1] = (line_pts[..., 1] + self.roi_size[1] / 2) \
                                        / self.roi_size[1] 
        return line_pts

    def _process_track_query_info(self, track_info):
        bs = len(track_info)
        all_scores = []
        all_lines = []
        for b_i in range(bs):
            embeds = track_info[b_i]['track_query_hs_embeds']
            scores = self.head.cls_branches[-1](embeds)
            coords = self.head.reg_branches[-1](embeds).sigmoid()
            coords = rearrange(coords, 'n1 (n2 n3) -> n1 n2 n3', n3=2)
            all_scores.append(scores)
            all_lines.append(coords)
        track_results = {
            'lines': all_lines,
            'scores': all_scores,
        }
        return track_results
    
    def select_memory_entries(self, history_metas, curr_meta):
        """
        Only used at test time, to select a subset from the long history bank
        """
        if len(history_metas) <= self.history_steps:
            return np.arange(len(history_metas))
        else:
            history_e2g_trans = np.array([item[0]['ego2global_translation'] for item in history_metas])[:, :2]
            curr_e2g_trans = np.array(curr_meta[0]['ego2global_translation'])[:2]
            dists = np.linalg.norm(history_e2g_trans - curr_e2g_trans[None, :], axis=1)

            sorted_indices = np.argsort(dists)
            sorted_dists = dists[sorted_indices]
            covered = np.zeros_like(sorted_indices).astype(np.bool_)
            selected_ids = []
            for dist_range in self.mem_select_dist_ranges[::-1]:
                outter_valid_flags = (sorted_dists >= dist_range) & ~covered
                if outter_valid_flags.any():
                    pick_id = np.where(outter_valid_flags)[0][0]     
                    covered[pick_id:] = True
                else:
                    inner_valid_flags = (sorted_dists < dist_range) & ~covered
                    if inner_valid_flags.any():
                        pick_id = np.where(inner_valid_flags)[0][-1]
                        covered[pick_id] = True
                    else:
                        return np.arange(len(history_metas))[-4:]
                selected_ids.append(pick_id)

            selected_mem_ids = sorted_indices[np.array(selected_ids)]

            return selected_mem_ids

    #####################################################################
    # 
    # Debugging visualization of the temporal propagation supervision
    # 
    ##################################################################### 

    def _viz_temporal_supervision(self, outputs_prev, all_track_info, gts, gts_prev, semantic_mask, 
                                  semantic_mask_prev, img_metas, img_metas_prev, timestep):
        """For debugging use: draw the visualization of the track queries and the corresponding
        matched G.T. information..."""
        import os
        from ..utils.renderer_track import Renderer
        viz_dir = './viz/debug_noisy_trans'
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        cat2id = {
            'ped_crossing': 0,
            'divider': 1,
            'boundary': 2,
        }
        renderer = Renderer(cat2id, self.roi_size, 'nusc')

        for b_i in range(len(all_track_info)):
            track_info = all_track_info[b_i]
            # prev pred info
            prev_pred_lines = outputs_prev['lines'][b_i]
            prev_pred_scores = outputs_prev['scores'][b_i]
            prev_target_inds = track_info['prev_target_ind']
            prev_out_inds = track_info['prev_out_ind']
            gt_prev2cur = track_info['gt_prev2cur']
            prev_scores, prev_labels = prev_pred_scores.max(-1)
            prev_scores = prev_scores.sigmoid()
            prev_lines = rearrange(prev_pred_lines[prev_out_inds], 'n (k c) -> n k c', c=2)
            prev_labels = prev_labels[prev_out_inds]
            prev_lines = self._denorm_lines(prev_lines)
            prev_scores = prev_scores[prev_out_inds]
            out_path_prev = os.path.join(viz_dir, f't={timestep}_{b_i}_prev.png')
            renderer.render_bev_from_vectors(prev_lines, prev_labels, out_path_prev, 
                id_info=prev_target_inds, score_info=prev_scores)

            # gt info
            gt_labels = gts['labels'][b_i]
            gt_lines = torch.clip(gts['lines'][b_i][:, 0], 0, 1)
            gt_lines = rearrange(gt_lines, 'n (k c) -> n k c', c=2)
            gt_lines = self._denorm_lines(gt_lines)
            out_path_gt = os.path.join(viz_dir, f't={timestep}_{b_i}_gt.png')
            gt_ids = np.arange(len(gt_lines))
            renderer.render_bev_from_vectors(gt_lines, gt_labels, out_path_gt, id_info=gt_ids)
            gt_semantic = semantic_mask[b_i].cpu().numpy()
            out_path_gt_semantic = os.path.join(viz_dir, f't={timestep}_{b_i}_gt_semantic.png')
            renderer.render_bev_from_mask(gt_semantic, out_path_gt_semantic)

            # gt info for prev frame
            gt_labels = gts_prev['labels'][b_i]
            gt_lines = torch.clip(gts_prev['lines'][b_i][:, 0], 0, 1)
            gt_lines = rearrange(gt_lines, 'n (k c) -> n k c', c=2)
            gt_lines = self._denorm_lines(gt_lines)
            out_path_gt = os.path.join(viz_dir, f't={timestep}_{b_i}_prev_gt.png')
            gt_ids = np.arange(len(gt_lines))
            renderer.render_bev_from_vectors(gt_lines, gt_labels, out_path_gt, id_info=gt_ids)
            gt_semantic = semantic_mask_prev[b_i].cpu().numpy()
            out_path_gt_semantic = os.path.join(viz_dir, f't={timestep}_{b_i}_prev_gt_semantic.png')
            renderer.render_bev_from_mask(gt_semantic, out_path_gt_semantic)

            # track query info
            track_to_cur_gt_ids = track_info['track_to_cur_gt_ids']
            trans_track_lines = track_info['trans_track_query_boxes']
            trans_track_lines = rearrange(trans_track_lines, 'n (k c) -> n k c', c=2)
            trans_track_lines = self._denorm_lines(trans_track_lines)
            #tp_track_mask = ~track_info['track_queries_fal_pos_mask'][:-100]
            trans_track_lines = trans_track_lines
            track_labels = track_info['track_query_labels']
            track_scores = track_info['track_query_scores']
            out_path_track = os.path.join(viz_dir, f't={timestep}_{b_i}_track.png')
            renderer.render_bev_from_vectors(trans_track_lines, track_labels, out_path_track, 
                id_info=track_to_cur_gt_ids, score_info=track_scores)

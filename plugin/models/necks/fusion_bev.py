import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet3d.registry import MODELS as MMDET3D_MODELS


@MMDET_MODELS.register_module()
@MMDET3D_MODELS.register_module()
class FusionBEV(nn.Module):
    """Cross-attention BEV fusion between camera and auxiliary modalities.

    The module consumes camera and auxiliary (LiDAR/aerial) BEV feature maps
    provided as separate tensors that share the same spatial resolution. The
    camera BEV queries the auxiliary BEV through multi-head attention and the
    refined signal is merged with the original stack via a lightweight
    residual block.

    Args:
        in_channels (int): ``cam_channels + aux_channels``.
        out_channels (int): Output channels, must match ``cam_channels``.
        cam_channels (int, optional): Number of channels in the camera BEV.
            Defaults to ``out_channels`` for backward compatibility.
        num_heads (int): Number of attention heads.
        attn_channels (int, optional): Hidden channels used inside attention.
            Defaults to ``out_channels // 2``.
        ffn_channels (int, optional): Hidden size of the FFN. Defaults to
            ``attn_channels * 4``.
        attn_dropout (float): Dropout applied inside the attention module.
        proj_dropout (float): Dropout for FFN projections.
        spatial_reduction (int): Factor to downsample auxiliary BEV before
            generating keys/values. ``1`` keeps the original resolution.
        use_positional_encoding (bool): Whether to add learned positional
            encodings to queries and keys before attention.
        bev_h (int, optional): Camera BEV height when positional encoding is used.
        bev_w (int, optional): Camera BEV width when positional encoding is used.
        bidirectional (bool): Enable auxiliary-to-camera attention in addition to
            the default direction.
        share_proj (bool): Share query/key projections when `bidirectional` is True.
        share_attn (bool): Share attention weights between directions when
            `bidirectional` is True.
        reverse_spatial_reduction (int, optional): Downsampling factor for the
            reverse attention pass. Defaults to ``spatial_reduction``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        cam_channels=None,
        num_heads=8,
        attn_channels=None,
        ffn_channels=None,
        attn_dropout=0.0,
        proj_dropout=0.0,
        spatial_reduction=1,
        use_positional_encoding=False,
        bev_h=None,
        bev_w=None,
        bidirectional=True,
        share_proj=False,
        share_attn=False,
        reverse_spatial_reduction=None,
        gate_cfg=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if cam_channels is None:
            cam_channels = out_channels
        if cam_channels != out_channels:
            raise ValueError('out_channels must equal cam_channels for FusionBEV.')
        if in_channels < cam_channels:
            raise ValueError('in_channels must be >= cam_channels for FusionBEV.')

        self.cam_channels = cam_channels
        self.aux_channels = in_channels - cam_channels
        self.num_heads = num_heads
        self.spatial_reduction = max(spatial_reduction, 1)
        self.use_positional_encoding = use_positional_encoding
        self.bev_h = bev_h
        self.bev_w = bev_w
        if reverse_spatial_reduction is None:
            reverse_spatial_reduction = spatial_reduction
        self.reverse_spatial_reduction = max(reverse_spatial_reduction, 1)

        self.bidirectional = bool(bidirectional and self.aux_channels > 0)
        self.share_proj = bool(share_proj and self.bidirectional)
        self.share_attn = bool(share_attn and self.bidirectional)

        if attn_channels is None:
            attn_channels = max(out_channels // 2, num_heads)
        if attn_channels % num_heads != 0:
            raise ValueError(
                f'attn_channels ({attn_channels}) must be divisible by '
                f'num_heads ({num_heads}).')
        self.attn_channels = attn_channels

        if ffn_channels is None:
            ffn_channels = attn_channels * 4

        self.q_proj = nn.Conv2d(self.cam_channels, attn_channels, kernel_size=1, bias=False)
        if self.aux_channels > 0:
            self.k_proj = nn.Conv2d(self.aux_channels, attn_channels, kernel_size=1, bias=False)
            self.v_proj = nn.Conv2d(self.aux_channels, attn_channels, kernel_size=1, bias=False)
        else:
            self.k_proj = None
            self.v_proj = None

        self.attn = nn.MultiheadAttention(
            embed_dim=attn_channels,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(attn_channels)
        self.attn_out_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(attn_channels, ffn_channels),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(ffn_channels, attn_channels),
            nn.Dropout(proj_dropout),
        )
        self.ffn_norm = nn.LayerNorm(attn_channels)
        self.ffn_out_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        self.out_proj = nn.Conv2d(attn_channels, self.cam_channels, kernel_size=1, bias=False)

        if self.bidirectional:
            if self.share_proj:
                if self.k_proj is None:
                    raise ValueError('Cannot share projections without auxiliary channels.')
                self.aux_q_proj = self.k_proj
                self.cam_k_proj = self.q_proj
            else:
                self.aux_q_proj = nn.Conv2d(self.aux_channels, attn_channels, kernel_size=1, bias=False)
                self.cam_k_proj = nn.Conv2d(self.cam_channels, attn_channels, kernel_size=1, bias=False)
            self.cam_v_proj = nn.Conv2d(self.cam_channels, attn_channels, kernel_size=1, bias=False)
            if self.share_attn:
                self.reverse_attn = self.attn
            else:
                self.reverse_attn = nn.MultiheadAttention(
                    embed_dim=attn_channels,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                )
            self.rev_attn_norm = nn.LayerNorm(attn_channels)
            self.rev_attn_out_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
            self.rev_ffn = nn.Sequential(
                nn.Linear(attn_channels, ffn_channels),
                nn.GELU(),
                nn.Dropout(proj_dropout),
                nn.Linear(ffn_channels, attn_channels),
                nn.Dropout(proj_dropout),
            )
            self.rev_ffn_norm = nn.LayerNorm(attn_channels)
            self.rev_ffn_out_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()
            self.aux_out_proj = nn.Conv2d(attn_channels, self.aux_channels, kernel_size=1, bias=False)
        else:
            self.aux_q_proj = None
            self.cam_k_proj = None
            self.cam_v_proj = None
            self.reverse_attn = None
            self.rev_attn_norm = None
            self.rev_attn_out_dropout = None
            self.rev_ffn = None
            self.rev_ffn_norm = None
            self.rev_ffn_out_dropout = None
            self.aux_out_proj = None

        if self.use_positional_encoding:
            if self.bev_h is None or self.bev_w is None:
                raise ValueError('bev_h and bev_w must be provided when using positional encoding.')
            cam_tokens = self.bev_h * self.bev_w
            aux_h = max(self.bev_h // self.spatial_reduction, 1)
            aux_w = max(self.bev_w // self.spatial_reduction, 1)
            aux_tokens = aux_h * aux_w

            self.cam_positional_embedding = nn.Embedding(cam_tokens, attn_channels)
            self.aux_positional_embedding = nn.Embedding(aux_tokens, attn_channels)
            self.register_buffer('cam_pos_indices', torch.arange(cam_tokens, dtype=torch.long), persistent=False)
            self.register_buffer('aux_pos_indices', torch.arange(aux_tokens, dtype=torch.long), persistent=False)
            self.aux_h = aux_h
            self.aux_w = aux_w

            if self.bidirectional:
                self.aux_query_positional_embedding = nn.Embedding(cam_tokens, attn_channels)
                self.register_buffer('aux_query_pos_indices',
                                     torch.arange(cam_tokens, dtype=torch.long),
                                     persistent=False)
                rev_cam_h = max(self.bev_h // self.reverse_spatial_reduction, 1)
                rev_cam_w = max(self.bev_w // self.reverse_spatial_reduction, 1)
                rev_cam_tokens = rev_cam_h * rev_cam_w
                self.cam_key_positional_embedding = nn.Embedding(rev_cam_tokens, attn_channels)
                self.register_buffer('cam_key_pos_indices',
                                     torch.arange(rev_cam_tokens, dtype=torch.long),
                                     persistent=False)
                self.rev_cam_h = rev_cam_h
                self.rev_cam_w = rev_cam_w
            else:
                self.aux_query_positional_embedding = None
                self.cam_key_positional_embedding = None
                self.aux_query_pos_indices = None
                self.cam_key_pos_indices = None
                self.rev_cam_h = None
                self.rev_cam_w = None
        else:
            self.cam_positional_embedding = None
            self.aux_positional_embedding = None
            self.aux_query_positional_embedding = None
            self.cam_key_positional_embedding = None
            self.aux_query_pos_indices = None
            self.cam_key_pos_indices = None
            self.aux_h = None
            self.aux_w = None
            self.rev_cam_h = None
            self.rev_cam_w = None

        # Optional reliability-aware spatial gate using branch confidences:
        # - pre_attn: aux is spatially modulated before participating in
        #   attention and subsequent fusion.
        # - post_fuse_blend: legacy post-hoc blend between fused and camera
        #   branches.
        self.gate_enabled = False
        self.gate_apply_mode = 'post_fuse_blend'
        self.gate_in_mode = 'cam_aux_absdiff'
        self.gate_hidden_channels = 64
        self.gate_proj_channels = 64
        self.gate_normalize_inputs = False
        self.gate_lambda = 0.0
        self.gate_entropy_lambda = 0.0
        self.gate_entropy_eps = 1e-6
        self.gate_log_stats = True
        self.gate_init_prob = 0.2
        self.gate_temperature = 1.0
        self.gate_masked_lambda = 0.0
        self.gate_cam_proj = None
        self.gate_aux_proj = None
        self.gate_net = None
        self.gate_cam_conf_head = None
        self.gate_aux_conf_head = None
        self.gate_schedule_enabled = False
        self.gate_schedule_start_iter = 8000
        self.gate_schedule_end_iter = 25000
        self.gate_schedule_lambda_masked_target = 0.02
        self.gate_schedule_lambda_gate_after = 1e-4
        self.gate_schedule_lambda_entropy_after = 0.0
        if gate_cfg is not None and gate_cfg.get('enabled', False):
            self.gate_enabled = True
            self.gate_apply_mode = str(gate_cfg.get('apply_mode', 'post_fuse_blend'))
            self.gate_in_mode = str(gate_cfg.get('in_mode', 'cam_aux_absdiff'))
            self.gate_hidden_channels = int(gate_cfg.get('hidden_channels', 64))
            self.gate_proj_channels = int(gate_cfg.get('proj_channels', self.gate_hidden_channels))
            self.gate_normalize_inputs = bool(gate_cfg.get('normalize_inputs', False))
            self.gate_lambda = float(gate_cfg.get('lambda_gate', 1e-3))
            self.gate_entropy_lambda = float(gate_cfg.get('lambda_entropy', 0.0))
            self.gate_entropy_eps = float(gate_cfg.get('entropy_eps', 1e-6))
            self.gate_log_stats = bool(gate_cfg.get('log_stats', True))
            self.gate_init_prob = float(gate_cfg.get('init_prob', 0.2))
            self.gate_temperature = float(gate_cfg.get('temperature', 1.0))
            self.gate_masked_lambda = float(gate_cfg.get('lambda_masked_gate', 0.0))
            if self.gate_apply_mode == 'post_fuse':
                # Backward-compatible alias.
                self.gate_apply_mode = 'post_fuse_blend'
            if not (0.0 < self.gate_init_prob < 1.0):
                raise ValueError('gate init_prob must be within (0, 1).')
            if self.gate_temperature <= 0:
                raise ValueError('gate temperature must be > 0.')
            if self.gate_hidden_channels < 1:
                raise ValueError('gate hidden_channels must be >= 1.')
            if self.gate_proj_channels < 1:
                raise ValueError('gate proj_channels must be >= 1.')
            if self.gate_lambda < 0:
                raise ValueError('gate lambda_gate must be >= 0.')
            if self.gate_entropy_lambda < 0:
                raise ValueError('gate lambda_entropy must be >= 0.')
            if self.gate_masked_lambda < 0:
                raise ValueError('gate lambda_masked_gate must be >= 0.')
            if self.gate_entropy_eps <= 0:
                raise ValueError('gate entropy_eps must be > 0.')
            if self.aux_channels <= 0:
                raise ValueError('gate_cfg requires auxiliary channels > 0.')
            if self.gate_apply_mode not in ('pre_attn', 'post_fuse_blend'):
                raise ValueError("gate apply_mode must be one of {'pre_attn', 'post_fuse_blend'}.")

            if self.gate_in_mode == 'confidence_ratio':
                self.gate_cam_conf_head = nn.Conv2d(self.cam_channels, 1, kernel_size=1, bias=True)
                self.gate_aux_conf_head = nn.Conv2d(self.aux_channels, 1, kernel_size=1, bias=True)
            elif self.gate_in_mode in ('cam_aux', 'cam_aux_absdiff'):
                # Align cam/aux into a dedicated gate feature space to make
                # abs-diff and local reliability cues more meaningful.
                self.gate_cam_proj = nn.Conv2d(self.cam_channels, self.gate_proj_channels, kernel_size=1, bias=False)
                self.gate_aux_proj = nn.Conv2d(self.aux_channels, self.gate_proj_channels, kernel_size=1, bias=False)
                gate_in_channels = self.gate_proj_channels * 2
                if self.gate_in_mode == 'cam_aux_absdiff':
                    gate_in_channels += self.gate_proj_channels
                self.gate_net = nn.Sequential(
                    nn.Conv2d(gate_in_channels, self.gate_hidden_channels, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.gate_hidden_channels, self.gate_hidden_channels, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.gate_hidden_channels, 1, kernel_size=1, bias=True),
                )
            else:
                raise ValueError(
                    "gate in_mode must be one of {'cam_aux', 'cam_aux_absdiff', 'confidence_ratio'}.")

            gate_schedule_cfg = gate_cfg.get('loss_schedule', None)
            if gate_schedule_cfg is not None and gate_schedule_cfg.get('enabled', False):
                self.gate_schedule_enabled = True
                self.gate_schedule_start_iter = int(gate_schedule_cfg.get('start_iter', 8000))
                self.gate_schedule_end_iter = int(gate_schedule_cfg.get('end_iter', 25000))
                self.gate_schedule_lambda_masked_target = float(
                    gate_schedule_cfg.get('lambda_masked_gate_target', 0.02))
                self.gate_schedule_lambda_gate_after = float(
                    gate_schedule_cfg.get('lambda_gate_after', 1e-4))
                self.gate_schedule_lambda_entropy_after = float(
                    gate_schedule_cfg.get('lambda_entropy_after', 0.0))
                if self.gate_schedule_start_iter < 0:
                    raise ValueError('gate loss_schedule start_iter must be >= 0.')
                if self.gate_schedule_end_iter <= self.gate_schedule_start_iter:
                    raise ValueError('gate loss_schedule end_iter must be > start_iter.')
                if self.gate_schedule_lambda_masked_target < 0:
                    raise ValueError('gate loss_schedule lambda_masked_gate_target must be >= 0.')
                if self.gate_schedule_lambda_gate_after < 0:
                    raise ValueError('gate loss_schedule lambda_gate_after must be >= 0.')
                if self.gate_schedule_lambda_entropy_after < 0:
                    raise ValueError('gate loss_schedule lambda_entropy_after must be >= 0.')

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _build_gate_map(self, cam, aux, force_camera_only=False):
        if self.gate_in_mode == 'confidence_ratio':
            cam_conf_logits = self.gate_cam_conf_head(cam)
            aux_conf_logits = self.gate_aux_conf_head(aux)
            conf_logits = torch.cat([cam_conf_logits, aux_conf_logits], dim=1) / self.gate_temperature
            gate_map = torch.softmax(conf_logits, dim=1)[:, 1:2]
        else:
            cam_gate = self.gate_cam_proj(cam)
            aux_gate = self.gate_aux_proj(aux)
            if self.gate_normalize_inputs:
                cam_gate = F.normalize(cam_gate, dim=1)
                aux_gate = F.normalize(aux_gate, dim=1)
            gate_input = [cam_gate, aux_gate]
            if self.gate_in_mode == 'cam_aux_absdiff':
                gate_input.append((cam_gate - aux_gate).abs())
            gate_logits = self.gate_net(torch.cat(gate_input, dim=1)) / self.gate_temperature
            gate_map = torch.sigmoid(gate_logits)
        # Keep the gate heads in the autograd graph even when forcing camera-only.
        if force_camera_only:
            gate_map = gate_map * 0.0
        # g is the auxiliary branch weight in [0, 1], shape [B, 1, H, W].
        return gate_map

    def _get_gate_loss_weights(self, current_iter=None):
        lambda_gate = self.gate_lambda
        lambda_entropy = self.gate_entropy_lambda
        lambda_masked = self.gate_masked_lambda

        if (not self.gate_schedule_enabled) or (current_iter is None):
            return lambda_gate, lambda_entropy, lambda_masked

        curr_iter = int(current_iter)
        if curr_iter <= self.gate_schedule_start_iter:
            return 0.0, 0.0, 0.0
        if curr_iter < self.gate_schedule_end_iter:
            ratio = ((curr_iter - self.gate_schedule_start_iter)
                     / float(self.gate_schedule_end_iter - self.gate_schedule_start_iter))
            return 0.0, 0.0, self.gate_schedule_lambda_masked_target * ratio
        return (
            self.gate_schedule_lambda_gate_after,
            self.gate_schedule_lambda_entropy_after,
            self.gate_schedule_lambda_masked_target,
        )

    def _build_gate_info(self, gate_map, aerial_drop_mask=None, force_camera_only=False, current_iter=None):
        lambda_gate, lambda_entropy, lambda_masked = self._get_gate_loss_weights(current_iter=current_iter)
        gate_sample_mean_raw = gate_map.mean(dim=(1, 2, 3))
        gate_sample_mean = gate_sample_mean_raw.detach()
        gate_mean = gate_map.mean()
        gate_mean_reg = gate_mean * lambda_gate if lambda_gate > 0 else gate_mean.new_tensor(0.0)

        gate_entropy = -(
            gate_map.clamp_min(self.gate_entropy_eps).log() * gate_map
            + (1 - gate_map).clamp_min(self.gate_entropy_eps).log() * (1 - gate_map)
        ).mean()
        gate_entropy_penalty = (gate_mean.new_tensor(1.0) - gate_entropy / math.log(2.0)).clamp(0.0, 1.0)
        gate_entropy_reg = (
            gate_entropy_penalty * lambda_entropy
            if lambda_entropy > 0 else gate_mean.new_tensor(0.0)
        )
        gate_masked_mean = gate_mean.new_tensor(0.0)
        gate_unmasked_mean = gate_mean.detach()
        gate_masked_reg = gate_mean.new_tensor(0.0)
        if aerial_drop_mask is not None:
            if aerial_drop_mask.shape != gate_map.shape:
                raise ValueError(
                    f'aerial_drop_mask shape {tuple(aerial_drop_mask.shape)} '
                    f'must match gate shape {tuple(gate_map.shape)}.')
            mask = aerial_drop_mask.float()
            mask_sum = mask.sum()
            if mask_sum.item() > 0:
                gate_masked_mean = (gate_map * mask).sum() / mask_sum
                if lambda_masked > 0:
                    gate_masked_reg = gate_masked_mean * lambda_masked
            unmask = 1.0 - mask
            unmask_sum = unmask.sum()
            if unmask_sum.item() > 0:
                gate_unmasked_mean = ((gate_map * unmask).sum() / unmask_sum).detach()
            else:
                gate_unmasked_mean = gate_mean.detach()

        if force_camera_only:
            gate_mean_reg = gate_mean.new_tensor(0.0)
            gate_entropy_reg = gate_mean.new_tensor(0.0)
            gate_masked_reg = gate_mean.new_tensor(0.0)

        gate_reg = gate_mean_reg + gate_entropy_reg + gate_masked_reg
        gate_p90 = gate_map.new_tensor(0.0)
        if self.gate_log_stats:
            gate_p90 = torch.quantile(gate_map.detach().flatten(), 0.9)
        return dict(
            gate_mean=gate_mean.detach(),
            gate_sample_mean_raw=gate_sample_mean_raw,
            gate_sample_mean=gate_sample_mean,
            gate_p90=gate_p90.detach(),
            gate_reg=gate_reg,
            gate_mean_reg=gate_mean_reg.detach(),
            gate_entropy=gate_entropy.detach(),
            gate_entropy_penalty=gate_entropy_penalty.detach(),
            gate_entropy_reg=gate_entropy_reg.detach(),
            gate_masked_mean=gate_masked_mean.detach(),
            gate_unmasked_mean=gate_unmasked_mean,
            gate_masked_reg=gate_masked_reg.detach(),
            gate_lambda_eff=gate_mean.new_tensor(lambda_gate).detach(),
            gate_entropy_lambda_eff=gate_mean.new_tensor(lambda_entropy).detach(),
            gate_masked_lambda_eff=gate_mean.new_tensor(lambda_masked).detach(),
            gate_applied=True,
        )

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        if self.gate_enabled:
            if self.gate_in_mode == 'confidence_ratio':
                if self.gate_cam_conf_head is not None:
                    nn.init.constant_(self.gate_cam_conf_head.weight, 0.0)
                    if self.gate_cam_conf_head.bias is not None:
                        nn.init.constant_(self.gate_cam_conf_head.bias, math.log(1.0 - self.gate_init_prob))
                if self.gate_aux_conf_head is not None:
                    nn.init.constant_(self.gate_aux_conf_head.weight, 0.0)
                    if self.gate_aux_conf_head.bias is not None:
                        nn.init.constant_(self.gate_aux_conf_head.bias, math.log(self.gate_init_prob))
            elif self.gate_net is not None:
                # Keep hidden gate convs at default/Kaiming init so spatial
                # variation can emerge early; only pin the final logit layer
                # to the requested prior gate probability.
                final_layer = self.gate_net[-1]
                nn.init.constant_(final_layer.weight, 0.0)
                init_logit = math.log(self.gate_init_prob / (1.0 - self.gate_init_prob))
                nn.init.constant_(final_layer.bias, init_logit)

    def forward(self,
                cam,
                aux=None,
                return_gate_info=False,
                force_camera_only=False,
                aerial_drop_mask=None,
                current_iter=None):
        """Fuse camera and auxiliary BEV tensors.

        Args:
            cam (Tensor): Camera BEV tensor of shape ``(B, cam_channels, H, W)``.
            aux (Tensor, optional): Auxiliary BEV tensor of shape
                ``(B, aux_channels, H, W)``. Required when `aux_channels > 0`.
            return_gate_info (bool): Whether to return gate statistics and
                regularization value.
            force_camera_only (bool): Force gate to zero map when gating is
                enabled, suppressing auxiliary contribution.
            aerial_drop_mask (Tensor, optional): Binary mask of shape
                ``(B, 1, H, W)`` marking spatial aerial-dropout cells. When
                provided, and ``gate_cfg.lambda_masked_gate > 0``, gate values
                on masked cells are penalized.
            current_iter (int, optional): Global training iteration used by
                gate loss schedules.

        Returns:
            Tensor | Tuple[Tensor, Dict[str, Tensor]]: Fused BEV tensor and
            optional gate metadata.
        """
        if cam.shape[1] != self.cam_channels:
            raise ValueError(
                f'Camera tensor has {cam.shape[1]} channels, expected {self.cam_channels}.')

        gate_info = None
        if self.aux_channels == 0:
            if aux is not None:
                raise ValueError('FusionBEV configured without auxiliary channels but aux tensor provided.')
            output = self.block(cam)
            if return_gate_info:
                zero = output.new_tensor(0.0)
                zero_vec = output.new_zeros((output.shape[0],))
                gate_info = dict(
                    gate_mean=zero,
                    gate_sample_mean_raw=zero_vec,
                    gate_sample_mean=zero_vec,
                    gate_p90=zero,
                    gate_reg=zero,
                    gate_mean_reg=zero,
                    gate_entropy=zero,
                    gate_entropy_penalty=zero,
                    gate_entropy_reg=zero,
                    gate_masked_mean=zero,
                    gate_unmasked_mean=zero,
                    gate_masked_reg=zero,
                    gate_lambda_eff=zero,
                    gate_entropy_lambda_eff=zero,
                    gate_masked_lambda_eff=zero,
                    gate_applied=False,
                )
                return output, gate_info
            return output

        if aux is None:
            raise ValueError('Auxiliary tensor required when FusionBEV has auxiliary channels.')
        if aux.shape[1] != self.aux_channels:
            raise ValueError(
                f'Auxiliary tensor has {aux.shape[1]} channels, expected {self.aux_channels}.')
        if cam.shape[-2:] != aux.shape[-2:]:
            raise RuntimeError('Camera and auxiliary BEV tensors must share the same spatial resolution.')

        batch_size, _, height, width = cam.shape
        gate_map = None
        aux_input = aux
        if self.gate_enabled:
            gate_map = self._build_gate_map(
                cam=cam,
                aux=aux,
                force_camera_only=force_camera_only,
            )
            if self.gate_apply_mode == 'pre_attn':
                aux_input = aux * gate_map

        q = self.q_proj(cam)
        if self.spatial_reduction > 1:
            target_h = max(height // self.spatial_reduction, 1)
            target_w = max(width // self.spatial_reduction, 1)
            aux_kv = F.adaptive_avg_pool2d(aux_input, (target_h, target_w))
        else:
            aux_kv = aux_input
        k = self.k_proj(aux_kv)
        v = self.v_proj(aux_kv)

        q_tokens = q.flatten(2).transpose(1, 2)
        k_tokens = k.flatten(2).transpose(1, 2)
        v_tokens = v.flatten(2).transpose(1, 2)

        if self.use_positional_encoding:
            if height != self.bev_h or width != self.bev_w:
                raise ValueError(
                    f'Camera BEV spatial size {(height, width)} does not match configured {(self.bev_h, self.bev_w)}.')
            cam_pos = self.cam_positional_embedding(self.cam_pos_indices).unsqueeze(0).to(
                device=q_tokens.device, dtype=q_tokens.dtype)
            q_tokens = q_tokens + cam_pos

            aux_h, aux_w = aux_kv.shape[-2:]
            if aux_h != self.aux_h or aux_w != self.aux_w:
                raise ValueError(
                    f'Aux BEV spatial size {(aux_h, aux_w)} does not match configured {(self.aux_h, self.aux_w)}.')
            aux_pos = self.aux_positional_embedding(self.aux_pos_indices).unsqueeze(0).to(
                device=k_tokens.device, dtype=k_tokens.dtype)
            k_tokens = k_tokens + aux_pos

        q_norm = self.attn_norm(q_tokens)
        attn_out, _ = self.attn(q_norm, k_tokens, v_tokens)
        attn_out = self.attn_out_dropout(attn_out)
        attn_out = attn_out + q_tokens

        ffn_input = self.ffn_norm(attn_out)
        ffn_out = self.ffn(ffn_input)
        ffn_out = self.ffn_out_dropout(ffn_out)
        ffn_out = ffn_out + attn_out

        fused = ffn_out.transpose(1, 2).reshape(batch_size, self.attn_channels, height, width)
        cam_refined = self.out_proj(fused) + cam

        aux_refined = aux_input
        if self.bidirectional:
            aux_queries = self.aux_q_proj(aux_input)
            aux_q_tokens = aux_queries.flatten(2).transpose(1, 2)
            if self.use_positional_encoding and self.aux_query_positional_embedding is not None:
                aux_query_pos = self.aux_query_positional_embedding(self.aux_query_pos_indices).unsqueeze(0).to(
                    device=aux_q_tokens.device, dtype=aux_q_tokens.dtype)
                aux_q_tokens = aux_q_tokens + aux_query_pos

            cam_kv = cam_refined
            if self.reverse_spatial_reduction > 1:
                rev_h = max(height // self.reverse_spatial_reduction, 1)
                rev_w = max(width // self.reverse_spatial_reduction, 1)
                cam_kv = F.adaptive_avg_pool2d(cam_kv, (rev_h, rev_w))
            else:
                rev_h, rev_w = height, width
            cam_k = self.cam_k_proj(cam_kv)
            cam_v = self.cam_v_proj(cam_kv)
            cam_k_tokens = cam_k.flatten(2).transpose(1, 2)
            cam_v_tokens = cam_v.flatten(2).transpose(1, 2)

            if self.use_positional_encoding and self.cam_key_positional_embedding is not None:
                if self.rev_cam_h is not None and self.rev_cam_w is not None:
                    if rev_h != self.rev_cam_h or rev_w != self.rev_cam_w:
                        raise ValueError(
                            f'Camera BEV size {(rev_h, rev_w)} does not match configured '
                            f'{(self.rev_cam_h, self.rev_cam_w)} for reverse_spatial_reduction={self.reverse_spatial_reduction}.')
                cam_key_pos = self.cam_key_positional_embedding(self.cam_key_pos_indices).unsqueeze(0).to(
                    device=cam_k_tokens.device, dtype=cam_k_tokens.dtype)
                cam_k_tokens = cam_k_tokens + cam_key_pos

            aux_q_norm = self.rev_attn_norm(aux_q_tokens)
            reverse_attn_out, _ = self.reverse_attn(aux_q_norm, cam_k_tokens, cam_v_tokens)
            reverse_attn_out = self.rev_attn_out_dropout(reverse_attn_out)
            reverse_attn_out = reverse_attn_out + aux_q_tokens

            rev_ffn_input = self.rev_ffn_norm(reverse_attn_out)
            rev_ffn_out = self.rev_ffn(rev_ffn_input)
            rev_ffn_out = self.rev_ffn_out_dropout(rev_ffn_out)
            rev_ffn_out = rev_ffn_out + reverse_attn_out

            aux_fused = rev_ffn_out.transpose(1, 2).reshape(batch_size, self.attn_channels, height, width)
            aux_refined = self.aux_out_proj(aux_fused) + aux_input

        fused_stack = torch.cat([cam_refined, aux_refined], dim=1)
        if self.gate_enabled and self.gate_apply_mode == 'pre_attn':
            # Also gate auxiliary channels before the conv fusion block so the
            # block cannot trivially bypass pre-attention suppression.
            fused_stack = torch.cat([cam_refined, gate_map * aux_refined], dim=1)
        fused_output = self.block(fused_stack)

        if self.gate_enabled:
            if self.gate_apply_mode == 'post_fuse_blend':
                cam_branch = cam_refined
                aux_branch = fused_output
                output = gate_map * aux_branch + (1 - gate_map) * cam_branch
            else:
                output = fused_output
            gate_info = self._build_gate_info(
                gate_map=gate_map,
                aerial_drop_mask=aerial_drop_mask,
                force_camera_only=force_camera_only,
                current_iter=current_iter,
            )
        else:
            output = fused_output
            if return_gate_info:
                zero = output.new_tensor(0.0)
                zero_vec = output.new_zeros((output.shape[0],))
                gate_info = dict(
                    gate_mean=zero,
                    gate_sample_mean_raw=zero_vec,
                    gate_sample_mean=zero_vec,
                    gate_p90=zero,
                    gate_reg=zero,
                    gate_mean_reg=zero,
                    gate_entropy=zero,
                    gate_entropy_penalty=zero,
                    gate_entropy_reg=zero,
                    gate_masked_mean=zero,
                    gate_unmasked_mean=zero,
                    gate_masked_reg=zero,
                    gate_lambda_eff=zero,
                    gate_entropy_lambda_eff=zero,
                    gate_masked_lambda_eff=zero,
                    gate_applied=False,
                )

        if return_gate_info:
            return output, gate_info
        return output

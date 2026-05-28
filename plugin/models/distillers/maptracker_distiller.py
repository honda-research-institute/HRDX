import copy
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet3d.registry import MODELS as MMDET3D_MODELS

from ..mapers.base_mapper import BaseMapper, MAPPERS


def _maybe_load_config(cfg_spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve a model configuration dictionary from a dict or config path."""
    cfg_spec = copy.deepcopy(cfg_spec)
    config_path = cfg_spec.pop('config', None)
    model_cfg = cfg_spec.pop('model_cfg', None)

    if model_cfg is None and config_path is None:
        raise ValueError('Each teacher specification must provide either '
                         '"config" (path to config file) or "model_cfg".')

    if config_path is not None:
        cfg = Config.fromfile(config_path)
        model_cfg = cfg.model

    if model_cfg is None:
        raise ValueError('Unable to resolve teacher model configuration.')

    return model_cfg, cfg_spec


@MAPPERS.register_module()
class MapTrackerDistiller(BaseMapper):
    """Wrapper that distills BEV, segmentation, and head logits from teachers."""

    def __init__(
        self,
        student_cfg: Any,
        teacher_cfgs: List[Dict[str, Any]],
        distill_weight: float = 1.0,
        loss_reduction: str = 'mean',
        train_cfg: Optional[Dict[str, Any]] = None,
        test_cfg: Optional[Dict[str, Any]] = None,
        student_pretrained: Optional[str] = None,
        student_map_location: str = 'cpu',
        bev_kd_weight: float = 1.0,
        seg_kd_weight: float = 1.0,
        head_kd_weight: float = 0.0,
        head_cls_weight: float = 1.0,
        head_point_weight: float = 1.0,
        head_attr_weight: float = 1.0,
        head_attr_temperature: float = 2.0,
        **kwargs,
    ):
        super().__init__()

        if not teacher_cfgs:
            raise ValueError('teacher_cfgs must contain at least one teacher.')

        resolved_student_cfg = self._resolve_student_cfg(student_cfg)
        student_cfg_build = copy.deepcopy(resolved_student_cfg)
        if train_cfg is not None:
            student_cfg_build.setdefault('train_cfg', train_cfg)
        if test_cfg is not None:
            student_cfg_build.setdefault('test_cfg', test_cfg)
        self.student = MMDET3D_MODELS.build(student_cfg_build)
        if student_pretrained is not None:
            checkpoint_path = self._resolve_path(student_pretrained)
            ckpt_meta = load_checkpoint(
                self.student,
                checkpoint_path,
                map_location=student_map_location,
                strict=False,
                revise_keys=[(r'^module\.', ''), (r'^student\.', '')],
            )
            self.student_ckpt_meta = ckpt_meta
        else:
            self.student_ckpt_meta = None

        # Mirror the student's data preprocessor so the runner finds it.
        if hasattr(self.student, 'data_preprocessor'):
            self.data_preprocessor = self.student.data_preprocessor

        self.teacher_models = nn.ModuleList()
        self.teacher_meta: List[Dict[str, Any]] = []

        for idx, teacher_spec in enumerate(teacher_cfgs):
            model_cfg, extra_cfg = _maybe_load_config(teacher_spec)
            teacher_cfg_build = copy.deepcopy(model_cfg)
            teacher = MMDET3D_MODELS.build(teacher_cfg_build)

            checkpoint = extra_cfg.pop('checkpoint', None)
            map_location = extra_cfg.pop('map_location', 'cpu')
            ckpt_meta = None
            if checkpoint is not None:
                checkpoint_path = self._resolve_path(checkpoint)
                ckpt_meta = load_checkpoint(
                    teacher,
                    checkpoint_path,
                    map_location=map_location,
                    strict=False,
                )

            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False

            name = extra_cfg.pop('name', f'teacher{idx}')
            weight = float(extra_cfg.pop('weight', 1.0))
            evaluate = bool(extra_cfg.pop('evaluate', False))
            eval_dataloader = extra_cfg.pop('eval_dataloader', None)

            if extra_cfg:
                # Keep track of any remaining metadata for completeness.
                extra_meta = copy.deepcopy(extra_cfg)
            else:
                extra_meta = {}

            self.teacher_models.append(teacher)
            self.teacher_meta.append(
                dict(
                    name=name,
                    weight=weight,
                    checkpoint=checkpoint,
                    ckpt_meta=ckpt_meta,
                    evaluate=evaluate,
                    eval_dataloader=eval_dataloader,
                    extra=extra_meta,
                )
            )

        self.distill_weight = float(distill_weight)
        self.loss_reduction = loss_reduction
        self.bev_kd_weight = float(bev_kd_weight)
        self.seg_kd_weight = float(seg_kd_weight)
        self.head_kd_weight = float(head_kd_weight)
        self.head_cls_weight = float(head_cls_weight)
        self.head_point_weight = float(head_point_weight)
        self.head_attr_weight = float(head_attr_weight)
        self.head_attr_temperature = float(head_attr_temperature)

    @staticmethod
    def _resolve_path(path: str) -> str:
        if osp.isabs(path):
            candidate = path
        else:
            candidate = osp.abspath(osp.join(os.getcwd(), path))
        if not osp.exists(candidate):
            raise FileNotFoundError(f'Checkpoint path "{path}" could not be resolved.')
        return candidate

    def _resolve_student_cfg(self, student_cfg: Any) -> Dict[str, Any]:
        """Normalise the student configuration into a detector dict."""
        if isinstance(student_cfg, dict):
            return copy.deepcopy(student_cfg)

        if isinstance(student_cfg, Config):
            return copy.deepcopy(student_cfg.model)

        if isinstance(student_cfg, str):
            cfg_path = student_cfg
            if not osp.isabs(cfg_path):
                cfg_path = osp.abspath(osp.join(os.getcwd(), cfg_path))
            if not osp.exists(cfg_path):
                raise FileNotFoundError(f'Student config path "{student_cfg}" is invalid.')
            cfg = Config.fromfile(cfg_path)
            return copy.deepcopy(cfg.model)

        raise TypeError(
            'student_cfg must be a dict, Config, or path to a config file; '
            f'got type {type(student_cfg).__name__}.'
        )

    def train(self, mode: bool = True):
        """Ensure teachers stay in eval mode while student follows training mode."""
        super().train(mode)
        self.student.train(mode)
        for teacher in self.teacher_models:
            teacher.eval()
        return self

    @property
    def num_iter(self) -> Optional[int]:
        return getattr(self.student, 'num_iter', None)

    @num_iter.setter
    def num_iter(self, value: int) -> None:
        setattr(self.student, 'num_iter', value)
        for teacher in self.teacher_models:
            setattr(teacher, 'num_iter', value)

    def init_weights(self, pretrained: Optional[str] = None):
        self.student.init_weights(pretrained)

    def _unwrap_tensor(self, value):
        """Recursively unwrap common container types to obtain a tensor."""
        if torch.is_tensor(value):
            return value
        if hasattr(value, 'data'):
            return self._unwrap_tensor(value.data)
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            tensors = [self._unwrap_tensor(item) for item in value]
            tensors = [tensor for tensor in tensors if tensor is not None]
            if not tensors:
                return None
            try:
                return torch.stack(tensors, dim=0)
            except (RuntimeError, TypeError):
                return tensors[0]
        return None

    def _build_segmentation_mask(
        self,
        batch_data: Dict[str, Any],
        target_shape: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Construct a spatial mask highlighting GT map regions for distillation."""
        semantic_mask = batch_data.get('semantic_mask', None)
        if semantic_mask is None:
            return None

        semantic_mask = self._unwrap_tensor(semantic_mask)
        if semantic_mask is None or not torch.is_tensor(semantic_mask):
            return None

        if semantic_mask.dim() < 4:
            return None

        mask = semantic_mask.detach()
        # Semantic masks originate from OpenCV rasterisation (image coords, y down).
        # Flip vertically so they align with the BEV frame (y up) used by the model.
        mask = torch.flip(mask, dims=[-2])
        mask = (mask > 0).any(dim=1, keepdim=True).float()
        if mask.numel() == 0:
            return None

        if mask.shape[-2:] != target_shape:
            mask = F.interpolate(mask, size=target_shape, mode='nearest')

        return mask.to(device=device, dtype=dtype)

    def _bev_loss(
        self,
        student_bev: torch.Tensor,
        teacher_bev: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if spatial_mask is not None:
            if spatial_mask.shape[1] == 1:
                spatial_mask = spatial_mask.expand(-1, student_bev.shape[1], -1, -1)
            mse = F.mse_loss(student_bev, teacher_bev, reduction='none')
            weighted = mse * spatial_mask
            valid = spatial_mask.sum()
            if valid.item() <= 0:
                return student_bev.new_tensor(0.0)
            return weighted.sum() / valid

        if self.loss_reduction == 'sum':
            return F.mse_loss(student_bev, teacher_bev, reduction='sum')
        return F.mse_loss(student_bev, teacher_bev, reduction='mean')

    def _attr_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if student_logits is None or teacher_logits is None:
            return None

        if mask is not None:
            valid = mask.bool()
            if valid.ndim > 1:
                valid = valid.view(valid.size(0), -1).any(dim=1)
        else:
            valid = torch.ones(teacher_logits.shape[0], dtype=torch.bool, device=teacher_logits.device)

        if valid.sum() == 0:
            return None

        temperature = max(self.head_attr_temperature, 1e-5)
        teacher_prob = F.softmax(teacher_logits[valid] / temperature, dim=-1)
        student_log_prob = F.log_softmax(student_logits[valid] / temperature, dim=-1)
        kd = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')
        return kd * (temperature * temperature)

    def _map_head_kd_loss(
        self,
        student_preds: Dict[str, List[torch.Tensor]],
        teacher_preds: Dict[str, List[torch.Tensor]],
        teacher_pos_inds: List[Optional[torch.Tensor]],
        teacher_gt: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor], int]:
        """Compute classification, regression, and attribute KD losses."""
        if (
            student_preds is None
            or teacher_preds is None
            or teacher_pos_inds is None
            or not hasattr(self.student, 'head')
            or getattr(self.student, 'skip_vector_head', False)
        ):
            return None, None, {}, 0

        assigner = getattr(self.student.head, 'assigner', None)
        cls_loss_fn = getattr(self.student.head, 'loss_cls', None)
        reg_loss_fn = getattr(self.student.head, 'loss_reg', None)
        if assigner is None or cls_loss_fn is None or reg_loss_fn is None:
            return None, None, {}, 0

        attr_names: List[str] = getattr(self.student.head, 'attr_names', [])

        requires_permute = False
        reg_cost_module = getattr(getattr(assigner, 'cost', None), 'reg_cost', None)
        if reg_cost_module is not None:
            requires_permute = bool(getattr(reg_cost_module, 'permute', False))

        student_scores_list = student_preds.get('scores')
        student_lines_list = student_preds.get('lines')
        teacher_scores_list = teacher_preds.get('scores')
        teacher_lines_list = teacher_preds.get('lines')
        student_attr_dict = student_preds.get('attr', {}) if attr_names else {}
        teacher_attr_dict = teacher_preds.get('attr', {}) if attr_names else {}

        if (
            student_scores_list is None
            or student_lines_list is None
            or teacher_scores_list is None
            or teacher_lines_list is None
        ):
            return None, None, {}, 0

        batch_size = len(student_scores_list)
        if batch_size == 0:
            return None, None, {}, 0

        collected_student_scores: List[torch.Tensor] = []
        collected_teacher_labels: List[torch.Tensor] = []
        collected_student_lines: List[torch.Tensor] = []
        collected_teacher_lines: List[torch.Tensor] = []
        collected_student_attrs: Dict[str, List[torch.Tensor]] = {name: [] for name in attr_names}
        collected_teacher_attrs: Dict[str, List[torch.Tensor]] = {name: [] for name in attr_names}
        collected_teacher_attr_masks: Dict[str, List[torch.Tensor]] = {name: [] for name in attr_names}

        total_matches = 0

        for b_idx in range(batch_size):
            if b_idx >= len(teacher_pos_inds):
                continue
            pos_inds_b = teacher_pos_inds[b_idx]
            if pos_inds_b is None:
                continue

            if torch.is_tensor(pos_inds_b):
                if pos_inds_b.numel() == 0:
                    continue
                pos_inds_b = pos_inds_b.to(
                    device=student_scores_list[b_idx].device, dtype=torch.long)
            else:
                pos_inds_b = torch.tensor(
                    pos_inds_b,
                    device=student_scores_list[b_idx].device,
                    dtype=torch.long,
                )
                if pos_inds_b.numel() == 0:
                    continue

            teacher_scores_b = teacher_scores_list[b_idx]
            teacher_lines_b = teacher_lines_list[b_idx]
            student_scores_b = student_scores_list[b_idx]
            student_lines_b = student_lines_list[b_idx]

            teacher_scores_pos = teacher_scores_b[pos_inds_b]
            teacher_lines_pos = teacher_lines_b[pos_inds_b]

            if teacher_scores_pos.numel() == 0:
                continue

            if teacher_gt is not None:
                teacher_labels_all = teacher_gt.get('labels', None)
                if teacher_labels_all is not None and b_idx < len(teacher_labels_all):
                    teacher_labels_pos = teacher_labels_all[b_idx][pos_inds_b]
                else:
                    teacher_labels_pos = teacher_scores_pos.argmax(dim=-1)
            else:
                teacher_labels_pos = teacher_scores_pos.argmax(dim=-1)

            teacher_labels_pos = teacher_labels_pos.to(
                device=student_scores_b.device, dtype=torch.long)
            teacher_lines_pos = teacher_lines_pos.to(
                device=student_lines_b.device, dtype=student_lines_b.dtype)
            if requires_permute and teacher_lines_pos.dim() == 2:
                teacher_lines_pos = teacher_lines_pos.unsqueeze(1)

            with torch.no_grad():
                assign_result, gt_permute_idx, _ = assigner.assign(
                    preds=dict(lines=student_lines_b, scores=student_scores_b),
                    gts=dict(lines=teacher_lines_pos, labels=teacher_labels_pos),
                    track_info=None,
                    gt_bboxes_ignore=None,
                )

            assigned_gt_inds = getattr(assign_result, 'gt_inds', None)
            if assigned_gt_inds is None:
                continue

            assigned_gt_inds = assigned_gt_inds.to(student_scores_b.device)
            matched_mask = assigned_gt_inds > 0
            if matched_mask.sum() == 0:
                continue

            student_scores_matched = student_scores_b[matched_mask]
            student_lines_matched = student_lines_b[matched_mask]

            teacher_indices = assigned_gt_inds[matched_mask] - 1
            if requires_permute and gt_permute_idx is not None:
                gt_permute_idx = gt_permute_idx.to(student_scores_b.device)
                matched_permute = gt_permute_idx[matched_mask, teacher_indices]
                teacher_lines_matched = teacher_lines_pos[teacher_indices, matched_permute]
            else:
                teacher_lines_matched = teacher_lines_pos[teacher_indices]
            teacher_labels_matched = teacher_labels_pos[teacher_indices]

            collected_student_scores.append(student_scores_matched)
            collected_teacher_labels.append(teacher_labels_matched)
            collected_student_lines.append(student_lines_matched)
            collected_teacher_lines.append(teacher_lines_matched.detach())

            for attr_name in attr_names:
                student_attr_logits = student_attr_dict.get(attr_name, [None] * batch_size)[b_idx]
                teacher_attr_logits = teacher_attr_dict.get(attr_name, [None] * batch_size)[b_idx] \
                    if attr_name in teacher_attr_dict else None

                if student_attr_logits is None or teacher_attr_logits is None:
                    continue

                student_attr_matched = student_attr_logits[matched_mask]
                teacher_attr_matched = teacher_attr_logits[pos_inds_b]
                teacher_attr_matched = teacher_attr_matched[teacher_indices]

                collected_student_attrs[attr_name].append(student_attr_matched)
                collected_teacher_attrs[attr_name].append(teacher_attr_matched.detach())

                mask_tensor = None
                if teacher_gt is not None:
                    attr_masks_all = teacher_gt.get('attr_masks', {}).get(attr_name, None)
                    if attr_masks_all is not None and b_idx < len(attr_masks_all):
                        mask_tensor = attr_masks_all[b_idx][pos_inds_b]
                        mask_tensor = mask_tensor[teacher_indices]
                if mask_tensor is not None:
                    collected_teacher_attr_masks[attr_name].append(mask_tensor.detach())
                else:
                    mask_fallback = torch.ones(
                        teacher_attr_matched.shape[0],
                        dtype=torch.bool,
                        device=teacher_attr_matched.device,
                    )
                    collected_teacher_attr_masks[attr_name].append(mask_fallback)

            total_matches += teacher_labels_matched.numel()

        if not collected_student_scores:
            return None, None, {}, total_matches

        student_scores_all = torch.cat(collected_student_scores, dim=0)
        teacher_labels_all = torch.cat(collected_teacher_labels, dim=0)
        student_lines_all = torch.cat(collected_student_lines, dim=0)
        teacher_lines_all = torch.cat(collected_teacher_lines, dim=0).to(
            device=student_lines_all.device, dtype=student_lines_all.dtype
        )

        cls_weights = student_scores_all.new_ones(teacher_labels_all.shape[0])
        cls_loss = cls_loss_fn(
            student_scores_all,
            teacher_labels_all,
            cls_weights,
            avg_factor=max(teacher_labels_all.numel(), 1),
        )

        line_weights = student_lines_all.new_ones(student_lines_all.shape)
        reg_loss = reg_loss_fn(
            student_lines_all,
            teacher_lines_all,
            line_weights,
            avg_factor=max(teacher_lines_all.shape[0], 1),
        )

        attr_losses: Dict[str, torch.Tensor] = {}
        for attr_name in attr_names:
            if not collected_student_attrs[attr_name]:
                continue
            student_cat = torch.cat(collected_student_attrs[attr_name], dim=0)
            teacher_cat = torch.cat(collected_teacher_attrs[attr_name], dim=0)
            mask_cat = torch.cat(collected_teacher_attr_masks[attr_name], dim=0)
            attr_loss = self._attr_kd_loss(student_cat, teacher_cat, mask_cat)
            if attr_loss is not None:
                attr_losses[attr_name] = attr_loss

        return cls_loss, reg_loss, attr_losses, total_matches

    def forward_train(self, **data):
        teacher_bevs: List[torch.Tensor] = []
        teacher_seg_logits: List[Optional[torch.Tensor]] = []
        teacher_head_preds: List[Optional[Dict[str, List[torch.Tensor]]]] = []
        teacher_head_pos_inds: List[Optional[List[torch.Tensor]]] = []
        teacher_head_gt: List[Optional[Dict[str, List[torch.Tensor]]]] = []

        for teacher in self.teacher_models:
            with torch.no_grad():
                teacher.forward_train(**data)

                teacher_bev = getattr(teacher, 'latest_bev_feats', None)
                if teacher_bev is None:
                    raise RuntimeError('Teacher did not expose BEV features for distillation.')
                teacher_bevs.append(teacher_bev.detach())

                seg_logits = getattr(teacher, 'latest_seg_logits', None)
                teacher_seg_logits.append(seg_logits.detach() if seg_logits is not None else None)

                head_preds = getattr(teacher, 'latest_head_preds', None)
                if head_preds is not None:
                    copied = dict(
                        lines=[tensor.detach() for tensor in head_preds.get('lines', [])],
                        scores=[tensor.detach() for tensor in head_preds.get('scores', [])],
                    )
                    attr_dict = head_preds.get('attr', None)
                    if attr_dict is not None:
                        copied['attr'] = {
                            name: [tensor.detach() for tensor in tensors]
                            for name, tensors in attr_dict.items()
                        }
                    teacher_head_preds.append(copied)
                else:
                    teacher_head_preds.append(None)

                pos_inds = getattr(teacher, 'latest_head_pos_inds', None)
                if pos_inds is not None:
                    teacher_head_pos_inds.append(
                        [inds.detach() if torch.is_tensor(inds) else inds for inds in pos_inds]
                    )
                else:
                    teacher_head_pos_inds.append(None)

                gt_info = getattr(teacher, 'latest_head_gt', None)
                if gt_info is not None:
                    copied_gt = {}
                    for key, value in gt_info.items():
                        if isinstance(value, dict):
                            copied_gt[key] = {
                                sub_key: [
                                    tensor.detach() if torch.is_tensor(tensor) else tensor
                                    for tensor in sub_list
                                ]
                                for sub_key, sub_list in value.items()
                            }
                        else:
                            copied_gt[key] = [
                                tensor.detach() if torch.is_tensor(tensor) else tensor
                                for tensor in value
                            ]
                    teacher_head_gt.append(copied_gt)
                else:
                    teacher_head_gt.append(None)

        student_loss, log_vars, num_samples = self.student.forward_train(**data)

        student_bev = getattr(self.student, 'latest_bev_feats', None)
        if student_bev is None:
            raise RuntimeError('Student did not expose BEV features for distillation.')
        student_seg_logits = getattr(self.student, 'latest_seg_logits', None)
        student_head_preds = getattr(self.student, 'latest_head_preds', None)
        student_head_pos_inds = getattr(self.student, 'latest_head_pos_inds', None)
        student_head_gt = getattr(self.student, 'latest_head_gt', None)

        seg_mask = self._build_segmentation_mask(
            data,
            target_shape=student_bev.shape[-2:],
            device=student_bev.device,
            dtype=student_bev.dtype,
        )

        kd_loss = student_bev.new_tensor(0.0)
        kd_log_items: Dict[str, float] = {}

        for idx, meta in enumerate(self.teacher_meta):
            teacher_bev = teacher_bevs[idx]
            teacher_seg = teacher_seg_logits[idx]
            head_preds = teacher_head_preds[idx]
            head_pos = teacher_head_pos_inds[idx]
            head_gt = teacher_head_gt[idx]

            if self.bev_kd_weight > 0:
                loss_single = self._bev_loss(student_bev, teacher_bev, spatial_mask=seg_mask)
                weighted_loss = meta['weight'] * self.bev_kd_weight * loss_single
                kd_loss = kd_loss + weighted_loss
                kd_log_items[f"kd_bev_{meta['name']}"] = float(weighted_loss.detach())

            if (
                self.seg_kd_weight > 0
                and student_seg_logits is not None
                and teacher_seg is not None
            ):
                seg_teacher = teacher_seg
                if seg_teacher.shape[-2:] != student_seg_logits.shape[-2:]:
                    seg_teacher = F.interpolate(
                        seg_teacher,
                        size=student_seg_logits.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                seg_student = student_seg_logits
                seg_diff = (seg_student - seg_teacher) ** 2
                if seg_mask is not None:
                    seg_mask_resized = seg_mask
                    if seg_mask_resized.shape[-2:] != seg_student.shape[-2:]:
                        seg_mask_resized = F.interpolate(
                            seg_mask_resized,
                            size=seg_student.shape[-2:],
                            mode='nearest',
                        )
                    if seg_mask_resized.shape[1] == 1:
                        seg_mask_resized = seg_mask_resized.expand(-1, seg_student.shape[1], -1, -1)
                    seg_weighted = seg_diff * seg_mask_resized
                    valid = seg_mask_resized.sum()
                    if valid.item() > 0:
                        seg_loss = seg_weighted.sum() / valid
                    else:
                        seg_loss = seg_student.new_tensor(0.0)
                else:
                    seg_loss = seg_diff.mean()
                seg_weighted = meta['weight'] * self.seg_kd_weight * seg_loss
                kd_loss = kd_loss + seg_weighted
                kd_log_items[f"kd_seg_{meta['name']}"] = float(seg_weighted.detach())

            if (
                self.head_kd_weight > 0
                and student_head_preds is not None
                and head_preds is not None
                and head_pos is not None
            ):
                cls_loss, reg_loss, attr_losses, num_matches = self._map_head_kd_loss(
                    student_head_preds,
                    head_preds,
                    head_pos,
                    head_gt,
                )

                if cls_loss is not None:
                    cls_weighted = (
                        meta['weight'] * self.head_kd_weight * self.head_cls_weight * cls_loss
                    )
                    kd_loss = kd_loss + cls_weighted
                    kd_log_items[f"kd_head_cls_{meta['name']}"] = float(cls_weighted.detach())

                if reg_loss is not None:
                    reg_weighted = (
                        meta['weight'] * self.head_kd_weight * self.head_point_weight * reg_loss
                    )
                    kd_loss = kd_loss + reg_weighted
                    kd_log_items[f"kd_head_point_{meta['name']}"] = float(reg_weighted.detach())

                if attr_losses and self.head_attr_weight > 0:
                    for attr_name, attr_loss in attr_losses.items():
                        attr_weighted = (
                            meta['weight']
                            * self.head_kd_weight
                            * self.head_attr_weight
                            * attr_loss
                        )
                        kd_loss = kd_loss + attr_weighted
                        kd_log_items[
                            f"kd_head_attr_{attr_name}_{meta['name']}"
                        ] = float(attr_weighted.detach())

                if num_matches:
                    kd_log_items[f"kd_head_matches_{meta['name']}"] = float(num_matches)

        kd_loss = self.distill_weight * kd_loss
        total_loss = student_loss + kd_loss

        log_vars = dict(log_vars)
        log_vars['loss_kd'] = float(kd_loss.detach())
        for key, val in kd_log_items.items():
            log_vars[key] = val
        log_vars['total'] = float(total_loss.detach())

        return total_loss, log_vars, num_samples

    @torch.no_grad()
    def forward_test(self, *args, **kwargs):
        return self.student.forward_test(*args, **kwargs)

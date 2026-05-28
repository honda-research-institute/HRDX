import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear
from mmengine.model.weight_init import bias_init_with_prob, xavier_init
from plugin.utils.fp16_utils import force_fp32
from plugin.utils.registry import build_mmdet_module
from plugin.utils.mmdet_compat import (multi_apply, reduce_mean,
                                                build_assigner, build_sampler,
                                                inverse_sigmoid)
from mmengine.structures import InstanceData

from mmdet.registry import MODELS as MMDET_MODELS
from einops import rearrange


@MMDET_MODELS.register_module(force=True)
class MapDetectorHead(nn.Module):

    def __init__(self, 
                 num_queries,
                 num_classes=3,
                 in_channels=128,
                 embed_dims=256,
                 score_thr=0.1,
                 num_points=20,
                 coord_dim=2,
                 roi_size=(60, 30),
                 different_heads=True,
                 predict_refine=False,
                 bev_pos=None,
                 sync_cls_avg_factor=True,
                 bg_cls_weight=0.,
                 trans_loss_weight=0.0,
                 transformer=dict(),
                 loss_cls=dict(),
                 loss_reg=dict(),
                 assigner=dict(),
                 attribute_heads=None,
                 attribute_use_focal_loss=False,
                 attribute_focal_loss_gamma=2.0,
                 attribute_focal_loss_alpha=0.25,
                ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.different_heads = different_heads
        self.predict_refine = predict_refine
        self.bev_pos = bev_pos
        self.num_points = num_points
        self.coord_dim = coord_dim
        
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.bg_cls_weight = bg_cls_weight
        
        self.trans_loss_weight = trans_loss_weight
        # NOTE: below is a simple MLP to transform the query from prev-frame to cur-frame,
        # we moved the propagation part outside,
            
        self.register_buffer('roi_size', torch.tensor(roi_size, dtype=torch.float32))
        origin = (-roi_size[0]/2, -roi_size[1]/2)
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32))

        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)

        self.transformer = build_mmdet_module(
            transformer,
            extra_scopes=(
                'plugin_transformer',
                'plugin_transformer_layer_sequence',
                'plugin_transformer_layer',
                'plugin_attention',
                'plugin_ffn',
                'plugin_positional_encoding',
            ))

        self.loss_cls = build_mmdet_module(loss_cls)
        self.loss_reg = build_mmdet_module(loss_reg)
        self.assigner = build_assigner(assigner)

        self.attribute_heads_cfg: Dict[str, Dict[str, object]] = attribute_heads or {}
        self.attr_names: List[str] = list(self.attribute_heads_cfg.keys())
        self.attribute_specs: Dict[str, Dict[str, object]] = {}
        self.attr_branches = nn.ModuleDict()
        self.attribute_use_focal_loss = attribute_use_focal_loss
        self.attr_focal_gamma = float(attribute_focal_loss_gamma)
        self.attr_focal_alpha: Optional[float] = (
            float(attribute_focal_loss_alpha) if attribute_focal_loss_alpha is not None else None
        )

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        
        self._init_embedding()
        self._init_branch()
        self._init_attribute_branches()
        self.init_weights()


    def init_weights(self):
        """Initialize weights of the DeformDETR head."""

        for p in self.input_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        xavier_init(self.reference_points_embed, distribution='uniform', bias=0.)

        self.transformer.init_weights()

        # init prediction branch
        for m in self.reg_branches:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
        for attr_branch_list in self.attr_branches.values():
            for attr_branch in attr_branch_list:
                if hasattr(attr_branch, 'weight') and attr_branch.weight is not None:
                    nn.init.xavier_uniform_(attr_branch.weight)
                if hasattr(attr_branch, 'bias') and attr_branch.bias is not None:
                    nn.init.constant_(attr_branch.bias, 0.)

        # focal loss init
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            if isinstance(self.cls_branches, nn.ModuleList):
                for m in self.cls_branches:
                    if hasattr(m, 'bias'):
                        nn.init.constant_(m.bias, bias_init)
            else:
                m = self.cls_branches
                nn.init.constant_(m.bias, bias_init)
        
        if hasattr(self, 'query_alpha'):
            for m in self.query_alpha:
                for param in m.parameters():
                    if param.dim() > 1:
                        nn.init.zeros_(param)

    def _init_embedding(self):
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=self.embed_dims//2,
            normalize=True
        )
        positional_encoding['_scope_'] = 'mmdet'
        self.bev_pos_embed = build_mmdet_module(positional_encoding)

        # query_pos_embed & query_embed
        self.query_embedding = nn.Embedding(self.num_queries,
                                            self.embed_dims)

        self.reference_points_embed = nn.Linear(self.embed_dims, self.num_points * 2)
        
    def _init_branch(self,):
        """Initialize classification branch and regression branch of head."""
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = Linear(self.embed_dims, self.cls_out_channels)

        reg_branch = [
            Linear(self.embed_dims, 2*self.embed_dims),
            nn.LayerNorm(2*self.embed_dims),
            nn.ReLU(),
            Linear(2*self.embed_dims, 2*self.embed_dims),
            nn.LayerNorm(2*self.embed_dims),
            nn.ReLU(),
            Linear(2*self.embed_dims, self.num_points * self.coord_dim),
        ]
        reg_branch = nn.Sequential(*reg_branch)

        num_layers = self.transformer.decoder.num_layers
        if self.different_heads:
            cls_branches = nn.ModuleList(
                [copy.deepcopy(cls_branch) for _ in range(num_layers)])
            reg_branches = nn.ModuleList(
                [copy.deepcopy(reg_branch) for _ in range(num_layers)])
        else:
            cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_layers)])
            reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_layers)])

        self.reg_branches = reg_branches
        self.cls_branches = cls_branches

    def _init_attribute_branches(self):
        """Initialize attribute classification branches."""
        if not self.attr_names:
            return

        num_layers = self.transformer.decoder.num_layers
        for attr_name in self.attr_names:
            cfg = self.attribute_heads_cfg[attr_name]
            classes = cfg.get('classes')
            if not classes:
                raise ValueError(f'Attribute "{attr_name}" requires a non-empty "classes" list.')
            if not isinstance(classes, list):
                raise TypeError(f'Attribute "{attr_name}" classes must be a list of strings.')

            applies_to = cfg.get('applies_to', [])
            if not isinstance(applies_to, (list, tuple)):
                raise TypeError(f'"applies_to" for attribute "{attr_name}" must be a list of class indices.')

            default_label = cfg.get('default', classes[0])
            if default_label not in classes:
                raise ValueError(
                    f'Default value "{default_label}" for attribute "{attr_name}" '
                    f'is not included in its classes list {classes}.'
                )

            loss_weight = float(cfg.get('loss_weight', 1.0))
            value_to_index = {value: idx for idx, value in enumerate(classes)}
            spec = dict(
                classes=classes,
                num_classes=len(classes),
                applies_to=list(applies_to),
                applies_to_set=set(applies_to),
                default=default_label,
                default_index=value_to_index[default_label],
                value_to_index=value_to_index,
                loss_weight=loss_weight,
            )
            self.attribute_specs[attr_name] = spec

            branch = Linear(self.embed_dims, len(classes))
            if self.different_heads:
                branch_list = nn.ModuleList([copy.deepcopy(branch) for _ in range(num_layers)])
            else:
                branch_list = nn.ModuleList([branch for _ in range(num_layers)])
            self.attr_branches[attr_name] = branch_list

    def _compute_attribute_loss(self,
                                logits: torch.Tensor,
                                targets: torch.Tensor,
                                normalizer: torch.Tensor) -> torch.Tensor:
        """Compute attribute loss with optional focal weighting."""
        if self.attribute_use_focal_loss:
            return self._attribute_focal_loss(logits, targets, normalizer)
        ce_loss = F.cross_entropy(logits, targets, reduction='sum')
        return ce_loss / normalizer

    def _attribute_focal_loss(self,
                              logits: torch.Tensor,
                              targets: torch.Tensor,
                              normalizer: torch.Tensor) -> torch.Tensor:
        """Multi-class focal loss for attribute predictions."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        gamma = self.attr_focal_gamma
        alpha = self.attr_focal_alpha
        alpha_factor = ce_loss.new_full(ce_loss.shape, 1.0 if alpha is None else float(alpha))
        focal_weight = alpha_factor * (1 - pt).pow(gamma)
        focal_loss = focal_weight * ce_loss
        return focal_loss.sum() / normalizer

    def _prepare_context(self, bev_features):
        """Prepare class label and vertex context."""
        device = bev_features.device

        # Add 2D coordinate grid embedding
        B, C, H, W = bev_features.shape
        bev_mask = bev_features.new_zeros(B, H, W)
        bev_pos_embeddings = self.bev_pos_embed(bev_mask) # (bs, embed_dims, H, W)
        bev_features = self.input_proj(bev_features) + bev_pos_embeddings # (bs, embed_dims, H, W)
    
        assert list(bev_features.shape) == [B, self.embed_dims, H, W]
        return bev_features

    def forward_train(self, bev_features, img_metas, gts, track_query_info=None, memory_bank=None, return_matching=False):
        '''
        Args:
            bev_feature (List[Tensor]): shape [B, C, H, W]
                feature in bev view
        Outs:
            preds_dict (list[dict]):
                lines (Tensor): Classification score of all
                    decoder layers, has shape
                    [bs, num_query, 2*num_points]
                scores (Tensor):
                    [bs, num_query,]
        '''

        bev_features = self._prepare_context(bev_features)

        bs, C, H, W = bev_features.shape
        img_masks = bev_features.new_zeros((bs, H, W))
        # pos_embed = self.positional_encoding(img_masks)
        pos_embed = None

        query_embedding = self.query_embedding.weight[None, ...].repeat(bs, 1, 1) # [B, num_q, embed_dims]
        input_query_num = self.num_queries

        init_reference_points = self.reference_points_embed(query_embedding).sigmoid() # (bs, num_q, 2*num_pts)
        init_reference_points = init_reference_points.view(-1, self.num_queries, self.num_points, 2) # (bs, num_q, num_pts, 2)
        
        assert list(init_reference_points.shape) == [bs, self.num_queries, self.num_points, 2]
        assert list(query_embedding.shape) == [bs, self.num_queries, self.embed_dims]

        # Prepare the propagated track queries, concat with the original dummy queries
        if track_query_info is not None and 'track_query_hs_embeds' in track_query_info[0]:
            new_query_embeds = []
            new_init_ref_pts = []
            for b_i in range(bs):
                new_queries = torch.cat([track_query_info[b_i]['track_query_hs_embeds'], query_embedding[b_i], 
                           track_query_info[b_i]['pad_hs_embeds']], dim=0)
                new_query_embeds.append(new_queries)
                init_ref = rearrange(init_reference_points[b_i], 'n k c -> n (k c)', c=2)
                new_ref = torch.cat([track_query_info[b_i]['trans_track_query_boxes'], init_ref, 
                           track_query_info[b_i]['pad_query_boxes']], dim=0)
                new_ref = rearrange(new_ref, 'n (k c) -> n k c', c=2)
                new_init_ref_pts.append(new_ref)
                #print('length of track queries', track_query_info[b_i]['track_query_hs_embeds'].shape[0])


            # concat to get the track+dummy queries
            query_embedding = torch.stack(new_query_embeds, dim=0)
            init_reference_points = torch.stack(new_init_ref_pts, dim=0)
            query_kp_mask = torch.stack([t['query_padding_mask'] for t in track_query_info], dim=0)
        else:
            query_kp_mask = query_embedding.new_zeros((bs, self.num_queries), dtype=torch.bool)
        
        # outs_dec: (num_layers, num_qs, bs, embed_dims)
        inter_queries, init_reference, inter_references = self.transformer(
            mlvl_feats=[bev_features,],
            mlvl_masks=[img_masks.type(torch.bool)],
            query_embed=query_embedding,
            mlvl_pos_embeds=[pos_embed], # not used
            memory_query=None,
            init_reference_points=init_reference_points,
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
            predict_refine=self.predict_refine,
            query_key_padding_mask=query_kp_mask, # mask used in self-attn,
            memory_bank=memory_bank,
        )

        outputs = []
        for layer_idx, queries in enumerate(inter_queries):
            reg_points = inter_references[layer_idx] # (bs, num_q, num_points, 2)
            bs_layer = reg_points.shape[0]
            reg_points = reg_points.view(bs_layer, -1, 2*self.num_points) # (bs, num_q, 2*num_points)

            scores = self.cls_branches[layer_idx](queries) # (bs, num_q, num_classes)

            attr_logits_dict = {}
            if self.attr_names:
                for attr_name in self.attr_names:
                    branch = self.attr_branches[attr_name][layer_idx]
                    attr_logits_dict[attr_name] = branch(queries)

            reg_points_list = []
            scores_list = []
            attr_list_dict = {name: [] for name in self.attr_names}
            for b_idx in range(scores.shape[0]):
                # padding queries should not be output
                reg_points_list.append(reg_points[b_idx])
                scores_list.append(scores[b_idx])
                for attr_name in self.attr_names:
                    attr_list_dict[attr_name].append(attr_logits_dict[attr_name][b_idx])

            pred_dict = {
                'lines': reg_points_list,
                'scores': scores_list,
                'attr': attr_list_dict
            }
            if return_matching:
                pred_dict['hs_embeds'] = queries
            outputs.append(pred_dict)

        # Pass in the track query information to massage the cost matrix
        loss_dict, det_match_idxs, det_match_gt_idxs, gt_info_list, matched_reg_cost = \
                self.loss(gts=gts, preds=outputs, track_info=track_query_info)

        if return_matching:
            return loss_dict, outputs[-1], det_match_idxs[-1], det_match_gt_idxs[-1], matched_reg_cost[-1], gt_info_list[-1]
        else:
            return outputs, loss_dict, det_match_idxs, det_match_gt_idxs, gt_info_list
    
    def forward_test(self, bev_features, img_metas, track_query_info=None, memory_bank=None):
        '''
        Args:
            bev_feature (List[Tensor]): shape [B, C, H, W]
                feature in bev view
        Outs:
            preds_dict (list[dict]):
                lines (Tensor): Classification score of all
                    decoder layers, has shape
                    [bs, num_query, 2*num_points]
                scores (Tensor):
                    [bs, num_query,]
        '''

        bev_features = self._prepare_context(bev_features)

        bs, C, H, W = bev_features.shape
        assert bs == 1, 'Only support bs=1 per-gpu for inference'
        
        img_masks = bev_features.new_zeros((bs, H, W))
        # pos_embed = self.positional_encoding(img_masks)
        pos_embed = None

        query_embedding = self.query_embedding.weight[None, ...].repeat(bs, 1, 1) # [B, num_q, embed_dims]
        input_query_num = self.num_queries
        # num query: self.num_query + self.topk
        
        init_reference_points = self.reference_points_embed(query_embedding).sigmoid() # (bs, num_q, 2*num_pts)
        init_reference_points = init_reference_points.view(-1, self.num_queries, self.num_points, 2) # (bs, num_q, num_pts, 2)
        
        assert list(init_reference_points.shape) == [bs, input_query_num, self.num_points, 2]
        assert list(query_embedding.shape) == [bs, input_query_num, self.embed_dims]

        # Prepare the propagated track queries, concat with the original dummy queries
        if track_query_info is not None and 'track_query_hs_embeds' in track_query_info[0]:
            prev_hs_embed = torch.stack([t['track_query_hs_embeds'] for t in track_query_info])
            prev_boxes = torch.stack([t['trans_track_query_boxes'] for t in track_query_info])
            prev_boxes = rearrange(prev_boxes, 'b n (k c) -> b n k c', c=2)

            # concat to get the track+dummy queries
            query_embedding = torch.cat([prev_hs_embed, query_embedding], dim=1)
            init_reference_points = torch.cat([prev_boxes, init_reference_points], dim=1)
            
        query_kp_mask = query_embedding.new_zeros((bs, query_embedding.shape[1]), dtype=torch.bool)

        # outs_dec: (num_layers, num_qs, bs, embed_dims)
        inter_queries, init_reference, inter_references = self.transformer(
            mlvl_feats=[bev_features,],
            mlvl_masks=[img_masks.type(torch.bool)],
            query_embed=query_embedding,
            mlvl_pos_embeds=[pos_embed], # not used
            memory_query=None,
            init_reference_points=init_reference_points,
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
            predict_refine=self.predict_refine,
            query_key_padding_mask=query_kp_mask, # mask used in self-attn,
            memory_bank=memory_bank,
        )

        outputs = []
        for layer_idx, queries in enumerate(inter_queries):
            reg_points = inter_references[layer_idx] # (bs, num_q, num_points, 2)
            bs_layer = reg_points.shape[0]
            reg_points = reg_points.view(bs_layer, -1, 2*self.num_points) # (bs, num_q, 2*num_points)
            scores = self.cls_branches[layer_idx](queries) # (bs, num_q, num_classes)

            attr_logits_dict = {}
            if self.attr_names:
                for attr_name in self.attr_names:
                    branch = self.attr_branches[attr_name][layer_idx]
                    attr_logits_dict[attr_name] = branch(queries)

            reg_points_list = []
            scores_list = []
            attr_list_dict = {name: [] for name in self.attr_names}
            for b_idx in range(scores.shape[0]):
                reg_points_list.append(reg_points[b_idx])
                scores_list.append(scores[b_idx])
                for attr_name in self.attr_names:
                    attr_list_dict[attr_name].append(attr_logits_dict[attr_name][b_idx])

            pred_dict = {
                'lines': reg_points_list,
                'scores': scores_list,
                'hs_embeds': queries,
                'attr': attr_list_dict,
            }
            outputs.append(pred_dict)

        return outputs

    @force_fp32(apply_to=('score_pred', 'lines_pred', 'gt_lines'))
    def _get_target_single(self,
                           score_pred,
                           lines_pred,
                           gt_labels,
                           gt_lines,
                           track_info=None,
                           gt_attrs=None,
                           gt_bboxes_ignore=None):
        """
            Compute regression and classification targets for one image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                score_pred (Tensor): Box score logits from a single decoder layer
                    for one image. Shape [num_query, cls_out_channels].
                lines_pred (Tensor):
                    shape [num_query, 2*num_points]
                gt_labels (torch.LongTensor)
                    shape [num_gt, ]
                gt_lines (Tensor):
                    shape [num_gt, 2*num_points].
                
            Returns:
                tuple[Tensor]: a tuple containing the following for one sample.
                    - labels (LongTensor): Labels of each image.
                        shape [num_query, 1]
                    - label_weights (Tensor]): Label weights of each image.
                        shape [num_query, 1]
                    - lines_target (Tensor): Lines targets of each image.
                        shape [num_query, num_points, 2]
                    - lines_weights (Tensor): Lines weights of each image.
                        shape [num_query, num_points, 2]
                    - pos_inds (Tensor): Sampled positive indices for each image.
                    - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_pred_lines = len(lines_pred)
        # assigner and sampler

        if gt_attrs is None:
            gt_attrs = {}
        attr_targets: Dict[str, torch.Tensor] = {}
        attr_masks: Dict[str, torch.Tensor] = {}
        sample_attr_labels: Dict[str, torch.Tensor] = {}
        sample_attr_masks: Dict[str, torch.Tensor] = {}

        for attr_name in self.attr_names:
            specs = self.attribute_specs[attr_name]
            default_index = specs['default_index']
            attr_targets[attr_name] = score_pred.new_full(
                (num_pred_lines,), default_index, dtype=torch.long)
            attr_masks[attr_name] = score_pred.new_zeros(
                (num_pred_lines,), dtype=torch.bool)

            if attr_name in gt_attrs:
                sample_attr = gt_attrs[attr_name]
                sample_attr_labels[attr_name] = sample_attr['labels'].to(score_pred.device)
                sample_attr_masks[attr_name] = sample_attr['mask'].to(score_pred.device).bool()
            else:
                sample_attr_labels[attr_name] = score_pred.new_full(
                    (len(gt_lines),), default_index, dtype=torch.long)
                sample_attr_masks[attr_name] = score_pred.new_zeros(
                    (len(gt_lines),), dtype=torch.bool)
        
        # We massage the matching cost here using the track info, following
        # the 3-type supervision of TrackFormer/MOTR
        assign_result, gt_permute_idx, matched_reg_cost = self.assigner.assign(preds=dict(lines=lines_pred, scores=score_pred,),
                                             gts=dict(lines=gt_lines,
                                                      labels=gt_labels, ),
                                             track_info=track_info,
                                             gt_bboxes_ignore=gt_bboxes_ignore)
        pred_instances = InstanceData()
        pred_instances.priors = lines_pred
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_lines
        gt_instances.labels = gt_labels
        sampling_result = self.sampler.sample(
            assign_result, pred_instances, gt_instances)
        num_gt = len(gt_lines)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lines.new_full(
                (num_pred_lines, ), self.num_classes, dtype=torch.long) # (num_q, )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_lines.new_ones(num_pred_lines) # (num_q, )

        lines_target = torch.zeros_like(lines_pred) # (num_q, 2*num_pts)
        lines_weights = torch.zeros_like(lines_pred) # (num_q, 2*num_pts)
        
        if num_gt > 0:
            if gt_permute_idx is not None: # using permute invariant label
                # gt_permute_idx: (num_q, num_gt)
                # pos_inds: which query is positive
                # pos_gt_inds: which gt each pos pred is assigned
                # single_matched_gt_permute_idx: which permute order is matched
                single_matched_gt_permute_idx = gt_permute_idx[
                    pos_inds, pos_gt_inds
                ]
                lines_target[pos_inds] = gt_lines[pos_gt_inds, single_matched_gt_permute_idx].type(
                    lines_target.dtype) # (num_q, 2*num_pts)
            else:
                lines_target[pos_inds] = sampling_result.pos_gt_bboxes.type(
                    lines_target.dtype) # (num_q, 2*num_pts)
        
        lines_weights[pos_inds] = 1.0 # (num_q, 2*num_pts)

        if len(pos_inds) > 0:
            for attr_name in self.attr_names:
                attr_labels_gt = sample_attr_labels[attr_name]
                attr_masks_gt = sample_attr_masks[attr_name]
                attr_targets[attr_name][pos_inds] = attr_labels_gt[pos_gt_inds]
                attr_masks[attr_name][pos_inds] = attr_masks_gt[pos_gt_inds]

        # normalization
        # n = lines_weights.sum(-1, keepdim=True) # (num_q, 1)
        # lines_weights = lines_weights / n.masked_fill(n == 0, 1) # (num_q, 2*num_pts)
        # [0, ..., 0] for neg ind and [1/npts, ..., 1/npts] for pos ind

        return (labels, label_weights, lines_target, lines_weights,
                pos_inds, neg_inds, pos_gt_inds, matched_reg_cost,
                attr_targets, attr_masks)

    # @force_fp32(apply_to=('preds', 'gts'))
    def get_targets(self, preds, gts, track_info=None, gt_bboxes_ignore_list=None):
        """
            Compute regression and classification targets for a batch image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                preds (dict): 
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                tuple: a tuple containing the following targets.
                    - labels_list (list[Tensor]): Labels for all images.
                    - label_weights_list (list[Tensor]): Label weights for all \
                        images.
                    - lines_targets_list (list[Tensor]): Lines targets for all \
                        images.
                    - lines_weight_list (list[Tensor]): Lines weights for all \
                        images.
                    - num_total_pos (int): Number of positive samples in all \
                        images.
                    - num_total_neg (int): Number of negative samples in all \
                        images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        # format the inputs
        gt_labels = gts['labels']
        gt_lines = gts['lines']

        lines_pred = preds['lines']

        gt_attrs_raw = gts.get('attrs', {}) if isinstance(gts, dict) else {}
        if self.attr_names:
            gt_attrs_list = []
            num_samples = len(gt_labels)
            for sample_idx in range(num_samples):
                sample_attr = {}
                for attr_name in self.attr_names:
                    if attr_name in gt_attrs_raw:
                        sample_attr[attr_name] = {
                            'labels': gt_attrs_raw[attr_name]['labels'][sample_idx],
                            'mask': gt_attrs_raw[attr_name]['mask'][sample_idx],
                        }
                gt_attrs_list.append(sample_attr)
        else:
            gt_attrs_list = [dict() for _ in range(len(gt_labels))]

        if track_info is None:
            track_info = [track_info for _ in range(len(gt_labels))]

        (labels_list, label_weights_list,
        lines_targets_list, lines_weights_list,
        pos_inds_list, neg_inds_list, pos_gt_inds_list, matched_reg_cost,
        attr_targets_list, attr_masks_list) = multi_apply(
            self._get_target_single, preds['scores'], lines_pred,
            gt_labels, gt_lines, track_info, gt_attrs_list,
            gt_bboxes_ignore=gt_bboxes_ignore_list)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        if track_info[0] is not None:
            # remove the padding elements from the neg counting
            padding_mask = torch.cat([t['query_padding_mask'] for t in track_info], dim=0)
            num_padding = padding_mask.sum()
            num_total_neg -= num_padding
        
        new_gts = dict(
            labels=labels_list, # list[Tensor(num_q, )], length=bs
            label_weights=label_weights_list, # list[Tensor(num_q, )], length=bs, all ones
            lines=lines_targets_list, # list[Tensor(num_q, 2*num_pts)], length=bs
            lines_weights=lines_weights_list, # list[Tensor(num_q, 2*num_pts)], length=bs
        )
        if self.attr_names:
            attr_labels_dict = {name: [] for name in self.attr_names}
            attr_masks_dict = {name: [] for name in self.attr_names}
            for sample_targets, sample_masks in zip(attr_targets_list, attr_masks_list):
                for attr_name in self.attr_names:
                    attr_labels_dict[attr_name].append(sample_targets[attr_name])
                    attr_masks_dict[attr_name].append(sample_masks[attr_name])
            new_gts['attr_labels'] = attr_labels_dict
            new_gts['attr_masks'] = attr_masks_dict

        return new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list, matched_reg_cost

    # @force_fp32(apply_to=('preds', 'gts'))
    def loss_single(self,
                    preds,
                    gts,
                    track_info=None,
                    gt_bboxes_ignore_list=None,
                    reduction='none'):
        """
            Loss function for outputs from a single decoder layer of a single
            feature level.
            Args:
                preds (dict): 
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components for outputs from
                    a single decoder layer.
        """

        # Get target for each sample
        new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list, matched_reg_cost =\
            self.get_targets(preds, gts, track_info, gt_bboxes_ignore_list)

        # Batched all data
        # for k, v in new_gts.items():
        #     new_gts[k] = torch.stack(v, dim=0) # tensor (bs, num_q, ...)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                preds['scores'][0].new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if track_info is not None:
            cat_padding_mask = torch.cat([t['query_padding_mask'] for t in track_info], dim=0)
            padding_loss_mask = ~cat_padding_mask

        # Classification loss
        # since the inputs needs the second dim is the class dim, we permute the prediction.
        pred_scores = torch.cat(preds['scores'], dim=0) # (bs*num_q, cls_out_channles)
        cls_scores = pred_scores.reshape(-1, self.cls_out_channels) # (bs*num_q, cls_out_channels)
        cls_labels = torch.cat(new_gts['labels'], dim=0).reshape(-1) # (bs*num_q, )
        cls_weights = torch.cat(new_gts['label_weights'], dim=0).reshape(-1) # (bs*num_q, )
        if track_info is not None:
            cls_weights = cls_weights * padding_loss_mask.float()
        
        loss_cls = self.loss_cls(
            cls_scores, cls_labels, cls_weights, avg_factor=cls_avg_factor)
        
        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        pred_lines = torch.cat(preds['lines'], dim=0)
        gt_lines = torch.cat(new_gts['lines'], dim=0)
        line_weights = torch.cat(new_gts['lines_weights'], dim=0)
        if track_info is not None:
            line_weights = line_weights * padding_loss_mask[:, None].float()

        assert len(pred_lines) == len(gt_lines)
        assert len(gt_lines) == len(line_weights)

        loss_reg = self.loss_reg(
            pred_lines, gt_lines, line_weights, avg_factor=num_total_pos)

        loss_dict = dict(
            cls=loss_cls,
            reg=loss_reg,
        )

        if self.attr_names:
            attr_pred_concat = {
                attr_name: torch.cat(preds['attr'][attr_name], dim=0)
                for attr_name in self.attr_names
            }
            attr_label_concat = {
                attr_name: torch.cat(new_gts['attr_labels'][attr_name], dim=0)
                for attr_name in self.attr_names
            }
            attr_mask_concat = {
                attr_name: torch.cat(new_gts['attr_masks'][attr_name], dim=0).bool()
                for attr_name in self.attr_names
            }
            padding_bool_mask = None
            if track_info is not None:
                padding_bool_mask = padding_loss_mask.bool()

            for attr_name in self.attr_names:
                logits_attr = attr_pred_concat[attr_name]
                targets_attr = attr_label_concat[attr_name]
                masks_attr = attr_mask_concat[attr_name]
                if padding_bool_mask is not None:
                    masks_attr = masks_attr & padding_bool_mask
                if masks_attr.any():
                    valid_logits = logits_attr[masks_attr]
                    valid_targets = targets_attr[masks_attr]
                    denom = masks_attr.sum().clamp(min=1).float()
                    attr_loss = self._compute_attribute_loss(valid_logits, valid_targets, denom)
                else:
                    attr_loss = logits_attr.sum() * 0.
                attr_loss = attr_loss * float(self.attribute_specs[attr_name]['loss_weight'])
                loss_dict[f'attr_{attr_name}'] = attr_loss

        new_gts_info = {
            'labels': new_gts['labels'],
            'lines': new_gts['lines'],
        }
        if self.attr_names:
            new_gts_info['attrs'] = {
                'labels': new_gts['attr_labels'],
                'mask': new_gts['attr_masks'],
            }

        return loss_dict, pos_inds_list, pos_gt_inds_list, matched_reg_cost, new_gts_info
    
    @force_fp32(apply_to=('gt_lines_list', 'preds_dicts'))
    def loss(self,
             gts,
             preds,
             gt_bboxes_ignore=None,
             track_info=None,
             reduction='mean',
            ):
        """
            Loss Function.
            Args:
                gts (list[dict]): list length: num_layers
                    dict {
                        'label': list[tensor(num_gts, )], list length: batchsize,
                        'line': list[tensor(num_gts, 2*num_points)], list length: batchsize,
                        ...
                    }
                preds (list[dict]): list length: num_layers
                    dict {
                        'lines': tensor(bs, num_queries, 2*num_points),
                        'scores': tensor(bs, num_queries, class_out_channels),
                    }
                    
                gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                    which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        track_info = [track_info for _ in range(len(gts))]
        # Since there might have multi layer
        losses, pos_inds_lists, pos_gt_inds_lists, matched_reg_costs, gt_info_list = multi_apply(
            self.loss_single, preds, gts, track_info, reduction=reduction)

        # Format the losses
        loss_dict = dict()
        # loss from the last decoder layer
        for k, v in losses[-1].items():
            loss_dict[k] = v
        
        # Loss from other decoder layers
        num_dec_layer = 0
        for loss in losses[:-1]:
            for k, v in loss.items():
                loss_dict[f'd{num_dec_layer}.{k}'] = v
            num_dec_layer += 1

        return loss_dict, pos_inds_lists, pos_gt_inds_lists, gt_info_list, matched_reg_costs
    
    def post_process(self, preds_dict, tokens, track_dict=None, thr=0.0):
        lines = preds_dict['lines'] # List[Tensor(num_queries, 2*num_points)]
        bs = len(lines)
        scores = preds_dict['scores'] # (bs, num_queries, 3)

        results = []
        for i in range(bs):
            tmp_vectors = lines[i]
            device = tmp_vectors.device
            # set up the prop_flags
            tmp_prop_flags = torch.zeros(tmp_vectors.shape[0], dtype=torch.bool, device=device)
            tmp_prop_flags[-100:] = 0
            tmp_prop_flags[:-100] = 1
            num_preds, num_points2 = tmp_vectors.shape
            tmp_vectors = tmp_vectors.view(num_preds, num_points2//2, 2)

            attr_outputs = preds_dict.get('attr', {})
            sample_attr_logits = {}
            sample_attr_preds = {}
            has_attrs = bool(self.attr_names)
            if self.loss_cls.use_sigmoid:
                tmp_scores, tmp_labels = scores[i].max(-1)
                tmp_scores = tmp_scores.sigmoid()
                pos = tmp_scores > thr
            else:
                assert self.num_classes + 1 == self.cls_out_channels
                tmp_scores, tmp_labels = scores[i].max(-1)
                bg_cls = self.cls_out_channels
                pos = tmp_labels != bg_cls

            tmp_vectors = tmp_vectors[pos]
            tmp_scores = tmp_scores[pos]
            tmp_labels = tmp_labels[pos]
            tmp_prop_flags = tmp_prop_flags[pos]

            if has_attrs:
                for attr_name in self.attr_names:
                    attr_list = attr_outputs.get(attr_name, [])
                    if not attr_list:
                        sample_attr_logits[attr_name] = []
                        sample_attr_preds[attr_name] = []
                        continue
                    logits_full = attr_list[i]
                    if logits_full.shape[0] != num_preds:
                        logits_full = logits_full[:num_preds]
                    if logits_full.numel() == 0:
                        sample_attr_logits[attr_name] = []
                        sample_attr_preds[attr_name] = []
                        continue
                    logits_filtered = logits_full[pos]
                    if logits_filtered.numel() == 0:
                        sample_attr_logits[attr_name] = []
                        sample_attr_preds[attr_name] = []
                        continue
                    sample_attr_logits[attr_name] = logits_filtered.detach().cpu().numpy()
                    pred_indices = torch.argmax(logits_filtered, dim=-1)
                    sample_attr_preds[attr_name] = pred_indices.detach().cpu().numpy()

            if len(tmp_scores) == 0:
                single_result = {
                'vectors': [],
                'scores': [],
                'labels': [],
                'props': [],
                'token': tokens[i],
                'attr_logits': {attr_name: [] for attr_name in self.attr_names},
                'attr_preds': {attr_name: [] for attr_name in self.attr_names},
            }
            else:
                attr_result = {}
                attr_pred_result = {}
                for attr_name in self.attr_names:
                    attr_result[attr_name] = sample_attr_logits.get(attr_name, [])
                    attr_pred_result[attr_name] = sample_attr_preds.get(attr_name, [])
                single_result = {
                    'vectors': tmp_vectors.detach().cpu().numpy(),
                    'scores': tmp_scores.detach().cpu().numpy(),
                    'labels': tmp_labels.detach().cpu().numpy(),
                    'props': tmp_prop_flags.detach().cpu().numpy(),
                    'token': tokens[i],
                    'attr_logits': attr_result,
                    'attr_preds': attr_pred_result,
                }
            if len(tmp_scores) == 0 and has_attrs:
                single_result['attr_logits'] = {attr_name: [] for attr_name in self.attr_names}
                single_result['attr_preds'] = {attr_name: [] for attr_name in self.attr_names}

            # also save the tracking information for analyzing
            if track_dict is not None and len(track_dict['lines'])>0:
                tmp_track_scores = track_dict['scores'][i]
                tmp_track_vectors = track_dict['lines'][i]
                tmp_track_scores, tmp_track_labels = tmp_track_scores.max(-1)
                tmp_track_scores = tmp_track_scores.sigmoid()
                single_result['track_scores'] = tmp_track_scores.detach().cpu().numpy()
                single_result['track_vectors'] = tmp_track_vectors.detach().cpu().numpy()
                single_result['track_labels'] = tmp_track_labels.detach().cpu().numpy()
            else:
                single_result['track_scores'] = []
                single_result['track_vectors'] = []
                single_result['track_labels'] = []

            results.append(single_result)
    
        return results
    
    def prepare_temporal_propagation(self, preds_dict, scene_name, local_idx, memory_bank=None, 
                        thr_track=0.1, thr_det=0.5):
        lines = preds_dict['lines'] # List[Tensor(num_queries, 2*num_points)]
        queries = preds_dict['hs_embeds']
        bs = len(lines)
        assert bs == 1, 'now only support bs=1 for temporal-evolving inference'
        scores = preds_dict['scores'] # (bs, num_queries, 3)

        first_frame = local_idx == 0

        tmp_vectors = lines[0]
        tmp_queries = queries[0]
        device = tmp_vectors.device

        # focal loss
        if self.loss_cls.use_sigmoid:
            tmp_scores, tmp_labels = scores[0].max(-1)
            tmp_scores = tmp_scores.sigmoid()
            pos_track = (tmp_scores[:-100] > thr_track).to(device)
            pos_det = (tmp_scores[-100:] > thr_det).to(device)
            pos = torch.cat([pos_track, pos_det], dim=0)
        else:
            raise RuntimeError('The experiment uses sigmoid for cls outputs')

        pos_vectors = tmp_vectors[pos]
        pos_labels = tmp_labels[pos]
        pos_queries = tmp_queries[pos]
        pos_scores = tmp_scores[pos]

        if first_frame:
            global_ids = torch.arange(len(pos_vectors), device=device)
            num_instance = len(pos_vectors)
        else:
            prop_ids = self.prop_info['global_ids'].to(device)
            prop_num_instance = self.prop_info['num_instance']
            global_ids_track = prop_ids[pos_track]
            num_newborn = int(pos_det.sum())
            global_ids_newborn = torch.arange(num_newborn, device=device) + prop_num_instance
            global_ids = torch.cat([global_ids_track, global_ids_newborn])
            num_instance = prop_num_instance + num_newborn
            
        self.prop_info = {
            'vectors': pos_vectors.detach(),
            'queries': pos_queries.detach(),
            'scores': pos_scores.detach(),
            'labels': pos_labels.detach(),
            'scene_name': scene_name,
            'local_idx': local_idx,
            'global_ids': global_ids,
            'num_instance': num_instance,
        }

        if memory_bank is not None:
            if first_frame:
                num_tracks = 0
            else:
                num_tracks = self.prop_active_tracks
            pos_out_inds = torch.where(pos)[0]
            prev_out = {
                'hs_embeds': queries,
                'scores': scores,
            }
            memory_bank.update_memory(0, first_frame, pos_out_inds, prev_out, num_tracks, local_idx, memory_bank.curr_t)
            self.prop_active_tracks = len(pos_out_inds)
        
        save_pos_results = {
            'vectors': pos_vectors.cpu().numpy(),
            'scores': pos_scores.cpu().numpy(),
            'labels': pos_labels.cpu().numpy(),
            'global_ids': global_ids.cpu().numpy(),
            'scene_name': scene_name,
            'local_idx': local_idx,
            'num_instance': num_instance,
        }

        return save_pos_results
    
    def get_track_info(self, scene_name, local_idx):
        prop_info = self.prop_info
        assert prop_info['scene_name'] == scene_name and (prop_info['local_idx']+1 == local_idx or \
            prop_info['local_idx'] == local_idx)
            
        vectors = prop_info['vectors']
        queries = prop_info['queries']
        device = queries.device

        target = {}
        target['track_query_hs_embeds'] = queries
        target['track_query_boxes'] = vectors
        track_info = [target, ]

        return track_info
    
    def get_self_iter_track_query(self, preds_dict):
        num_tracks = self.prop_active_tracks

        lines = preds_dict['lines'] # List[Tensor(num_queries, 2*num_points)]
        queries = preds_dict['hs_embeds']
        bs = len(lines)
        assert bs == 1, 'now only support bs=1 for temporal-evolving inference'
        scores = preds_dict['scores'] # (bs, num_queries, 3)

        queries = queries[0][:num_tracks]
        vectors = lines[0][:num_tracks]

        target = {}
        target['track_query_hs_embeds'] = queries
        target['track_query_boxes'] = vectors
        track_info = [target, ]
        return track_info


    
    def clear_temporal_cache(self):
        self.prop_info = None

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
    
    def eval(self):
        super().eval()

    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

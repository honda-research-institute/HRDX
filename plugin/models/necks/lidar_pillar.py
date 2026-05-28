import math
import torch
import torch.nn as nn
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet3d.registry import MODELS as MMDET3D_MODELS


class _PFNLayer(nn.Module):
    """Pillar Feature Net layer (pointwise MLP + max pool) as in SECOND.

    Processes per-point features and performs a max reduction within each pillar.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (N, Fin)
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


@MMDET_MODELS.register_module()
@MMDET3D_MODELS.register_module()
class LidarPillarEncoder(nn.Module):
    """
    SECOND-style pillar encoder with BEV flattening (as in BEVFusion).

    - Voxelize points into pillars on the BEV grid.
    - Build per-point features with cluster and pillar-center offsets.
    - Apply a PFN (MLP + BN + ReLU) and max-reduce per pillar.
    - Scatter per-pillar features to a dense BEV pseudo-image.

    Inputs
      points: list[Tensor(N_i, C)] or Tensor(B, N, C), C>=3 (x, y, z [, intensity]).
      img_metas: unused here.

    Output
      Tensor: (B, out_channels, H, W)
    """

    def __init__(
        self,
        roi_size,
        bev_h,
        bev_w,
        in_channels=4,
        out_channels=64,
        z_min=-5.0,
        z_max=5.0,
        max_points_per_pillar=32,
        include_cluster_center=True,
        include_pillar_center=True,
        with_distance=False,
        scatter_type='max',  # or 'sum', 'mean'
    ):
        super().__init__()
        self.roi_size = roi_size
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_min = z_min
        self.z_max = z_max
        self.max_points_per_pillar = max_points_per_pillar
        self.include_cluster_center = include_cluster_center
        self.include_pillar_center = include_pillar_center
        self.with_distance = with_distance
        self.scatter_type = scatter_type

        # Compute pillar (voxel) size on the BEV plane
        self.dx = roi_size[0] / bev_w
        self.dy = roi_size[1] / bev_h
        self.x_min = -roi_size[0] / 2.0
        self.y_max = roi_size[1] / 2.0

        feat_in = in_channels
        if include_cluster_center:
            feat_in += 3  # x_c, y_c, z_c (cluster mean offsets)
        if include_pillar_center:
            feat_in += 2  # x_p, y_p (pillar center offsets)
        if with_distance:
            feat_in += 1

        self.pfn = _PFNLayer(feat_in, out_channels)
        self.out_channels = out_channels

    def init_weights(self):
        # Initialize linear and BN layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _mask_and_grid(self, pts):
        """Filter points within ROI and z-range; compute grid indices.

        Returns:
            mask (N,), gx (N,), gy (N,)
        """
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        mask = (
            (x >= self.x_min) & (x < self.x_min + self.dx * self.bev_w) &
            (y > self.y_max - self.dy * self.bev_h) & (y <= self.y_max) &
            (z >= self.z_min) & (z <= self.z_max)
        )
        if not mask.any():
            return mask, None, None
        x_sel, y_sel = x[mask], y[mask]
        gx = torch.clamp(((x_sel - self.x_min) / self.dx).floor().long(), 0, self.bev_w - 1)
        gy = torch.clamp(((self.y_max - y_sel) / self.dy).floor().long(), 0, self.bev_h - 1)
        return mask, gx, gy

    def _build_point_features(self, pts_sel, gx, gy, inv, counts):
        """Build per-point features (SECOND PFN-style).

        pts_sel: (Ns, C)
        gx, gy: (Ns,)
        inv: (Ns,) mapping to unique pillars
        counts: (num_pillars,)
        """
        x = pts_sel[:, 0]
        y = pts_sel[:, 1]
        z = pts_sel[:, 2]
        has_intensity = pts_sel.shape[1] >= 4
        i = pts_sel[:, 3] if has_intensity else torch.zeros_like(x)

        # Cluster means per pillar
        num_pillars = counts.shape[0]
        one = torch.ones_like(x)
        sum_x = torch.zeros(num_pillars, device=x.device).scatter_add_(0, inv, x)
        sum_y = torch.zeros(num_pillars, device=x.device).scatter_add_(0, inv, y)
        sum_z = torch.zeros(num_pillars, device=x.device).scatter_add_(0, inv, z)
        mean_x = sum_x / counts.clamp(min=1)
        mean_y = sum_y / counts.clamp(min=1)
        mean_z = sum_z / counts.clamp(min=1)
        mx = mean_x[inv]
        my = mean_y[inv]
        mz = mean_z[inv]

        # Pillar center coordinates per point
        cx = self.x_min + (gx.to(torch.float32) + 0.5) * self.dx
        cy = self.y_max - (gy.to(torch.float32) + 0.5) * self.dy

        feats = [x, y, z, i]
        if self.include_cluster_center:
            feats += [x - mx, y - my, z - mz]
        if self.include_pillar_center:
            feats += [x - cx, y - cy]
        if self.with_distance:
            dist = torch.sqrt(x.pow(2) + y.pow(2) + z.pow(2) + 1e-9)
            feats += [dist]

        feats = torch.stack(feats, dim=1)  # (Ns, Fin)
        return feats

    def _pfn_pool(self, point_feats, inv, num_pillars):
        """Apply PFN per point and max-reduce within each pillar.

        point_feats: (Ns, Fin)
        inv: (Ns,) indices mapping points -> pillar [0 .. num_pillars-1]
        returns: (num_pillars, C)
        """
        point_emb = self.pfn(point_feats)  # (Ns, C)

        # Segment max by pillar via sorting and per-segment reduction
        order = torch.argsort(inv)
        inv_sorted = inv[order]
        feats_sorted = point_emb[order]

        # Find segment boundaries
        N = inv_sorted.numel()
        if N == 0:
            return torch.zeros((num_pillars, self.out_channels), device=point_emb.device)
        is_start = torch.ones(N, dtype=torch.bool, device=inv_sorted.device)
        is_start[1:] = inv_sorted[1:] != inv_sorted[:-1]
        start_idx = torch.nonzero(is_start, as_tuple=False).flatten()
        start_idx = torch.cat([start_idx, torch.tensor([N], device=start_idx.device)])

        out = torch.empty((num_pillars, feats_sorted.shape[1]), device=feats_sorted.device)
        # Loop per pillar (kept small: H*W typical ~5k)
        for k in range(num_pillars):
            s = start_idx[k].item()
            e = start_idx[k + 1].item()
            seg = feats_sorted[s:e]
            if seg.numel() == 0:
                out[k] = 0
            else:
                out[k], _ = seg.max(dim=0)
        return out

    def _scatter_to_bev(self, pillar_feats, unique_ids, H, W):
        """Scatter per-pillar features to dense (C, H, W)."""
        C = pillar_feats.shape[1]
        bev = torch.zeros((C, H, W), device=pillar_feats.device)
        gy = unique_ids // W
        gx = unique_ids % W
        bev[:, gy, gx] = pillar_feats.t()
        return bev

    def _encode_single(self, pts_b: torch.Tensor):
        device = next(self.pfn.parameters()).device
        if pts_b.numel() == 0:
            return torch.zeros((self.out_channels, self.bev_h, self.bev_w), device=device)

        pts_b = pts_b.to(device).float()
        # Drop padded rows (all zeros) and NaNs
        if pts_b.ndim == 2:
            zero_mask = (pts_b.abs().sum(dim=1) == 0)
            if zero_mask.any():
                pts_b = pts_b[~zero_mask]
        pts_b = pts_b[~torch.isnan(pts_b).any(dim=1)]
        mask, gx, gy = self._mask_and_grid(pts_b)
        if not mask.any():
            return torch.zeros((self.out_channels, self.bev_h, self.bev_w), device=device)

        pts_sel = pts_b[mask]
        gx = gx.to(torch.long)
        gy = gy.to(torch.long)

        # Unique pillars and inverse map
        grid_flat = gy * self.bev_w + gx
        unique_ids, inv = torch.unique(grid_flat, return_inverse=True)
        # Count points per pillar and cap to max_points_per_pillar by random subsampling
        counts = torch.bincount(inv, minlength=unique_ids.shape[0])

        if self.max_points_per_pillar is not None and self.max_points_per_pillar > 0:
            # For pillars with too many points, randomly sample K per pillar
            # Build indices to keep
            keep_mask = torch.zeros_like(inv, dtype=torch.bool)
            # Sort by inv to form contiguous blocks
            order = torch.argsort(inv)
            inv_sorted = inv[order]
            N = inv_sorted.numel()
            is_start = torch.ones(N, dtype=torch.bool, device=inv_sorted.device)
            is_start[1:] = inv_sorted[1:] != inv_sorted[:-1]
            start_idx = torch.nonzero(is_start, as_tuple=False).flatten()
            start_idx = torch.cat([start_idx, torch.tensor([N], device=start_idx.device)])
            for k in range(unique_ids.shape[0]):
                s = start_idx[k].item()
                e = start_idx[k + 1].item()
                seg_len = e - s
                if seg_len <= 0:
                    continue
                if seg_len <= self.max_points_per_pillar:
                    keep_mask[order[s:e]] = True
                else:
                    idx = torch.randperm(seg_len, device=device)[: self.max_points_per_pillar]
                    keep_mask[order[s + idx]] = True
            # Apply mask
            pts_sel = pts_sel[keep_mask]
            gx = gx[keep_mask]
            gy = gy[keep_mask]
            inv = inv[keep_mask]
            counts = torch.bincount(inv, minlength=unique_ids.shape[0])

        # Build per-point PFN input features
        point_feats = self._build_point_features(pts_sel, gx, gy, inv, counts)
        # PFN + max reduce
        pillar_feats = self._pfn_pool(point_feats, inv, unique_ids.shape[0])  # (P, C)
        # Scatter to dense BEV
        bev = self._scatter_to_bev(pillar_feats, unique_ids, self.bev_h, self.bev_w)  # (C, H, W)
        return bev

    def forward(self, points, img_metas=None):
        # Normalize to list[Tensor]
        if points is None:
            raise ValueError('LiDAR points are required when LidarPillarEncoder is enabled.')
        if isinstance(points, torch.Tensor):
            assert points.dim() == 3, 'Tensor points must be (B, N, C)'
            pts_list = [p for p in points]
        elif isinstance(points, (list, tuple)):
            pts_list = []
            for p in points:
                if isinstance(p, torch.Tensor):
                    pts_list.append(p)
                else:
                    pts_list.append(torch.tensor(p))
        else:
            raise TypeError(f'Unsupported points type: {type(points)}')

        bevs = [self._encode_single(pts_b) for pts_b in pts_list]
        bevs = torch.stack(bevs, dim=0)  # (B, C, H, W)
        return bevs

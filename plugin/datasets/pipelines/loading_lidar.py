import os
import numpy as np
from mmdet3d.registry import TRANSFORMS
try:
    from pyquaternion import Quaternion
except Exception:
    Quaternion = None


@TRANSFORMS.register_module(force=True)
class LoadLidarPointsFromFile(object):
    """
    Minimal LiDAR points loader.

    Looks up a file path in the sample dict and loads LiDAR points into
    `results['points']` as a float32 NumPy array.

    - Supported formats: `.pcd.bin` (NuScenes), generic `.bin` (KITTI‑style), `.npy`.
    - Dimension handling: if `load_dim` is None, infer 5 for `.pcd.bin` and 4 for `.bin`.

    Args:
        file_key: preferred key in `results` to read the path from.
        fallback_keys: additional keys to probe if `file_key` is absent.
        load_dim: number of floats per point in file; if None, infer by extension.
        use_dim: indices to keep from loaded points (e.g., (0,1,2,3) for xyz,i).
        filter_roi: optional ((xmin,xmax),(ymin,ymax),(zmin,zmax)) filter to KEEP only points inside.
        remove_close: if True, remove points close to LiDAR to suppress ego-vehicle returns.
        close_radius: cylindrical XY radius (meters) for near-ego removal when `close_bounds` is None.
        close_bounds: optional ((xmin,xmax),(ymin,ymax),(zmin,zmax)) BOX to REMOVE points inside.
        close_z: optional (zmin,zmax) used with `close_radius`; None means ignore Z in removal.
    """

    def __init__(self,
                 file_key='lidar_path',
                 fallback_keys=('lidar_filename', 'lidar_file', 'lidar_path'),
                 load_dim=None,
                 use_dim=(0, 1, 2, 3),
                 filter_roi=None,
                 remove_close=True,
                 close_radius=1.6,
                 close_bounds=None,
                 close_z=None,
                 sample_num=None,
                 shuffle=True,
                 produce_ego_aligned=True,
                 ego_aligned_key='points_ego_aligned'):
        self.file_key = file_key
        self.fallback_keys = tuple(fallback_keys) if fallback_keys else ()
        self.load_dim = load_dim
        self.use_dim = tuple(use_dim)
        self.filter_roi = filter_roi
        # Near-ego removal configuration
        self.remove_close = bool(remove_close)
        self.close_radius = float(close_radius) if close_radius is not None else None
        self.close_bounds = close_bounds  # expected shape: ((xmin,xmax),(ymin,ymax),(zmin,zmax))
        self.close_z = tuple(close_z) if close_z is not None else None
        self.sample_num = int(sample_num) if sample_num is not None else None
        self.shuffle = bool(shuffle)
        self.produce_ego_aligned = bool(produce_ego_aligned)
        self.ego_aligned_key = str(ego_aligned_key) if ego_aligned_key else None

    def __call__(self, results):
        path = results.get(self.file_key, None)
        if path is None:
            for k in self.fallback_keys:
                if k in results:
                    path = results[k]
                    break
        # If nothing found or file missing, leave results unchanged.
        if path is None or (isinstance(path, str) and not os.path.isfile(path)):
            return results

        if isinstance(path, list):
            # if multiple sweeps provided, concatenate after potential transform in future
            pts_list = [self._load_one(p) for p in path]
            pts = np.concatenate(pts_list, axis=0) if len(pts_list) > 0 else np.zeros((0, len(self.use_dim)), dtype=np.float32)
        else:
            pts = self._load_one(path)

        # Keep-only ROI filter
        if self.filter_roi is not None and pts.shape[0] > 0:
            x_rng, y_rng, z_rng = self.filter_roi
            mask = (
                (pts[:, 0] >= x_rng[0]) & (pts[:, 0] <= x_rng[1]) &
                (pts[:, 1] >= y_rng[0]) & (pts[:, 1] <= y_rng[1]) &
                (pts[:, 2] >= z_rng[0]) & (pts[:, 2] <= z_rng[1])
            )
            pts = pts[mask]

        # Remove near-ego points (ego-vehicle/roof/bonnet) before sampling
        if self.remove_close and pts.shape[0] > 0:
            if self.close_bounds is not None:
                xb, yb, zb = self.close_bounds
                in_box = (
                    (pts[:, 0] >= xb[0]) & (pts[:, 0] <= xb[1]) &
                    (pts[:, 1] >= yb[0]) & (pts[:, 1] <= yb[1]) &
                    (pts[:, 2] >= zb[0]) & (pts[:, 2] <= zb[1])
                )
                pts = pts[~in_box]
            elif self.close_radius is not None:
                r2 = pts[:, 0] * pts[:, 0] + pts[:, 1] * pts[:, 1]
                in_cyl = r2 <= (self.close_radius * self.close_radius)
                if self.close_z is not None:
                    zmin, zmax = self.close_z
                    in_cyl = in_cyl & (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
                pts = pts[~in_cyl]

        # Enforce fixed-size sampling and padding if requested
        if self.sample_num is not None:
            n, d = pts.shape
            target = self.sample_num
            if n >= target and target > 0:
                if self.shuffle:
                    idx = np.random.choice(n, target, replace=False)
                else:
                    # take first target points deterministically
                    idx = np.arange(target)
                pts = pts[idx]
            elif n < target and target > 0:
                pad = np.zeros((target - n, d), dtype=pts.dtype)
                if self.shuffle and n > 0:
                    perm = np.random.permutation(n)
                    pts = pts[perm]
                pts = np.concatenate([pts, pad], axis=0)

        pts = pts.astype(np.float32, copy=False)
        results['points'] = pts

        # Optionally create an ego-axes-aligned copy (rotation only; no translation)
        # This aligns LiDAR points with the BEV/vector frame that is ego-oriented
        # but centered at the LiDAR origin (as used in NuscDataset.get_sample).
        if self.produce_ego_aligned and self.ego_aligned_key and 'lidar2ego_rotation' in results and Quaternion is not None:
            try:
                R = Quaternion(results['lidar2ego_rotation']).rotation_matrix.astype(np.float32)
                xyz = pts[:, :3]
                xyz_e = xyz.dot(R.T)  # row-vector form: v_E = v_L @ R^T
                pts_e = pts.copy()
                pts_e[:, :3] = xyz_e
                results['points'] = pts_e
            except Exception:
                # Fail silently if quaternion not available/malformed
                pass
        return results

    def _load_one(self, path):
        # Determine loader and point dimension
        lower = path.lower()
        if lower.endswith('.pcd.bin') or lower.endswith('.bin'):
            raw = np.fromfile(path, dtype=np.float32)
            # Infer dim if not set explicitly
            if self.load_dim is None:
                inferred = 5 if lower.endswith('.pcd.bin') else 4
                # If file size doesn't divide evenly, try alternate common dims
                if raw.size % inferred != 0:
                    for alt in (4, 5, 6):
                        if raw.size % alt == 0:
                            inferred = alt
                            break
                load_dim = inferred
            else:
                load_dim = int(self.load_dim)
            if raw.size % load_dim != 0:
                raise ValueError(f'File size {raw.size} not divisible by load_dim {load_dim} for {path}')
            pts = raw.reshape(-1, load_dim)
        elif lower.endswith('.npy'):
            pts = np.load(path)
            if pts.ndim != 2:
                pts = pts.reshape(-1, pts.shape[-1])
        elif lower.endswith('.pcd'):
            pts = self._load_pcd(path)
        else:
            raise ValueError(f'Unsupported lidar file type: {path}')

        # Select requested channels
        pts = pts[:, list(self.use_dim)] if self.use_dim is not None else pts
        return pts

    @staticmethod
    def _load_pcd(path):
        """Parse a PCL .pcd file (binary DATA section) into an (N, F) float32 array.

        Supports the standard PCL v0.7 header with FIELDS / SIZE / TYPE / COUNT
        / POINTS / DATA lines. Returns one column per field (with COUNT>1
        flattened so an `rgb` field stays a single column).
        """
        type_map = {'F': 'f', 'I': 'i', 'U': 'u'}
        with open(path, 'rb') as f:
            meta = {}
            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f'Unexpected EOF in PCD header: {path}')
                text = line.decode('ascii', errors='replace').strip()
                if not text or text.startswith('#'):
                    continue
                parts = text.split()
                meta[parts[0]] = parts[1:]
                if parts[0] == 'DATA':
                    break
            fields = meta.get('FIELDS', [])
            sizes = [int(s) for s in meta.get('SIZE', [])]
            types = meta.get('TYPE', [])
            counts = [int(c) for c in meta.get('COUNT', ['1'] * len(fields))]
            npoints = int(meta.get('POINTS', ['0'])[0])
            data_kind = meta.get('DATA', ['ascii'])[0]
            if data_kind != 'binary':
                raise ValueError(
                    f'PCD loader only supports DATA binary, got {data_kind!r}: {path}')
            dtype_fields = []
            for fname, fsize, ftype, fcount in zip(fields, sizes, types, counts):
                np_type = f'{type_map[ftype]}{fsize}'
                dtype_fields.append(
                    (fname, np_type, fcount) if fcount > 1 else (fname, np_type))
            dtype = np.dtype(dtype_fields)
            raw = f.read(npoints * dtype.itemsize)
        arr = np.frombuffer(raw, dtype=dtype, count=npoints)
        cols = []
        for fname in fields:
            col = arr[fname]
            # Re-interpret `rgb` packed float as raw bytes -> single channel intensity proxy
            # (lowest byte of the packed RGB int). Keeps the column count predictable.
            if fname == 'rgb' and col.dtype.kind == 'f':
                col = col.view(np.uint32).astype(np.float32)
            cols.append(col.astype(np.float32).reshape(npoints, -1))
        return np.concatenate(cols, axis=1)

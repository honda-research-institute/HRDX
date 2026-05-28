#!/usr/bin/env python
"""Fast per-sample Chamfer-based metrics from an existing submission file.

Implementation notes:
  1. Uses orjson/ujson for fast JSON loading (5.87 GB → ~30s)
  2. Pure numpy interpolation (no Shapely)
  3. Lightweight numpy Chamfer distance (no torch cdist full expansion)
  4. Multiprocessing across tokens

Usage:
    python tools/analysis/extract_per_sample_ap_fast.py \
        --config plugin/configs/maptracker_aerial_only/rdx_dataset/maptracker_rdx_stage2_warmup_aerial_only.py \
        --submission work_dirs/maptracker_rdx_stage2_warmup_aerial_only/val_results/submission_vector.json \
        --out work_dirs/analysis/aerial_only_stage2_ap.csv

    # Quickest mode – skip per-class AP, just compute mean Chamfer:
    python tools/analysis/extract_per_sample_ap_fast.py \
        --config <CONFIG> --submission <JSON> --out <CSV> --chamfer-only

    # Then visualize:
    python tools/analysis/plot_loss_heatmap.py \
        --csv work_dirs/analysis/aerial_only_stage2_ap.csv \
        --loss-col mean_chamfer_dist --out-dir work_dirs/analysis/ap_maps/
"""

import argparse
import csv
import importlib
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

INTERP_NUM = 100  # 100 is enough for ranking; original used 200
THRESHOLDS = [0.5, 1.0, 1.5]


# ---------------------------------------------------------------------------
# Fast JSON loading
# ---------------------------------------------------------------------------
def load_json_fast(path):
    """Load JSON using the fastest available library."""
    t0 = time.time()
    raw = open(path, 'rb').read()
    print(f'  Read {len(raw)/1e9:.2f} GB in {time.time()-t0:.1f}s')

    t1 = time.time()
    try:
        import orjson
        data = orjson.loads(raw)
        lib = 'orjson'
    except ImportError:
        try:
            import ujson
            data = ujson.loads(raw)
            lib = 'ujson'
        except ImportError:
            import json
            data = json.loads(raw)
            lib = 'json (slow – install orjson for ~5× speedup)'
    del raw
    print(f'  Parsed JSON with {lib} in {time.time()-t1:.1f}s')
    return data


def load_predictions_fast(path):
    """Load predictions from either submission JSON or pos_predictions PKL.

    Supported formats:
      - submission_vector.json: {"results": {token: {...}}}
      - *.pkl produced by BaseMapDataset.format_results (pos_predictions.pkl):
        list[dict(vectors, labels, scores, meta=...)]
    """
    suffix = os.path.splitext(path)[1].lower()

    if suffix == '.json':
        submission = load_json_fast(path)
        if isinstance(submission, dict) and 'results' in submission:
            return submission['results']
        if isinstance(submission, dict):
            return submission
        raise TypeError(f'Unexpected JSON submission type: {type(submission)}')

    if suffix in ('.pkl', '.pickle'):
        t0 = time.time()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f'  Loaded PKL in {time.time()-t0:.1f}s')

        if isinstance(data, dict):
            if 'results' in data and isinstance(data['results'], dict):
                return data['results']
            return data

        if isinstance(data, list):
            results = {}
            missing_token = 0
            for item in data:
                if not isinstance(item, dict):
                    continue
                token = item.get('token')
                if token is None:
                    meta = item.get('meta', {})
                    if isinstance(meta, dict):
                        token = meta.get('token')
                if token is None:
                    missing_token += 1
                    continue
                results[token] = item
            print(f'  Parsed list PKL into {len(results)} token entries '
                  f'({missing_token} missing token)')
            return results

        raise TypeError(f'Unsupported PKL prediction type: {type(data)}')

    raise ValueError(f'Unsupported prediction file extension: {suffix}')


def _vector_to_polyline(vector):
    """Normalize vector encoding to shape (N, 2)."""
    arr = np.asarray(vector, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr

    flat = arr.reshape(-1)
    if flat.size < 4 or flat.size % 2 != 0:
        return None
    return flat.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Pure-numpy interpolation (replaces Shapely LineString)
# ---------------------------------------------------------------------------
def interp_fixed_num_np(pts, num=INTERP_NUM):
    """Interpolate a polyline to `num` equally-spaced points. Pure numpy."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return np.tile(pts[0] if len(pts) else np.zeros(2), (num, 1))
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cum_lengths[-1]
    if total < 1e-12:
        return np.tile(pts[0], (num, 1))
    targets = np.linspace(0, total, num)
    # Find segment index for each target
    seg_idx = np.searchsorted(cum_lengths, targets, side='right') - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_lengths) - 1)
    # Local parameter within each segment
    t = (targets - cum_lengths[seg_idx]) / np.maximum(seg_lengths[seg_idx], 1e-12)
    t = np.clip(t, 0, 1)
    result = pts[seg_idx] + t[:, None] * diffs[seg_idx]
    return result


# ---------------------------------------------------------------------------
# Fast Chamfer distance (numpy only, no torch)
# ---------------------------------------------------------------------------
def chamfer_dist_np(a, b):
    """Symmetric Chamfer distance between two (N,2) and (M,2) point arrays."""
    # a: (N,D), b: (M,D)
    # dist_matrix: (N,M)
    diff = a[:, None, :] - b[None, :, :]           # (N,M,D)
    sq_dist = (diff ** 2).sum(axis=2)               # (N,M)
    d_a2b = np.sqrt(sq_dist.min(axis=1)).mean()     # mean of min-per-row
    d_b2a = np.sqrt(sq_dist.min(axis=0)).mean()     # mean of min-per-col
    return (d_a2b + d_b2a) / 2.0


def chamfer_dist_batch(pred_lines, gt_lines):
    """Chamfer distance matrix (num_pred, num_gt) using numpy.

    pred_lines: (M, P, 2), gt_lines: (N, P, 2)
    Returns: (M, N) distance matrix
    """
    M, P, D = pred_lines.shape
    N = gt_lines.shape[0]
    dist = np.empty((M, N), dtype=np.float64)
    for i in range(M):
        pi = pred_lines[i]  # (P, D)
        for j in range(N):
            gj = gt_lines[j]  # (P, D)
            dd = pi[:, None, :] - gj[None, :, :]  # (P, P, D)
            sq = (dd ** 2).sum(axis=2)  # (P, P)
            dist[i, j] = (np.sqrt(sq.min(axis=1)).mean() + np.sqrt(sq.min(axis=0)).mean()) / 2.0
    return dist


# ---------------------------------------------------------------------------
# Per-token metric computation (runs in worker processes)
# ---------------------------------------------------------------------------
def compute_token_metrics(args):
    """Compute metrics for one token. Designed for multiprocessing."""
    token, pred, gt, id2cat, thresholds, chamfer_only = args

    # Penalty distance for unmatched predictions / GTs.
    # Should be larger than typical matched Chamfer (P99 ≈ 6) but finite
    # so the metric stays plottable.  Using the ROI diagonal / 2 is a
    # reasonable upper-bound on within-ROI Chamfer.  60x30 ROI ⇒ ~33.5.
    UNMATCHED_PENALTY = 15.0

    metrics = {}
    all_chamfer = []           # matched-pair Chamfer (original metric)
    all_chamfer_penalized = [] # includes penalties for FP & FN
    total_pred = 0
    total_gt = 0

    # Parse pred into per-class lists once
    pred_by_class = defaultdict(lambda: {'vectors': [], 'scores': []})
    pred_labels = pred.get('labels', [])
    pred_scores = pred.get('scores', [])
    pred_vectors_raw = pred.get('vectors', [])
    n_det = min(len(pred_labels), len(pred_scores), len(pred_vectors_raw))

    for i in range(n_det):
        lbl = int(pred_labels[i])
        score = float(pred_scores[i])
        poly = _vector_to_polyline(pred_vectors_raw[i])
        if poly is None or len(poly) < 2:
            continue
        pred_by_class[lbl]['vectors'].append(poly)
        pred_by_class[lbl]['scores'].append(score)

    for label_id, cat_name in id2cat.items():
        gt_vectors = gt.get(label_id, [])
        pv = pred_by_class.get(label_id, {'vectors': [], 'scores': []})
        pred_vectors = pv['vectors']
        pred_scores = pv['scores']

        n_gt = len(gt_vectors)
        n_pred = len(pred_vectors)
        total_gt += n_gt
        total_pred += n_pred

        # Trivial cases
        if n_gt == 0 and n_pred == 0:
            # Neither GT nor predictions – skip this class entirely
            # (not included in mean_AP; standard mAP convention)
            if not chamfer_only:
                for thr in thresholds:
                    metrics[f'AP@{thr}_{cat_name}'] = float('nan')
            continue
        if n_gt == 0:
            # False positive predictions for a class with no GT
            if not chamfer_only:
                for thr in thresholds:
                    metrics[f'AP@{thr}_{cat_name}'] = float('nan')  # no GT → exclude
            # Penalise every false-positive prediction
            all_chamfer_penalized.extend([UNMATCHED_PENALTY] * n_pred)
            continue
        if n_pred == 0:
            # All GT missed
            if not chamfer_only:
                for thr in thresholds:
                    metrics[f'AP@{thr}_{cat_name}'] = 0.0
            all_chamfer.append(float('inf'))  # missed everything
            all_chamfer_penalized.extend([UNMATCHED_PENALTY] * n_gt)
            continue

        # Interpolate with numpy (fast)
        pred_lines = np.stack([interp_fixed_num_np(v) for v in pred_vectors])
        gt_lines = np.stack([interp_fixed_num_np(np.array(v)) for v in gt_vectors])

        # Chamfer distance matrix
        dist_mat = chamfer_dist_batch(pred_lines, gt_lines)  # (M, N)

        scores = np.array(pred_scores)
        sorted_idx = np.argsort(-scores)  # descending score

        # Precompute closest GT for each detection (same as official AP.py)
        dist_min = dist_mat.min(axis=1)      # (M,)
        dist_argmin = dist_mat.argmin(axis=1) # (M,)

        if not chamfer_only:
            for thr in thresholds:
                tp = np.zeros(n_pred, dtype=np.float32)
                fp = np.zeros(n_pred, dtype=np.float32)
                gt_covered = np.zeros(n_gt, dtype=bool)

                for rank, det_idx in enumerate(sorted_idx):
                    if dist_min[det_idx] <= thr:
                        matched_gt_idx = dist_argmin[det_idx]
                        if not gt_covered[matched_gt_idx]:
                            gt_covered[matched_gt_idx] = True
                            tp[rank] = 1
                        else:
                            fp[rank] = 1
                    else:
                        fp[rank] = 1

                # Compute AP as area under the precision-recall curve
                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(fp)
                recalls = cum_tp / n_gt
                precisions = cum_tp / (cum_tp + cum_fp + np.finfo(np.float32).eps)

                # VOC-style AP (monotone envelope + area)
                mrec = np.concatenate(([0.0], recalls, [1.0]))
                mpre = np.concatenate(([0.0], precisions, [0.0]))
                for i in range(len(mpre) - 1, 0, -1):
                    mpre[i - 1] = max(mpre[i - 1], mpre[i])
                idx = np.where(mrec[1:] != mrec[:-1])[0]
                ap_val = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
                metrics[f'AP@{thr}_{cat_name}'] = float(ap_val)

        # Mean Chamfer for matched pairs (greedy at thr=1.5)
        gt_covered = np.zeros(n_gt, dtype=bool)
        n_matched = 0
        for det_idx in sorted_idx:
            matched_gt_idx = dist_argmin[det_idx]
            if not gt_covered[matched_gt_idx]:
                d = float(dist_mat[det_idx, matched_gt_idx])
                all_chamfer.append(d)
                all_chamfer_penalized.append(d)
                gt_covered[matched_gt_idx] = True
                n_matched += 1

        # Penalise unmatched GTs (false negatives) and excess preds (false positives)
        n_unmatched_gt = n_gt - n_matched
        n_unmatched_pred = max(0, n_pred - n_matched)
        all_chamfer_penalized.extend([UNMATCHED_PENALTY] * n_unmatched_gt)
        all_chamfer_penalized.extend([UNMATCHED_PENALTY] * n_unmatched_pred)

    # Aggregate
    if not chamfer_only:
        # Only average over classes that had GT (NaN = no GT → excluded)
        ap_vals = [v for k, v in metrics.items() if k.startswith('AP@') and not np.isnan(v)]
        metrics['mean_AP'] = float(np.mean(ap_vals)) if ap_vals else float('nan')
        metrics['num_classes_with_gt'] = len(ap_vals) // len(thresholds) if thresholds else 0
    metrics['mean_chamfer_dist'] = float(np.mean(all_chamfer)) if all_chamfer else float('nan')
    metrics['penalized_chamfer'] = float(np.mean(all_chamfer_penalized)) if all_chamfer_penalized else float('nan')
    metrics['num_pred'] = total_pred
    metrics['num_gt'] = total_gt

    return token, metrics


# ---------------------------------------------------------------------------
# Lat/Lon lookup
# ---------------------------------------------------------------------------
def build_latlon_lookup(ann_file):
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)
    lookup = {}
    samples = data if isinstance(data, list) else data.get('samples', data.get('infos', []))
    for sample in samples:
        token = sample.get('token')
        llh = sample.get('lat_long_heading', [None, None, None])
        if token is not None:
            lookup[token] = {
                'lat': float(llh[0]) if llh[0] is not None else None,
                'lon': float(llh[1]) if llh[1] is not None else None,
                'heading': float(llh[2]) if llh[2] is not None else None,
                'scene_name': sample.get('scene_name', ''),
            }
    return lookup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Fast per-sample AP extraction')
    p.add_argument('--config', required=True, help='Config .py path')
    p.add_argument('--submission', required=True,
                   help='Path to prediction file: submission_vector.json or pos_predictions.pkl')
    p.add_argument('--out', default='work_dirs/analysis/per_sample_ap.csv')
    p.add_argument('--ann-file', default=None,
                   help='Override annotation pkl for lat/lon lookup')
    p.add_argument('--chamfer-only', action='store_true',
                   help='Skip per-class AP, only compute mean Chamfer (fastest)')
    p.add_argument('--workers', type=int, default=0,
                   help='Num parallel workers (0 = auto, 1 = sequential)')
    p.add_argument('--interp-num', type=int, default=INTERP_NUM,
                   help='Interpolation points per line (default 100, original used 200)')
    return p.parse_args()


def main():
    args = parse_args()
    global INTERP_NUM
    INTERP_NUM = args.interp_num

    t_start = time.time()

    # Load config to get eval_config and class mapping
    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    plugin_dirs = cfg.plugin_dir if isinstance(cfg.plugin_dir, list) else [cfg.plugin_dir]
    for pd in plugin_dirs:
        importlib.import_module(pd.rstrip('/').replace('/', '.').rstrip('.'))

    # Load submission
    print(f'Loading predictions from {args.submission} ...')
    results = load_predictions_fast(args.submission)
    print(f'  {len(results)} tokens in predictions')

    # Build GT (this is fast — it loads a small pkl + vectorization)
    print('Building GT evaluator ...')
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')
    from plugin.datasets.evaluation.vector_eval import VectorEvaluate
    evaluator = VectorEvaluate(Config(cfg.eval_config))
    id2cat = evaluator.id2cat
    gts = evaluator.gts
    print(f'  {len(gts)} tokens in GT')

    # Lat/lon
    ann_file = args.ann_file or cfg.eval_config.get('ann_file',
                cfg.data.get('val', {}).get('ann_file', ''))
    print(f'Building lat/lon lookup from {ann_file} ...')
    latlon_lookup = build_latlon_lookup(ann_file)

    # Prepare work items
    tokens = sorted(gts.keys())
    empty_pred = {'vectors': [], 'scores': [], 'labels': []}
    work_items = [
        (tok, results.get(tok, empty_pred), gts[tok], id2cat,
         THRESHOLDS, args.chamfer_only)
        for tok in tokens
    ]

    # Process
    n_workers = args.workers if args.workers > 0 else min(cpu_count(), 16)
    print(f'Computing per-sample metrics for {len(tokens)} tokens '
          f'(workers={n_workers}, chamfer_only={args.chamfer_only}) ...')

    t_compute = time.time()
    if n_workers == 1:
        # Sequential — easier to debug
        results_list = []
        for i, item in enumerate(work_items):
            results_list.append(compute_token_metrics(item))
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t_compute
                rate = (i + 1) / elapsed
                eta = (len(work_items) - i - 1) / rate
                print(f'  {i+1}/{len(work_items)} ({rate:.1f} tok/s, ETA {eta:.0f}s)')
    else:
        with Pool(n_workers) as pool:
            results_list = []
            for i, result in enumerate(pool.imap_unordered(compute_token_metrics, work_items, chunksize=64)):
                results_list.append(result)
                if (i + 1) % 2000 == 0:
                    elapsed = time.time() - t_compute
                    rate = (i + 1) / elapsed
                    eta = (len(work_items) - i - 1) / rate
                    print(f'  {i+1}/{len(work_items)} ({rate:.1f} tok/s, ETA {eta:.0f}s)')

    print(f'  Computed metrics in {time.time()-t_compute:.1f}s')

    # Assemble rows
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for token, metrics in results_list:
        geo = latlon_lookup.get(token, {'lat': None, 'lon': None,
                                         'heading': None, 'scene_name': ''})
        row = {
            'token': token,
            'scene_name': geo.get('scene_name', ''),
            'lat': geo['lat'],
            'lon': geo['lon'],
            'heading': geo['heading'],
        }
        row.update(metrics)
        rows.append(row)

    # Write CSV
    if not rows:
        print('No results!')
        return

    fieldnames = list(rows[0].keys())
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_time = time.time() - t_start
    print(f'\nWrote {len(rows)} rows to {args.out} in {total_time:.1f}s total')

    # Summary
    chamfer_dists = [r['mean_chamfer_dist'] for r in rows
                     if not np.isnan(r['mean_chamfer_dist'])]
    pen_chamfers = [r['penalized_chamfer'] for r in rows
                    if not np.isnan(r['penalized_chamfer'])]
    if not args.chamfer_only:
        mean_aps = [r['mean_AP'] for r in rows]
        print(f'  mean_AP: mean={np.mean(mean_aps):.4f}, std={np.std(mean_aps):.4f}')
    if chamfer_dists:
        print(f'  chamfer_dist: mean={np.mean(chamfer_dists):.4f}, '
              f'median={np.median(chamfer_dists):.4f}, max={np.max(chamfer_dists):.4f}')
    if pen_chamfers:
        print(f'  penalized_chamfer: mean={np.mean(pen_chamfers):.4f}, '
              f'median={np.median(pen_chamfers):.4f}, max={np.max(pen_chamfers):.4f}')

    # Worst 10
    sort_key = 'mean_AP' if not args.chamfer_only else 'mean_chamfer_dist'
    reverse = args.chamfer_only  # worst = highest chamfer or lowest AP
    sorted_rows = sorted(rows, key=lambda r: r.get(sort_key, 0),
                         reverse=reverse)
    print(f'\nBottom-10 samples by {sort_key}:')
    for r in sorted_rows[:10]:
        ap_str = f"AP={r['mean_AP']:.3f}, " if 'mean_AP' in r else ''
        cd = r['mean_chamfer_dist']
        cd_str = f'{cd:.3f}' if not np.isnan(cd) else 'nan'
        print(f"  token={r['token']}, scene={r['scene_name']}, "
              f"{ap_str}chamfer={cd_str}, lat={r['lat']}, lon={r['lon']}")


if __name__ == '__main__':
    main()

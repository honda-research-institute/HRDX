#!/usr/bin/env python
"""Extract per-sample loss for a trained MapTracker model.

Iterates over every validation (or train) sample with batch_size=1,
runs a forward_train pass, and records per-sample losses alongside
geographic metadata (lat, lon, heading) from the dataset pkl.

Outputs a CSV:
    token, scene_name, lat, lon, heading, loss_total, loss_cls, loss_reg,
    loss_seg, loss_seg_dice, loss_<attr>...

Usage:
    python tools/analysis/extract_per_sample_loss.py \
        --config plugin/configs/maptracker_aerial_only/rdx_dataset/maptracker_rdx_stage3_joint_finetune_aerial_only.py \
        --checkpoint work_dirs/maptracker_rdx_stage3_joint_finetune_aerial_only/latest.pth \
        --out work_dirs/analysis/aerial_only_val_losses.csv \
        --split val

    # To also extract losses for the camera-only model (for delta comparison):
    python tools/analysis/extract_per_sample_loss.py \
        --config plugin/configs/maptracker/rdx_dataset/maptracker_rdx_stage3_joint_finetune.py \
        --checkpoint work_dirs/maptracker_rdx_stage3_joint_finetune/latest.pth \
        --out work_dirs/analysis/camera_only_val_losses.csv \
        --split val
"""

import argparse
import csv
import importlib
import os
import sys
import pickle
import warnings

import torch
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope

# ---- path setup ----
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description='Extract per-sample loss')
    p.add_argument('--config', required=True, help='Path to config .py')
    p.add_argument('--checkpoint', required=True, help='Path to trained checkpoint .pth')
    p.add_argument('--out', default='work_dirs/analysis/per_sample_losses.csv',
                   help='Output CSV path')
    p.add_argument('--split', default='val', choices=['val', 'train', 'test'],
                   help='Which dataset split to iterate over')
    p.add_argument('--max-samples', type=int, default=None,
                   help='Limit number of samples (for debugging)')
    p.add_argument('--device', default='cuda:0', help='Device')
    return p.parse_args()


def build_latlon_lookup(ann_file):
    """Build a {token -> (lat, lon, heading)} dict from the dataset pkl."""
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


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # Register all plugin modules
    plugin_dirs = cfg.plugin_dir if isinstance(cfg.plugin_dir, list) else [cfg.plugin_dir]
    for plugin_dir in plugin_dirs:
        module_path = plugin_dir.rstrip('/').replace('/', '.').rstrip('.')
        importlib.import_module(module_path)

    init_default_scope('mmdet3d')

    # ---- Build the lat/lon lookup from the pkl ----
    split_cfg = cfg.data[args.split]
    ann_file = split_cfg['ann_file']
    print(f'Building lat/lon lookup from {ann_file} ...')
    latlon_lookup = build_latlon_lookup(ann_file)
    print(f'  Found {len(latlon_lookup)} tokens with lat/lon metadata')

    # ---- Build dataset ----
    # We need the TRAINING pipeline (with GT) but iterate deterministically
    # over the val/train split. We override the pipeline to include GT loading.
    from mmengine.registry import DATASETS as MMENGINE_DATASETS
    from plugin.datasets.builder import build_dataloader

    # Use the eval_config pipeline (has GT vectorize + rasterize) but also
    # load aerial images. We build a custom dataset config for loss extraction.
    loss_dataset_cfg = dict(split_cfg)

    # Use the train pipeline keys but in test_mode=False so we get GT
    # We need: VectorizeMap, RasterizeMap (semantic_mask), LoadAerialImageFromFile,
    #          LoadMultiViewImagesFromFiles, FormatBundleMap, Collect3D
    # The easiest approach: use the training pipeline but disable augmentation
    loss_pipeline = []
    train_pipeline = cfg.data.get('train', {}).get('pipeline', cfg.train_pipeline)

    for step in train_pipeline:
        step_type = step.get('type', '')
        # Skip augmentation steps
        if step_type in ('RandomFlip3D', 'PhotoMetricDistortion3D',
                         'PhotoMetricDistortionMultiViewImage',
                         'RandomScaleImageMultiViewImage'):
            continue
        loss_pipeline.append(step)

    loss_dataset_cfg['pipeline'] = loss_pipeline
    loss_dataset_cfg['test_mode'] = False  # so we get GT vectors
    loss_dataset_cfg['seq_split_num'] = -1  # no sequence splitting
    loss_dataset_cfg['multi_frame'] = False  # single frame only
    loss_dataset_cfg['matching'] = False

    dataset = MMENGINE_DATASETS.build(Config(loss_dataset_cfg))
    print(f'Dataset has {len(dataset)} samples')

    # ---- Build model ----
    from mmengine.registry import MODELS
    model = MODELS.build(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'Warning: missing keys: {missing[:5]}...')
    if unexpected:
        print(f'Warning: unexpected keys: {unexpected[:5]}...')

    model = model.to(args.device)
    model.eval()  # batch norm in eval mode, but we call forward_train

    # ---- Iterate and collect losses ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Determine CSV columns from first sample
    all_rows = []
    num_samples = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))

    print(f'Extracting per-sample losses for {num_samples} samples...')
    loss_keys = None

    for i in range(num_samples):
        if i % 100 == 0:
            print(f'  [{i}/{num_samples}]')

        try:
            data = dataset[i]
        except Exception as e:
            warnings.warn(f'Failed to load sample {i}: {e}')
            continue

        if data is None:
            continue

        # Extract token from img_metas
        img_metas = data.get('img_metas')
        if hasattr(img_metas, 'data'):
            img_metas = img_metas.data
        token = img_metas.get('token', f'idx_{i}')
        scene_name = img_metas.get('scene_name', '')

        # Move tensors to device and add batch dim
        def to_device_batch(x):
            if isinstance(x, torch.Tensor):
                return x.unsqueeze(0).to(args.device)
            return x

        try:
            with torch.no_grad():
                # Prepare inputs for forward_train
                img = to_device_batch(data.get('img').data if hasattr(data.get('img'), 'data') else data.get('img'))
                vectors = data.get('vectors')
                if hasattr(vectors, 'data'):
                    vectors = vectors.data
                vectors = [vectors]  # batch dim

                semantic_mask = data.get('semantic_mask')
                if hasattr(semantic_mask, 'data'):
                    semantic_mask = semantic_mask.data
                if isinstance(semantic_mask, torch.Tensor):
                    semantic_mask = semantic_mask.unsqueeze(0).to(args.device)

                aerial_img = data.get('aerial_img')
                if aerial_img is not None:
                    if hasattr(aerial_img, 'data'):
                        aerial_img = aerial_img.data
                    if isinstance(aerial_img, torch.Tensor):
                        aerial_img = aerial_img.unsqueeze(0).to(args.device)

                # Reset model state for each sample (no temporal context)
                model.latest_bev_feats = None
                model.latest_seg_logits = None
                model.latest_head_preds = None
                model.latest_head_pos_inds = None
                model.latest_head_gt = None
                if hasattr(model, 'history_bev_feats_all'):
                    model.history_bev_feats_all = []
                    model.history_img_metas_all = []
                if hasattr(model, 'tracked_query_length'):
                    model.tracked_query_length = {}

                img_metas_list = [img_metas]

                # Create local2global_info (identity for single frame)
                local2global_info = [{
                    'ego2global_translation': img_metas.get('ego2global_translation', [0, 0, 0]),
                    'ego2global_rotation': img_metas.get('ego2global_rotation', np.eye(3).tolist()),
                }]

                result = model.forward_train(
                    img=img,
                    vectors=vectors,
                    semantic_mask=semantic_mask,
                    aerial_img=aerial_img,
                    img_metas=img_metas_list,
                    all_prev_data=None,
                    all_local2global_info=local2global_info,
                )

                # result is (loss, log_vars, num_sample) based on the model code
                loss_tensor, log_vars, num_sample = result

        except Exception as e:
            warnings.warn(f'Forward failed for sample {i} (token={token}): {e}')
            continue

        # Build row
        geo = latlon_lookup.get(token, {'lat': None, 'lon': None, 'heading': None, 'scene_name': scene_name})
        row = {
            'token': token,
            'scene_name': geo.get('scene_name', scene_name),
            'sample_idx': i,
            'lat': geo['lat'],
            'lon': geo['lon'],
            'heading': geo['heading'],
            'loss_total': loss_tensor.item(),
        }

        # Add individual loss components
        for k, v in sorted(log_vars.items()):
            row[f'loss_{k}'] = v

        if loss_keys is None:
            loss_keys = sorted(log_vars.keys())

        all_rows.append(row)

    # ---- Write CSV ----
    if not all_rows:
        print('No samples processed successfully!')
        return

    fieldnames = ['token', 'scene_name', 'sample_idx', 'lat', 'lon', 'heading', 'loss_total']
    if loss_keys:
        fieldnames += [f'loss_{k}' for k in loss_keys]

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'\nDone! Wrote {len(all_rows)} rows to {args.out}')

    # Print summary statistics
    total_losses = [r['loss_total'] for r in all_rows]
    print(f'  Loss stats: mean={np.mean(total_losses):.4f}, '
          f'std={np.std(total_losses):.4f}, '
          f'min={np.min(total_losses):.4f}, '
          f'max={np.max(total_losses):.4f}')

    # Top-10 worst samples
    sorted_rows = sorted(all_rows, key=lambda r: r['loss_total'], reverse=True)
    print('\nTop-10 worst samples:')
    for r in sorted_rows[:10]:
        print(f"  token={r['token']}, scene={r['scene_name']}, "
              f"lat={r['lat']}, lon={r['lon']}, loss={r['loss_total']:.4f}")


if __name__ == '__main__':
    main()

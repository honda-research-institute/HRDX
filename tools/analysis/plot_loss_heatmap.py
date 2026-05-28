#!/usr/bin/env python
"""Visualize per-sample losses on an interactive geo-heatmap.

Reads the CSV produced by `extract_per_sample_loss.py` and generates:
  1. An interactive Folium HTML map with colored markers (red=high loss)
  2. A static matplotlib scatter plot with hexbin aggregation
  3. Per-scene boxplot of losses

Usage:
    python tools/analysis/plot_loss_heatmap.py \
        --csv work_dirs/analysis/aerial_only_val_losses.csv \
        --out-dir work_dirs/analysis/maps/

    # Compare aerial-only vs camera-only:
    python tools/analysis/plot_loss_heatmap.py \
        --csv work_dirs/analysis/aerial_only_val_losses.csv \
        --csv-baseline work_dirs/analysis/camera_only_val_losses.csv \
        --out-dir work_dirs/analysis/maps/
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable


def parse_args():
    p = argparse.ArgumentParser(description='Plot loss heatmap on map')
    p.add_argument('--csv', required=True, help='CSV from extract_per_sample_loss.py')
    p.add_argument('--csv-baseline', default=None,
                   help='Optional baseline CSV for delta comparison (e.g., camera-only)')
    p.add_argument('--out-dir', default='work_dirs/analysis/maps/')
    p.add_argument('--loss-col', default='loss_total', help='Which loss column to plot')
    p.add_argument('--percentile-clip', type=float, default=95.0,
                   help='Clip colors at this percentile to avoid outlier domination')
    p.add_argument('--hexbin-gridsize', type=int, default=40,
                   help='Gridsize for hexbin aggregation')
    return p.parse_args()


def plot_folium_map(df, loss_col, out_path, title='Per-Sample Loss Map', clip_pct=95):
    """Interactive Folium map with colored circle markers."""
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print('folium not installed. Skipping interactive map.')
        print('  pip install folium')
        return

    valid = df.dropna(subset=['lat', 'lon', loss_col])
    if len(valid) == 0:
        print('No valid lat/lon data for Folium map.')
        return

    center_lat = valid['lat'].mean()
    center_lon = valid['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                   tiles='OpenStreetMap')

    vmin = valid[loss_col].quantile(0.05)
    vmax = valid[loss_col].quantile(clip_pct / 100)
    cmap = plt.cm.RdYlGn_r  # red = high loss, green = low

    for _, row in valid.iterrows():
        val = row[loss_col]
        norm_val = np.clip((val - vmin) / (vmax - vmin + 1e-8), 0, 1)
        rgba = cmap(norm_val)
        color = mcolors.to_hex(rgba)

        popup_text = (
            f"<b>Token:</b> {row['token']}<br>"
            f"<b>Scene:</b> {row.get('scene_name', 'N/A')}<br>"
            f"<b>{loss_col}:</b> {val:.4f}<br>"
            f"<b>Lat:</b> {row['lat']:.6f}<br>"
            f"<b>Lon:</b> {row['lon']:.6f}"
        )

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{val:.3f}",
        ).add_to(m)

    # Also add a heatmap layer (togglable)
    heat_data = valid[['lat', 'lon', loss_col]].values.tolist()
    HeatMap(heat_data, name='Loss Heatmap', min_opacity=0.3,
            radius=15, blur=10, max_zoom=17).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(out_path)
    print(f'Folium map saved to {out_path}')


def plot_scatter_and_hexbin(df, loss_col, out_dir, clip_pct=95, gridsize=40):
    """Static matplotlib scatter + hexbin plots."""
    valid = df.dropna(subset=['lat', 'lon', loss_col])
    if len(valid) == 0:
        print('No valid data for scatter plot.')
        return

    vmin = valid[loss_col].quantile(0.05)
    vmax = valid[loss_col].quantile(clip_pct / 100)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Scatter plot
    ax = axes[0]
    sc = ax.scatter(valid['lon'], valid['lat'],
                    c=valid[loss_col], cmap='RdYlGn_r',
                    vmin=vmin, vmax=vmax,
                    s=8, alpha=0.6, edgecolors='none')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Per-Sample {loss_col} (scatter)')
    plt.colorbar(sc, ax=ax, label=loss_col)

    # Hexbin aggregation (mean loss per hex)
    ax = axes[1]
    hb = ax.hexbin(valid['lon'], valid['lat'],
                   C=valid[loss_col], reduce_C_function=np.mean,
                   gridsize=gridsize, cmap='RdYlGn_r',
                   vmin=vmin, vmax=vmax,
                   mincnt=1)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Mean {loss_col} per region (hexbin)')
    plt.colorbar(hb, ax=ax, label=f'mean {loss_col}')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'loss_scatter_hexbin.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Scatter/hexbin plot saved to {out_path}')


def plot_per_scene_boxplot(df, loss_col, out_dir, top_n=30):
    """Boxplot of losses per scene, sorted by median loss."""
    valid = df.dropna(subset=[loss_col])
    if 'scene_name' not in valid.columns or valid['scene_name'].nunique() == 0:
        print('No scene_name data for boxplot.')
        return

    # Compute median loss per scene
    scene_medians = valid.groupby('scene_name')[loss_col].median().sort_values(ascending=False)
    top_scenes = scene_medians.head(top_n).index.tolist()
    subset = valid[valid['scene_name'].isin(top_scenes)]

    fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.3)))
    scene_order = scene_medians.loc[top_scenes].index.tolist()
    data_by_scene = [subset[subset['scene_name'] == s][loss_col].values for s in scene_order]

    bp = ax.boxplot(data_by_scene, vert=False, patch_artist=True,
                    labels=[s[:30] for s in scene_order])

    # Color boxes by median
    medians = [np.median(d) for d in data_by_scene]
    norm = plt.Normalize(min(medians), max(medians))
    cmap = plt.cm.RdYlGn_r
    for patch, med in zip(bp['boxes'], medians):
        patch.set_facecolor(cmap(norm(med)))
        patch.set_alpha(0.7)

    ax.set_xlabel(loss_col)
    ax.set_title(f'Top-{top_n} scenes by median {loss_col}')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'loss_per_scene_boxplot.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Per-scene boxplot saved to {out_path}')


def plot_loss_histogram(df, loss_col, out_dir):
    """Overall loss distribution histogram."""
    valid = df.dropna(subset=[loss_col])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valid[loss_col], bins=100, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(valid[loss_col].median(), color='red', linestyle='--', label=f'median={valid[loss_col].median():.3f}')
    ax.axvline(valid[loss_col].mean(), color='orange', linestyle='--', label=f'mean={valid[loss_col].mean():.3f}')
    ax.set_xlabel(loss_col)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {loss_col}')
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'loss_histogram.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Histogram saved to {out_path}')


def plot_delta_map(df_aerial, df_baseline, loss_col, out_dir, clip_pct=95):
    """Plot Δloss = aerial - baseline on the map. Red = aerial worse."""
    merged = df_aerial.merge(df_baseline, on='token', suffixes=('_aerial', '_baseline'))
    col_a = f'{loss_col}_aerial'
    col_b = f'{loss_col}_baseline'

    if col_a not in merged.columns or col_b not in merged.columns:
        print(f'Could not find {col_a} or {col_b} after merge.')
        return

    merged['delta_loss'] = merged[col_a] - merged[col_b]

    lat_col = 'lat_aerial' if 'lat_aerial' in merged.columns else 'lat'
    lon_col = 'lon_aerial' if 'lon_aerial' in merged.columns else 'lon'

    valid = merged.dropna(subset=[lat_col, lon_col, 'delta_loss'])
    if len(valid) == 0:
        print('No valid data for delta map.')
        return

    # Symmetric color range
    abs_max = valid['delta_loss'].abs().quantile(clip_pct / 100)
    vmin, vmax = -abs_max, abs_max

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Scatter
    ax = axes[0]
    sc = ax.scatter(valid[lon_col], valid[lat_col],
                    c=valid['delta_loss'], cmap='RdBu_r',
                    vmin=vmin, vmax=vmax,
                    s=8, alpha=0.6, edgecolors='none')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Δloss (aerial − baseline)\nRed = aerial worse, Blue = aerial better')
    plt.colorbar(sc, ax=ax, label='Δloss')

    # Hexbin
    ax = axes[1]
    hb = ax.hexbin(valid[lon_col], valid[lat_col],
                   C=valid['delta_loss'], reduce_C_function=np.mean,
                   gridsize=40, cmap='RdBu_r',
                   vmin=vmin, vmax=vmax, mincnt=1)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Mean Δloss per region (hexbin)')
    plt.colorbar(hb, ax=ax, label='mean Δloss')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'delta_loss_scatter_hexbin.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Delta map saved to {out_path}')

    # Also save an interactive folium map for delta
    try:
        import folium
        from folium.plugins import HeatMap

        center_lat = valid[lat_col].mean()
        center_lon = valid[lon_col].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        cmap_mpl = plt.cm.RdBu_r

        for _, row in valid.iterrows():
            val = row['delta_loss']
            norm_val = np.clip((val - vmin) / (vmax - vmin + 1e-8), 0, 1)
            rgba = cmap_mpl(norm_val)
            color = mcolors.to_hex(rgba)

            popup_text = (
                f"<b>Token:</b> {row['token']}<br>"
                f"<b>Δloss:</b> {val:.4f}<br>"
                f"<b>Aerial loss:</b> {row[col_a]:.4f}<br>"
                f"<b>Baseline loss:</b> {row[col_b]:.4f}<br>"
                f"<b>Lat:</b> {row[lat_col]:.6f}<br>"
                f"<b>Lon:</b> {row[lon_col]:.6f}"
            )

            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=5, color=color, fill=True,
                fill_color=color, fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"Δ={val:.3f}",
            ).add_to(m)

        folium.LayerControl().add_to(m)
        out_html = os.path.join(out_dir, 'delta_loss_map.html')
        m.save(out_html)
        print(f'Delta folium map saved to {out_html}')
    except ImportError:
        pass

    # Print top-20 worst delta samples
    sorted_delta = valid.sort_values('delta_loss', ascending=False)
    print('\nTop-20 samples where aerial is worse than baseline:')
    for _, r in sorted_delta.head(20).iterrows():
        print(f"  token={r['token']}, Δ={r['delta_loss']:.4f}, "
              f"aerial={r[col_a]:.4f}, baseline={r[col_b]:.4f}, "
              f"lat={r[lat_col]:.6f}, lon={r[lon_col]:.6f}")

    # Save the merged CSV
    merged_out = os.path.join(out_dir, 'delta_losses.csv')
    merged.to_csv(merged_out, index=False)
    print(f'Merged delta CSV saved to {merged_out}')


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading {args.csv} ...')
    df = pd.read_csv(args.csv)
    print(f'  {len(df)} samples, columns: {list(df.columns)}')

    # Replace inf values with NaN for plotting (inf chamfer = no prediction)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ---- Single model analysis ----
    plot_loss_histogram(df, args.loss_col, args.out_dir)
    plot_scatter_and_hexbin(df, args.loss_col, args.out_dir,
                           clip_pct=args.percentile_clip,
                           gridsize=args.hexbin_gridsize)
    plot_folium_map(df, args.loss_col,
                    os.path.join(args.out_dir, 'loss_map.html'),
                    clip_pct=args.percentile_clip)
    plot_per_scene_boxplot(df, args.loss_col, args.out_dir)

    # ---- Delta comparison ----
    if args.csv_baseline:
        print(f'\nLoading baseline {args.csv_baseline} ...')
        df_baseline = pd.read_csv(args.csv_baseline)
        print(f'  {len(df_baseline)} samples')
        plot_delta_map(df, df_baseline, args.loss_col, args.out_dir,
                       clip_pct=args.percentile_clip)

    print('\nAll plots saved to', args.out_dir)


if __name__ == '__main__':
    main()

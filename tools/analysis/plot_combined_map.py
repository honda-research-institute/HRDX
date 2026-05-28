#!/usr/bin/env python
"""
Create a combined Folium map showing per-sample loss (penalized_chamfer or
mean_AP) for multiple models as toggleable layers.

Usage:
    python tools/analysis/plot_combined_map.py \
        --csvs  csv1.csv csv2.csv csv3.csv \
        --names "Model A" "Model B" "Model C" \
        --loss-col penalized_chamfer \
        --out combined_map.html
"""

import argparse, pathlib
import numpy as np
import pandas as pd

try:
    import folium
    from folium.plugins import HeatMap
except ImportError:
    raise SystemExit("pip install folium")

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
import matplotlib.colors as mcolors
import matplotlib.cm as mcm


def value_to_hex(val, vmin, vmax, cmap_name="RdYlGn_r"):
    """Map a scalar *val* in [vmin, vmax] → hex colour string."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    sm = mcm.ScalarMappable(norm=norm, cmap=mcm.get_cmap(cmap_name))
    r, g, b, _ = sm.to_rgba(val)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="Paths to per-sample AP CSVs")
    ap.add_argument("--names", nargs="+", default=None,
                    help="Display names for each CSV (same order)")
    ap.add_argument("--loss-col", default="penalized_chamfer",
                    help="Column to visualise (default: penalized_chamfer)")
    ap.add_argument("--out", default="work_dirs/analysis/combined_loss_map.html",
                    help="Output HTML file")
    ap.add_argument("--sample-frac", type=float, default=0.15,
                    help="Fraction of points to plot (0-1, default 0.15 for speed)")
    ap.add_argument("--radius", type=float, default=4,
                    help="Circle marker radius in pixels")
    args = ap.parse_args()

    if args.names is None:
        args.names = [pathlib.Path(p).stem for p in args.csvs]
    assert len(args.names) == len(args.csvs), "Mismatch between --csvs and --names"

    loss_col = args.loss_col

    # ---- Load all CSVs --------------------------------------------------
    dfs = {}
    for csv_path, name in zip(args.csvs, args.names):
        df = pd.read_csv(csv_path)
        assert loss_col in df.columns, f"{loss_col} not in {csv_path}"
        # Replace inf with column max (finite) for colour mapping
        finite = df[loss_col].replace([np.inf, -np.inf], np.nan)
        fill_val = finite.max()
        df[loss_col] = df[loss_col].replace([np.inf, -np.inf], fill_val)
        dfs[name] = df
        med = df[loss_col].median()
        mn = df[loss_col].mean()
        print(f"  {name}: {len(df)} rows, {loss_col} mean={mn:.4f} median={med:.4f}")

    # ---- Global colour scale --------------------------------------------
    all_vals = pd.concat([d[loss_col] for d in dfs.values()])
    vmin = float(all_vals.quantile(0.02))
    vmax = float(all_vals.quantile(0.98))
    print(f"\nColour scale: {loss_col} in [{vmin:.3f}, {vmax:.3f}] (2nd-98th pctile)")

    # ---- Build Folium map -----------------------------------------------
    # Centre on median lat/lon of first CSV
    first_df = list(dfs.values())[0]
    centre = [first_df["lat"].median(), first_df["lon"].median()]
    m = folium.Map(location=centre, zoom_start=11, tiles="CartoDB positron")

    layer_colours = ["blue", "red", "green", "orange", "purple"]

    for idx, (name, df) in enumerate(dfs.items()):
        fg = folium.FeatureGroup(name=name, show=(idx == 0))  # show first layer by default

        # Subsample for performance
        if args.sample_frac < 1.0:
            df_plot = df.sample(frac=args.sample_frac, random_state=42)
        else:
            df_plot = df
        print(f"  Plotting {len(df_plot)} points for '{name}'")

        for _, row in df_plot.iterrows():
            val = row[loss_col]
            colour = value_to_hex(val, vmin, vmax)
            popup_text = (
                f"<b>{name}</b><br>"
                f"scene: {row.get('scene_name','')}<br>"
                f"{loss_col}: {val:.4f}<br>"
                f"mean_AP: {row.get('mean_AP', ''):.4f}<br>"
                f"token: {row.get('token','')}"
            )
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=args.radius,
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=0.7,
                weight=1,
                popup=folium.Popup(popup_text, max_width=350),
            ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ---- Legend ----------------------------------------------------------
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 13px;">
      <b>{loss_col}</b><br>
      <span style="color:{value_to_hex(vmin, vmin, vmax)};">&#9632;</span> {vmin:.2f} (low)
      &nbsp;&nbsp;
      <span style="color:{value_to_hex((vmin+vmax)/2, vmin, vmax)};">&#9632;</span> {(vmin+vmax)/2:.2f}
      &nbsp;&nbsp;
      <span style="color:{value_to_hex(vmax, vmin, vmax)};">&#9632;</span> {vmax:.2f} (high)
      <br><i>Toggle layers in top-right panel</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ---- Save -----------------------------------------------------------
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    print(f"\nSaved combined map to {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

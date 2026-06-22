import math
import os
import os.path as osp
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import av2.geometry.interpolate as interp_utils
import cv2
import matplotlib

matplotlib.use("agg")

import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from PIL import Image, ImageEnhance

from plugin.datasets.visualize.renderer import points_ego2img
from plugin.datasets.visualize.download_satellite_images import download_export


# -----------------------------------------------------------------------------
# Styled palettes
# -----------------------------------------------------------------------------

BASE_CATEGORY_COLORS = {
    "divider": "#607D8B",
    "boundary": "#546E7A",
    "ped_crossing": "#F6A57F",
    "centerline": "#F5B26A",
    "drivable_area": "#8FBF9F",
    "Parking spots": "#AF7AC5",
    "Lane center lines": "#F5B26A",
    "Non-drivable areas": "#B0BEC5",
    "Bike lanes": "#4CAF50",
    "Road boundary": "#3E5F8A",
    "Lane lines": "#2BB0B5",
    "Text on the road": "#EC7263",
    "Crosswalks": "#C19CFA",
    "Stop Line": "#FFFFFF",
    "Intersection Centerlines": "#FFB300",
    "Intersection Centerline": "#FFB300",
}

TEXT_ATTR_COLORS = {
    "Straight arrow": "#6A9FB5",
    "Turn right arrow": "#F28E2B",
    "left turn arrow": "#4E79A7",
    "Railway crossing": "#9C755F",
    "RXR": "#8C6D31",
    "Keep Clear": "#AF7AA1",
    "YIELD (AHEAD)": "#D37295",
    "SLOW": "#FF9D9A",
    "Stop text": "#C1666B",
    "Merge Advisory Arrow": "#8FD0A9",
    "ONLY": "#F1CE63",
    "PED XING": "#76B7B2",
    "Other class": "#FFB5A7",
    "SCHOOL": "#B07AA1",
    "HOV Lane": "#6F9F77",
    "BUS/TAXI Lane": "#3F3F3F",
    "U-Turn": "#E15759",
}

PARKING_ATTR_COLORS = {
    "empty": "#7FB069",
    "not empty ": "#C7A27C",
}

LANE_CENTER_COMBO_COLORS = {
    ("Straight", "lane centerline"): "#F6B87D",
    ("Right", "lane centerline"): "#F2A96C",
    ("Left", "lane centerline"): "#F8C48E",
    ("U-turn", "lane centerline"): "#EFA06A",
    ("Straight", "Intersection centerline"): "#6FB1E4",
    ("Right", "Intersection centerline"): "#F0625D",
    ("Left", "Intersection centerline"): "#58B88C",
    ("U-turn", "Intersection centerline"): "#9C7BD4",
}

LANE_CENTER_ATTR_COLORS = {
    "Straight": "#F6B87D",
    "Right": "#F2A96C",
    "Left": "#F8C48E",
    "U-turn": "#EFA06A",
}


CLASS_STYLES = {
    "Road boundary": {"linewidth": 6.44, "linestyle": "-", "alpha": 0.98},
    "Lane lines": {"linewidth": 3.45, "linestyle": "-", "alpha": 0.92, "arrow": True},
    "Lane center lines": {"linewidth": 1.61, "linestyle": "-", "alpha": 0.95, "arrow": True},
    "Bike lanes": {"linewidth": 2.99, "linestyle": "-", "alpha": 0.9},
    "Non-drivable areas": {"linewidth": 1.84, "linestyle": "-", "alpha": 0.8},
    "Crosswalks": {"linewidth": 1.725, "linestyle": "-", "alpha": 0.95, "fill_alpha": 0.35},
    "Parking spots": {"linewidth": 2.3, "linestyle": "-", "alpha": 0.85},
    "Text on the road": {"linewidth": 2.07, "linestyle": "-", "alpha": 0.9},
    "drivable_area": {"linewidth": 1.61, "linestyle": "-", "alpha": 0.75},
    "centerline": {"linewidth": 1.61, "linestyle": "-", "alpha": 0.9, "arrow": True},
    "boundary": {"linewidth": 2.99, "linestyle": "-", "alpha": 0.9},
    "divider": {"linewidth": 2.3, "linestyle": (0, (4, 8)), "alpha": 0.85},
    "ped_crossing": {"linewidth": 2.07, "linestyle": "-", "alpha": 0.9, "fill_alpha": 0.35},
}

POLYGONAL_CATEGORIES = {
    "Crosswalks",
    "ped_crossing",
    "drivable_area",
    "Non-drivable areas",
    "Parking spots",
}


ColorLike = Union[str, Tuple[int, int, int], Tuple[float, float, float]]


def _color_to_mpl(color: ColorLike, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """Convert a hex/int RGB color to RGBA tuple for Matplotlib."""
    if isinstance(color, str):
        rgba = mcolors.to_rgba(color, alpha=alpha)
    else:
        if max(color) > 1.0:
            norm = tuple(c / 255.0 for c in color)
        else:
            norm = color
        rgba = (norm[0], norm[1], norm[2], alpha)
    return rgba


def _to_pil(img_like: Union[str, np.ndarray, Image.Image]) -> Optional[Image.Image]:
    """Convert supported backgrounds to RGBA PIL image."""
    if isinstance(img_like, Image.Image):
        return img_like.convert("RGBA")
    if isinstance(img_like, str):
        return Image.open(img_like).convert("RGBA")

    if isinstance(img_like, np.ndarray):
        arr = img_like
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        return Image.fromarray(arr).convert("RGBA")

    return None


def _preprocess_background(img: Image.Image) -> Image.Image:
    """Apply the requested desaturation, gamma, and overlay adjustments."""
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Color(rgb).enhance(0.65)

    arr = np.asarray(rgb).astype(np.float32) / 255.0
    gamma = 1.1
    arr = np.power(arr, gamma)
    overlay = 0.18
    arr = np.clip(arr * (1.0 - overlay) + overlay * 0.65, 0.0, 1.0)

    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGBA")


def _generate_plain_background(width_m: float, height_m: float, ppm: int = 10) -> Image.Image:
    """Create a neutral grey background without grid markings."""
    w_px = int(round(width_m * ppm))
    h_px = int(round(height_m * ppm))
    canvas = np.full((h_px, w_px, 3), 28, dtype=np.uint8)

    yy, xx = np.mgrid[0:h_px, 0:w_px]
    cx = w_px / 2.0
    cy = h_px / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist_norm = np.clip(dist / max(w_px, h_px), 0.0, 1.0)
    shading = (1.0 + 0.25 * (1.0 - dist_norm))[:, :, None]
    canvas = np.clip(canvas.astype(np.float32) * shading, 20, 60).astype(np.uint8)

    return Image.fromarray(canvas).convert("RGBA")


def _rgba_to_bgr(rgba: Tuple[float, float, float, float]) -> Tuple[int, int, int]:
    """Convert an RGBA color in [0,1] to 8-bit OpenCV BGR."""
    clipped = [np.clip(c, 0.0, 1.0) for c in rgba[:3]]
    r, g, b = (int(round(component * 255.0)) for component in clipped)
    return (b, g, r)


def _resolve_dash_pattern(style: Any, thickness: int) -> Tuple[int, int]:
    """Convert a matplotlib dash style into pixel dash and gap lengths."""
    base_dash = 6.0
    base_gap = 6.0
    if isinstance(style, tuple) and len(style) >= 2:
        pattern = style[1]
        if isinstance(pattern, (list, tuple)) and len(pattern) >= 2:
            base_dash = float(pattern[0])
            base_gap = float(pattern[1])
        elif isinstance(pattern, (list, tuple)) and len(pattern) == 1:
            base_dash = base_gap = float(pattern[0])
    elif isinstance(style, str) and style not in ("-", "", "solid"):
        base_dash = 6.0
        base_gap = 4.0
    scale = max(1.0, thickness / 2.0)
    dash = int(max(2, round(base_dash * scale)))
    gap = int(max(2, round(base_gap * scale)))
    return dash, gap


def _draw_dashed_polyline(
    image: np.ndarray,
    pts: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    dash_length: int,
    gap_length: int,
) -> None:
    """Draw a dashed polyline onto an image."""
    if pts.shape[0] < 2:
        return

    dash = max(1, dash_length)
    gap = max(1, gap_length)
    for i in range(len(pts) - 1):
        p_start = pts[i].astype(np.float32)
        p_end = pts[i + 1].astype(np.float32)
        segment = p_end - p_start
        seg_len = float(np.linalg.norm(segment))
        if seg_len < 1e-3:
            continue
        direction = segment / seg_len
        distance = 0.0
        draw_phase = True
        while distance < seg_len:
            step = dash if draw_phase else gap
            next_distance = min(distance + step, seg_len)
            if draw_phase:
                seg_start = p_start + direction * distance
                seg_end = p_start + direction * next_distance
                cv2.line(
                    image,
                    tuple(np.round(seg_start).astype(int)),
                    tuple(np.round(seg_end).astype(int)),
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )
            distance = next_distance
            draw_phase = not draw_phase


def _draw_polyline_on_overlay(
    overlay: np.ndarray,
    uv: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    line_style: Any,
) -> None:
    """Draw a polyline with optional dash style onto an overlay."""
    pts = np.round(uv).astype(np.int32)
    if pts.shape[0] < 2:
        return

    dashed = False
    if isinstance(line_style, tuple):
        dashed = True
    elif isinstance(line_style, str) and line_style not in ("-", "", "solid"):
        dashed = True

    if dashed:
        dash_len, gap_len = _resolve_dash_pattern(line_style, thickness)
        _draw_dashed_polyline(overlay, pts, color, thickness, dash_len, gap_len)
    else:
        cv2.polylines(
            overlay,
            [pts],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def _draw_arrow_on_overlay(
    overlay: np.ndarray,
    uv: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    tip_length: float = 0.18,
) -> None:
    """Draw an arrowhead at the end of a polyline."""
    if uv.shape[0] < 2:
        return
    start = tuple(np.round(uv[-2]).astype(int))
    end = tuple(np.round(uv[-1]).astype(int))
    if start == end:
        return
    thickness_px = max(1, thickness)
    tip = float(np.clip(tip_length, 0.05, 0.5))
    cv2.arrowedLine(overlay, start, end, color, thickness_px, cv2.LINE_AA, 0, tip)


def _blend_overlay(base: np.ndarray, overlay: np.ndarray, alpha: float) -> None:
    """Blend overlay into base image with the provided alpha."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 1e-3:
        return
    if alpha >= 0.999:
        base[:] = overlay
    else:
        cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0, dst=base)


def _ensure_xyz(points: np.ndarray) -> np.ndarray:
    """Ensure points are Nx3 in ego frame (z defaults to 0)."""
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    if arr.shape[1] >= 3:
        return arr[:, :3].astype(np.float32)
    zeros = np.zeros((arr.shape[0], 1), dtype=np.float32)
    return np.concatenate([arr[:, :2], zeros], axis=1)


def _densify_points(points: np.ndarray, max_points: int = 500) -> np.ndarray:
    """Arc-length interpolate a set of points for smoother rendering."""
    if points.shape[0] < 2:
        return points
    try:
        return np.asarray(interp_utils.interp_arc(t=max_points, points=points), dtype=np.float32)
    except Exception:
        return points


def _project_polyline_to_image(
    points_xyz: np.ndarray,
    ego2cam: np.ndarray,
    intrinsic: np.ndarray,
    distortion: Optional[np.ndarray],
    img_shape: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    """Project XYZ points into pixel coordinates, returning valid uv."""
    if points_xyz.shape[0] == 0:
        return None

    uv, depth = points_ego2img(points_xyz, ego2cam, intrinsic, distortion)
    if uv.size == 0:
        return None

    if distortion is not None:
        uv_ref, depth_ref = points_ego2img(points_xyz, ego2cam, intrinsic, None)
    else:
        uv_ref, depth_ref = uv, depth

    h, w = img_shape[:2]
    valid = (
        (depth_ref > 0.0)
        & (uv_ref[:, 0] >= 0.0)
        & (uv_ref[:, 0] < (w - 1))
        & (uv_ref[:, 1] >= 0.0)
        & (uv_ref[:, 1] < (h - 1))
    )
    if not np.any(valid):
        return None
    return uv[valid]


def _is_lane_centerline(category: Optional[str], attrs: Optional[Dict[str, Any]]) -> bool:
    """Determine whether a vector corresponds to a lane or intersection centerline."""
    name = (category or "").lower()
    if "center" in name and "line" in name:
        return True

    if attrs:
        variant = attrs.get("category")
        if isinstance(variant, (list, tuple)):
            variant = variant[0] if variant else None
        if isinstance(variant, str):
            variant_name = variant.lower()
            if "center" in variant_name and "line" in variant_name:
                return True

    return False


def _prepare_arrow_path(points: np.ndarray, max_points: int = 600) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return a densified 3D path and cumulative XY arc-lengths for arrow sampling."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts, None

    if pts.shape[1] < 3:
        zeros = np.zeros((pts.shape[0], 3 - pts.shape[1]), dtype=np.float32)
        pts = np.concatenate([pts, zeros], axis=1)

    dense = _densify_points(pts, max_points=max_points)
    dense_xy = dense[:, :2]

    diffs = np.diff(dense_xy, axis=0)
    if diffs.shape[0] == 0:
        return dense, None
    seg_lens = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
    if cumulative[-1] < 1e-3:
        return dense, None
    return dense, cumulative


def _interpolate_path_point(points: np.ndarray, cumulative: np.ndarray, s: float) -> np.ndarray:
    """Interpolate a point along the path at arc-length position s."""
    s = float(np.clip(s, cumulative[0], cumulative[-1]))
    idx = int(np.searchsorted(cumulative, s))
    idx = np.clip(idx, 1, points.shape[0] - 1)
    prev_s = cumulative[idx - 1]
    next_s = cumulative[idx]
    if next_s <= prev_s + 1e-6:
        return points[idx].copy()
    ratio = (s - prev_s) / (next_s - prev_s)
    return points[idx - 1] + (points[idx] - points[idx - 1]) * ratio


def _sample_arrow_parameters(
    cumulative: np.ndarray,
    interval: float,
    min_arrow_len: float,
) -> List[Tuple[float, float]]:
    """Return start/end arc-length positions for arrowheads along the path."""
    total_len = float(cumulative[-1])
    if total_len < 1e-3:
        return []

    if total_len >= interval:
        centers = np.arange(interval, total_len, interval, dtype=np.float32)
    else:
        centers = np.array([total_len * 0.5], dtype=np.float32)

    samples: List[Tuple[float, float]] = []
    for center in centers:
        arrow_len = max(min_arrow_len, min(interval * 0.6, total_len * 0.2))
        start_s = max(0.0, center - arrow_len * 0.35)
        end_s = min(total_len, center + arrow_len * 0.65)
        if end_s - start_s < 1e-3:
            continue
        samples.append((start_s, end_s))
    return samples


def _draw_interval_arrowheads(
    ax: plt.Axes,
    pts: np.ndarray,
    color_hex: str,
    interval: float = 8.5,
    min_arrow_len: float = 6.0,
    mutation_scale: float = 45.0,
) -> None:
    """Draw arrowheads that follow the curve at regular intervals."""
    pts_xyz = np.concatenate(
        [pts, np.zeros((pts.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    path_xyz, cumulative = _prepare_arrow_path(pts_xyz)
    if cumulative is None:
        return
    samples = _sample_arrow_parameters(cumulative, interval, min_arrow_len)
    if not samples:
        return

    arrow_color = _color_to_mpl(color_hex, 0.95)
    for start_s, end_s in samples:
        start_pt = _interpolate_path_point(path_xyz, cumulative, start_s)
        end_pt = _interpolate_path_point(path_xyz, cumulative, end_s)
        if np.linalg.norm(end_pt[:2] - start_pt[:2]) < 1e-3:
            continue
        patch = FancyArrowPatch(
            tuple(start_pt[:2]),
            tuple(end_pt[:2]),
            arrowstyle="-|>",
            linewidth=0.0,
            color=arrow_color,
            mutation_scale=mutation_scale,
        )
        patch.set_zorder(6)
        ax.add_patch(patch)


def _draw_interval_arrowheads_on_image(
    image: np.ndarray,
    pts_xyz: np.ndarray,
    ego2cam: np.ndarray,
    intrinsic: np.ndarray,
    distortion: Optional[np.ndarray],
    color_bgr: Tuple[int, int, int],
    thickness: int,
    interval: float = 7.0,
    min_arrow_len: float = 4.0,
    tip_length: float = 0.4,
) -> None:
    """Project and draw arrowheads that follow the curve on camera images."""
    path_xyz, cumulative = _prepare_arrow_path(pts_xyz)
    if cumulative is None:
        return
    samples = _sample_arrow_parameters(cumulative, interval, min_arrow_len)
    if not samples:
        return

    line_thickness = max(1, thickness)
    for start_s, end_s in samples:
        center_s = 0.5 * (start_s + end_s)
        ahead_s = min(center_s + 3.0, cumulative[-1])
        start_pt = _interpolate_path_point(path_xyz, cumulative, center_s)
        end_pt = _interpolate_path_point(path_xyz, cumulative, ahead_s)
        arrow_xyz = np.array([start_pt, end_pt], dtype=np.float32)
        if min(start_pt[2], end_pt[2]) > 0.5:
            continue
        uv, depth = points_ego2img(arrow_xyz, ego2cam, intrinsic, distortion)
        uv = np.asarray(uv, dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32)
        if uv.shape[0] != 2 or depth.shape[0] != 2 or uv.shape[1] != 2:
            continue
        if np.any(depth <= 0.0):
            continue
        if not np.isfinite(uv).all():
            continue
        avg_depth = float(np.mean(depth))
        if not (2.0 <= avg_depth <= 45.0):
            continue
        height, width = image.shape[:2]
        center_uv = uv[0]
        ahead_uv = uv[1]
        direction = ahead_uv - center_uv
        norm = float(np.linalg.norm(direction))
        if norm < 1e-3:
            continue
        direction /= norm
        arrow_len_px = 40.0
        start_uv_f = center_uv - direction * (arrow_len_px * 0.3)
        end_uv_f = center_uv + direction * (arrow_len_px * 0.7)
        if not np.isfinite(start_uv_f).all() or not np.isfinite(end_uv_f).all():
            continue
        if center_uv[1] < height * 0.4:
            continue
        start_uv = (int(round(float(start_uv_f[0]))), int(round(float(start_uv_f[1]))))
        end_uv = (int(round(float(end_uv_f[0]))), int(round(float(end_uv_f[1]))))
        if start_uv == end_uv:
            continue
        if not (
            0 <= start_uv[0] < width
            and 0 <= start_uv[1] < height
            and 0 <= end_uv[0] < width
            and 0 <= end_uv[1] < height
        ):
            continue
        try:
            cv2.arrowedLine(
                image,
                start_uv,
                end_uv,
                color_bgr,
                line_thickness,
                cv2.LINE_AA,
                0,
                tip_length,
            )
        except cv2.error:
            continue


def _draw_scale_bar(ax: plt.Axes, roi_w: float, roi_h: float) -> None:
    """Draw a 10 m scale bar in the corner of the plot."""
    bar_len = min(10.0, roi_w * 0.4)
    x0 = ax.get_xlim()[0] + roi_w * 0.05
    y0 = ax.get_ylim()[0] + roi_h * 0.07
    rect = Rectangle((x0, y0), bar_len, roi_h * 0.01, facecolor="black", alpha=0.8)
    ax.add_patch(rect)
    ax.text(
        x0 + bar_len / 2.0,
        y0 + roi_h * 0.02,
        f"{bar_len:.0f} m",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
        path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],
    )


def _build_legend(ax: plt.Axes) -> None:
    """Attach a legend describing the styled categories."""
    handles: List[Union[Line2D, Polygon]] = []
    labels: List[str] = []

    for category in LEGEND_ORDER:
        color = BASE_CATEGORY_COLORS.get(category, "#666666")
        style = CLASS_STYLES.get(category, {})
        if category == "Crosswalks":
            patch = Polygon(
                [[0, 0]],
                closed=True,
                facecolor=_color_to_mpl(color, 0.25),
                edgecolor=_color_to_mpl(color, 0.9),
                linewidth=style.get("linewidth", 1.5),
            )
            handles.append(patch)
        else:
            line = Line2D(
                [0, 1],
                [0, 0],
                color=color,
                linewidth=style.get("linewidth", 2.0),
                linestyle=style.get("linestyle", "-"),
            )
            handles.append(line)
        labels.append(category)

    ax.legend(handles, labels, loc="upper right", framealpha=0.9, facecolor="white", fontsize=8)


def _vector_to_array(vec: Union[np.ndarray, Iterable]) -> np.ndarray:
    """Normalize a vector representation to a numpy array."""
    if isinstance(vec, np.ndarray):
        return vec
    if hasattr(vec, "coords"):
        return np.asarray(vec.coords)
    try:
        return np.array(vec)
    except Exception:
        return np.asarray(list(vec))


def _resolve_color(category: str, attrs: Optional[Dict[str, Any]]) -> str:
    """Get the styled color for a category, optionally using attributes."""
    if attrs:
        if category == "Text on the road":
            key = attrs.get("Text on the road", [None])[0]
            if key in TEXT_ATTR_COLORS:
                return TEXT_ATTR_COLORS[key]
        elif category == "Parking spots":
            key = attrs.get("attribute")
            if key in PARKING_ATTR_COLORS:
                return PARKING_ATTR_COLORS[key]
        elif category == "Lane lines":
            color_attr = attrs.get("Color")
            if isinstance(color_attr, (list, tuple)):
                color_attr = color_attr[0]
            if isinstance(color_attr, str) and color_attr.lower().startswith("y"):
                return "#FFD54F"
            return "#FFFFFF"
        elif category in ("Lane center lines", "Intersection Centerline"):
            attr = attrs.get("attribute")
            cat = attrs.get("category")
            if not cat:
                cat = "lane centerline" if category == "Lane center lines" else "Intersection centerline"
            if attr and cat and (attr[0], cat) in LANE_CENTER_COMBO_COLORS:
                return LANE_CENTER_COMBO_COLORS[(attr[0], cat)]
            if attr and attr[0] in LANE_CENTER_ATTR_COLORS:
                return LANE_CENTER_ATTR_COLORS[attr[0]]

    return BASE_CATEGORY_COLORS.get(category, "#666666")


def _resolve_linestyle(category: str, attrs: Optional[Dict[str, Any]], default: Any) -> Any:
    """Determine the matplotlib linestyle tuple/string based on attributes."""
    if category == "Lane lines" and attrs:
        line_type = attrs.get("Type")
        if isinstance(line_type, (list, tuple)):
            line_type = line_type[0]
        if isinstance(line_type, str) and line_type.lower().startswith("dash"):
            return (0, (6, 6))
        return "-"
    return default


def _compute_local_frame(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute a local coordinate frame using the first segment orientation."""
    center = pts.mean(axis=0)
    coords = pts - center

    forward = None
    if pts.shape[0] >= 2:
        for i in range(1, pts.shape[0]):
            vec = pts[i] - pts[i - 1]
            norm = np.linalg.norm(vec)
            if norm > 1e-3:
                forward = vec / norm
                break

    if forward is None:
        forward = np.array([1.0, 0.0], dtype=np.float32)

    left = np.array([-forward[1], forward[0]], dtype=np.float32)
    if np.linalg.norm(left) < 1e-6:
        left = np.array([0.0, 1.0], dtype=np.float32)
    left /= np.linalg.norm(left)

    R = np.column_stack([forward, left])
    local = coords @ R
    half_w = max(np.max(np.abs(local[:, 0])), 1e-3)
    half_h = max(np.max(np.abs(local[:, 1])), half_w * 0.5, 1e-3)
    return center, R, half_w, half_h


def _draw_text_on_road_symbol(
    ax: plt.Axes,
    pts: np.ndarray,
    attrs: Optional[Dict[str, Any]],
    ped_crossings: Optional[List[np.ndarray]] = None,
) -> None:
    """Render a symbolic representation of a road text marking inside its bounds."""
    if attrs:
        value = attrs.get("Text on the road")
        if isinstance(value, (list, tuple)) and value:
            kind = value[0]
        else:
            kind = "Other class"
    else:
        kind = "Other class"

    center_est = pts.mean(axis=0)
    target = None
    if ped_crossings:
        ped_arr = np.asarray(ped_crossings, dtype=np.float32)
        if ped_arr.ndim == 2 and ped_arr.shape[0] > 0:
            dists = np.linalg.norm(ped_arr - center_est, axis=1)
            if np.any(np.isfinite(dists)):
                idx = int(np.nanargmin(dists))
                if np.isfinite(dists[idx]) and dists[idx] > 1e-3:
                    target = ped_arr[idx]

    center, R_box, half_w, half_h = _compute_local_frame(pts)
    rect_outer = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ]
    )
    rect_global = rect_outer @ R_box.T + center
    ax.add_patch(
        Polygon(
            rect_global,
            closed=True,
            facecolor=(1.0, 1.0, 1.0, 0.08),
            edgecolor=(1.0, 1.0, 1.0, 0.25),
            linewidth=1.0,
            linestyle="--",
        )
    )

    R_symbol = R_box
    half_forward = half_w
    half_side = half_h
    if target is not None:
        diff = target - center
        dist = np.linalg.norm(diff)
        if dist > 1e-3:
            local_diff = R_box.T @ diff
            if abs(local_diff[0]) >= abs(local_diff[1]):
                direction = 1.0 if local_diff[0] >= 0 else -1.0
                R_symbol = np.column_stack([R_box[:, 0] * direction, R_box[:, 1] * direction])
            else:
                direction = 1.0 if local_diff[1] >= 0 else -1.0
                forward_vec = R_box[:, 1] * direction
                left_vec = -R_box[:, 0] * direction
                R_symbol = np.column_stack([forward_vec, left_vec])
                half_forward, half_side = half_h, half_w

    effective_forward = half_forward * 0.9
    effective_side = half_side * 0.9
    symbol_color = "white"
    linewidth = 2.6
    mutation_scale = max(12.0, min(effective_forward, effective_side) * 40.0)
    rotation_deg = math.degrees(math.atan2(R_symbol[1, 0], R_symbol[0, 0]))

    def to_global(local_pts: np.ndarray) -> np.ndarray:
        return local_pts @ R_symbol.T + center

    def add_line(local_pts: Iterable[Tuple[float, float]], **kwargs) -> None:
        arr = np.asarray(local_pts, dtype=np.float32)
        transformed = to_global(arr)
        ax.plot(
            transformed[:, 0],
            transformed[:, 1],
            color=symbol_color,
            linewidth=kwargs.get("linewidth", linewidth),
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    def add_arrow(
        start_local: Tuple[float, float],
        end_local: Tuple[float, float],
        **kwargs,
    ) -> None:
        start = to_global(np.asarray(start_local))
        end = to_global(np.asarray(end_local))
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle=kwargs.get("arrowstyle", "-|>"),
            connectionstyle=kwargs.get("connectionstyle", "arc3,rad=0.0"),
            linewidth=kwargs.get("linewidth", linewidth),
            color=symbol_color,
            mutation_scale=kwargs.get("mutation_scale", mutation_scale),
        )
        ax.add_patch(arrow)

    def render_text_lines(lines: List[str], **kwargs) -> None:
        if not lines:
            return

        base_font = kwargs.get(
            "fontsize",
            max(6, min(36, effective_forward * 16.0)),
        )
        weight = kwargs.get("fontweight", "bold")
        n_lines = len(lines)
        if n_lines == 1 or effective_side < 1e-3:
            positions = [center]
        else:
            usable_height = effective_side * 0.9
            line_gap = usable_height / max(n_lines - 1, 1)
            offsets = np.linspace(usable_height / 2.0, -usable_height / 2.0, n_lines)
            positions = [center + R_symbol[:, 1] * off for off in offsets]

        for pos, line in zip(positions, lines):
            ax.text(
                pos[0],
                pos[1],
                line.upper(),
                color=symbol_color,
                ha="center",
                va="center",
                fontsize=base_font,
                fontweight=weight,
                rotation=rotation_deg,
                rotation_mode="anchor",
            )

    def format_text_lines(raw: str) -> List[str]:
        text = raw.upper()
        for ch in "/()-":
            text = text.replace(ch, " ")
        tokens = [tok for tok in text.replace("\n", " ").split() if tok]
        return tokens if tokens else [raw.upper()]

    kind_normalized = (kind or "Other class").lower()

    if "straight arrow" in kind_normalized:
        add_arrow(
            (-effective_forward * 0.6, 0.0),
            (effective_forward * 0.6, 0.0),
            mutation_scale=mutation_scale,
            linewidth=linewidth,
        )
    elif "turn right" in kind_normalized:
        stem_x = 0.0
        add_line(
            [
                (-effective_forward * 0.6, 0.0),
                (stem_x, 0.0),
            ]
        )
        add_line(
            [
                (stem_x, 0.0),
                (stem_x, effective_side * 0.4),
            ]
        )
        add_arrow(
            (stem_x, effective_side * 0.2),
            (stem_x + effective_forward * 0.35, effective_side * 0.45),
            connectionstyle="arc3,rad=-0.7",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
        )
    elif "left turn" in kind_normalized:
        stem_x = 0.0
        add_line(
            [
                (-effective_forward * 0.6, 0.0),
                (stem_x, 0.0),
            ]
        )
        add_line(
            [
                (stem_x, 0.0),
                (stem_x, effective_side * 0.4),
            ]
        )
        add_arrow(
            (stem_x, effective_side * 0.2),
            (stem_x - effective_forward * 0.35, effective_side * 0.45),
            connectionstyle="arc3,rad=0.7",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
        )
    elif "u-turn" in kind_normalized:
        add_arrow(
            (effective_forward * 0.5, 0.0),
            (-effective_forward * 0.4, 0.0),
            connectionstyle="arc3,rad=0.9",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
        )
    elif "merge advisory arrow" in kind_normalized:
        add_arrow(
            (-effective_forward * 0.6, 0.0),
            (effective_forward * 0.6, 0.0),
            mutation_scale=mutation_scale,
            linewidth=linewidth,
        )
        add_line(
            [
                (-effective_forward * 0.1, -effective_side * 0.4),
                (effective_forward * 0.1, 0.0),
            ]
        )
    elif "railway" in kind_normalized or kind_normalized == "rxr":
        render_text_lines(["RXR"])
    elif "yield" in kind_normalized:
        render_text_lines(["YIELD"])
    elif "slow" in kind_normalized:
        render_text_lines(["SLOW"])
    elif "stop" in kind_normalized:
        render_text_lines(["STOP"])
    elif "only" in kind_normalized:
        render_text_lines(["ONLY"])
    elif "keep clear" in kind_normalized:
        render_text_lines(["KEEP", "CLEAR"])
    elif "ped xing" in kind_normalized:
        render_text_lines(["PED", "XING"])
    elif "school" in kind_normalized:
        render_text_lines(["SCHOOL"])
    elif "hov lane" in kind_normalized:
        render_text_lines(["HOV"])
    elif "bus/taxi lane" in kind_normalized:
        render_text_lines(["BUS", "TAXI"])
    else:
        label = kind.upper() if isinstance(kind, str) else "TEXT"
        render_text_lines(format_text_lines(label))


class StyledRenderer:
    """Render BEV vectors with enhanced styling for qualitative inspection."""

    def __init__(self, cat2id: Dict[str, int], roi_size: Tuple[float, float], dataset: str = "av2") -> None:
        self.cat2id = cat2id
        self.id2cat = {v: k for k, v in cat2id.items()}
        self.roi_size = roi_size
        if dataset == "av2":
            self.cam_names = [
                "ring_front_center",
                "ring_front_right",
                "ring_front_left",
                "ring_rear_right",
                "ring_rear_left",
                "ring_side_right",
                "ring_side_left",
            ]
        elif dataset == "nusc":
            self.cam_names = [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
        else:
            self.cam_names = ["cam_1", "cam_2", "cam_3", "cam_4", "cam_5", "cam_6"]

    def render_camera_views_from_vectors_pretty(
        self,
        vectors: Dict[int, List],
        imgs: Iterable[np.ndarray],
        ego2cams: Iterable[np.ndarray],
        intrinsics: Iterable[np.ndarray],
        out_dir: str,
        distortions: Optional[Iterable[np.ndarray]] = None,
        idx: Optional[int] = None,
        draw_scores: bool = False,
        id_info: Optional[Dict[int, List[Any]]] = None,
        thickness_scale: float = 1.6,
        arrow_tip_length: float = 0.18,
        draw_lane_arrows: bool = True,
    ) -> None:
        """Project styled vectors onto camera views with MapTracker aesthetics.

        Args:
            vectors: Mapping from class id to iterable of vector geometries.
            imgs: Sequence of BGR camera images aligned with `ego2cams` and `intrinsics`.
            ego2cams: Sequence of 4x4 ego-to-camera extrinsics.
            intrinsics: Sequence of 3x3 camera intrinsics.
            out_dir: Base directory where per-camera outputs are stored.
            distortions: Optional per-camera distortion coefficients for projection.
            idx: Optional frame index suffix for filenames.
            draw_scores: Whether to render detection scores if provided.
            id_info: Optional mapping {label: [ids]} for per-vector annotations.
            thickness_scale: Multiplier applied to style linewidth values to obtain pixel thickness.
            arrow_tip_length: Relative arrowhead size for directional classes.
            draw_lane_arrows: If True, renders lane centerline direction arrows on top of camera images.
        """
        imgs_list = list(imgs)
        ego2cams_list = [np.asarray(mat) for mat in ego2cams]
        intrinsics_list = [np.asarray(mat) for mat in intrinsics]

        if not (len(imgs_list) == len(ego2cams_list) == len(intrinsics_list)):
            raise ValueError("imgs, ego2cams, and intrinsics must have matching lengths.")

        if distortions is not None:
            distortions_list = list(distortions)
            if len(distortions_list) != len(imgs_list):
                raise ValueError("distortions length must match number of cameras.")
        else:
            distortions_list = [None] * len(imgs_list)

        os.makedirs(out_dir, exist_ok=True)

        for cam_idx, (img, ego2cam, intrinsic, distortion) in enumerate(
            zip(imgs_list, ego2cams_list, intrinsics_list, distortions_list)
        ):
            if isinstance(img, (str, os.PathLike)):
                img_bgr = cv2.imread(str(img))
                if img_bgr is None:
                    continue
            else:
                img_bgr = np.asarray(img).copy()

            if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
                raise ValueError("Each camera image must be an HxWx3 BGR array.")

            ego2cam = np.asarray(ego2cam, dtype=np.float32)
            intrinsic = np.asarray(intrinsic, dtype=np.float32)
            distortion = None if distortion is None else np.asarray(distortion, dtype=np.float32)

            for label, vector_list in vectors.items():
                category = self.id2cat.get(label, str(label))
                style = CLASS_STYLES.get(category, {"linewidth": 2.0, "linestyle": "-", "alpha": 1.0})

                for vec_idx, vector in enumerate(vector_list):
                    vector_payload = vector
                    score = None
                    prop = False
                    if draw_scores and isinstance(vector, tuple) and len(vector) == 3:
                        vector_payload, score, prop = vector

                    attrs = getattr(vector_payload, "attrs", getattr(vector, "attrs", None))
                    vec_array = _vector_to_array(vector_payload)
                    if vec_array is None or len(vec_array) == 0:
                        continue

                    pts_xyz = _ensure_xyz(vec_array)
                    if pts_xyz.shape[0] < 2:
                        continue

                    base_color_hex = _resolve_color(category, attrs)
                    rgba = _color_to_mpl(base_color_hex, alpha=1.0)
                    color_bgr = _rgba_to_bgr(rgba)
                    line_style = _resolve_linestyle(category, attrs, style.get("linestyle", "-"))
                    line_alpha = float(style.get("alpha", 1.0))
                    fill_alpha = float(style.get("fill_alpha", 0.0))
                    arrow_flag = bool(style.get("arrow", False))
                    thickness = max(1, int(round(style.get("linewidth", 2.0) * thickness_scale)))
                    is_lane_centerline = _is_lane_centerline(category, attrs)

                    target_points = int(min(500, max(50, pts_xyz.shape[0] * 8)))
                    dense_xyz = _densify_points(pts_xyz, max_points=target_points)
                    uv_line = _project_polyline_to_image(
                        dense_xyz,
                        ego2cam=ego2cam,
                        intrinsic=intrinsic,
                        distortion=distortion,
                        img_shape=img_bgr.shape,
                    )
                    if uv_line is None or uv_line.shape[0] < 2:
                        continue

                    if category in POLYGONAL_CATEGORIES and pts_xyz.shape[0] >= 3 and fill_alpha > 0.0:
                        poly_xyz = pts_xyz
                        if not np.allclose(poly_xyz[0], poly_xyz[-1]):
                            poly_xyz = np.vstack([poly_xyz, poly_xyz[0]])
                        uv_poly = _project_polyline_to_image(
                            poly_xyz,
                            ego2cam=ego2cam,
                            intrinsic=intrinsic,
                            distortion=distortion,
                            img_shape=img_bgr.shape,
                        )
                        if uv_poly is not None and uv_poly.shape[0] >= 3:
                            overlay_fill = img_bgr.copy()
                            cv2.fillPoly(
                                overlay_fill,
                                [np.round(uv_poly).astype(np.int32)],
                                color=color_bgr,
                            )
                            _blend_overlay(img_bgr, overlay_fill, fill_alpha)

                    overlay_line = img_bgr.copy()
                    _draw_polyline_on_overlay(
                        overlay_line,
                        uv_line,
                        color=color_bgr,
                        thickness=thickness,
                        line_style=line_style,
                    )
                    if arrow_flag and not is_lane_centerline and category != "Text on the road":
                        _draw_arrow_on_overlay(
                            overlay_line,
                            uv_line,
                            color=color_bgr,
                            thickness=thickness,
                            tip_length=arrow_tip_length,
                        )
                    _blend_overlay(img_bgr, overlay_line, line_alpha)

                    label_text: Optional[str] = None
                    if draw_scores and score is not None:
                        label_text = f"{score:.2f}{'p' if prop else ''}"
                    elif id_info:
                        vec_ids = id_info.get(label)
                        if vec_ids and vec_idx < len(vec_ids):
                            label_text = str(vec_ids[vec_idx])

                    if label_text:
                        mid_pt = uv_line[uv_line.shape[0] // 2]
                        cv2.putText(
                            img_bgr,
                            label_text,
                            (int(round(mid_pt[0])), int(round(mid_pt[1]))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color_bgr,
                            1,
                            lineType=cv2.LINE_AA,
                        )

            cam_name = self.cam_names[cam_idx] if cam_idx < len(self.cam_names) else f"cam_{cam_idx}"
            cam_dir = os.path.join(out_dir, cam_name)
            os.makedirs(cam_dir, exist_ok=True)
            filename = f"styled_projected_{idx}.jpg" if idx is not None else "styled_projected.jpg"
            cv2.imwrite(os.path.join(cam_dir, filename), img_bgr)

    def render_bev_from_vectors_pretty(
        self,
        vectors: Dict[int, List],
        out_dir: Optional[str] = None,
        draw_scores: bool = False,
        specified_path: Optional[str] = None,
        id_info: Optional[Dict[int, List[Any]]] = None,
        background: Optional[Union[str, np.ndarray, Image.Image]] = None,
        arcgis_service_url: Optional[str] = None,
        center_lon: Optional[float] = None,
        center_lat: Optional[float] = None,
        rotation_cw_deg: float = 0.0,
        ppm: int = 10,
        arcgis_fmt: str = "png",
        arcgis_transparent: bool = False,
        arcgis_print_url: bool = False,
        arcgis_extra: Optional[Dict[str, Any]] = None,
        frame_number: Optional[int] = None,
    ) -> str:
        """Render vectors with upgraded styling.

        Args:
            vectors: Mapping from class id to iterable of polylines (Nx2 or Nx3).
            out_dir: Directory to save the rendering.
            draw_scores: Whether to annotate detection scores if present.
            specified_path: Optional output path override.
            id_info: Optional mapping of class id to vector ids for annotations.
            background: Optional background (path/array/PIL). If omitted, a neutral grid is used.
            arcgis_service_url: Optional ArcGIS export endpoint (mirrors legacy behavior).
            center_lon, center_lat: Coordinates required when using `arcgis_service_url`.
            rotation_cw_deg: Rotation passed to ArcGIS export.
            ppm: Pixels per meter for background sizing.
            arcgis_fmt, arcgis_transparent, arcgis_print_url, arcgis_extra: ArcGIS options.
            frame_number: Included for API parity (unused).

        Returns:
            Path to the saved image.
        """
        del frame_number  # unused but kept for API compatibility

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        map_path = specified_path or (
            os.path.join(out_dir, "map_styled.jpg") if out_dir else "map_styled.jpg"
        )

        roi_w, roi_h = float(self.roi_size[0]), float(self.roi_size[1])
        bg_extent = [-roi_w / 2.0, roi_w / 2.0, -roi_h / 2.0, roi_h / 2.0]

        bg_img: Optional[Image.Image] = None
        tmp_bg_path: Optional[str] = None

        if background is not None:
            bg_img = _to_pil(background)
        elif arcgis_service_url and (center_lon is not None) and (center_lat is not None):
            buffer_m = (max(roi_w, roi_h) / 2.0) * (1.0 / math.cos(math.radians(center_lat)))
            target_width_px = int(round(roi_w * ppm))
            target_height_px = int(round(roi_h * ppm))
            max_side = 4096
            min_side = 256
            target_width_px = max(min_side, min(max_side, target_width_px))
            target_height_px = max(min_side, min(max_side, target_height_px))
            size_px = max(target_width_px, target_height_px)

            tmp_bg_path = os.path.join(out_dir or ".", f"_bev_bg_styled.{arcgis_fmt.lower()}")
            extra = arcgis_extra or {}
            download_export(
                lon=center_lon,
                lat=center_lat,
                buffer_m=buffer_m,
                size=size_px,
                fmt=arcgis_fmt,
                rotation=rotation_cw_deg,
                out=tmp_bg_path,
                service_export_url=arcgis_service_url,
                dpi=96,
                transparent=arcgis_transparent,
                use_bbox4326=False,
                print_url=arcgis_print_url,
                **extra,
            )
            bg_img = Image.open(tmp_bg_path).convert("RGBA")
            if (bg_img.width != target_width_px) or (bg_img.height != target_height_px):
                left = max(0, int(round((bg_img.width - target_width_px) / 2.0)))
                top = max(0, int(round((bg_img.height - target_height_px) / 2.0)))
                right = min(bg_img.width, left + target_width_px)
                bottom = min(bg_img.height, top + target_height_px)
                bg_img = bg_img.crop((left, top, right, bottom))
        else:
            bg_img = _generate_plain_background(roi_w, roi_h, ppm=max(5, ppm))

        if bg_img is not None:
            bg_img = _preprocess_background(bg_img)

        fig = plt.figure(figsize=(roi_w / 2.0, roi_h / 2.0), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor((0, 0, 0, 0))
        ax.set_xlim(-roi_w / 2.0, roi_w / 2.0)
        ax.set_ylim(-roi_h / 2.0, roi_h / 2.0)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        if bg_img is not None:
            ax.imshow(bg_img, extent=bg_extent, alpha=1.0)

        ped_crossing_centers: List[np.ndarray] = []
        for label, vector_list in vectors.items():
            category = self.id2cat.get(label, str(label))
            if category in ("Crosswalks", "ped_crossing"):
                for vec in vector_list:
                    base_vec = vec
                    if isinstance(base_vec, tuple) and len(base_vec) >= 1:
                        base_vec = base_vec[0]
                    arr = _vector_to_array(base_vec)
                    try:
                        pts_arr = np.asarray(arr, dtype=np.float32)
                    except Exception:
                        continue
                    if pts_arr.ndim < 2 or pts_arr.shape[0] == 0:
                        continue
                    ped_crossing_centers.append(np.mean(pts_arr[:, :2], axis=0))

        plotted_categories: Dict[str, bool] = {}

        for label, vector_list in vectors.items():
            category = self.id2cat.get(label, str(label))

            for idx_vec, vector in enumerate(vector_list):
                attrs = getattr(vector, "attrs", None)
                vec_array = vector
                score = None
                prop = False
                if draw_scores:
                    if isinstance(vector, tuple) and len(vector) == 3:
                        vec_array, score, prop = vector
                if isinstance(vec_array, list):
                    vec_array = _vector_to_array(vec_array)

                if vec_array is None or len(vec_array) == 0:
                    continue

                pts = np.asarray(vec_array)
                pts = pts[:, :2]
                is_lane_centerline = _is_lane_centerline(category, attrs)

                color_hex = _resolve_color(category, attrs)
                style = CLASS_STYLES.get(category, {"linewidth": 2.0, "linestyle": "-", "alpha": 0.9})
                alpha = style.get("alpha", 0.9)
                linewidth = style.get("linewidth", 2.0)
                linestyle = _resolve_linestyle(category, attrs, style.get("linestyle", "-"))
                rgba = _color_to_mpl(color_hex, alpha)

                if category in POLYGONAL_CATEGORIES and len(pts) >= 3:
                    closed_pts = pts
                    if not np.allclose(pts[0], pts[-1]):
                        closed_pts = np.vstack([pts, pts[0]])
                    patch = Polygon(
                        closed_pts,
                        closed=True,
                        facecolor=_color_to_mpl(color_hex, style.get("fill_alpha", 0.3)),
                        edgecolor=rgba,
                        linewidth=linewidth,
                    )
                    ax.add_patch(patch)
                else:
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        color=color_hex,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                        alpha=alpha,
                    )

                    if style.get("arrow") and pts.shape[0] >= 2 and not is_lane_centerline and category != "Text on the road":
                        start = pts[-2]
                        end = pts[-1]
                        arrow = FancyArrowPatch(
                            start,
                            end,
                            arrowstyle="-|>",
                            mutation_scale=10,
                            linewidth=0.0,
                            color=color_hex,
                            alpha=alpha,
                        )
                        ax.add_patch(arrow)

                    if is_lane_centerline:
                        _draw_interval_arrowheads(ax, pts, color_hex)

                if category == "Text on the road":
                    label_text = None
                else:
                    if draw_scores and score is not None:
                        label_text = f"{score:.2f}{'p' if prop else ''}"
                    elif id_info:
                        vec_ids = id_info.get(label)
                        if vec_ids:
                            label_text = str(vec_ids[idx_vec])
                        else:
                            label_text = None
                    else:
                        label_text = None

                    if label_text:
                        mid_idx = pts.shape[0] // 2
                        ax.text(
                            pts[mid_idx, 0],
                            pts[mid_idx, 1],
                            label_text,
                            fontsize=8,
                            ha="center",
                            va="center",
                            color=color_hex,
                            path_effects=[patheffects.withStroke(linewidth=1.2, foreground="white")],
                        )

                plotted_categories[category] = True

        fig.savefig(map_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        if tmp_bg_path and osp.exists(tmp_bg_path):
            try:
                os.remove(tmp_bg_path)
            except OSError:
                pass

        return map_path

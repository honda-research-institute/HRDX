import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PARENT_DIR)
for path in (PARENT_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.append(path)

import argparse
import pickle
import shutil
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from glob import glob
from mmengine.config import Config
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS
from matplotlib.patches import Patch

from plugin.datasets.visualize.renderer import COLOR_MAPS_BGR

from tracking.cmap_utils.match_utils import *


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--out_dir', 
        required=True,
        default='demo',
        help='directory where visualize results will be saved')
    parser.add_argument(
        '--data_path',
        required=True,
        default="",
        help='directory to submission file')
    parser.add_argument(
        '--scene_id',
        type=str, 
        nargs='+',
        default=None,
        help='scene_id to visulize')
    parser.add_argument(
        '--option',
        default="vis-gt",
        help='vis-gt or vis-pred')
    parser.add_argument(
        '--line_opacity',
        default=0.75,
        type=float,
        help='Line simplification tolerance'
    )
    parser.add_argument(
        '--overwrite',
        default=1,
        type=int,
        help='whether to overwrite the existing images'
    )
    parser.add_argument(
        '--dpi',
        default=20,
        type=int,
        help='whether to merge boundary lines'
    )
    parser.add_argument(
        '--front-cam',
        default='CAM_FRONT',
        dest='front_cam',
        help='camera key used as the forward-facing image panel',
    )
    parser.add_argument(
        '--aerial-root',
        dest='aerial_root',
        help='Optional root directory containing aerial imagery tiles (overrides dataset config)',
    )
    parser.add_argument(
        '--pred-semantic',
        dest='pred_semantic',
        action='append',
        help='Optional path to submission file (json/pkl) containing semantic masks for predictions',
    )
    parser.add_argument(
        '--raster-fallback',
        dest='raster_fallback',
        action='store_true',
        help='Rasterize vectors when semantic masks are unavailable (disabled by default)',
    )

    args = parser.parse_args()

    return args

LABEL_COLORS = ['b', 'r', 'g', 'c', 'y', 'm', 'orange', 'purple', 'brown', 'pink', 'olive']

_CAR_IMAGE_RGBA: Optional[np.ndarray] = None


class ProgressLogger:
    """Lightweight progress reporter for per-scene visualization loops."""

    def __init__(self, scene_name: str, total: int, phase: str) -> None:
        self.scene_name = scene_name
        self.total = total
        self.phase = phase
        self._reported = set()
        if total <= 0:
            self.interval = 0
        else:
            self.interval = max(1, total // 20)

    def update(self, index: int) -> None:
        if self.total <= 0:
            return
        iteration = index + 1
        if iteration == 1 or iteration == self.total or (
            self.interval > 0 and iteration % self.interval == 0 and iteration not in self._reported
        ):
            percent = int(round(iteration * 100.0 / self.total))
            print(
                f'[{self.phase}] Scene "{self.scene_name}": {iteration}/{self.total} frames ({percent}%)',
                flush=True,
            )
            self._reported.add(iteration)


def _get_car_image() -> np.ndarray:
    global _CAR_IMAGE_RGBA
    if _CAR_IMAGE_RGBA is None:
        car_path = os.path.join(PROJECT_ROOT, 'resources', 'car-orange.png')
        if not os.path.isfile(car_path):
            raise FileNotFoundError(f'Car icon not found at {car_path}')
        with Image.open(car_path) as img:
            _CAR_IMAGE_RGBA = np.array(img.convert('RGBA'))
    return _CAR_IMAGE_RGBA

class VideoRecorder:
    """Stream frames to disk-backed writers to avoid accumulating them in memory."""

    def __init__(
        self,
        output_path: str,
        fps: int = 10,
        scale: Optional[float] = None,
        temp_root: Optional[str] = None,
    ) -> None:
        self.output_path = output_path
        self.scale = scale
        self.fps = fps
        self._writer = None
        self._size: Optional[Tuple[int, int]] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._frame_idx = 0
        self._use_disk = False
        self._ext = Path(output_path).suffix.lower()
        output_parent = os.path.dirname(os.path.abspath(output_path))
        resolved_temp_root = temp_root or output_parent or tempfile.gettempdir()
        self._temp_root = resolved_temp_root

    def _init_disk_backend(self) -> None:
        if self._temp_dir is None:
            try:
                os.makedirs(self._temp_root, exist_ok=True)
            except Exception:
                self._temp_root = tempfile.gettempdir()
            self._temp_dir = tempfile.TemporaryDirectory(prefix='vis_frames_', dir=self._temp_root)
        self._use_disk = True

    def _ensure_writer(self, size: Tuple[int, int]) -> None:
        if self._size is None:
            self._size = size
        if self._writer is not None or self._use_disk:
            return
        if self._ext in {'.mp4', '.mov', '.mkv', '.avi'}:
            try:
                self._writer = imageio.get_writer(
                    self.output_path,
                    format='ffmpeg',
                    fps=self.fps,
                    macro_block_size=None,
                )
            except Exception:
                self._init_disk_backend()
        else:
            # fallback for formats that don't stream efficiently (e.g., GIF)
            self._init_disk_backend()

    def _write_frame_to_disk(self, image: Image.Image) -> None:
        if self._temp_dir is None:
            raise RuntimeError('Temporary directory not initialised for disk writer')
        if image.size != self._size:
            image = image.resize(self._size, Image.Resampling.LANCZOS)
        frame_path = os.path.join(self._temp_dir.name, f'frame_{self._frame_idx:06d}.png')
        image.convert('RGB').save(frame_path, format='PNG', compress_level=1)
        self._frame_idx += 1

    def add_frame(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        image = Image.fromarray(frame).convert('RGBA')
        if self.scale is not None:
            w, h = image.size
            target_w = max(1, int(round(w * self.scale)))
            target_h = max(1, int(round(h * self.scale)))
            image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)

        size = image.size
        self._ensure_writer(size)
        if self._writer is not None:
            if image.size != self._size:
                image = image.resize(self._size, Image.Resampling.LANCZOS)
            frame_array = np.asarray(image.convert('RGB'))
            self._writer.append_data(frame_array)
        else:
            if self._size is None:
                self._size = size
            self._write_frame_to_disk(image)

    def _finalize_from_disk(self) -> None:
        if self._frame_idx == 0 or self._temp_dir is None:
            return
        frame_pattern = os.path.join(self._temp_dir.name, 'frame_%06d.png')
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            cmd = [
                ffmpeg_path,
                '-y',
                '-framerate',
                str(self.fps),
                '-i',
                frame_pattern,
            ]
            if self._ext == '.gif':
                cmd.extend([self.output_path])
            else:
                cmd.extend([
                    '-pix_fmt',
                    'yuv420p',
                    self.output_path,
                ])
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except subprocess.CalledProcessError:
                # fall back to pure imageio below
                pass

        frame_files = sorted(Path(self._temp_dir.name).glob('frame_*.png'))
        if not frame_files:
            return
        writer_kwargs = {'fps': self.fps}
        if self._ext == '.gif':
            format_name = 'gif'
            writer_kwargs['loop'] = 0
        else:
            format_name = 'ffmpeg'
            writer_kwargs['macro_block_size'] = None

        with imageio.get_writer(self.output_path, format=format_name, **writer_kwargs) as writer:
            for frame_path in frame_files:
                writer.append_data(imageio.imread(frame_path))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._use_disk and self._temp_dir is not None:
            self._finalize_from_disk()
            self._temp_dir.cleanup()
            self._temp_dir = None
        self._size = None
        self._frame_idx = 0
        self._use_disk = False


class SemanticMaskResolver:
    """Lazily resolve semantic masks from optional fallback submissions."""

    def __init__(
        self,
        initial_results: Optional[Dict[str, dict]] = None,
        fallback_paths: Optional[List[str]] = None,
    ) -> None:
        self._cache: Dict[str, np.ndarray] = {}
        self._sources: List[Dict[str, dict]] = []
        if isinstance(initial_results, dict):
            self._sources.append(initial_results)
        unique_paths = []
        if fallback_paths:
            seen = set()
            for path in fallback_paths:
                if not path or path in seen:
                    continue
                unique_paths.append(path)
                seen.add(path)
        self._fallback_paths = unique_paths
        self._loaded_paths = set()

    def _load_results_from_path(self, path: str) -> Optional[Dict[str, dict]]:
        try:
            data = load(path)
        except Exception:
            try:
                with open(path, 'rb') as fp:
                    data = pickle.load(fp)
            except Exception:
                return None
        if isinstance(data, dict) and 'results' in data and isinstance(data['results'], dict):
            print(f'[SemanticMaskResolver] Loaded semantic masks from {path}', flush=True)
            return data['results']
        return None

    def get(self, token: str) -> Optional[np.ndarray]:
        if token in self._cache:
            return self._cache[token]
        for source in self._sources:
            entry = source.get(token)
            if entry is None:
                continue
            mask = entry.get('semantic_mask')
            if mask is not None:
                self._cache[token] = mask
                return mask
        for path in list(self._fallback_paths):
            if path in self._loaded_paths:
                continue
            results = self._load_results_from_path(path)
            self._loaded_paths.add(path)
            if results:
                self._sources.append(results)
                entry = results.get(token)
                if entry:
                    mask = entry.get('semantic_mask')
                    if mask is not None:
                        self._cache[token] = mask
                        return mask
        return None


def _resolve_image_path(path: Optional[str], dataset) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    base = getattr(dataset, 'img_data_root', None)
    if base is None:
        base = getattr(dataset, 'data_root', None)
    if base is not None:
        candidate = os.path.join(base, path).replace('cam_1','cam_5')
        if os.path.exists(candidate):
            return candidate
    return path


def _load_dataset_image(path: Optional[str], dataset, reference_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    resolved = _resolve_image_path(path, dataset)
    return _load_image_or_blank(resolved, reference_shape)


def render_legend_image(output_path: str, cat2id: Dict[str, int]) -> None:
    """Render a legend image showing class colors."""
    if not cat2id:
        return
    entries = sorted(cat2id.items(), key=lambda x: x[1])
    patches = []
    for cat, idx in entries:
        color_bgr = COLOR_MAPS_BGR.get(cat)
        if color_bgr is not None:
            b, g, r = color_bgr
            patch_color = (r / 255.0, g / 255.0, b / 255.0)
        else:
            patch_color = LABEL_COLORS[idx % len(LABEL_COLORS)]
        patches.append(Patch(facecolor=patch_color, edgecolor='k', label=cat))

    fig_height = max(1.5, 0.35 * len(patches))
    fig, ax = plt.subplots(figsize=(4, fig_height))
    ax.axis('off')
    legend = ax.legend(
        handles=patches,
        loc='center left',
        bbox_to_anchor=(0, 0.5),
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_fontsize(10)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', transparent=False, facecolor='white')
    plt.close(fig)


def _prepare_for_cv(image: np.ndarray) -> np.ndarray:
    if image is None:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def _resize_to_height(img: np.ndarray, target_height: int) -> np.ndarray:
    if img.shape[0] == target_height:
        return img
    scale = target_height / img.shape[0]
    target_width = max(1, int(round(img.shape[1] * scale)))
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _pad_to_width(img: np.ndarray, target_width: int) -> np.ndarray:
    if img.shape[1] >= target_width:
        return img
    pad_width = target_width - img.shape[1]
    pad = np.zeros((img.shape[0], pad_width, 3), dtype=np.uint8)
    return cv2.hconcat([img, pad])


def _add_gap(imgs: List[np.ndarray], gap: int, axis: int) -> np.ndarray:
    if not imgs:
        raise ValueError('No images provided for concatenation')
    if gap <= 0 or len(imgs) == 1:
        return cv2.hconcat(imgs) if axis == 1 else cv2.vconcat(imgs)

    gap_shape = (imgs[0].shape[0], gap, 3) if axis == 1 else (gap, imgs[0].shape[1], 3)
    pieces = []
    for idx, img in enumerate(imgs):
        pieces.append(img)
        if idx < len(imgs) - 1:
            pieces.append(np.zeros(gap_shape, dtype=np.uint8))
    return cv2.hconcat(pieces) if axis == 1 else cv2.vconcat(pieces)


def _annotate_panel(img: np.ndarray, text: str) -> np.ndarray:
    annotated = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = annotated.shape[:2]
    min_dim = max(1, min(h, w))
    scale = max(0.2, min(0.55, min_dim / 360.0))
    thickness = max(1, int(round(scale * 1.5)))
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 10, 10 + text_size[1]
    cv2.rectangle(
        annotated,
        (x - 8, y - text_size[1] - 8),
        (x + text_size[0] + 8, y + baseline + 8),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(annotated, text, (x, y), font, scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    return annotated


def assemble_quadrants(
    q1: np.ndarray,
    q2: np.ndarray,
    q3: np.ndarray,
    q4: np.ndarray,
    gap: int = 20,
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    default_labels = ['Map Prediction', 'Map Ground Truth', 'Front Camera', 'Aerial View']
    label_list = labels if labels is not None else default_labels
    if len(label_list) != 4:
        raise ValueError('labels must contain exactly four entries')
    panels = [
        (q1, label_list[0]),
        (q2, label_list[1]),
        (q3, label_list[2]),
        (q4, label_list[3]),
    ]
    return assemble_panels(panels=panels, columns=2, gap=gap)


def assemble_panels(
    panels: List[Tuple[Optional[np.ndarray], str]],
    columns: int,
    gap: int = 20,
) -> np.ndarray:
    if columns < 1:
        raise ValueError('columns must be a positive integer')
    if not panels:
        raise ValueError('panels must contain at least one entry')

    prepared = [
        _annotate_panel(_prepare_for_cv(panel), label)
        for panel, label in panels
    ]

    rows = []
    for row_start in range(0, len(prepared), columns):
        row_imgs = prepared[row_start:row_start + columns]
        target_height = max(img.shape[0] for img in row_imgs)
        row_imgs = [_resize_to_height(img, target_height) for img in row_imgs]
        row = _add_gap(row_imgs, gap, axis=1)
        rows.append(row)

    target_width = max(row.shape[1] for row in rows)
    rows = [_pad_to_width(row, target_width) for row in rows]
    composite_bgr = _add_gap(rows, gap, axis=0)
    return cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)


def _colorize_semantic_mask(semantic_mask, id2cat: Dict[int, str]) -> Optional[np.ndarray]:
    if semantic_mask is None:
        return None

    mask_obj = semantic_mask
    # Handle object arrays produced by numpy
    if isinstance(mask_obj, np.ndarray) and mask_obj.dtype == object:
        mask_obj = mask_obj.tolist()

    label_map: Optional[np.ndarray] = None
    mask_np: Optional[np.ndarray] = None

    if isinstance(mask_obj, (list, tuple)):
        if len(mask_obj) == 0:
            return None
        first_item = mask_obj[0]
        # Typical RasterizeMap instance output: [mask, label]
        if (
            isinstance(first_item, (list, tuple))
            and len(first_item) >= 2
            and not np.isscalar(first_item[0])
        ):
            mask_sample = np.asarray(first_item[0])
            if mask_sample.ndim == 3:
                mask_sample = mask_sample[0]
            if mask_sample.ndim < 2:
                mask_sample = None
            if mask_sample is not None:
                h, w = mask_sample.shape[-2:]
                label_map = np.zeros((h, w), dtype=np.uint8)
                for mask_entry in mask_obj:
                    mask_arr, label = mask_entry[:2]
                    mask_arr = np.asarray(mask_arr)
                    if mask_arr.ndim == 3:
                        mask_arr = mask_arr[0]
                    if mask_arr.shape != (h, w):
                        continue
                    mask_bool = mask_arr.astype(bool)
                    label_map[mask_bool] = int(label) + 1
        if label_map is None:
            # Attempt to stack the masks along the channel dimension
            try:
                mask_np = np.stack([np.asarray(m) for m in mask_obj], axis=0)
            except Exception:
                return None
    else:
        mask_np = np.asarray(mask_obj)

    if label_map is None:
        if mask_np is None or mask_np.size == 0:
            return None
        if mask_np.ndim == 3:
            c, h, w = mask_np.shape
            label_map = np.zeros((h, w), dtype=np.uint8)
            drivable_label = next((idx for idx, cat in id2cat.items() if cat == 'drivable_area'), None)
            if drivable_label is not None and drivable_label < c:
                label_map[mask_np[drivable_label] == 1] = drivable_label + 1
            for label in range(c):
                if label == drivable_label:
                    continue
                label_map[mask_np[label] == 1] = label + 1
        elif mask_np.ndim == 2:
            label_map = mask_np.astype(np.uint8)
        else:
            return None

    bev_bgr = np.ones((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8) * 255
    for label, cat in id2cat.items():
        color_bgr = COLOR_MAPS_BGR.get(cat, (128, 128, 128))
        bev_bgr[label_map == (label + 1)] = color_bgr
    bev_rgb = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2RGB)
    bev_rgb = np.flipud(bev_rgb)
    return bev_rgb


def _rasterize_vectors_to_mask(
    vectors: Dict[int, np.ndarray],
    roi_size: Tuple[float, float],
    canvas_size: Tuple[int, int],
    thickness: int = 3,
) -> Optional[np.ndarray]:
    if not vectors:
        return None
    canvas_w, canvas_h = canvas_size
    if canvas_w <= 0 or canvas_h <= 0:
        return None
    label_map = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    scale_x = canvas_w / float(roi_size[0])
    scale_y = canvas_h / float(roi_size[1])
    half_w = canvas_w / 2.0
    half_h = canvas_h / 2.0

    for label, vecs in vectors.items():
        if vecs is None or len(vecs) == 0:
            continue
        vec_array = np.asarray(vecs)
        if vec_array.ndim != 3 or vec_array.shape[-1] < 2:
            continue
        for vec in vec_array:
            pts = vec[:, :2]
            if len(pts) < 2:
                continue
            xs = (pts[:, 0] * scale_x) + half_w
            ys = (pts[:, 1] * scale_y) + half_h
            pts_px = np.stack([xs, ys], axis=-1)
            pts_px = np.clip(np.round(pts_px), [0, 0], [canvas_w - 1, canvas_h - 1]).astype(np.int32)
            if label == 0:
                cv2.fillPoly(label_map, [pts_px], color=int(label) + 1)
            else:
                cv2.polylines(label_map, [pts_px], False, color=int(label) + 1, thickness=thickness, lineType=cv2.LINE_AA)
    return label_map


def _create_placeholder_panel(size: Tuple[int, int], message: str) -> np.ndarray:
    width, height = size
    width = int(round(width))
    height = int(round(height))
    width = max(10, width)
    height = max(10, height)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.2, min(0.5, min(width, height) / 280.0))
    thickness = max(1, int(round(scale * 1.5)))
    text = message
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(5, (width - text_size[0]) // 2)
    y = max(text_size[1] + 5, height // 2)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    return img

def plot_one_frame_results(vectors, id_info, roi_size, scene_dir, args):
    """Render a single frame of vector results into an RGB array."""
    roi_w, roi_h = float(roi_size[0]), float(roi_size[1])
    fig, ax = plt.subplots(figsize=(roi_w, roi_h), dpi=args.dpi)
    ax.set_xlim(-roi_w / 2, roi_w / 2)
    ax.set_ylim(-roi_h / 2, roi_h / 2)
    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    ax.set_autoscale_on(False)

    car_img = _get_car_image()
    ax.imshow(car_img, extent=[-2.2, 2.2, -2, 2])

    for label, vecs in vectors.items():
        if len(vecs) == 0:
            continue
        if label == 0:
            label_text = 'P'
        elif label == 1:
            label_text = 'D'
        elif label == 2:
            label_text = 'B'
        else:
            label_text = f'L{label}'

        color = LABEL_COLORS[label] if 0 <= label < len(LABEL_COLORS) else 'm'

        for vec_idx, vec in enumerate(vecs):
            pts = np.asarray(vec[:, :2])
            x, y = pts[:, 0], pts[:, 1]
            ax.plot(x, y, 'o-', color=color, linewidth=25, markersize=20, alpha=args.line_opacity)
            vec_id = None
            if id_info is not None and label in id_info:
                label_info = id_info[label]
                if isinstance(label_info, dict):
                    vec_id = label_info.get(vec_idx)
                elif isinstance(label_info, (list, tuple)) and vec_idx < len(label_info):
                    vec_id = label_info[vec_idx]
            mid_idx = len(x) // 2

            if -roi_h / 2 <= y[mid_idx] < -roi_h / 2 + 2:
                text_y = y[mid_idx] + 2
            elif roi_h / 2 - 2 < y[mid_idx] <= roi_h / 2:
                text_y = y[mid_idx] - 2
            else:
                text_y = y[mid_idx]

            if -roi_w / 2 <= x[mid_idx] < -roi_w / 2 + 4:
                text_x = x[mid_idx] + 4
            elif roi_w / 2 - 4 < x[mid_idx] <= roi_w / 2:
                text_x = x[mid_idx] - 4
            else:
                text_x = x[mid_idx]

            if vec_id is not None:
                ax.text(text_x, text_y, f'{label_text}{vec_id}', fontsize=80, color=color)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    frame_rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
    frame_rgb = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    if scene_dir:
        try:
            imageio.imwrite(os.path.join(scene_dir, 'temp.png'), frame_rgb)
        except Exception:
            pass
    return frame_rgb
    
def _extract_vectors_from_dataset(dataset, sample_idx, origin, roi_size):
    sample = dataset[sample_idx]
    vectors_container = sample.get('vectors')
    if vectors_container is not None:
        vectors_data = vectors_container.data if hasattr(vectors_container, 'data') else vectors_container
    else:
        vectors_data = {}
    semantic_mask = sample.get('semantic_mask')
    if semantic_mask is not None and hasattr(semantic_mask, 'data'):
        semantic_mask = semantic_mask.data
    extracted = {}
    for label, vecs in vectors_data.items():
        if len(vecs) == 0:
            continue
        vec_array = np.asarray(vecs)
        extracted[label] = vec_array * roi_size + origin
    return extracted, semantic_mask


def _load_image_or_blank(path: Optional[str], reference_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if path is None or not os.path.exists(path):
        if reference_shape is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        h, w = reference_shape
        return np.zeros((h, w, 3), dtype=np.uint8)
    try:
        return imageio.imread(path)
    except Exception:
        if reference_shape is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        h, w = reference_shape
        return np.zeros((h, w, 3), dtype=np.uint8)


def build_aerial_lookup(aerial_root: Optional[str]) -> Dict[str, str]:
    if not aerial_root:
        return {}
    search_pattern = os.path.join(aerial_root, '*', '*.png')
    mapping: Dict[str, Tuple[int, str]] = {}
    for path in glob(search_pattern):
        name = os.path.basename(path)
        if name.endswith('_map_overlay.png'):
            priority = 0
        elif name.endswith('_basemap.png'):
            priority = 2
        else:
            priority = 1
        token = name.split('.')[0].split('_')[-1]
        if token not in mapping or priority < mapping[token][0]:
            mapping[token] = (priority, path)
    return {token: info[1] for token, info in mapping.items()}


def vis_pred_data(scene_name, args, dataset, pred_results, origin, roi_size, canvas_size, aerial_lookup, id2cat,
                 semantic_resolver: Optional[SemanticMaskResolver] = None, use_raster_fallback=False):

    scene_dir = os.path.join(args.out_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    token2pred = {}
    if isinstance(pred_results, dict) and 'results' in pred_results:
        for token, entry in pred_results['results'].items():
            token2pred[token] = entry
    elif isinstance(pred_results, list):
        for entry in pred_results:
            token = (
                entry.get('token')
                or entry.get('sample_token')
                or entry.get('sample_token_seq')
                or entry.get('meta', {}).get('token')
                or entry.get('meta', {}).get('sample_token')
            )
            if token is None:
                continue
            token2pred[token] = entry
    else:
        raise TypeError('Unsupported prediction results format for visualization')

    if not token2pred:
        raise ValueError('No prediction entries found for visualization')

    per_frame_writer = VideoRecorder(os.path.join(scene_dir, 'per_frame_pred.gif'))
    composite_writer = VideoRecorder(os.path.join(scene_dir, 'per_frame_pred_quadrants.mp4'))
    seg_writer = VideoRecorder(os.path.join(scene_dir, 'per_frame_pred_segmentation.mp4'))
    has_frames = False

    scene_indices = dataset.scene_name2idx.get(scene_name, [])
    progress = ProgressLogger(scene_name, len(scene_indices), 'Prediction')
    g2l_id_mapping = dict()
    label_ins_counter = defaultdict(int)

    for frame_idx, global_idx in enumerate(scene_indices):
        progress.update(frame_idx)
        sample = dataset.samples[global_idx]
        sample_token = sample['token']
        pred_entry = token2pred.get(sample_token)
        if pred_entry is None:
            continue

        vectors_raw = pred_entry.get('vectors', [])
        if len(vectors_raw) == 0:
            continue
        vectors_np = np.asarray(vectors_raw)
        if vectors_np.ndim == 1:
            if len(vectors_np) % 2 != 0:
                continue
            vectors_np = vectors_np.reshape(1, -1, 2)
        elif vectors_np.ndim == 2:
            if vectors_np.shape[-1] % 2 != 0:
                continue
            num_points = vectors_np.shape[-1] // 2
            vectors_np = vectors_np.reshape(-1, num_points, 2)
        elif vectors_np.ndim != 3:
            continue
        vectors = vectors_np
        if np.abs(vectors).max() <= 1:
            vectors = vectors * roi_size + origin

        labels = np.asarray(pred_entry.get('labels', []), dtype=np.int64)
        if labels.shape[0] != vectors.shape[0]:
            labels = labels[:vectors.shape[0]]

        global_ids_raw = pred_entry.get('global_ids')
        if global_ids_raw is None:
            global_ids = np.arange(vectors.shape[0])
        else:
            global_ids = np.asarray(global_ids_raw, dtype=np.int64)
            if global_ids.shape[0] != vectors.shape[0]:
                global_ids = np.arange(vectors.shape[0])

        per_label_results = defaultdict(list)
        for ins_idx in range(vectors.shape[0]):
            label = int(labels[ins_idx]) if ins_idx < len(labels) else 0
            global_id = int(global_ids[ins_idx])
            if global_id not in g2l_id_mapping:
                local_idx = label_ins_counter.get(label, 0)
                g2l_id_mapping[global_id] = (label, local_idx)
                label_ins_counter[label] = label_ins_counter.get(label, 0) + 1
            else:
                stored_label, local_idx = g2l_id_mapping[global_id]
                if label != stored_label:
                    local_idx = label_ins_counter.get(label, 0)
                    g2l_id_mapping[global_id] = (label, local_idx)
                    label_ins_counter[label] = label_ins_counter.get(label, 0) + 1
            per_label_results[label].append((vectors[ins_idx], global_id, g2l_id_mapping[global_id][1]))

        curr_vectors = defaultdict(list)
        id_info = {}
        for label, results in per_label_results.items():
            vec_results = [item[0] for item in results]
            local_ids = [item[2] for item in results]
            curr_vectors[label] = np.stack(vec_results, axis=0)
            id_info[label] = {idx: ins_id for idx, ins_id in enumerate(local_ids)}

        viz_image = plot_one_frame_results(curr_vectors, id_info, roi_size, scene_dir, args)
        per_frame_writer.add_frame(viz_image)
        has_frames = True

        gt_vectors, gt_semantic_mask = _extract_vectors_from_dataset(dataset, global_idx, origin, roi_size)
        gt_image = plot_one_frame_results(gt_vectors, None, roi_size, scene_dir, args)
        seg_mask_raw = pred_entry.get('semantic_mask')
        if seg_mask_raw is None and semantic_resolver is not None:
            seg_mask_raw = semantic_resolver.get(sample_token)
        if seg_mask_raw is None and use_raster_fallback:
            seg_mask_raw = _rasterize_vectors_to_mask(curr_vectors, roi_size, canvas_size)
        seg_pred_img = _colorize_semantic_mask(seg_mask_raw, id2cat)
        seg_gt_img = _colorize_semantic_mask(gt_semantic_mask, id2cat)
        if seg_pred_img is None:
            msg = 'Segmentation Prediction Missing (use --pred-semantic)' if not use_raster_fallback \
                else 'Segmentation Prediction N/A'
            seg_pred_img = _create_placeholder_panel(canvas_size, msg)
        if seg_gt_img is None:
            seg_gt_img = _create_placeholder_panel(canvas_size, 'Segmentation Ground Truth N/A')

        raw_sample = sample
        cams = raw_sample['cams']
        if args.front_cam in cams:
            front_path = _resolve_image_path(cams[args.front_cam]['img_fpath'], dataset)
        else:
            first_cam = next(iter(cams.values()))
            front_path = _resolve_image_path(first_cam['img_fpath'], dataset)
        front_img = _load_dataset_image(front_path, dataset)

        aerial_img = None
        if getattr(args, 'enable_aerial_panel', False):
            aerial_path = raw_sample.get('aerial_image_path')
            if aerial_path is None and aerial_lookup:
                aerial_path = aerial_lookup.get(raw_sample['token'])
            if aerial_path is not None:
                aerial_img = _load_dataset_image(aerial_path, dataset, reference_shape=front_img.shape[:2])

        panels = [
            (viz_image, 'Vector Prediction'),
            (gt_image, 'Vector Ground Truth'),
            (seg_pred_img, 'Segmentation Prediction'),
            (seg_gt_img, 'Segmentation Ground Truth'),
            (front_img, args.front_cam),
        ]
        if getattr(args, 'enable_aerial_panel', False):
            panels.append((aerial_img, 'Aerial View'))
        composite = assemble_panels(panels=panels, columns=3)
        composite_writer.add_frame(composite)

        if seg_pred_img is not None or seg_gt_img is not None:
            seg_pair = assemble_panels(
                panels=[
                    (seg_pred_img, 'Segmentation Prediction'),
                    (seg_gt_img, 'Segmentation Ground Truth'),
                ],
                columns=2,
            )
            seg_writer.add_frame(seg_pair)
            del seg_pair
        del curr_vectors, gt_vectors, gt_image, viz_image, seg_pred_img, seg_gt_img, front_img
        if aerial_img is not None:
            del aerial_img
    if not has_frames:
        per_frame_writer.close()
        composite_writer.close()
        seg_writer.close()
        print(f'Warning: no predictions found for scene {scene_name}; skipping visualization.')
        return
    per_frame_writer.close()
    composite_writer.close()
    seg_writer.close()
        
def vis_gt_data(scene_name, args, dataset, scene_name2idx, gt_data, origin, roi_size, canvas_size, aerial_lookup, id2cat):
    gt_info = gt_data[scene_name]
    sample_ids = gt_info['sample_ids']
    instance_ids_seq = gt_info['instance_ids']

    scene_dir = os.path.join(args.out_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    per_frame_writer = VideoRecorder(os.path.join(scene_dir, 'per_frame_gt.gif'))
    composite_writer = VideoRecorder(os.path.join(scene_dir, 'per_frame_gt_quadrants.mp4'))
    seg_writer = VideoRecorder(os.path.join(scene_dir, 'per_frame_gt_segmentation.mp4'))
    cam_writers: Dict[str, VideoRecorder] = {}
    has_frames = False

    scene_indices = scene_name2idx[scene_name]
    total_frames = min(len(scene_indices), len(sample_ids), len(instance_ids_seq))
    progress = ProgressLogger(scene_name, total_frames, 'Ground Truth')
    for frame_idx in range(total_frames):
        progress.update(frame_idx)
        global_idx = scene_indices[frame_idx]
        sample_idx = sample_ids[frame_idx]
        id_info = instance_ids_seq[frame_idx]
        # collect images for each camera
        sample_meta = dataset.samples[global_idx]
        cams = sample_meta['cams']
        front_img = None
        fallback_image = None
        for cam, info in cams.items():
            cam_path = _resolve_image_path(info['img_fpath'], dataset)
            img = _load_dataset_image(cam_path, dataset)
            writer = cam_writers.get(cam)
            if writer is None:
                writer = VideoRecorder(os.path.join(scene_dir, f'{cam}.gif'), scale=0.3)
                cam_writers[cam] = writer
            writer.add_frame(img)
            if cam == args.front_cam:
                front_img = img
            if fallback_image is None:
                fallback_image = img
        if front_img is None:
            front_img = fallback_image
        if front_img is None:
            front_img = _create_placeholder_panel(canvas_size, f'{args.front_cam} Image N/A')

        aerial_img = None
        if getattr(args, 'enable_aerial_panel', False):
            aerial_path = sample_meta.get('aerial_image_path')
            if aerial_path is None and aerial_lookup:
                aerial_path = aerial_lookup.get(sample_meta['token'])
            if aerial_path is not None:
                aerial_img = _load_dataset_image(aerial_path, dataset, reference_shape=front_img.shape[:2])

        sample_info = dataset[sample_idx]
        vectors_container = sample_info.get('vectors')
        vectors_data = vectors_container.data if hasattr(vectors_container, 'data') else vectors_container
        curr_vectors = {}
        for label, vecs in vectors_data.items():
            if len(vecs) > 0:
                curr_vectors[label] = vecs * roi_size + origin
            else:
                curr_vectors[label] = vecs

        viz_image = plot_one_frame_results(curr_vectors, id_info, roi_size, scene_dir, args)
        per_frame_writer.add_frame(viz_image)
        has_frames = True

        semantic_mask = sample_info.get('semantic_mask')
        if semantic_mask is not None and hasattr(semantic_mask, 'data'):
            semantic_mask = semantic_mask.data
        seg_img = _colorize_semantic_mask(semantic_mask, id2cat)
        if seg_img is None:
            seg_img = _create_placeholder_panel(canvas_size, 'Segmentation Ground Truth N/A')

        panels = [
            (viz_image, 'Map Ground Truth'),
            (seg_img, 'Segmentation Ground Truth'),
            (front_img, 'Front Camera'),
        ]
        if getattr(args, 'enable_aerial_panel', False):
            panels.append((aerial_img, 'Aerial View'))
        composite = assemble_panels(panels=panels, columns=2)
        composite_writer.add_frame(composite)

        seg_frame = assemble_panels(
            panels=[(seg_img, 'Segmentation Ground Truth')],
            columns=1,
        )
        seg_writer.add_frame(seg_frame)

        del sample_info, curr_vectors, viz_image, seg_img, front_img, seg_frame
        if aerial_img is not None:
            del aerial_img

    if not has_frames:
        per_frame_writer.close()
        composite_writer.close()
        seg_writer.close()
        for writer in cam_writers.values():
            writer.close()
        print(f'Warning: no ground-truth frames found for scene {scene_name}; skipping visualization.')
        return

    per_frame_writer.close()
    composite_writer.close()
    seg_writer.close()
    for writer in cam_writers.values():
        writer.close()
    
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    import_plugin(cfg)
    dataset = DATASETS.build(cfg.match_config)

    cat2id = getattr(dataset, 'cat2id', None)
    if cat2id is None and hasattr(cfg, 'cat2id'):
        cat2id = cfg.cat2id
    if cat2id is None:
        cat2id = {}
    id2cat = {idx: cat for cat, idx in cat2id.items()}
    args.enable_legend = bool(cat2id)

    aerial_lookup: Dict[str, str] = {}
    dataset_lookup = getattr(dataset, 'sample_token2aerial_fpath', None)
    if dataset_lookup:
        aerial_lookup = dict(dataset_lookup)
    else:
        aerial_root = args.aerial_root
        data_cfg = getattr(cfg, 'data', None)
        if aerial_root is None and data_cfg is not None:
            for split in ('test', 'val', 'train'):
                split_cfg = data_cfg.get(split) if isinstance(data_cfg, dict) else getattr(data_cfg, split, None)
                if split_cfg is None:
                    continue
                candidate = split_cfg.get('aerial_data_root') if isinstance(split_cfg, dict) else getattr(split_cfg, 'aerial_data_root', None)
                if candidate:
                    aerial_root = candidate
                    break
        if aerial_root:
            aerial_lookup = build_aerial_lookup(os.path.abspath(aerial_root))
    dataset_has_aerial = False
    for sample in dataset.samples:
        if sample.get('aerial_image_path'):
            dataset_has_aerial = True
            break
    args.enable_aerial_panel = dataset_has_aerial or bool(aerial_lookup)
    if not args.enable_aerial_panel:
        print('Aerial imagery unavailable; skipping aerial panels.')

    scene_name2idx = {}
    scene_name2token = {}
    for idx, sample in enumerate(dataset.samples):
        scene = sample['scene_name']
        if scene not in scene_name2idx:
            scene_name2idx[scene] = []
            scene_name2token[scene] = []
        scene_name2idx[scene].append(idx)

    if args.data_path == "":
        data = {}
    elif args.option == "vis-gt": # visulize GT option
        data = load(args.data_path)
    elif args.option == "vis-pred":
        try:
            data = load(args.data_path)
        except Exception:
            with open(args.data_path, 'rb') as fp:
                data = pickle.load(fp)

    all_scene_names = sorted(list(scene_name2idx.keys()))
    scene_info_list = []
    for single_scene_name in all_scene_names:
        scene_info_list.append((single_scene_name, args))

    roi_size = torch.tensor(cfg.roi_size).numpy()
    origin = torch.tensor(cfg.pc_range[:2]).numpy()
    canvas_size = tuple(getattr(cfg, 'canvas_size', (200, 100)))

    semantic_resolver = None
    if args.option == "vis-pred":
        initial_results = None
        if isinstance(data, dict) and isinstance(data.get('results'), dict):
            initial_results = data['results']
        fallback_paths: List[str] = []
        if args.pred_semantic:
            for sem_path in args.pred_semantic:
                fallback_paths.append(os.path.abspath(sem_path))
        if args.data_path.endswith('.pkl'):
            default_sem_path = os.path.join(os.path.dirname(args.data_path), 'submission_vector.json')
            if os.path.isfile(default_sem_path):
                fallback_paths.append(os.path.abspath(default_sem_path))
        semantic_resolver = SemanticMaskResolver(
            initial_results=initial_results,
            fallback_paths=fallback_paths,
        )
    
    for scene_name in all_scene_names:

        if args.scene_id is not None and scene_name not in args.scene_id:
            continue
        scene_dir = os.path.join(args.out_dir,scene_name)
        if os.path.exists(scene_dir) and len(os.listdir(scene_dir)) > 0 and not args.overwrite:
            print(f"Scene {scene_name} already generated, skipping...")
            continue
        os.makedirs(scene_dir,exist_ok=True)
        if args.option == "vis-gt":
            vis_gt_data(scene_name=scene_name, args=args, dataset=dataset, 
                scene_name2idx=scene_name2idx, gt_data=data,origin=origin,roi_size=roi_size,
                canvas_size=canvas_size, aerial_lookup=aerial_lookup, id2cat=id2cat)
        elif args.option == "vis-pred":
            vis_pred_data(
                scene_name=scene_name,
                args=args,
                dataset=dataset,
                pred_results=data,
                origin=origin,
                roi_size=roi_size,
                canvas_size=canvas_size,
                aerial_lookup=aerial_lookup,
                id2cat=id2cat,
                semantic_resolver=semantic_resolver,
                use_raster_fallback=args.raster_fallback,
            )
        if getattr(args, 'enable_legend', False):
            legend_path = os.path.join(scene_dir, 'legend.png')
            if not os.path.exists(legend_path):
                render_legend_image(legend_path, dataset.cat2id)

if __name__ == '__main__':
    main()

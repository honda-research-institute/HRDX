import argparse
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from plugin.datasets.rdx_dataset import RDXDataset
from plugin.datasets.visualize.styled_renderer import StyledRenderer
from plugin.datasets.visualize.renderer import Image as _Image  # placeholder to avoid lint


CAMERA_GRID_LAYOUT = [
    ["cam_1", "cam_5", "cam_2"],
    ["cam_4", "cam_6", "cam_3"],
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render camera overlays with arrows and local aerial backgrounds without vector overlays."
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default="2025_05_01_12_54_30",
        help="Scene identifier to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: vis_global/<scene_id>_local_aerial).",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=1,
        help="Index offset within the selected scene to start rendering.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum number of frames to render from the start offset. Use <=0 for all.",
    )
    parser.add_argument(
        "--skip-plain-bev",
        action="store_true",
        help="If set, skip the plain BEV render.",
    )
    parser.add_argument(
        "--skip-aerial",
        action="store_true",
        help="If set, skip the aerial background render.",
    )
    parser.add_argument(
        "--skip-camera",
        action="store_true",
        help="If set, skip rendering camera projections.",
    )
    parser.add_argument(
        "--skip-camera-grid",
        action="store_true",
        help="If set, skip composing the 2x3 camera mosaic.",
    )
    parser.add_argument(
        "--car-overlay",
        type=str,
        default="docs/fig/car_img.png",
        help="Path to a PNG of the car top-down sprite (with transparency).",
    )
    parser.add_argument(
        "--car-scale",
        type=float,
        default=0.1,
        help="Relative scale (0-1) of the car overlay with respect to the min(width,height) of the image.",
    )
    return parser.parse_args()


def _load_camera_images(dataset: RDXDataset, sample: dict) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    filenames = sample.get("img_filenames") or []
    for cam_path in filenames:
        resolved = cam_path
        mount = getattr(dataset, "data_server_mountpoint", None)
        if isinstance(mount, str) and isinstance(cam_path, str):
            prefix = "/data/"
            if cam_path.startswith(prefix):
                resolved = os.path.join(mount, cam_path[len(prefix) :])

        with open(resolved, "rb") as fh:
            payload = fh.read(dataset.PAYLOAD_SIZE_8BIT)
        bayer = np.frombuffer(payload, dtype=np.uint8).reshape((1860, 2880))
        image_bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)
        images.append(image_bgr)
    return images


def _load_car_sprite(path: str) -> Optional[Image.Image]:
    if not path or not os.path.isfile(path):
        return None
    try:
        return Image.open(path).convert("RGBA")
    except Exception:
        return None


def _overlay_car(
    base_img: Image.Image,
    car_sprite: Optional[Image.Image],
    scale: float,
) -> Image.Image:
    if car_sprite is None or scale <= 0.0:
        return base_img
    w, h = base_img.size
    sprite = car_sprite.copy()
    target_size = int(min(w, h) * scale)
    if target_size <= 0:
        return base_img
    ratio = target_size / max(sprite.width, sprite.height)
    new_size = (
        max(1, int(round(sprite.width * ratio))),
        max(1, int(round(sprite.height * ratio))),
    )
    sprite = sprite.resize(new_size, Image.BICUBIC)
    canvas = base_img.convert("RGBA")
    x = (w - new_size[0]) // 2
    y = (h - new_size[1]) // 2
    canvas.alpha_composite(sprite, (x, y))
    return canvas.convert("RGB")


def _load_local_aerial(dataset: RDXDataset, sample: dict, background: Tuple[int, int, int]) -> Image.Image:
    """Fetch the ego-aligned local aerial crop (without overlays)."""
    cam_location = sample.get("img_filenames", [None])[4] if sample.get("img_filenames") else None
    cam_seq = None
    if cam_location:
        stem = os.path.splitext(os.path.basename(cam_location))[0]
        try:
            cam_seq = int(stem)
        except ValueError:
            cam_seq = stem

    crop_path = dataset._resolve_aerial_crop_path(
        scene_name=sample.get("scene_name"),
        frame_number=sample.get("frame_number"),
        token=sample.get("token"),
        cam_5_seq_num=cam_seq,
    )
    if crop_path and os.path.isfile(crop_path):
        return Image.open(crop_path).convert("RGB")
    return Image.new("RGB", (1024, 1024), background)


def main() -> None:
    args = _parse_args()

    coords_dim = 3
    roi_size = (60, 60)
    meta = dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False,
        output_format="vector",
    )
    cat2id = {
        "Parking spots": 0,
        "Lane center lines": 1,
        "Non-drivable areas": 2,
        "Bike lanes": 3,
        "Road boundary": 4,
        "Lane lines": 5,
        "Text on the road": 6,
        "Crosswalks": 7,
    }

    dataset = RDXDataset(
        data_root="./datasets/RDX/HRDX_annotations",
        data_server_mountpoint="./datasets/RDX/server_root",
        ann_file="./datasets/RDX/all_data_1_every_5m.pkl",
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=[
            dict(
                type="VectorizeMap",
                coords_dim=coords_dim,
                simplify=False,
                sample_dist=0.1,
                normalize=False,
                roi_size=roi_size,
            ),
            dict(type="FormatBundleMap"),
            dict(
                type="Collect3D",
                keys=["vectors"],
                meta_keys=[
                    "token",
                    "ego2img",
                    "sample_idx",
                    "ego2global_translation",
                    "ego2global_rotation",
                    "img_shape",
                    "scene_name",
                ],
            ),
        ],
        interval=1,
        test_mode=True,
    )

    styled = StyledRenderer(cat2id=cat2id, roi_size=roi_size, dataset="rdx")
    car_sprite = _load_car_sprite(args.car_overlay)

    scene_id = args.scene_id
    scene_indices = [
        idx for idx, sample in enumerate(dataset.samples) if sample["scene_name"] == scene_id
    ]
    if not scene_indices:
        raise ValueError(f"No samples found for scene '{scene_id}'")

    out_dir = args.output_dir or os.path.join("vis_global", f"{scene_id}_local_aerial")
    os.makedirs(out_dir, exist_ok=True)

    try:
        from mmcv.parallel import DataContainer
    except ImportError:
        DataContainer = None

    start_offset = max(0, args.start_offset)
    end_offset = None if args.max_frames <= 0 else start_offset + args.max_frames
    target_indices = scene_indices[start_offset:end_offset]

    for sample_idx in tqdm(target_indices):
        sample = deepcopy(dataset.get_sample(sample_idx))
        data = dataset.pipeline(sample)

        vectors = data.get("vectors")
        if DataContainer and isinstance(vectors, DataContainer):
            vectors = vectors.data
        if vectors is None:
            continue

        frame_number = sample.get("frame_number")
        frame_tag = f"{frame_number:06d}" if isinstance(frame_number, int) else f"idx_{sample_idx}"

        if not args.skip_plain_bev:
            bev_dir = os.path.join(out_dir, "plain_bev")
            os.makedirs(bev_dir, exist_ok=True)
            plain_path = os.path.join(bev_dir, f"styled_plain_{frame_tag}.jpg")
            styled.render_bev_from_vectors_pretty(
                vectors=vectors,
                out_dir=out_dir,
                specified_path=plain_path,
                background=None,
            )
            if car_sprite is not None:
                bev_img = Image.open(plain_path).convert("RGB")
                bev_img = _overlay_car(bev_img, car_sprite, args.car_scale)
                bev_img.save(plain_path)
                bev_img.close()

        if not args.skip_aerial:
            aerial_img = _load_local_aerial(dataset, sample, background=(24, 24, 24))
            aerial_dir = os.path.join(out_dir, "local_aerial")
            os.makedirs(aerial_dir, exist_ok=True)
            aerial_out = os.path.join(aerial_dir, f"local_aerial_{frame_tag}.jpg")
            aerial_img = _overlay_car(aerial_img, car_sprite, args.car_scale)
            aerial_img.save(aerial_out)
            aerial_img.close()

        mosaic_sources: Dict[str, Image.Image] = {}

        if not args.skip_camera:
            images = _load_camera_images(dataset, sample)
            camera_dir = os.path.join(out_dir, "camera_views")
            render_idx = frame_number if isinstance(frame_number, int) else sample_idx
            styled.render_camera_views_from_vectors_pretty(
                vectors=vectors,
                imgs=images,
                ego2cams=sample["cam_extrinsics"],
                intrinsics=sample["cam_intrinsics"],
                distortions=sample.get("cam_distortion_coeffs"),
                out_dir=camera_dir,
                idx=render_idx,
                draw_lane_arrows=True,
                thickness_scale=3.2,
            )
            for cam_row in CAMERA_GRID_LAYOUT:
                for cam_name in cam_row:
                    cam_dir = os.path.join(camera_dir, cam_name)
                    filename = (
                        f"styled_projected_{render_idx}.jpg" if frame_number is not None else "styled_projected.jpg"
                    )
                    img_path = os.path.join(cam_dir, filename)
                    if not os.path.isfile(img_path):
                        candidates = sorted(
                            [
                                p
                                for p in os.listdir(cam_dir)
                                if p.startswith("styled_projected") and p.endswith(".jpg")
                            ]
                        )
                        if candidates:
                            img_path = os.path.join(cam_dir, candidates[-1])
                    if os.path.isfile(img_path):
                        bgr = cv2.imread(img_path)
                        if bgr is not None:
                            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            mosaic_sources[cam_name] = Image.fromarray(rgb)

        if mosaic_sources and not args.skip_camera_grid:
            cols = len(CAMERA_GRID_LAYOUT[0])
            rows = len(CAMERA_GRID_LAYOUT)
            padding = 0  # no whitespace between tiles
            sample_tile = next(iter(mosaic_sources.values()))
            tile_w = sample_tile.width
            tile_h = sample_tile.height
            canvas = Image.new(
                "RGB",
                (
                    padding * (cols + 1) + cols * tile_w,
                    padding * (rows + 1) + rows * tile_h,
                ),
                (15, 15, 15),
            )

            def place_tile(img: Image.Image, dest: Tuple[int, int]) -> None:
                ratio = min(tile_w / img.width, tile_h / img.height)
                resized = img.resize((int(round(img.width * ratio)), int(round(img.height * ratio))), Image.BICUBIC)
                panel = Image.new("RGB", (tile_w, tile_h), (15, 15, 15))
                offset = ((tile_w - resized.width) // 2, (tile_h - resized.height) // 2)
                panel.paste(resized, offset)
                canvas.paste(panel, dest)

            for row_idx, row in enumerate(CAMERA_GRID_LAYOUT):
                for col_idx, cam_name in enumerate(row):
                    tile = mosaic_sources.get(cam_name)
                    if tile is None:
                        continue
                    x = padding + col_idx * (tile_w + padding)
                    y = padding + row_idx * (tile_h + padding)
                    place_tile(tile, (x, y))
            mosaic_dir = os.path.join(out_dir, "camera_grids")
            os.makedirs(mosaic_dir, exist_ok=True)
            grid_path = os.path.join(mosaic_dir, f"camera_grid_{frame_tag}.jpg")
            canvas.save(grid_path)
            canvas.close()


if __name__ == "__main__":
    main()

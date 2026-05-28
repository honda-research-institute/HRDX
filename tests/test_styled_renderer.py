import argparse
import os
from copy import deepcopy
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from plugin.datasets.rdx_dataset import RDXDataset
from plugin.datasets.visualize.styled_renderer import StyledRenderer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render styled BEV and camera projections.")
    parser.add_argument(
        "--no-bev-plain",
        action="store_true",
        help="Skip rendering the plain (no-background) BEV image.",
    )
    parser.add_argument(
        "--no-bev-aerial",
        action="store_true",
        help="Skip rendering the aerial-background BEV image.",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Skip rendering the camera projections.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vis_global/suppl_material",
        help="Override the base output directory (default: vis_global/<scene_id>).",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=147,
        help="Index offset within the selected scene to start rendering.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3000,
        help="Maximum number of frames to render from the start offset. Use <=0 for all.",
    )
    return parser.parse_args()


def _load_camera_images(dataset: RDXDataset, sample: dict) -> List[np.ndarray]:
    """Decode all camera frames referenced by the sample into RGB images."""
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
        image = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)
        images.append(image)

    return images


def main() -> None:
    """Render BEV and camera outputs with StyledRenderer for qualitative QA."""
    args = _parse_args()

    coords_dim = 3
    roi_size = (150,100)
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
        ann_file="./datasets/RDX/train_1_every_5m.pkl",
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
    scene_id = "2025_05_01_12_54_30"
    scene_indices = [
        sample_idx for sample_idx, sample in enumerate(dataset.samples) if sample["scene_name"] == scene_id
    ]
    if not scene_indices:
        raise ValueError(f"No samples found for scene '{scene_id}'")

    out_dir = args.output_dir or os.path.join("vis_global", scene_id)
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
        print(f"Sample group idx: {sample['group']}")
        data = dataset.pipeline(sample)

        vectors = data.get("vectors")
        if DataContainer and isinstance(vectors, DataContainer):
            vectors = vectors.data
        if vectors is None:
            continue

        frame_number = sample.get("frame_number")
        frame_tag = f"{frame_number:06d}" if isinstance(frame_number, int) else f"idx_{sample_idx}"

        if not args.no_bev_plain:
            plain_path = os.path.join(out_dir, f"styled_grid_{frame_tag}.jpg")
            styled.render_bev_from_vectors_pretty(
                vectors=vectors,
                out_dir=out_dir,
                specified_path=plain_path,
                background=None,
            )

        if not args.no_bev_aerial:
            lat_lon_heading = sample.get("lat_long_heading") or [None, None, None]
            center_lat, center_lon, heading_deg = (lat_lon_heading + [None, None, None])[:3]
            if center_lat is None or center_lon is None:
                raise ValueError(
                    f"Sample {sample_idx} missing latitude/longitude for aerial imagery download."
                )

            aerial_path = os.path.join(out_dir, f"styled_aerial_{frame_tag}.jpg")
            styled.render_bev_from_vectors_pretty(
                vectors=vectors,
                out_dir=out_dir,
                specified_path=aerial_path,
                background=None,
                arcgis_service_url="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export",
                center_lon=center_lon,
                center_lat=center_lat,
                rotation_cw_deg=(heading_deg or 0.0) - 90.0,
                ppm=10,
                arcgis_transparent=False,
                arcgis_print_url=False,
            )

        if not args.no_camera:
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
            )


if __name__ == "__main__":
    main()

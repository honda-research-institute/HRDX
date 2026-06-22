import os.path as osp
import os
import av2.geometry.interpolate as interp_utils
import numpy as np
import copy
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from plugin.datasets.visualize.download_satellite_images import download_export
import math
try:
    from pyquaternion import Quaternion
except Exception:
    Quaternion = None

matplotlib.use('agg') # prevent memory leak for drawing figures in a loop

def remove_nan_values(uv,depth):
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    depth_valid = depth[is_uv_valid]
    return uv_valid,depth_valid

def points_ego2img(pts_ego, extrinsics, intrinsics,distortion=None):
    pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
    pts_cam_4d = extrinsics @ pts_ego_4d.T
    pts_cam = pts_cam_4d[:3, :].T 
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    #uv = (intrinsics @ pts_cam_4d[:3, :]).T
    uv, _ = cv2.projectPoints(pts_cam_4d.T[:, :3], np.zeros(3), np.zeros(3), intrinsics, distortion)
    uv = uv.reshape(-1, 2) 
    depth = pts_cam[:,2]    
    uv, depth = remove_nan_values(uv,depth)

    return uv, depth

def draw_polyline_ego_on_img(polyline_ego, img_bgr, extrinsics, intrinsics, color_bgr, thickness,distortions=None):
    if polyline_ego.shape[1] == 2:
        zeros = np.zeros((polyline_ego.shape[0], 1))
        polyline_ego = np.concatenate([polyline_ego, zeros], axis=1)

    polyline_ego = interp_utils.interp_arc(t=500, points=polyline_ego)
    h, w, c = img_bgr.shape

    if distortions is not None:
        uv, depth = points_ego2img(polyline_ego, extrinsics, intrinsics,distortion=distortions)
        uv_undistorted, depth_undistorted = points_ego2img(polyline_ego, extrinsics, intrinsics)
        is_valid_x = np.logical_and(0 <= uv_undistorted[:, 0], uv_undistorted[:, 0] < w - 1)
        is_valid_y = np.logical_and(0 <= uv_undistorted[:, 1], uv_undistorted[:, 1] < h - 1)
        is_valid_z = depth_undistorted > 0

    else:
        uv, depth = points_ego2img(polyline_ego, extrinsics, intrinsics)

        is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
        is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
        is_valid_z = depth > 0

    is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

    if is_valid_points.sum() == 0:
        return
    
    uv = np.round(uv[is_valid_points]).astype(np.int32)

    draw_visible_polyline_cv2(
        copy.deepcopy(uv),
        valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
        image=img_bgr,
        color=color_bgr,
        thickness_px=thickness,
    )

def draw_visible_polyline_cv2(line, valid_pts_bool, image, color, thickness_px):
    """Draw a polyline onto an image using given line segments.

    Args:
        line: Array of shape (K, 2) representing the coordinates of line.
        valid_pts_bool: Array of shape (K,) representing which polyline coordinates are valid for rendering.
            For example, if the coordinate is occluded, a user might specify that it is invalid.
            Line segments touching an invalid vertex will not be rendered.
        image: Array of shape (H, W, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        thickness_px: thickness (in pixels) to use when rendering the polyline.
    """
    line = np.round(line).astype(int)  # type: ignore
    for i in range(len(line) - 1):

        if (not valid_pts_bool[i]) or (not valid_pts_bool[i + 1]):
            continue

        x1 = line[i][0]
        y1 = line[i][1]
        x2 = line[i + 1][0]
        y2 = line[i + 1][1]

        # Use anti-aliasing (AA) for curves
        image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv2.LINE_AA)


COLOR_MAPS_BGR = {
    # bgr colors
    'divider': (0, 0, 255),
    'boundary': (0, 255, 0),
    'ped_crossing': (255, 0, 0),
    'centerline': (51, 183, 255),
    'drivable_area': (171, 255, 255),
    'Parking spots': (128, 0, 128),
    'Lane center lines': (0, 140, 255),
    'Non-drivable areas': (128, 128, 128),
    'Bike lanes': (255, 255, 0),
    'Road boundary': (19, 69, 139),
    'Lane lines': (255, 191, 0),
    'Text on the road': (0, 0, 0),
    'Crosswalks': (0, 0, 128),
    'lane_line': (255, 191, 0),
    'stop_line': (0, 0, 255),
    'road_boundary': (19, 69, 139),
    'lane_centerline': (0, 140, 255),
    'intersection_centerline': (0, 140, 255),
    'text_on_road': (0, 0, 0),
    'non_drivable_area': (128, 128, 128),
    'parking_spot': (128, 0, 128),
    'crosswalk': (0, 0, 128),
    'bike_lane': (255, 255, 0),
    'blocked_area': (0, 165, 255),
}

COLOR_MAPS_PLT = {
    'divider': 'r',
    'boundary': 'g',
    'ped_crossing': 'b',
    'centerline': 'orange',
    'drivable_area': 'y',
    'Parking spots': 'purple',
    'Lane center lines': 'darkorange',
    'Non-drivable areas': 'gray',
    'Bike lanes': 'cyan',
    'Road boundary': 'saddlebrown',
    'Lane lines': 'deepskyblue',
    'Text on the road': 'black',
    'Crosswalks': 'maroon',
    'lane_line': 'deepskyblue',
    'stop_line': 'crimson',
    'road_boundary': 'saddlebrown',
    'lane_centerline': 'darkorange',
    'intersection_centerline': 'darkorange',
    'text_on_road': 'black',
    'non_drivable_area': 'gray',
    'parking_spot': 'purple',
    'crosswalk': 'maroon',
    'bike_lane': 'cyan',
    'blocked_area': 'orange',
}

CATEGORY_CANONICAL = {
    'Lane lines': 'lane_line',
    'lane lines': 'lane_line',
    'lane_line': 'lane_line',
    'Stop lines': 'stop_line',
    'stop lines': 'stop_line',
    'stop_line': 'stop_line',
    'Road boundary': 'road_boundary',
    'road boundary': 'road_boundary',
    'road_boundary': 'road_boundary',
    'Lane center lines': 'lane_centerline',
    'lane center lines': 'lane_centerline',
    'lane_centerline': 'lane_centerline',
    'Intersection centerline': 'intersection_centerline',
    'intersection centerline': 'intersection_centerline',
    'intersection_centerline': 'intersection_centerline',
    'Text on the road': 'text_on_road',
    'text on the road': 'text_on_road',
    'text_on_road': 'text_on_road',
    'Non-drivable areas': 'non_drivable_area',
    'non-drivable areas': 'non_drivable_area',
    'non_drivable_area': 'non_drivable_area',
    'Parking spots': 'parking_spot',
    'parking spots': 'parking_spot',
    'parking_spot': 'parking_spot',
    'Crosswalks': 'crosswalk',
    'crosswalks': 'crosswalk',
    'crosswalk': 'crosswalk',
    'Bike lanes': 'bike_lane',
    'bike lanes': 'bike_lane',
    'bike_lane': 'bike_lane',
    '施工区域 Blocked areas': 'blocked_area',
    'Blocked areas': 'blocked_area',
    'blocked areas': 'blocked_area',
    'blocked_area': 'blocked_area',
}

CAM_NAMES_AV2 = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
    ]
CAM_NAMES_NUSC = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',]

CAM_NAMES_RDX = ['cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5', 'cam_6']

# 1) Text on the road
text_on_road_color_map = {
    "Straight arrow":         "#9B59B6",  # Purple
    "straight_arrow":         "#9B59B6",
    "Turn right arrow":       "#00BFFF",  # Deep Sky Blue
    "turn_right_arrow":       "#00BFFF",
    "left turn arrow":        "#FFFF00",  # Yellow
    "left_turn_arrow":        "#FFFF00",
    "Railway crossing":       "#6C3483",  # Dark Purple
    "RXR":                    "#008B8B",  # Dark Cyan
    "Keep Clear":             "#002050",  # Navy Blue
    "keep_clear":             "#002050",
    "YIELD (AHEAD)":          "#007FFF",  # Azure
    "yield_ahead":            "#007FFF",
    "SLOW":                   "#FF8C00",  # Dark Orange
    "Stop text":              "#000080",  # Navy
    "stop_text":              "#000080",
    "Merge Advisory Arrow":   "#E6BE8A",  # Pale Gold
    "merge_advisory_arrow":   "#E6BE8A",
    "ONLY":                   "#5F9EA0",  # Cadet Blue
    "PED XING":               "#E97451",  # Burnt Sienna
    "ped_xing":               "#E97451",
    "Other class":            "#D83B01",  # Vivid Orange
    "other_class":            "#D83B01",
    "SCHOOL":                 "#DDA0DD",  # Plum
    "HOV Lane":               "#FFFFFF",  # White
    "hov_lane":               "#FFFFFF",
    "BUS/TAXI Lane":          "#000000",  # Black
    "bus_taxi_lane":          "#000000",
    "other_text":             "#D83B01",
}

# 2) Parking spots
parking_spots_color_map = {
    "empty":     "#228B22",  # Forest Green
    "not empty ":"#36454F",  # Charcoal
    "not_empty": "#36454F",
}

# 3) Lane lines (full combos)
lane_line_combo_color_map = {
    ("Dashed",     "White",  "Single "): "#9B59B6",  # Purple
    ("Dashed",     "Yellow", "Single "): "#00BFFF",  # Deep Sky Blue
    ("Solid",      "White",  "Single "): "#FFFF00",  # Yellow
    ("Solid",      "Yellow", "Single "): "#6C3483",  # Dark Purple
    ("Solid",      "Other", "Single "): "#6C3483",  # Dark Purple
    ("Stop lines", "White",  "Single "): "#008B8B",  # Dark Cyan
    ("Stop lines", "Yellow", "Single "): "#002050",  # Navy Blue
    ("dashed", "white"): "#9B59B6",
    ("dashed", "yellow"): "#00BFFF",
    ("solid", "white"): "#FFFF00",
    ("solid", "yellow"): "#6C3483",
}

# 4a) Lane center lines (full combos)
lane_center_combo_color_map = {
    ("Straight",  "lane centerline"):         "#007FFF",  # Azure
    ("Right",     "lane centerline"):         "#FF8C00",  # Dark Orange
    ("Left",      "lane centerline"):         "#000080",  # Navy
    ("U-turn",    "lane centerline"):         "#E6BE8A",  # Pale Gold
    ("Straight",  "Intersection centerline"): "#5F9EA0",  # Cadet Blue
    ("Right",     "Intersection centerline"): "#E97451",  # Burnt Sienna
    ("Left",      "Intersection centerline"): "#D83B01",  # Vivid Orange
    ("U-turn",    "Intersection centerline"): "#DDA0DD",  # Plum
}

# 4b) Lane center lines fallback (attribute-only)
lane_center_attr_color_map = {
    "Straight": "#228B22",  # Forest Green
    "Right":    "#36454F",  # Charcoal
    "Left":     "#FFFDD0",  # Cream
    "U-turn":   "#1A1A1A",  # Very Dark Gray
    "straight": "#228B22",
    "right":    "#36454F",
    "left":     "#FFFDD0",
    "u_turn":   "#1A1A1A",
}

# 5) Master lookup
def get_color(category, attrs):
    """Return a color (hex string or matplotlib color) for a category."""
    canonical = CATEGORY_CANONICAL.get(category, category)
    attrs = attrs or {}

    if canonical == 'text_on_road':
        label = attrs.get('text_type')
        if isinstance(label, (list, tuple)) and label:
            label = label[0]
        if label is None and 'Text on the road' in attrs:
            raw = attrs['Text on the road']
            if isinstance(raw, (list, tuple)) and raw:
                label = raw[0]
            else:
                label = raw
        if label is not None:
            key = str(label).strip()
            key_normalized = key.lower().replace(' ', '_')
            return text_on_road_color_map.get(key_normalized, text_on_road_color_map.get(key, '#000000'))
        return '#000000'

    if canonical == 'parking_spot':
        status = attrs.get('parking_status')
        if isinstance(status, (list, tuple)) and status:
            status = status[0]
        if status is None and 'attribute' in attrs:
            raw = attrs['attribute']
            if isinstance(raw, (list, tuple)) and raw:
                status = raw[0]
            else:
                status = raw
        key = str(status).strip() if status is not None else 'empty'
        key_normalized = key.lower().replace(' ', '_')
        return parking_spots_color_map.get(key_normalized, parking_spots_color_map.get(key, "#228B22"))

    if canonical == 'lane_line':
        style = attrs.get('lane_style')
        color = attrs.get('lane_color')
        if style is None and 'Type' in attrs:
            style = attrs['Type']
        if color is None and 'Color' in attrs:
            color = attrs['Color']
        key_candidates = []
        if style is not None and color is not None:
            style_str = str(style).strip()
            color_str = str(color).strip()
            key_candidates.append((style_str, color_str, attrs.get('Single or Double', 'Single ')))
            key_candidates.append((style_str.lower(), color_str.lower()))
        for key_candidate in key_candidates:
            if key_candidate in lane_line_combo_color_map:
                return lane_line_combo_color_map[key_candidate]
        return COLOR_MAPS_PLT.get('lane_line', '#FFBF00')

    if canonical in ('lane_centerline', 'intersection_centerline'):
        movement = attrs.get('intersection_movement')
        if isinstance(movement, (list, tuple)) and movement:
            movement = movement[0]
        if movement is None and 'attribute' in attrs:
            raw = attrs['attribute']
            if isinstance(raw, (list, tuple)) and raw:
                movement = raw[0]
            else:
                movement = raw
        if movement is not None:
            movement_key = str(movement).strip()
            # Prefer lowercase canonical versions
            if canonical == 'intersection_centerline':
                combo_key = (movement_key, 'Intersection centerline')
                combo_key_lower = (movement_key.lower(), 'intersection_centerline')
                if combo_key in lane_center_combo_color_map:
                    return lane_center_combo_color_map[combo_key]
                if combo_key_lower in lane_center_combo_color_map:
                    return lane_center_combo_color_map[combo_key_lower]
            else:
                combo_key = (movement_key, 'lane centerline')
                combo_key_lower = (movement_key.lower(), 'lane centerline')
                if combo_key in lane_center_combo_color_map:
                    return lane_center_combo_color_map[combo_key]
                if combo_key_lower in lane_center_combo_color_map:
                    return lane_center_combo_color_map[combo_key_lower]
            if movement_key in lane_center_attr_color_map:
                return lane_center_attr_color_map[movement_key]
            movement_key_lower = movement_key.lower().replace(' ', '_')
            if movement_key_lower in lane_center_attr_color_map:
                return lane_center_attr_color_map[movement_key_lower]
        return lane_center_attr_color_map.get('straight', '#228B22')

    if canonical in COLOR_MAPS_PLT:
        return COLOR_MAPS_PLT[canonical]
    if category in COLOR_MAPS_PLT:
        return COLOR_MAPS_PLT[category]
    return 'g'


class Renderer(object):
    """Render map elements on image views.

    Args:
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        dataset (str): 'av2' or 'nusc'
    """

    def __init__(self, cat2id, roi_size, dataset='av2'):
        self.roi_size = roi_size
        self.cat2id = cat2id
        self.id2cat = {v: k for k, v in cat2id.items()}
        if dataset == 'av2':
            self.cam_names = CAM_NAMES_AV2
        elif dataset == 'nusc':
            self.cam_names = CAM_NAMES_NUSC
        elif dataset == 'rdx':
            self.cam_names = CAM_NAMES_RDX

    def render_aerial_with_vectors(self, vectors, aerial_img, out_dir,idx=None,
                                   draw_scores=False, id_info=None):
        """Convenience wrapper to render vectors over an aerial background.

        Args:
            vectors (dict): {label_id: [np.ndarray(Nx2 or Nx3), ...]}
            aerial_img (str | PIL.Image | np.ndarray | torch.Tensor): background crop
                aligned to the same ROI as vectors (meters in ego frame).
            out_path (str): output path for the rendered image.
            draw_scores (bool): draw per-instance scores if provided in vectors.
            id_info (dict): optional id mapping per label for track visualization.
        """
        # Accept torch tensor (C,H,W) as well; convert to numpy HWC
        try:
            import torch
            if isinstance(aerial_img, torch.Tensor):
                t = aerial_img.detach().cpu()
                if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                    t = t.permute(1, 2, 0)
                aerial_img = t.numpy()
        except Exception:
            pass

        # Ensure output directory exists
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if idx is not None:
            specified_path = os.path.join(out_dir,f'aerial_overlap_{idx}.jpg')
        return self.render_bev_from_vectors(
            vectors=vectors,
            out_dir=out_dir,
            draw_scores=draw_scores,
            id_info=id_info,
            background=aerial_img,
            specified_path=specified_path
        )

    def render_bev_from_vectors(
            self, vectors, out_dir=None, draw_scores=False, specified_path=None,
            id_info=None,
            # --- NEW OPTIONALS (all default to old behavior if unused) ---
            background=None,                 # str | PIL.Image | np.ndarray; if given, used as the base image
            arcgis_service_url=None,         # if set, will download background via /export
            center_lon=None, center_lat=None,# required only if arcgis_service_url is provided
            rotation_cw_deg=0,               # clockwise rotation passed to ArcGIS /export
            ppm=10,                           # pixels-per-meter for background request sizing
            arcgis_fmt="png",
            arcgis_transparent=False,
            arcgis_print_url=False,
            arcgis_extra=None,                # dict of extra kwargs to pass into download_export (e.g., timeout)
            frame_number=None
        ):
        '''Render bev segmentation using vectorized map elements.

        Args:
            vectors (dict): dict of vectorized map elements.
            out_dir (str): output directory
            draw_scores (bool): draw detection scores if (vector, score, prop).
            specified_path (str): explicit output path.
            id_info (dict): {label: [vec_id,...]} parallel to vectors.
            background: optional background image to draw on (path | PIL | np.ndarray).
            arcgis_service_url: if provided, will fetch background from this ArcGIS /export endpoint.
            center_lon, center_lat: center of the export (required if arcgis_service_url is used).
            rotation_cw_deg: clockwise rotation for ArcGIS /export (aligns image to your BEV axes).
            ppm: pixels-per-meter used to choose export size from self.roi_size.
            arcgis_fmt: output format for export (e.g., 'png').
            arcgis_transparent: request transparent background (if supported).
            arcgis_print_url: print the final export URL for debugging.
            arcgis_extra: extra kwargs dict for download_export (e.g., {'timeout': 120}).

        Notes:
            - If neither `background` nor `arcgis_service_url` is provided, this uses the original
            behavior: Image.open('resources/car.png') and imshow with fixed extent.
            - The background image is placed using extent=[-roi_x/2, roi_x/2, -roi_y/2, roi_y/2]
            so your vector coordinates in meters overlay directly.
        '''

        # -----------------------
        # Resolve output path (unchanged logic)
        # -----------------------
        if specified_path:
            map_path = specified_path
        else:
            map_path = os.path.join(out_dir, 'map.jpg')

        roi_w, roi_h = float(self.roi_size[0]), float(self.roi_size[1])

        # -----------------------
        # Resolve background image source
        # -----------------------
        bg_img = None
        bg_needs_cleanup = False  # if we create a temp file for background

        def _to_pil(img_like):
            """Convert various image-like inputs to a PIL RGBA image.

            Accepts:
            - PIL.Image
            - str path
            - numpy.ndarray (H,W), (H,W,1), (H,W,3), (H,W,4), or (C,H,W)
            - torch.Tensor with the same shape conventions as numpy
            """
            # PIL image
            if isinstance(img_like, Image.Image):
                return img_like.convert("RGBA")
            # File path
            if isinstance(img_like, str):
                return Image.open(img_like).convert("RGBA")

            # Torch tensor -> numpy
            try:
                import torch  # local import to avoid hard dependency at module import time
                if isinstance(img_like, torch.Tensor):
                    t = img_like.detach().cpu()
                    # Handle channel position
                    if t.ndim == 3:
                        # If CHW, move to HWC
                        if t.shape[0] in (1, 3, 4) and t.shape[1] > 4 and t.shape[2] > 4:
                            t = t.permute(1, 2, 0)
                    arr = t.numpy()
                else:
                    arr = None
            except Exception:
                arr = None

            # Numpy array branch (also used after torch -> numpy)
            if isinstance(img_like, np.ndarray) or arr is not None:
                arr = img_like if isinstance(img_like, np.ndarray) else arr

                # If CHW, convert to HWC
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and not (arr.shape[2] in (1, 3, 4)):
                    arr = np.transpose(arr, (1, 2, 0))

                # Ensure uint8 range
                if arr.dtype != np.uint8:
                    max_val = float(np.max(arr)) if arr.size > 0 else 1.0
                    scale = 255.0 if max_val <= 1.0 else 1.0
                    arr = np.clip(arr * scale, 0, 255).astype(np.uint8)

                # Expand grayscale to RGB
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                if arr.ndim == 3 and arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)

                return Image.fromarray(arr).convert("RGBA")

            return None

        # (A) If explicit background provided, use it
        if background is not None:
            bg_img = _to_pil(background)

        # (B) Else if ArcGIS URL provided, auto-download a square export sized to ROI via ppm
        elif arcgis_service_url and (center_lon is not None) and (center_lat is not None):
            # Choose a square buffer large enough to cover the ROI; downloader takes a single buffer
            phi = math.radians(center_lat)
            sec_phi = 1.0 / math.cos(phi)

            # if you keep the single square buffer:
            buffer_m = (max(roi_w, roi_h) / 2.0) * sec_phi

            # Choose pixel size from ppm; clamp to a service-friendly range
            target_width_px = int(round(roi_w * ppm))
            target_height_px = int(round(roi_h * ppm))
            max_side = 4096
            min_side = 256
            target_width_px = max(min_side, min(max_side, target_width_px))
            target_height_px = max(min_side, min(max_side, target_height_px))
            size_px = max(target_width_px, target_height_px)  # exporter uses square size

            # Build a temp path for the background image
            tmp_bg_path = os.path.join(out_dir or ".", f"_bev_bg.{arcgis_fmt.lower()}")
            bg_needs_cleanup = True

            extra = arcgis_extra or {}
            # Use the user's existing downloader (assumed imported/available)
            download_export(
                lon=center_lon,
                lat=center_lat,
                buffer_m=buffer_m,
                size=size_px,
                fmt=arcgis_fmt,
                rotation=rotation_cw_deg,   # clockwise rotation to align with BEV axes if desired
                out=tmp_bg_path,
                service_export_url=arcgis_service_url,
                dpi=96,
                transparent=arcgis_transparent,
                use_bbox4326=False,
                print_url=arcgis_print_url,
                **extra
            )
            bg_img = Image.open(tmp_bg_path).convert("RGBA")

            if (bg_img.width != target_width_px) or (bg_img.height != target_height_px):
                left = max(0, int(round((bg_img.width - target_width_px) / 2.0)))
                top = max(0, int(round((bg_img.height - target_height_px) / 2.0)))
                right = left + target_width_px
                bottom = top + target_height_px
                right = min(bg_img.width, right)
                bottom = min(bg_img.height, bottom)
                left = max(0, right - target_width_px)
                top = max(0, bottom - target_height_px)
                bg_img = bg_img.crop((left, top, right, bottom))

            draw = ImageDraw.Draw(bg_img)

            # Center coordinates
            cx, cy = bg_img.size[0] // 2, bg_img.size[1] // 2

            # Small dot (radius 3 px)
            r = 15
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="red")

        # (C) Else fall back to original car image
        bg_extent = [-roi_w / 2, roi_w / 2, -roi_h / 2, roi_h / 2]
        if bg_img is None:
            bg_img = Image.open('resources/car.png').convert("RGBA")
            # keep the original car footprint (~5 m x 4 m) instead of stretching to the ROI
            bg_extent = [-2.5, 2.5, -2.0, 2.0]

        # -----------------------
        # Plot (kept as close as possible to your original)
        # -----------------------
        fig = plt.figure(figsize=(self.roi_size[0], self.roi_size[1]))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-roi_w / 2, roi_w / 2)
        ax.set_ylim(-roi_h / 2, roi_h / 2)
        ax.axis('off')

        # Original had: ax.imshow(car_img, extent=[-2.5, 2.5, -2.0, 2.0])
        # Now: always place background across full ROI (meters) so vectors overlay directly.
        ax.imshow(bg_img, extent=bg_extent)
        ax.scatter(
            0.0,
            0.0,
            s=600,
            c='white',
            edgecolors='black',
            linewidths=2.5,
            zorder=10,
        )

        for label, vector_list in vectors.items():
            cat = self.id2cat[label]
            color = COLOR_MAPS_PLT[cat]
            for vec_i, vector in enumerate(vector_list):
                if draw_scores:
                    vector, score, prop = vector
                if isinstance(vector, list):
                    vector = np.array(vector)
                    from shapely.geometry import LineString
                    vector = np.array(LineString(vector).simplify(0.2).coords)
                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, 'o-', color=color, linewidth=1, markersize=25)
                if draw_scores:
                    if prop:
                        p = 'p'
                    else:
                        p = ''
                    score = round(score, 2)
                    mid_idx = len(x) // 2
                    ax.text(x[mid_idx], y[mid_idx], str(score)+p, fontsize=100, color=color)
                if id_info:
                    vec_id = id_info[label][vec_i]
                    mid_idx = len(x) // 2
                    ax.text(x[mid_idx], y[mid_idx], f'{cat[:1].upper()}{vec_id}', fontsize=100, color=color)

        fig.savefig(map_path, bbox_inches='tight', dpi=20)
        plt.clf()

        # Cleanup temp background file if used
        try:
            if bg_needs_cleanup:
                os.remove(tmp_bg_path)
        except Exception:
            pass

        return map_path

        
    def render_camera_views_from_vectors(self, vectors, imgs, extrinsics, 
            intrinsics, ego2cams, thickness, out_dir,distortions=None,idx= None):
        '''Project vectorized map elements to camera views.
        
        Args:
            vectors (dict): dict of vectorized map elements.
            imgs (tensor): images in bgr color.
            extrinsics (array): ego2img extrinsics, shape (4, 4)
            intrinsics (array): intrinsics, shape (3, 3) 
            thickness (int): thickness of lines to draw on images.
            out_dir (str): output directory
        '''

        for i in range(len(imgs)):
            img = imgs[i]
            extrinsic = extrinsics[i]
            intrinsic = intrinsics[i]
            ego2cam = ego2cams[i]
            img_bgr = copy.deepcopy(img)
            distortion = distortions[i] if distortions is not None else None

            for label, vector_list in vectors.items():
                cat = self.id2cat[label]
                color = COLOR_MAPS_BGR[cat]
                for vector in vector_list:
                    img_bgr = np.ascontiguousarray(img_bgr)
                    if isinstance(vector, list):
                        vector = np.array(vector)
                    draw_polyline_ego_on_img(vector, img_bgr, ego2cam, intrinsic, color, thickness,distortions=distortion)
            if idx is not None:
                cam_out_dir = osp.join(out_dir, self.cam_names[i])
                os.makedirs(cam_out_dir, exist_ok=True)
                out_path = osp.join(cam_out_dir, f'projected_{idx}.jpg')
            else:      
                out_path = osp.join(out_dir, self.cam_names[i]) + '.jpg'
            cv2.imwrite(out_path, img_bgr)

    def render_camera_views_from_vectors_with_attributes(self,vectors,imgs,extrinsics,intrinsics, ego2cams,thickness, out_dir,
        distortions=None,idx=None,time=None, scene_name=None, group_name=None,frame_number=None
    ):
        '''
        Project vectorized map elements to camera views, producing a 2x2 grid image:
        - Top-left: Road boundary, Non-drivable areas, Crosswalks, Bike lanes
        - Top-right: Text on the road, Parking spots
        - Bottom-left: Lane lines
        - Bottom-right: Lane & intersection centerlines
        '''
        # Define label groups
        cat_groups = {
            'tl': ['road_boundary', 'non_drivable_area', 'crosswalk', 'bike_lane'],
            'tr': ['text_on_road', 'parking_spot'],
            'bl': ['lane_line'],
            'br': ['lane_centerline', 'intersection_centerline'],
        }

        for i, img in enumerate(imgs):
            extrinsic = extrinsics[i]
            intrinsic = intrinsics[i]
            ego2cam = ego2cams[i]
            distortion = distortions[i] if distortions is not None else None

            # Prepare blank copies for each quadrant
            imgs_q = {
                'tl': img.copy(),
                'tr': img.copy(),
                'bl': img.copy(),
                'br': img.copy(),
            }

            # Draw each label into its quadrant image
            simple_color_cats = {'road_boundary', 'non_drivable_area', 'crosswalk', 'bike_lane'}

            for label, vector_list in vectors.items():
                cat = self.id2cat[label]
                canonical_cat = CATEGORY_CANONICAL.get(cat, cat)
                for group, cats in cat_groups.items():
                    if canonical_cat in cats:
                        for vector in vector_list:
                            img_q = imgs_q[group]
                            img_q = np.ascontiguousarray(img_q)
                            coords = np.array(vector) if isinstance(vector, list) else vector

                            # Determine color
                            if canonical_cat in simple_color_cats:
                                color_raw = COLOR_MAPS_BGR.get(canonical_cat, COLOR_MAPS_BGR.get(cat, (0, 255, 0)))
                            else:
                                attrs = vector.attrs if getattr(vector, 'attrs', None) else None
                                color_raw = get_color(canonical_cat, attrs) if attrs else COLOR_MAPS_BGR.get(canonical_cat, (0, 255, 0))
                            if isinstance(color_raw, str):
                                hex_str = color_raw.lstrip('#')
                                if len(hex_str) == 6:
                                    r = int(hex_str[0:2], 16)
                                    g = int(hex_str[2:4], 16)
                                    b = int(hex_str[4:6], 16)
                                    color = (b, g, r)
                                else:
                                    raise ValueError(f"Invalid hex color '{color_raw}' for label '{label}'")
                            else:
                                # Assume iterable of numeric components
                                arr = np.array(color_raw)
                                if arr.ndim == 1 and arr.size >= 3:
                                    # Take first three elements
                                    b, g, r = arr[:3]
                                    color = (int(b), int(g), int(r))
                                else:
                                    raise ValueError(f"Invalid color format for label '{label}': {color_raw}")
                            # Draw
                            draw_polyline_ego_on_img(
                                coords,
                                img_q,
                                ego2cam,
                                intrinsic,
                                color,
                                thickness,
                                distortions=distortion
                            )

            # Stack quadrants
            h, w = imgs_q['tl'].shape[:2]
            top = np.hstack((imgs_q['tl'], imgs_q['tr']))
            bottom = np.hstack((imgs_q['bl'], imgs_q['br']))
            grid = np.vstack((top, bottom))

            # Prepare output path
            if idx is not None:
                cam_dir = os.path.join(out_dir, self.cam_names[i])
                os.makedirs(cam_dir, exist_ok=True)
                if group_name is not None:
                    cam_dir = os.path.join(cam_dir, group_name)
                    os.makedirs(cam_dir, exist_ok=True)
                out_path = os.path.join(cam_dir, f'projected_{idx}_grid.jpg')
            else:
                if group_name is not None:
                    out_dir = os.path.join(out_dir, group_name)
                    os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f'{self.cam_names[i]}_grid.jpg')
            
            if time is not None and scene_name is not None:
                # Add text to the grid image
                time = str(time)
                scene_name = str(scene_name)
                frame_text = str(frame_number) if frame_number is not None else None
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3  # Larger font
                font_thickness = 4   # Thicker stroke to balance larger font
                color = (0, 255, 0)  # Green in BGR

                # Calculate text size to space the lines dynamically
                (time_width, time_height), _ = cv2.getTextSize(time, font, font_scale, font_thickness)
                (scene_width, scene_height), _ = cv2.getTextSize(scene_name, font, font_scale, font_thickness)
                (frame_width, frame_height), _ = cv2.getTextSize(frame_text, font, font_scale, font_thickness)
                x = 10
                y = 30 + time_height  # Add height to avoid clipping top of large text

                # Draw time
                cv2.putText(grid, time, (x, y), font, font_scale, color, font_thickness, cv2.LINE_AA)

                # Draw scene name below time, using calculated height
                y_scene = y + scene_height + 20  # extra 20 pixels spacing
                cv2.putText(grid, scene_name, (x, y_scene), font, font_scale, color, font_thickness, cv2.LINE_AA)
                # Draw frame number below scene name if provided
                if frame_text:
                    y_frame = y_scene + frame_height + 20
                    cv2.putText(grid, frame_text, (x, y_frame), font, font_scale, color, font_thickness, cv2.LINE_AA)

            # Save
            cv2.imwrite(out_path, grid)


    def render_bev_from_mask(self, semantic_mask, out_dir, flip=False):
        '''Render bev segmentation from semantic_mask.
        
        Args:
            semantic_mask (array): semantic mask.
            out_dir (str): output directory
        '''

        semantic_mask=np.asarray(semantic_mask)
        if len(semantic_mask.shape) == 3:
            c, h, w = semantic_mask.shape
        else:
            h, w = semantic_mask.shape
        
        bev_img = np.ones((3, h, w), dtype=np.uint8) * 255
        if 'drivable_area' in self.cat2id:
            drivable_area_mask = semantic_mask[self.cat2id['drivable_area']]
            bev_img[:, drivable_area_mask == 1] = \
                    np.array(COLOR_MAPS_BGR['drivable_area']).reshape(3, 1)
        
        for label in self.id2cat:
            cat = self.id2cat[label]
            if cat == 'drivable_area':
                continue
            if len(semantic_mask.shape) == 3:
                valid = (semantic_mask[label] == 1)
            else:
                valid = semantic_mask == (label + 1)
            bev_img[:, valid] = np.array(COLOR_MAPS_BGR[cat]).reshape(3, 1)

        #for label in range(c):
        #    cat = self.id2cat[label]
        #    if cat == 'drivable_area':
        #        continue
        #    mask = semantic_mask[label]
        #    valid = mask == 1
        #    bev_img[:, valid] = np.array(COLOR_MAPS_BGR[cat]).reshape(3, 1)

        out_path = osp.join(out_dir, 'semantic_map.jpg')
        if flip:
            bev_img_flipud = np.array([np.flipud(i) for i in bev_img], dtype=np.uint8)
            cv2.imwrite(out_path, bev_img_flipud.transpose((1, 2, 0)))
        else:
            cv2.imwrite(out_path, bev_img.transpose((1, 2, 0)))
            
        
    def render_lidar_on_cameras(
        self,
        points,
        imgs_or_paths,
        ego2cam_list,
        lidar2ego_translation,
        intrinsics_list,
        out_dir,
        idx=None,
        point_size=2,
        intensity_range=None,
    ):
        """Project LiDAR points onto each camera image and save overlays.

        Args:
            points: ndarray (N, >=3) in lidar/ego frame consistent with `ego2cam_list`.
            imgs_or_paths: list of BGR images or file paths, length == num cams.
            ego2cam_list: list of 4x4 ego-to-cam (here lidar-to-cam) extrinsics.
            intrinsics_list: list of 3x3 intrinsics per camera.
            out_dir: base output directory.
            idx: optional sample index for filenames.
            point_size: circle radius in pixels.
            intensity_range: optional (min,max) to normalize intensity; if None, inferred.
        """
        os.makedirs(out_dir, exist_ok=True)

        if points is None or len(points) == 0:
            return

        pts_arr = np.asarray(points, dtype=np.float32)
        pts_xyz = pts_arr[:, :3]
        has_intensity = pts_arr.shape[1] >= 4

        lidar2ego_translation = np.asarray(lidar2ego_translation, dtype=np.float32).reshape(3)
        lidar2ego_translation_transform = np.eye(4, dtype=np.float32)
        lidar2ego_translation_transform[:3, 3] = lidar2ego_translation

        for i, (img_src, ego2cam, K) in enumerate(zip(imgs_or_paths, ego2cam_list, intrinsics_list)):
            # load image if a path was provided
            img = cv2.imread(img_src) if isinstance(img_src, str) else img_src.copy()
            if img is None:
                continue

            h, w = img.shape[:2]
            uv, depth = points_ego2img(pts_xyz, np.array(ego2cam) @ lidar2ego_translation_transform, np.array(K))

            # filter valid projections
            if uv.size == 0:
                valid = np.zeros((0,), dtype=bool)
            else:
                valid = (
                    (depth > 0.0)
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < w - 1)
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < h - 1)
                )

            uv_v = uv[valid].astype(np.int32)
            depth_v = depth[valid]
            intens_v = pts_arr[valid, 3] if has_intensity else None

            # choose color by intensity if available, else fallback to depth-based normalization
            if has_intensity and intens_v.size > 0:
                if intensity_range is None:
                    lo, hi = np.percentile(intens_v, [1, 99])
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        lo, hi = 0.0, 1.0
                else:
                    lo, hi = intensity_range
                scale = max(1e-6, hi - lo)
                vals = np.clip((intens_v - lo) / scale, 0.0, 1.0)
                colors = (plt.get_cmap('turbo')(vals)[:, :3] * 255).astype(np.uint8)[:, ::-1]
                for (u, v), col in zip(uv_v, colors):
                    cv2.circle(img, (int(u), int(v)), point_size, (int(col[0]), int(col[1]), int(col[2])), thickness=-1, lineType=cv2.LINE_AA)
            else:
                # fallback: color by depth (closer=blue, farther=red using turbo)
                if depth_v.size > 0:
                    dmin, dmax = np.percentile(depth_v, [1, 99])
                    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
                        dmin, dmax = 0.0, 60.0
                    scale = max(1e-6, dmax - dmin)
                    vals = np.clip((depth_v - dmin) / scale, 0.0, 1.0)
                    colors = (plt.get_cmap('turbo')(vals)[:, :3] * 255).astype(np.uint8)[:, ::-1]
                    for (u, v), col in zip(uv_v, colors):
                        cv2.circle(img, (int(u), int(v)), point_size, (int(col[0]), int(col[1]), int(col[2])), thickness=-1, lineType=cv2.LINE_AA)


            cam_dir = os.path.join(out_dir, self.cam_names[i] if i < len(self.cam_names) else f'cam_{i}')
            os.makedirs(cam_dir, exist_ok=True)
            fname = f'lidar_overlay_{idx}.jpg' if idx is not None else 'lidar_overlay.jpg'
            cv2.imwrite(os.path.join(cam_dir, fname), img)

    def render_lidar_bev(
        self,
        points,
        out_dir,
        idx=None,
        px_per_meter=10,
        intensity_range=None,
        bg_color=(255, 255, 255),
        point_color=(0, 0, 255),
        vectors=None,
        vec_thickness=2,
        lidar2ego_rotation=None,
    ):
        """Rasterize LiDAR points onto a BEV canvas and save.

        Args:
            points: ndarray (N, >=3) in lidar/ego frame.
            out_dir: output directory.
            idx: optional sample index for filenames.
            px_per_meter: resolution. Canvas size = roi_size * px_per_meter.
            intensity_range: optional (min,max) to normalize intensity; if None, inferred.
            vectors: optional dict[label] -> list of polylines (Nx2 or Nx3) in ego frame to overlay.
            vec_thickness: thickness (px) for vector lines.
        """
        os.makedirs(out_dir, exist_ok=True)

        if points is None or len(points) == 0:
            return

        roi_x, roi_y = float(self.roi_size[0]), float(self.roi_size[1])
        W = int(round(roi_x * px_per_meter))
        H = int(round(roi_y * px_per_meter))
        canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)

        pts_arr = np.asarray(points, dtype=np.float32)
        pts = pts_arr[:, :3]
        # If provided, align axes to ego (rotation only; no translation)
        if lidar2ego_rotation is not None:
            try:
                from pyquaternion import Quaternion
                R = Quaternion(lidar2ego_rotation).rotation_matrix.astype(np.float32)
                pts = pts.dot(R.T)
            except Exception:
                pass
        has_intensity = pts_arr.shape[1] >= 4

        # map metric coords (x forward, y left) to image coords
        # u increases to the right, v increases downward
        u = (pts[:, 0] + roi_x / 2.0) * px_per_meter
        v = (roi_y / 2.0 - pts[:, 1]) * px_per_meter

        valid = (
            (u >= 0)
            & (u < W)
            & (v >= 0)
            & (v < H)
        )
        u = u[valid].astype(np.int32)
        v = v[valid].astype(np.int32)
        if has_intensity:
            intens = pts_arr[:, 3][valid]
            if intensity_range is None and intens.size > 0:
                lo, hi = np.percentile(intens, [1, 99])
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = 0.0, 1.0
            else:
                lo, hi = intensity_range if intensity_range is not None else (0.0, 1.0)
            vals = np.clip((intens - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
            bgr = (plt.get_cmap('turbo')(vals)[:, :3] * 255).astype(np.uint8)[:, ::-1]
            for uu, vv, col in zip(u, v, bgr):
                canvas[int(vv), int(uu)] = col
        else:
            for uu, vv in zip(u, v):
                canvas[int(vv), int(uu)] = point_color

        # Overlay vector annotations if provided
        if vectors is not None:
            for label, vector_list in vectors.items():
                cat = self.id2cat.get(label, None)
                color = COLOR_MAPS_BGR.get(cat, (0, 255, 0)) if cat is not None else (0, 255, 0)
                for vec in vector_list:
                    arr = np.array(vec) if isinstance(vec, list) else vec
                    if arr is None or len(arr) == 0:
                        continue
                    if arr.shape[1] >= 2:
                        xs = arr[:, 0]
                        ys = arr[:, 1]
                    else:
                        continue
                    uu = (xs + roi_x / 2.0) * px_per_meter
                    vv = (roi_y / 2.0 - ys) * px_per_meter
                    pts_pix = np.stack([uu, vv], axis=1).astype(np.int32)
                    for k in range(len(pts_pix) - 1):
                        p1 = tuple(pts_pix[k])
                        p2 = tuple(pts_pix[k + 1])
                        cv2.line(canvas, p1, p2, color=color, thickness=vec_thickness, lineType=cv2.LINE_AA)

        fname = f'bev_lidar_{idx}.jpg' if idx is not None else 'bev_lidar.jpg'
        cv2.imwrite(os.path.join(out_dir, fname), canvas)

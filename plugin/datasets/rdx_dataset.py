from .base_dataset import BaseMapDataset
from mmdet3d.registry import DATASETS
from .visualize.renderer import Renderer
from shapely.geometry import LineString, Polygon
from mmengine.fileio import load
import numpy as np
from .map_utils.RDXmap_extractor import RDXMapExtractor
from pyquaternion import Quaternion
from collections import defaultdict
import pickle
import os
from glob import glob
from scipy.spatial.transform import Rotation as R
from time import time
import cv2
from copy import deepcopy
import json
import re

@DATASETS.register_module()
class RDXDataset(BaseMapDataset):
    """RDX map dataset class.

    Args:
        data_root (str): root directory of the
    """
    def __init__(self, data_root, img_data_root, aerial_crop_root=None,
                 lidar_data_root=None, **kwargs):
        self.data_root = data_root
        self.img_data_root = img_data_root
        self.aerial_crop_root = aerial_crop_root
        self.lidar_data_root = lidar_data_root
        super().__init__(**kwargs)
        self.map_extractor = RDXMapExtractor(data_root, self.roi_size)
        self.renderer = Renderer(self.cat2id, self.roi_size, 'rdx')

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        
        start_time = time()
        ann = load(ann_file)

        # Some RDX eval configs point `ann_file` to `*_gt_tracks.pkl`, which
        # stores tracking metadata (dict) rather than per-frame samples (list).
        # For dataset loading we need sample records, so fall back to the paired
        # non-gt-tracks pickle when available.
        if isinstance(ann, dict):
            fallback_ann_file = ann_file.replace('_gt_tracks.pkl', '.pkl')
            if fallback_ann_file != ann_file and os.path.isfile(fallback_ann_file):
                print(
                    f'[{self.__class__.__name__}] ann_file "{ann_file}" is track meta; '
                    f'loading samples from "{fallback_ann_file}" instead.')
                ann = load(fallback_ann_file)
            elif 'samples' in ann and isinstance(ann['samples'], list):
                ann = ann['samples']
            else:
                raise TypeError(
                    f'Unsupported annotation payload in "{ann_file}": '
                    f'expected list or dict with "samples", got {type(ann)}')

        #ann = self.remove_empty(ann)
        samples = ann[::self.interval]
        
        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples    

    def remove_empty(self, train_data):
        """Filter out sessions with missing/empty group annotations."""
        sess2group = defaultdict(set)
        for item in train_data:
            sess2group[item['scene_name']].add(item['group'])

        missing_groups = set()
        for sess in sess2group:
            for grp in sess2group[sess]:
                pattern = os.path.join(
                    self.data_root,
                    sess,
                    f'*{grp}*.json'
                )  # TODO
                if len(glob(pattern)) == 0:
                    missing_groups.add((sess, grp))

        all_jsons = glob(f'{self.data_root}/*/*.json')  # TODO

        def extract_group_tag(filename: str) -> str:
            match = re.search(r'-(group_\d+)-', filename)
            if not match:
                raise ValueError(f'Missing group tag in {filename}')
            return match.group(1)

        for file_name in all_jsons:
            if file_name.endswith('.json'):
                with open(file_name, encoding='utf-8') as f:
                    data = json.load(f)
                    group_data = data.get('instances', [])
                    if len(group_data) == 0:
                        sess = file_name.split('/')[-2]  # TODO
                        grp = extract_group_tag(os.path.basename(file_name))
                        missing_groups.add((sess, grp))

        filtered_data = []
        print('before (empty/incomplete annotations) filtering', len(train_data))
        for item in train_data:
            if (item['scene_name'], item['group']) not in missing_groups:
                filtered_data.append(item)
        print('after (empty/incomplete annotations) filtering', len(filtered_data))
        return filtered_data

    def load_matching(self, matching_file):
        with open(matching_file, 'rb') as pf:
            data = pickle.load(pf)
        total_samples = 0
        for scene_name, info in data.items():
            total_samples += len(info['sample_ids'])
        assert total_samples == len(self.samples), 'Matching info not matched with data samples'
        self.matching_meta = data
        print(f'loaded matching meta for {len(data)} scenes')

    def _parse_ann(self, ann_path):
        ann = load(ann_path)
        label2geoms = {self.cat2id[k]: [] for k in self.cat2id}
        for inst in ann.get('instances', []):
            cat = inst.get('category')
            if cat not in self.cat2id:
                continue
            label = self.cat2id[cat]
            for shape in inst.get('shapes', []):
                shape_type = shape.get('type')
                data = shape.get('shapeData', {})
                verts = data.get('vertexes', [])
                if len(verts) == 0 and 'position' in data:
                    verts = data['vertexes'] if 'vertexes' in data else [data['position']]
                pts = []
                for v in verts:
                    if isinstance(v, dict):
                        pts.append(v.get('position', v))
                    else:
                        pts.append(v)
                pts = np.array(pts)
                if len(pts) == 0:
                    continue
                pts2d = pts[:, :2]
                if shape_type in ['POLYLINE3', 'CENTER_LINE3', 'CURVE3']:
                    geom = LineString(pts2d)
                else:
                    geom = Polygon(pts2d)
                label2geoms[label].append(geom)
        return label2geoms

    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        location = sample['scene_name']

        # lidar2ego = np.eye(4)
        # lidar2ego[:3,:3] = Quaternion(sample['lidar2ego_rotation']).rotation_matrix
        # lidar2ego[:3, 3] = sample['lidar2ego_translation']

        # ego2global = np.eye(4)
        # ego2global[:3,:3] = Quaternion(sample['e2g_rotation']).rotation_matrix
        # ego2global[:3, 3] = sample['e2g_translation']

        # NOTE: The original StreamMapNet uses the ego location to query the map,
        # to align with the lidar-centered setting in MapTR, we made some modifiactions 
        # here to switch to the lidar-center setting
        #lidar2global = ego2global @ lidar2ego
        timestamp = sample['timestamp']
        lidar2global_translation = sample['lidar2g_translation']
        lidar2global_rotation = sample['lidar2g_rotation']
        group_name= sample['group'] if 'group' in sample else None

        map_geoms,map_classes = self.map_extractor.get_map_geom(
            location,
            lidar2global_translation,
            lidar2global_rotation,
            group_name=group_name,
            token=sample.get('token')
        ) # transforms the vectors to lidar frame with caching
        self.renderer.map_classes = map_classes

        # lidar2global = np.eye(4)
        # lidar2global[:3,:3] = Quaternion(e2g_rotation).rotation_matrix
        # lidar2global[:3, 3] = lidar_shifted_e2g_translation
        # global2lidar = np.linalg.inv(lidar2global)
        
        # ego2lidar = global2lidar  @ ego2global

        extrinsics = []  # lidar2cam 4x4
        intrinsics = []  # 3x3 K
        distortions = []
        ego2imgs = []    # projection 4x4 (here: lidar2img)
        for c in sample['cams'].values():
            cam2lidar = c['extrinsics']
            intrinsic = c['intrinsics']

            lidar2cam_rt, K, dist = build_camera_matrices(cam2lidar, intrinsic)
            extrinsics.append(lidar2cam_rt)
            intrinsics.append(K)
            distortions.append(dist)

            viewpad = np.eye(4, dtype=float)
            viewpad[:3, :3] = K
            ego2img_rt = viewpad @ lidar2cam_rt  # naming kept as 'ego2img' to satisfy pipeline
            ego2imgs.append(ego2img_rt)

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v
        # Ensure dense class channels [0..num_classes-1] for downstream rasterization
        num_classes = max(self.cat2id.values()) + 1 if len(self.cat2id) > 0 else 0
        for lbl in range(num_classes):
            map_label2geom.setdefault(lbl, [])
        
        # ego2img_rts = []
        # ego2cam_rts = []
        # for c in sample['cams'].values():
        #     extrinsic, intrinsic = np.array(
        #         c['extrinsics']), np.array(c['intrinsics'])

        #     # ego coord to cam coord
        #     #ego2cam_rt = extrinsic

        #     cam2ego_rt = np.linalg.inv(extrinsic)
        #     cam2lidar_rt = ego2lidar @ cam2ego_rt
        #     lidar2cam_rt = np.linalg.inv(cam2lidar_rt)
        #     ego2cam_rt = lidar2cam_rt

        #     viewpad = np.eye(4)
        #     viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

        #     ego2img_rt = (viewpad @ ego2cam_rt)
        #     ego2cam_rts.append(ego2cam_rt)
        #     ego2img_rts.append(ego2img_rt)
        if self.aerial_crop_root != None:
            aerial_img_path = os.path.join(self.aerial_crop_root, sample['aerial_crop_path']) if sample['aerial_crop_path'] != None else None
            aerial_img_path = aerial_img_path.replace('.png', '.jpg') if aerial_img_path else None

        # Resolve lidar pcd path against lidar_data_root so the pipeline can open it.
        lidar_filepath = None
        if self.lidar_data_root is not None:
            rel = sample.get('lidar_filepath')
            if rel is not None and not os.path.isabs(rel):
                lidar_filepath = os.path.join(self.lidar_data_root, rel)
            else:
                lidar_filepath = rel

        # Resolve camera image filepaths against the mounted data server or local root.
        img_filenames = []
        for c in sample['cams'].values():
            p = c['img_fpath']
            img_filenames.append(self._resolve_image_path(p))

        input_dict = {
            'location': location,
            'token': sample['token'],
            'timestamp': timestamp,
            'img_filenames': img_filenames,
            # intrinsics are 3x3 Ks
            'cam_intrinsics': intrinsics,
            # extrinsics are 4x4 tranform matrix, **ego2cam**
            'cam_extrinsics': extrinsics,
            'cam_distortion_coeffs': distortions,
            'ego2img': ego2imgs,
            # 'ego2img': ego2img_rts,
            # 'ego2cam': ego2cam_rts,
            'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            'ego2global_translation': sample['lidar2g_translation'], 
            'ego2global_rotation': R.from_quat(sample['lidar2g_rotation']).as_matrix().tolist(),
            'sample_idx': sample['sample_idx'],
            'scene_name': sample['scene_name'],
            'lidar2ego_translation': sample['lidar2ego_translation'],
            'lidar2ego_rotation': sample['lidar2ego_rotation'],
            'lat_long_heading': sample['lat_long_heading'], # [latitude, longitude, heading(in degrees)]
            'map_classes': map_classes,
            'frame_number': sample['frame_number'],
            'group': group_name,
            'aerial_image_path': aerial_img_path if self.aerial_crop_root != None else None,
            'lidar_filepath': lidar_filepath,
        }

        return input_dict

    def _resolve_image_path(self, path):
        if not isinstance(path, str):
            return path
        candidate = path
        if self.img_data_root and not os.path.isabs(candidate):
            candidate = os.path.normpath(os.path.join(self.img_data_root, candidate))
        return candidate
    
    def render_vector_projection_on_camera(self, idx,out_dir=None):
        '''Render vectorized map elements on camera views.
        
        Args:
            sample (dict): sample dict containing vectors, images, extrinsics, intrinsics.
        '''
        sample = self.get_sample(idx)
        sample = deepcopy(sample)
        data = self.pipeline(sample)
        vectors = data['vectors'].data
        extrinsics = sample['cam_extrinsics']
        intrinsics = sample['cam_intrinsics']
        distortions = sample['cam_distortion_coeffs']
        time = sample['timestamp']
        scene_name = sample['scene_name']
        group_name = sample['group'] if 'group' in sample else None
        imgs=[]
        for i ,cam_loc in enumerate(sample['img_filenames']):

            image_filename = self._resolve_image_path(cam_loc)
            img = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f'Failed to load image: {image_filename}')
            
            imgs.append(img)
            cam_name = os.path.basename(cam_loc)
            #extrinsics[idx],intrinsics[idx] = build_camera_matrices(extrinsics[idx], intrinsics[idx])
        ego2cams = sample['cam_extrinsics']

        out_dir = os.path.join(out_dir, 'camera_views')
        os.makedirs(out_dir, exist_ok=True)
        

        self.renderer.render_camera_views_from_vectors_with_attributes(vectors, imgs, extrinsics, 
                intrinsics, ego2cams, 20, out_dir,distortions=distortions,
                idx=idx,time=time, scene_name=scene_name,group_name=group_name,frame_number=sample['frame_number'] if 'frame_number' in sample else None)
        
    def show_gt_on_satellite_img(self, idx, out_dir='demo/'):
        '''Visualize ground-truth.

        Args:
            idx (int): index of sample.
            out_dir (str): output directory.
        '''

        from plugin.utils.data_container import DataContainer
        from copy import deepcopy
        sample = self.get_sample(idx)
        sample = deepcopy(sample)
        data = self.pipeline(sample)

        #imgs = [mmcv.imread(i) for i in sample['img_filenames']]
        #cam_extrinsics = sample['cam_extrinsics']
        #cam_intrinsics = sample['cam_intrinsics']

        if 'vectors' in data:
            vectors = data['vectors']
            if isinstance(vectors, DataContainer):
                vectors = vectors.data

            #self.renderer.render_bev_from_vectors(vectors, out_dir)
            specified_path = os.path.join(out_dir, f'map_{idx}.jpg')
            self.renderer.render_bev_from_vectors(vectors, out_dir,specified_path=specified_path,    
                                                  arcgis_service_url="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export",
                                                center_lon=sample['lat_long_heading'][1], center_lat=sample['lat_long_heading'][0], # latitude
                                                rotation_cw_deg=sample['lat_long_heading'][2]-90,     # clockwise degrees
                                                ppm=10,                  # controls image sharpness
                                                arcgis_print_url=True,
                                                frame_number = sample['frame_number'],)

            #self.renderer.render_camera_views_from_vectors(vectors, imgs, 
            #    cam_extrinsics, cam_intrinsics, 2, out_dir)

        if 'semantic_mask' in data:
            semantic_mask = data['semantic_mask']
            if isinstance(semantic_mask, DataContainer):
                semantic_mask = semantic_mask.data
            
            self.renderer.render_bev_from_mask(semantic_mask, out_dir, flip=True)

    def _resolve_aerial_crop_path(
        self,
        scene_name: str,
        frame_number: int = None,
        token: str = None,
        cam_5_seq_num: int = None,
    ) -> str:
        """Find the ego-aligned aerial crop corresponding to a sample."""
        # New aerial crops follow {cam_5_seq_num}_{scene_name}_{county}.png under provider dirs.
        if cam_5_seq_num is None:
            return None

        root = self.aerial_crop_root
        if not root or not os.path.isdir(root):
            return None

        candidate_dirs = []
        scene_name= scene_name.split('+')[0]
        direct_scene = os.path.join(root, scene_name)
        if os.path.isdir(direct_scene):
            candidate_dirs.append(direct_scene)

        provider_priority = ['san_jose', 'san_mateo', 'san_francisco']
        priority_rank = {name: idx for idx, name in enumerate(provider_priority)}
        try:
            providers = os.listdir(root)
            providers.sort(key=lambda name: (priority_rank.get(name, len(priority_rank)), name))
        except OSError:
            providers = []

        for provider in providers:
            provider_scene = os.path.join(root, provider, scene_name)
            if os.path.isdir(provider_scene):
                candidate_dirs.append(provider_scene)

        if not candidate_dirs:
            return None

        seq_candidates = []
        try:
            seq_int = int(cam_5_seq_num)
            seq_candidates.extend([str(seq_int), f"{seq_int:06d}"])
        except (TypeError, ValueError):
            seq_candidates.append(str(cam_5_seq_num))
        seq_candidates = list(dict.fromkeys(seq_candidates))

        for scene_dir in candidate_dirs:
            for seq in seq_candidates:
                pattern = f"{seq}_{scene_name}_*.png"
                matches = sorted(glob(os.path.join(scene_dir, pattern)))
                if matches:
                    return matches[0]

        return None

    def show_gt_on_aligned_aerial(self, idx, out_dir='demo/'):
        """Overlay vector ground truth on pre-rendered ego-aligned aerial imagery."""
        try:
            from plugin.utils.data_container import DataContainer
        except ImportError:
            DataContainer = None

        sample = deepcopy(self.get_sample(idx))
        data = self.pipeline(sample)

        vectors = data.get('vectors')
        if DataContainer and isinstance(vectors, DataContainer):
            vectors = vectors.data
        cam_location = sample.get('img_filenames', [None])[4]
        cam_5_seq_num = int(os.path.splitext(os.path.basename(cam_location))[0])
        if vectors is None:
            return

        crop_path = self._resolve_aerial_crop_path(
            scene_name=sample['scene_name'],
            frame_number=sample.get('frame_number'),
            token=sample.get('token'),
            cam_5_seq_num=cam_5_seq_num,
        )
        if not crop_path:
            raise FileNotFoundError(f"Could not locate aerial crop for scene {sample['scene_name']} "
                                    f"frame {sample.get('frame_number')} token {sample.get('token')}")

        os.makedirs(out_dir, exist_ok=True)
        specified_path = os.path.join(out_dir, f"aerial_gt_{cam_5_seq_num}_{sample['scene_name']}.jpg")
        self.renderer.render_bev_from_vectors(
            vectors=vectors,
            out_dir=out_dir,
            specified_path=specified_path,
            background=crop_path,
            frame_number=sample.get('frame_number'),
        )


def build_camera_matrices(extrinsics: dict, intrinsics: dict):
    """
    Build both the 4×4 extrinsic (world‐to‐camera) transform and the 3×3 intrinsic matrix.

    Parameters
    ----------
    extrinsics : dict
        {
          'translation': [tx, ty, tz],
          'quaternion': [w, x, y, z]
        }
    intrinsics : dict
        {
          'fx': ...,
          'fy': ...,
          'cx': ...,
          'cy': ...,
        }

    Returns
    -------
    K : ndarray, shape (3, 3)
        Camera intrinsic matrix.
    T : ndarray, shape (4, 4)
        Homogeneous transform from world to camera.
    """
    # Unpack extrinsic params
    t = np.asarray(extrinsics['translation'], dtype=float)
    qx, qy, qz, qw = extrinsics['quaternion']
    # SciPy expects [x, y, z, w]
    R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # Build 4×4 transform
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_mat
    T[:3, 3] = t

    T=np.linalg.inv(T)  # Invert to get lidar2camera transform

    # Unpack intrinsics and build 3×3 K
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    dist= intrinsics['distortion']
    dist_coeffs = np.array([dist['k1'], dist['k2'], dist['p1'], dist['p2'], dist['k3']], dtype=np.float64)
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=float)

    return T,K,dist_coeffs

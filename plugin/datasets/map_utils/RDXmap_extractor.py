import os
import json
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from shapely.geometry import LineString, Polygon, box
from shapely import affinity
from scipy.spatial.transform import Rotation as R
from shapely.ops import transform as shp_transform
from collections import OrderedDict
from torch.utils.data import get_worker_info

from plugin.datasets.map_utils.rdx_schema import (
    BASE_CLASSES,
    ATTRIBUTE_SCHEMAS,
    normalize_category_and_attributes,
)


class RDXMapExtractor:
    """
    Generate ground-truth map geometries for the RDX dataset.

    Args:
        data_root (str): Path to directory containing annotation JSON files.
        roi_size (tuple or list): BEV range (height, width).
    """
    def __init__(self, data_root: str, roi_size: Union[List, Tuple]) -> None:
        self.data_root = data_root
        self.roi_size = tuple(roi_size)
        # Define local patch in ego coordinates
        self.local_patch = box(-self.roi_size[0]/2,
                               -self.roi_size[1]/2,
                                self.roi_size[0]/2,
                                self.roi_size[1]/2)
        # In-memory caches
        # (location, group_name) -> { 'instances': [...], 'map_classes': {...} }
        self._instance_cache: dict = {}
        # token -> per-token vectorized and clipped geometries in ego frame
        # Use a small LRU to bound memory
        self._vector_cache: OrderedDict = OrderedDict()
        #self._vector_cache_max = 4096*4
        self._wandb = None
        self._wandb_disabled = False
        self._wandb_log_pid: Optional[int] = None

    def get_map_geom(self,
                     location: str, # This will be ignored or used as subdirectory if needed
                     e2g_translation: np.ndarray,
                     e2g_rotation: np.ndarray,
                     map_classes: Dict[str, List[str]] = None,
                     group_name: str = None,
                     token: str = None
                     ) -> Dict[str, List[Union[LineString, Polygon]]]:
        """
        Load all annotation JSONs in the directory and return geometries
        in ego (local) coordinates.

        Args:
            location: Identifier used for path, can be directory name.
            e2g_translation: (3,) translation vector from ego to global.
            e2g_rotation: (4,) quaternion [x, y, z, w] from ego to global.
            map_classes: Dict of categories and attributes (defaults to all seen).

        Returns:
            A dict mapping each category to a list of Shapely geometries.
        """
        # 0) Load and cache raw instances for this (location, group)
        cache_key = (location, group_name)
        if cache_key not in self._instance_cache:
            instances = []
            dir_path = os.path.join(self.data_root, location.split('+')[0])
            if group_name is not None:
                pattern = f"-{group_name}-"
                for filename in os.listdir(dir_path):
                    if filename.endswith('.json') and pattern in filename:
                        json_path = os.path.join(dir_path, filename)
                        with open(json_path, 'r') as f:
                            data = json.load(f, encoding='utf-8')
                            instances.extend(data.get('instances', []))
            else:
                for file_name in os.listdir(dir_path):
                    if file_name.endswith('.json'):
                        json_path = os.path.join(dir_path, file_name)
                        with open(json_path, 'r') as f:
                            data = json.load(f, encoding='utf-8')
                            instances.extend(data.get('instances', []))

            # Build map_classes once for this scene/group if not provided
            if map_classes is None:
                map_classes = {cls: {'attributes': []} for cls in BASE_CLASSES}
                for attr_name, schema in ATTRIBUTE_SCHEMAS.items():
                    for cls in schema['applies_to']:  # type: ignore[index]
                        if cls in map_classes:
                            map_classes[cls]['attributes'].append(attr_name)

            self._instance_cache[cache_key] = {
                'instances': instances,
                'map_classes': map_classes
            }
        else:
            instances = self._instance_cache[cache_key]['instances']
            if map_classes is None:
                map_classes = self._instance_cache[cache_key]['map_classes']

        # 1) Vectorization cache per token (optional but fast-path)
        if token is not None and token in self._vector_cache:
            # move to end to mark as recently used
            geoms_cached, classes_cached = self._vector_cache.pop(token)
            self._vector_cache[token] = (geoms_cached, classes_cached)
            self._log_vector_cache_size()
            return geoms_cached, classes_cached

        # 2) Vectorize on the fly for this pose
        vector_map = VectorizedLocalMap(instances, self.roi_size, map_classes)
        geoms = vector_map.gen_vectorized_samples(e2g_translation, e2g_rotation)

        # 3) Insert into LRU cache
        if token is not None:
            self._vector_cache[token] = (geoms, map_classes)
            #if len(self._vector_cache) > self._vector_cache_max:
            #    self._vector_cache.popitem(last=False)
            self._log_vector_cache_size()

        return geoms, map_classes

    def _log_vector_cache_size(self) -> None:
        if self._wandb_disabled:
            return

        pid = os.getpid()
        if self._wandb_log_pid is None:
            self._wandb_log_pid = pid
        if pid != self._wandb_log_pid:
            return

        worker = get_worker_info()
        if worker is not None and worker.id != 0:
            return

        try:
            if self._wandb is None:
                import wandb
                if wandb.run is None:
                    return
                self._wandb = wandb
            self._wandb.log(
                {'train/vector_cache_size': len(self._vector_cache)},
                commit=False)
        except Exception:
            self._wandb_disabled = True

# ------------------------------------------------------------------------
# 1) New helper: behaves exactly like a List[geom], but holds attributes too
# ------------------------------------------------------------------------
class CategoryGeomList(list):
    def __init__(self,
                 geoms: List[Union[LineString, Polygon]],
                 attrs: List[Dict]):
        super().__init__(geoms)
        # parallel list of attribute-dicts
        self.attrs = attrs

    @property
    def attribute_keys(self) -> List[str]:
        """Return a sorted list of every attribute name seen for this category."""
        keys = set()
        for d in self.attrs:
            keys.update(d.keys())
        return sorted(keys)


class VectorizedLocalMap:
    """
    Transform annotation instances into Shapely geometries in the local frame.
    """
    def __init__(self,
                 instances: List[dict],
                 roi_size: Tuple[float, float],
                 map_classes: Dict[str, List[str]]
                 ):
        self.instances = instances
        self.roi_size = roi_size
        self.map_classes = map_classes
        # Local patch for clipping
        self.local_patch = box(-roi_size[0]/2,
                               -roi_size[1]/2,
                                roi_size[0]/2,
                                roi_size[1]/2)

    
    def gen_vectorized_samples(self,
                               e2g_translation: np.ndarray,
                               e2g_rotation: np.ndarray
                               ) -> Dict[str, List[Union[LineString, Polygon]]]:
        """
        Args:
            e2g_translation: (3,) translation vector from ego to global.
            e2g_rotation: (4,) quaternion [x,y,z,w] from ego to global.
        Returns:
            map_dict: {category: [geom,...]}
        """
        # Build rotation matrix: global -> ego
        rot3 = R.from_quat(e2g_rotation).as_matrix()   # 3×3 ego→global
        R_g2e = rot3.T                                 # global→ego
        t_g2e = -R_g2e.dot(e2g_translation)            # so that R_g2e @ p + t_g2e = p_local
        
        def project_3d(x, y, z=None):
            # 1) make sure z is a list of the right length
            if z is None:
                z = [0.0] * len(x)

            # 2) build a 3×N array from your coords
            coords = np.vstack((x, y, z))       # shape = (3, N)
            v = R_g2e.dot(coords)               # shape = (3, N)

            # 3) add each component of t_g2e to the corresponding row
            vx = v[0, :] + t_g2e[0]
            vy = v[1, :] + t_g2e[1]
            vz = v[2, :] + t_g2e[2]

            return vx, vy, vz

        # Initialize output
        # 2) Instead of one dict of lists, keep two parallel ones
        class_names = list(self.map_classes.keys())
        geom_buffers: Dict[str, List] = {cls: [] for cls in class_names}
        attr_buffers: Dict[str, List] = {cls: [] for cls in class_names}

        # Process each instance
        for inst in self.instances:
            cat = inst.get('category')
            for shape in inst.get('shapes', []):
                geom = self._shape_to_geom(shape)
                if geom is None:
                    continue
                attributes = shape.get('attributes', {})
                base_class, normalized_attrs = normalize_category_and_attributes(cat, attributes)
                if base_class is None or base_class not in geom_buffers:
                    continue
                # Transform to ego coords
                geom_local = shp_transform(project_3d, geom)
                if not geom_local.is_valid:
                    geom_local = geom_local.buffer(0)
                # Clip to patch
                geom_local = geom_local.intersection(self.local_patch)
                if geom_local.is_empty:
                    continue
                # Collect
                if isinstance(geom_local, (LineString, Polygon)):
                    geom_buffers[base_class].append(geom_local)
                    attr_buffers[base_class].append(normalized_attrs)
                else:
                    # multi-part geometry: flatten
                    for part in geom_local:
                        geom_buffers[base_class].append(part)
                        attr_buffers[base_class].append(normalized_attrs)
        # ----------------------------------------------------------------
        #  Wrap each list in our CategoryGeomList and return
        # ----------------------------------------------------------------
        map_dict: Dict[str, CategoryGeomList] = {}
        for cls in list(self.map_classes.keys()):
            geoms = geom_buffers[cls]
            attrs = attr_buffers[cls]
            map_dict[cls] = CategoryGeomList(geoms, attrs)

        return map_dict

    def _shape_to_geom(self, shape: dict) -> Union[LineString, Polygon, None]:
        """
        Convert a single annotation shape into a Shapely geometry.
        """
        typ = shape.get('type', '')
        data = shape.get('shapeData', {})
        # Extract XY points
        if 'vertexes' in data:
            pts = []
            for v in data['vertexes']:
                # Some formats use dicts, others lists
                pos = v.get('position') if isinstance(v, dict) else v
                if len(pos) >= 2:
                    pts.append((pos[0], pos[1],pos[2]))
            if len(pts) < 2:
                return None
        else:
            return None

        # Map types
        if typ in ['POLYLINE3', 'CENTER_LINE3', 'CURVE3']:
            return LineString(pts)
        elif typ in ['POLYGON3', 'RECTANGLE3']:
            # Ensure closed ring
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            return Polygon(pts)
        else:
            return None

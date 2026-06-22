import numpy as np
from plugin.utils.data_container import DataContainer as DC

try:  # pragma: no cover - optional dependency in mmdet3d>=1.4
    from mmdet3d.structures.points import BasePoints
except Exception:  # pragma: no cover - fallback when unavailable
    class BasePoints:  # type: ignore[override]
        pass
from mmdet3d.registry import TRANSFORMS
from plugin.utils.tensor import to_tensor


@TRANSFORMS.register_module()
class FormatBundleMap(object):
    """Format data for map tasks and then collect data for model input.

    These fields are formatted as follows.

    - img: (1) transpose, (2) to tensor, (3) to DataContainer (stack=True)
    - semantic_mask (if exists): (1) to tensor, (2) to DataContainer (stack=True)
    - vectors (if exists): (1) to DataContainer (cpu_only=True)
    - img_metas: (1) to DataContainer (cpu_only=True)
    """

    def __init__(self, process_img=True, 
                keys=['img', 'semantic_mask', 'vectors'], 
                meta_keys=['intrinsics', 'extrinsics']):
        
        self.process_img = process_img
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            pts = results['points']
            if isinstance(pts, BasePoints):
                results['points'] = DC(pts.tensor)
            elif isinstance(pts, list):
                # list of per-sample arrays/tensors
                results['points'] = DC(pts, stack=False)
            else:
                # single array/tensor
                results['points'] = DC(to_tensor(pts), stack=False)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if 'img' in results and self.process_img:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=True)

        # Optional aerial image (single RGB image per sample)
        if 'aerial_img' in results and self.process_img:
            aer = results['aerial_img']
            if aer is None:
                results['aerial_img'] = DC(None, stack=False)
            elif isinstance(aer, list):
                imgs = [np.ascontiguousarray(img.transpose(2, 0, 1)) for img in aer]
                tensors = [to_tensor(img) for img in imgs]
                results['aerial_img'] = DC(tensors, stack=False)
            else:
                img = np.ascontiguousarray(aer.transpose(2, 0, 1))
                results['aerial_img'] = DC(to_tensor(img), stack=False)
        
        if 'semantic_mask' in results:
            #results['semantic_mask'] = DC(to_tensor(results['semantic_mask']), stack=True)
            if isinstance(results['semantic_mask'], np.ndarray):
                results['semantic_mask'] = DC(to_tensor(results['semantic_mask']), stack=True,
                                              pad_dims=None)
            else:
                assert isinstance(results['semantic_mask'], list)
                results['semantic_mask'] = DC(results['semantic_mask'], stack=False)

        if 'vectors' in results:
            # vectors may have different sizes
            vectors = results['vectors']
            results['vectors'] = DC(vectors, stack=False, cpu_only=True)
        
        if 'polys' in results:
            results['polys'] = DC(results['polys'], stack=False, cpu_only=True)
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(process_img={self.process_img}, '
        return repr_str


@TRANSFORMS.register_module()
class Collect3D(object):
    """Collect data from the loader relevant to the keys.

    Args:
        keys (Sequence[str]): Keys of results to be collected.
        meta_keys (Sequence[str]): Meta keys to be collected into
            ``img_metas``. Defaults to ``('img_shape', 'ori_shape')`` but
            typically overridden via config.
    """

    def __init__(self,
                 keys,
                 meta_keys=('img_shape', 'ori_shape')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        for key in self.keys:
            if key not in results:
                raise KeyError(f'Key {key} not found in results')
            data[key] = results[key]

        meta = {key: results[key] for key in self.meta_keys if key in results}
        data['img_metas'] = DC(meta, cpu_only=True)
        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}(keys={self.keys}, '
                f'meta_keys={self.meta_keys})')

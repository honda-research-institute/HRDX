import os
import warnings

import cv2
import mmcv
import numpy as np
from mmdet3d.registry import TRANSFORMS

try:
    from mmdet.datasets.builder import PIPELINES
except ImportError:  # mmengine-style builds without legacy PIPELINES registry
    PIPELINES = None


@TRANSFORMS.register_module(force=True)
class LoadMultiViewImagesFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filenames']

        def _read_image(path: str):
            lower = path.lower()
            img = None
            if lower.endswith('.bin'):
                # Attempt to decode raw Bayer data in RDX format (1860x2880 RGGB)
                expected = 1860 * 2880
                if not os.path.exists(path):
                    raise FileNotFoundError(f'Image not found: {path}')
                try:
                    fsize = os.path.getsize(path)
                    if fsize == expected:
                        with open(path, 'rb') as f:
                            buf = f.read(expected)
                        bayer = np.frombuffer(buf, dtype=np.uint8).reshape((1860, 2880))
                        img = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)
                    else:
                        warnings.warn(
                            f'Unexpected .bin size {fsize} (expected {expected}) for {path}; '
                            'falling back to mmcv.imread', RuntimeWarning)
                except Exception:
                    # Defer to mmcv.imread fallback below
                    img = None

            if img is None:
                img = mmcv.imread(path, self.color_type)
            if img is None:
                raise FileNotFoundError(f'Failed to load image: {path}')
            return img

        img = [_read_image(name) for name in filename]
        if self.to_float32:
            img = [i.astype(np.float32) for i in img]
        results['img'] = img
        results['img_shape'] = [i.shape for i in img]
        results['ori_shape'] = [i.shape for i in img]
        # Set initial values for default meta_keys
        results['pad_shape'] = [i.shape for i in img]
        # results['scale_factor'] = 1.0
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__} (to_float32={self.to_float32}, '\
            f"color_type='{self.color_type}')"


@TRANSFORMS.register_module(force=True)
class LoadAerialImageFromFile(object):
    """Load a single aerial (satellite) image given a resolved filepath.

    Expects `results['aerial_image_path']` to be a string path to the
    lidar/ego-referenced satellite crop produced by AID4AD tools.

    Args:
        to_float32 (bool): Whether to convert the image to float32. Default: False.
        color_type (str): Color type for mmcv.imread. Default: 'unchanged'.
        normalize (bool): Whether to apply ImageNet normalization (scale to
            [0, 1] then subtract mean / divide by std).  Required when the
            downstream encoder is an ImageNet-pretrained ResNet. Default: True.
        img_norm_cfg (dict | None): Custom mean/std for normalization.
            Defaults to ImageNet values when *normalize* is True.
    """

    # Standard ImageNet statistics (RGB order)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, to_float32=False, color_type='unchanged',
                 normalize=True, img_norm_cfg=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.normalize = normalize
        if img_norm_cfg is not None:
            self.mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
            self.std = np.array(img_norm_cfg['std'], dtype=np.float32)
        else:
            self.mean = self.IMAGENET_MEAN
            self.std = self.IMAGENET_STD

    def __call__(self, results):
        # The dataset should provide the resolved path keyed by the sample token
        filename = results.get('aerial_image_path', None)
        if filename is None:
            warnings.warn('Aerial image path missing; dropping satellite modality for this sample.',
                          RuntimeWarning)
            results['aerial_img'] = None
            results['aerial_img_missing'] = True
            return results

        if not os.path.exists(filename):
            warnings.warn(f'Aerial image not found at {filename}; dropping satellite modality.',
                          RuntimeWarning)
            results['aerial_img'] = None
            results['aerial_img_missing'] = True
            return results

        try:
            img = mmcv.imread(filename, self.color_type)
        except Exception as exc:
            raise RuntimeError(
                f"LoadAerialImageFromFile: failed to read {filename}"
            ) from exc
        if img is None:
            raise FileNotFoundError(
                f"LoadAerialImageFromFile: mmcv.imread returned None for {filename}"
            )
        if self.to_float32:
            img = img.astype(np.float32)

        if self.normalize:
            # mmcv.imread returns BGR by default; convert to RGB for ImageNet stats
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = img[:, :, ::-1].copy()  # BGR -> RGB
            # Scale from [0, 255] to [0, 1], then normalize
            img = img / 255.0
            img = (img - self.mean) / self.std

        # Store under a dedicated field; FormatBundleMap will move to Tensor + DC
        results['aerial_img'] = img
        results['aerial_img_shape'] = img.shape
        results['aerial_img_missing'] = False
        results['aerial_ori_shape'] = img.shape
        results['aerial_img_fields'] = ['aerial_img']
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(to_float32={self.to_float32}, "
                f"color_type='{self.color_type}', normalize={self.normalize})")


if PIPELINES is not None:
    PIPELINES.register_module(force=True)(LoadMultiViewImagesFromFiles)
    PIPELINES.register_module(force=True)(LoadAerialImageFromFile)

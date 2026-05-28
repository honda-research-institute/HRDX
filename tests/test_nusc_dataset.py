import os
import os.path as osp
import numpy as np

from mmcv import Config
from mmdet3d.datasets import build_dataset
import matplotlib.pyplot as plt
import cv2
from tools.tracking.cmap_utils.match_utils import *

# Define config
cfg = Config.fromfile('plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage1_bev_pretrain.py')
import_plugin(cfg)


# Build dataset

dataset = build_dataset(cfg.match_config)


# Visualize all samples in the dataset
for idx in range(len(dataset)):
    sample = dataset.get_sample(idx)
    data = dataset.pipeline(sample)

    img_filenames = sample['img_filenames']
    out_dir = './datasets/nuscenes/vis_aerial_overlay'

    # Show ground truth
    dataset.show_gt(idx, out_dir=out_dir)

    # --- Visualize LiDAR projections ---
    #points = data['points'].data  # Nx(3 or 4), in lidar frame

    dataset.renderer.render_aerial_with_vectors(data['vectors'].data, data['aerial_img'].data, out_dir,idx)

    # Project LiDAR onto each camera view
    # dataset.renderer.render_lidar_on_cameras(
    #     points=points,
    #     imgs_or_paths=img_filenames,
    #     ego2cam_list=sample['ego2cam'],
    #     lidar2ego_translation=sample.get('lidar2ego_translation', None),
    #     intrinsics_list=sample['cam_intrinsics'],
    #     out_dir=out_dir,
    #     idx=sample['sample_idx'],
    #     point_size=2,
    # )

    # Render LiDAR BEV raster within ROI
    # dataset.renderer.render_lidar_bev(
    #     points=points,
    #     out_dir=out_dir,
    #     idx=sample['sample_idx'],
    #     px_per_meter=10,
    #     vectors=data['vectors'].data,
    #     #lidar2ego_rotation=sample.get('lidar2ego_rotation', None),
    # )
     
# Check if method exists
# if hasattr(dataset, 'render_vector_projection_on_camera'):
#     print("Rendering vector projection on camera view...")
#     vis = dataset.render_vector_projection_on_camera(data)

#     if isinstance(vis, str):
#         # If the method returns an image path
#         img = mmcv.imread(vis)
#         plt.imshow(mmcv.bgr2rgb(img))
#     else:
#         # If it returns an image array directly
#         plt.imshow(vis)

#     plt.axis('off')
#     plt.title('Vector Projection on Camera')
#     plt.show()
# else:
#     print("The dataset does not support render_vector_projection_on_camera()")

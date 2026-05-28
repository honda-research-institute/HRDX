from mmengine.config import read_base

with read_base():
    from ...maptracker.rdx_dataset.maptracker_rdx_stage1_bev_pretrain import *

# Teacher/student settings ----------------------------------------------------
model = dict(
    _delete_=True,
    type='MapTrackerDistiller',
    student_cfg='plugin/configs/maptracker/rdx_dataset/maptracker_rdx_stage1_bev_pretrain.py',
    student_pretrained='work_dirs/teacher_checkpoints/rdx_stage_1.pth',
    teacher_cfgs=[
        dict(
            config='plugin/configs/maptracker_aerial_fuse/rdx_dataset/'
                   'maptracker_rdx_stage1_bev_pretrain_aerial_fuse.py',
            checkpoint='work_dirs/teacher_checkpoints/rdx_stage_1.pth',
            weight=1.0,
            name='aerial_teacher',
        ),
    ],
    distill_weight=1,
    bev_kd_weight=5,
    seg_kd_weight=0,
    head_kd_weight=0.0,
    head_attr_weight=0.0,
)

# Pipelines (add aerial imagery) ---------------------------------------------
train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        roi_size=roi_size,
        sample_num=num_points,
        normalize=True,
        permute=permute,
    ),
    dict(
        type='RasterizeMap',
        roi_size=roi_size,
        coords_dim=coords_dim,
        canvas_size=canvas_size,
        thickness=thickness,
        semantic_mask=True,
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(
        type='CropTopMultiViewImages',
        crop_ratio=img_top_crop_ratio,
        change_intrinsics=True,
    ),
    dict(type='LoadAerialImageFromFile', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='ResizeMultiViewImages',
        size=img_size,
        change_intrinsics=True,
    ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(
        type='Collect3D',
        keys=['img', 'vectors', 'semantic_mask', 'aerial_img'],
        meta_keys=(
            'token', 'ego2img', 'sample_idx', 'ego2global_translation',
            'ego2global_rotation', 'img_shape', 'scene_name'),
    ),
]

test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(
        type='CropTopMultiViewImages',
        crop_ratio=img_top_crop_ratio,
        change_intrinsics=True,
    ),
    dict(type='LoadAerialImageFromFile', to_float32=True),
    dict(
        type='ResizeMultiViewImages',
        size=img_size,
        change_intrinsics=True,
    ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(
        type='Collect3D',
        keys=['img', 'aerial_img'],
        meta_keys=(
            'token', 'ego2img', 'sample_idx', 'ego2global_translation',
            'ego2global_rotation', 'img_shape', 'scene_name'),
    ),
]

# Dataset definitions --------------------------------------------------------
_aerial_root = './datasets/RDX/aerial_ego_aligned'

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        type='RDXDataset',
        data_root='./datasets/RDX/HRDX_annotations',
        img_data_root=img_data_root,
        aerial_crop_root=_aerial_root,
        ann_file='./datasets/RDX/train_1_every_5m.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=-2,
        matching=True,
        multi_frame=5,
        sampling_span=10,
    ),
    val=dict(
        type='RDXDataset',
        data_root='./datasets/RDX/HRDX_annotations',
        img_data_root=img_data_root,
        aerial_crop_root=_aerial_root,
        ann_file='./datasets/RDX/val_1_every_5m_gt_tracks.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
        eval_semantic=True,
    ),
    test=dict(
        type='RDXDataset',
        data_root='./datasets/RDX/HRDX_annotations',
        img_data_root=img_data_root,
        aerial_crop_root=_aerial_root,
        ann_file='./datasets/RDX/val_1_every_5m_gt_tracks.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=1,
        eval_semantic=True,
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

load_from = None

# Visualization / logging ----------------------------------------------------
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='StepAwareWandbVisBackend',
        init_kwargs=dict(
            project='maptracker-expts_RDX',
            name='maptracker_rdx_stage1_kd',
            tags=['stage_1', 'rdx', 'kd'],
        ),
        define_metric_cfg=[
            dict(name='train/*', step_metric='train/iter'),
            dict(name='val/*', step_metric='val/iter'),
            dict(name='train/iter', step_metric='train/iter'),
            dict(name='val/iter', step_metric='val/iter'),
            dict(name='iter', step_metric='train/iter'),
        ],
    ),
]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

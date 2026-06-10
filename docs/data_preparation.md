
# Data Preparation

Compared to the data preparation procedure of StreamMapNet or MapTR, we have one more step to generate the ground truth tracking information (Step 3).

We noticed that the track generation results can be slighly different when running on different machines (potentially because Shapely's behaviors are slightly different across different machines), **so please always run the Step 3 below on the training machine to generate the gt tracking information**.

## HRDX

Download the HRDX dataset from the official website:
[usa.honda-ri.com/hrdx](https://usa.honda-ri.com/hrdx).

HRDX ships as a refined release: per-sample pickles plus aligned
image / LiDAR / aerial roots. Place the bundle under `./datasets/RDX/`
so the shipped configs (which reference `./datasets/RDX/...` relative
paths) work without modification. The expected layout is:

```
datasets/RDX/
├── HRDX_annotations/                 # per-scene vector map JSONs
│   └── <scene_name>/*group_<N>*.json
├── 1by2.5x_jpegs/                    # 6 surround-view JPEGs (half-res)
│   └── <scene_name>/cam_{1..6}/<frame_id>.jpeg
├── aerial_ego_aligned/               # ego-aligned aerial crops by provider
│   └── {san_francisco,san_jose,san_mateo}/<scene_name>/{cam5_seq}_{scene}_{county}.png
├── pointclouds/                      # 128-beam LiDAR pcds
│   └── <scene_name>/<lidar_seq>.pcd
├── train_1_every_5m.pkl              # train samples
├── train_1_every_5m_gt_tracks.pkl
├── val_1_every_5m.pkl                # val samples
└── val_1_every_5m_gt_tracks.pkl
```

Each sample pickle is a list of per-frame dicts holding camera image
paths, calibration, ego/LiDAR poses, and modality-specific keys
(`aerial_crop_path` and `lidar_filepath`). The paired `_gt_tracks.pkl`
files carry tracking metadata for C-mAP evaluation and are auto-loaded
by `RDXDataset` when an eval config points at them.

The raw stage-01 sensor and annotation dumps are not distributed; only
the refined assets above. See the HRDX paper for the full
data-preparation methodology.

The legacy nuScenes / Argoverse2 instructions below remain useful for
reproducing the upstream MapTracker results.

## nuScenes
**Step 1.** Download [nuScenes](https://www.nuscenes.org/download) dataset to `./datasets/nuscenes`.


**Step 2.** Generate annotation files for NuScenes dataset (the same as StreamMapNet)

```
python tools/data_converter/nuscenes_converter.py --data-root ./datasets/nuscenes
```

Add ``--newsplit`` to generate the metadata for the new split (geographical-based split) provided by StreamMapNet.

**Step 3.** Generate the tracking ground truth by 

```
python tools/tracking/prepare_gt_tracks.py plugin/configs/maptracker/nuscenes_oldsplit/maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune.py  --out-dir tracking_gts/nuscenes --visualize
```

Add the ``--visualize`` flag to visualize the data with element IDs derived from our track generation process, or remove it to save disk memory.  

For generating the G.T. tracks of the new split, change the config file accordingly.


## Argoverse2

**Step 1.** Download [Argoverse2 (sensor)](https://argoverse.github.io/user-guide/getting_started.html#download-the-datasets) dataset to `./datasets/av2`.

**Step 2.** Generate annotation files for Argoverse2 dataset.

```
python tools/data_converter/argoverse_converter.py --data-root ./datasets/av2
```

**Step 3.** Generate the tracking ground truth by 

```
python tools/tracking/prepare_gt_tracks.py plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py  --out-dir tracking_gts/av2 --visualize
```


## Checkpoints

We provide the checkpoints at [this link](https://www.dropbox.com/scl/fo/miulg8q9oby7q2x5vemme/ALoxX1HyxGlfR9y3xlqfzeE?rlkey=i3rw4mbq7lacblc7xsnjkik1u&dl=0). Please download and place them as ``./work_dirs/pretrained_ckpts``.


## File structures

Make sure the final file structures look like below:

```
maptracker
├── mmdetection3d
├── tools
├── plugin
│   ├── configs
│   ├── models
│   ├── datasets
│   ├── ...
├── work_dirs
│   ├── pretrained_ckpts
│   │   ├── maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune
│   │   │   ├── latest.pth
│   │   ├── ...
│   ├── ....
├── datasets
│   ├── nuscenes
│   │   ├── maps <-- used
│   │   ├── samples <-- key frames
│   │   ├── v1.0-test <-- metadata
|   |   ├── v1.0-trainval <-- metadata and annotations
│   │   ├── nuscenes_map_infos_train_{newsplit}.pkl <-- train annotations
│   │   ├── nuscenes_map_infos_train_{newsplit}_gt_tracks.pkl <-- train gt tracks
│   │   ├── nuscenes_map_infos_val_{newsplit}.pkl <-- val annotations
│   │   ├── nuscenes_map_infos_val_{newsplit}_gt_trakcs.pkl <-- val gt tracks
│   ├── av2
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── maptrv2_val_samples_info.pkl <-- maptr's av2 metadata, used to align the val set (download from MapTR)
│   │   ├── av2_map_infos_train_{newsplit}.pkl <-- train annotations
│   │   ├── av2_map_infos_train_{newsplit}_gt_tracks.pkl <-- train gt tracks
│   │   ├── av2_map_infos_val_{newsplit}.pkl <-- val annotations
│   │   ├── av2_map_infos_val_{newsplit}_gt_trakcs.pkl <-- val gt tracks
│   ├── RDX                                                      <-- HRDX (refined release)
│   │   ├── HRDX_annotations           <-- per-scene vector map JSONs
│   │   ├── 1by2.5x_jpegs              <-- 6 surround-view JPEGs per scene
│   │   ├── aerial_ego_aligned         <-- ego-aligned aerial crops by provider
│   │   ├── pointclouds                <-- 128-beam LiDAR pcds per scene
│   │   ├── train_1_every_5m.pkl
│   │   ├── train_1_every_5m_gt_tracks.pkl
│   │   ├── val_1_every_5m.pkl
│   │   ├── val_1_every_5m_gt_tracks.pkl

```

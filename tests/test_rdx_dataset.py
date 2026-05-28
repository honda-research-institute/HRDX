import sys
import os
import cProfile
import pstats
import io
from tqdm import tqdm
from plugin.datasets.rdx_dataset import RDXDataset

def main():
    coords_dim = 3
    meta = dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False,
        output_format='vector'
    )
    roi_size = (60,30)

    cat2id = {
        'Parking spots': 0,
        'Lane center lines': 1,
        'Non-drivable areas': 2,
        'Bike lanes': 3,
        'Road boundary': 4,
        'Lane lines': 5,
        'Text on the road': 6,
        'Crosswalks': 7,
    }

    num_points = 20
    canvas_size = (200, 200)
    thickness = 1

    rdx_dataset = RDXDataset(
        data_root='./datasets/RDX/HRDX_annotations',
        img_data_root='./datasets/RDX/server_root',
        ann_file='./datasets/RDX/train_1_every_5m.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=[
            dict(
                type='VectorizeMap',
                coords_dim=coords_dim,
                simplify=False,
                sample_dist=0.1,
                normalize=False,
                roi_size=roi_size
            ),
            dict(type='FormatBundleMap'),
            dict(type='Collect3D', keys=['vectors'], meta_keys=[
                'token', 'ego2img', 'sample_idx', 'ego2global_translation',
                'ego2global_rotation', 'img_shape', 'scene_name'
            ])
        ],
        interval=1,
        test_mode=True
    )

    prev_time = 0
    for i, sample in tqdm(enumerate(rdx_dataset.samples), desc="Checking timestamps"):
        time = sample['timestamp']
        if prev_time > time:
            print("Time error: ", prev_time, time)
            print('Scene:', sample['scene_name'])
        prev_time = time

    print("Loaded samples:", len(rdx_dataset.samples))

    out_dir = './datasets/RDX/server_root/stage01/RDX_DATA/Intersections/sessions/2025_05_29_14_43_32/annotation_review/merged_delta_xy'
    os.makedirs(out_dir, exist_ok=True)

    scene_id = "2025_05_29_14_43_32"
    start_idx = 0

    scene_names=[]
    scene_indices = []
    for i in range(start_idx, len(rdx_dataset.samples)):
        if rdx_dataset.samples[i]['scene_name'] == scene_id:
            scene_indices.append(i)
        scene_names.append(rdx_dataset.samples[i]['scene_name'])
    scene_names = list(set(scene_names))
    gt_dir = os.path.join(out_dir, "gt_annotation")
    os.makedirs(gt_dir, exist_ok=True)
    aerial_dir = os.path.join(out_dir, "aerial_gt")
    os.makedirs(aerial_dir, exist_ok=True)

    for idx in tqdm(scene_indices[ 0:], desc=f"Rendering {scene_id}", total=len(scene_indices)):
        # rdx_dataset.show_gt(idx, gt_dir)
        #rdx_dataset.show_gt_on_satellite_img(idx, "./work_dirs/rdx_aerial_debug_sf")
        rdx_dataset.show_gt_on_aligned_aerial(idx, aerial_dir)
        #rdx_dataset.render_vector_projection_on_camera(idx, out_dir=out_dir)


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()

    main()

    # profiler.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    # ps.print_stats()
    # with open("rdx_dataset_profile.txt", "w") as f:
    #     f.write(s.getvalue())
    # print("Profiling complete. Output written to 'rdx_dataset_profile.txt'.")

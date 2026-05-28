import csv
import pickle
import random
from pathlib import Path
from typing import List, Dict, Any
import os
import json
import uuid
import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
import pickle
import random
from pathlib import Path
from typing import List, Dict, Any
import re


def get_csv_path(session: str, base_dir: Path, csv_filename: str) -> Path:
    """Construct the full path to a session’s CSV file."""
    return base_dir / session / csv_filename


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Read all rows from a CSV into a list of dicts."""
    with csv_path.open(newline='') as f:
        return list(csv.DictReader(f))
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]

def get_calibration(calibration_dir: str) -> Dict[str, Any]:
    cams_calibration: Dict[str, Any] = {}
    cam_files = sorted(
            Path(calibration_dir).glob("cam_*.json"),
            key=lambda p: natural_key(p.name)
        )
    for cam_file in cam_files:
                cam_name = cam_file.stem  # e.g. "cam_1"
                with cam_file.open("r") as f:
                    data = json.load(f)

                intr = data["intrinsic_params"]
                ext = data["extrinsic_params"]

                cams_calibration[cam_name] = {
                    "intrinsics": {
                        "fx": intr["fx"],
                        "fy": intr["fy"],
                        "cx": intr["cx"],
                        "cy": intr["cy"],
                        "distortion": {
                            "k1": intr["k1"],
                            "k2": intr["k2"],
                            "k3": intr["k3"],
                            "p1": intr["p1"],
                            "p2": intr["p2"],
                        }
                    },
                    "extrinsics": {
                        "translation": [ext["px"], ext["py"], ext["pz"]],
                        "quaternion": [
                            ext["quaternion"]["x"],
                            ext["quaternion"]["y"],
                            ext["quaternion"]["z"],
                            ext["quaternion"]["w"],
                        ]
                    }
                }
    return cams_calibration

def load_lip_lir(lip_lir_dir: str) -> np.ndarray:
        """Load the LIP and LIR files to create a transformation matrix.
        The LIP file contains the position, the LIR file contains the rotation angles,
        """
        lip_file = os.path.join(lip_lir_dir, "mobile.lip")
        lir_file = os.path.join(lip_lir_dir, "mobile.lir")
        vat_file = os.path.join(lip_lir_dir, "mobile.vat")   

        position = np.loadtxt(lip_file)[:3]
        rotation_angles = np.loadtxt(lir_file)[:3]
        vat_rotation = np.loadtxt(vat_file)[:3]
        
        vat_rotation = R.from_euler('ZYX',vat_rotation,degrees=True)
        rotation_matrix = R.from_euler('ZYX', rotation_angles, degrees=True)
        yaw_90 = R.from_euler('Z',[90],degrees=True)

        rotation_final= (vat_rotation.inv()*rotation_matrix*yaw_90)

        # Construct the 4x4 transformation matrix
        transformation_matrix = np.eye(4)  # Start with an identity matrix
        transformation_matrix[:3, :3] = rotation_final.as_matrix()  # Set rotation part
        transformation_matrix[:3, 3] = position  # Set translation part
        return transformation_matrix

def build_sample(
    session: str,
    row: Dict[str, str],
    idx: int,
    cams_calibration: Dict[str, Any],
    lidar2ego: np.ndarray,
    stage00_dir: str,
    group: str = None,
    frame_number: int = 0
) -> Dict[str, Any]:
    """Extract the fields you need from one CSV row into your sample dict."""
    session_name = session.split('/')[-1]
    unique_token = str(uuid.uuid4())  # Generating a unique token
    # cams = {
    #     'cam_1': {'img_fpath' : os.path.join(stage00_dir,session_name,"cam_1",f"{row.get('cam_1')}.bin")},
    #     'cam_2': {'img_fpath' : os.path.join(stage00_dir,session_name,"cam_2",f"{row.get('cam_2')}.bin")},
    #     'cam_3': {'img_fpath' : os.path.join(stage00_dir,session_name,"cam_3",f"{row.get('cam_3')}.bin")},
    #     'cam_4': {'img_fpath' : os.path.join(stage00_dir,session_name,"cam_4",f"{row.get('cam_4')}.bin")},
    #     'cam_5': {'img_fpath' : os.path.join(stage00_dir,session_name,"cam_5",f"{row.get('cam_5')}.bin")},
    #     'cam_6': {'img_fpath' : os.path.join(stage00_dir,session_name,"cam_6",f"{row.get('cam_6')}.bin")}
    # }
    cams = {}
    for cam_key, calib in cams_calibration.items():
        img_id = row.get(cam_key)
        img_fpath = os.path.join(
            stage00_dir, session_name, cam_key, f"{img_id}.bin"
        )
        cams[cam_key] = {
            "img_fpath": img_fpath,
            **calib
        }

    lidar2g_translation= np.array([
        float(row['pos_x']),
        float(row['pos_y']),
        float(row['pos_z'])
    ])

    # Extract quaternion and convert to rotation matrix
    lidar2g_rotation = np.array([
        float(row['quat_x']),
        float(row['quat_y']),
        float(row['quat_z']),
        float(row['quat_w'])
    ])
    lidar2g = np.eye(4)
    lidar2g[:3, :3] = R.from_quat(lidar2g_rotation).as_matrix()
    lidar2g[:3, 3] = lidar2g_translation
    ego2g = lidar2g.dot(np.linalg.inv(lidar2ego))

    latitude,longitude = float(row['latitude']),float(row['longitude'])
    heading = float(row['yaw'])  # in degrees
    lat_long_heading = np.array([latitude, longitude, heading])

    """Extract the fields you need from one CSV row into your sample dict."""
    return {
        'scene_name': session_name,
        'sample_idx': idx,
        'timestamp': float(row['time']),
        'lidar2ego_translation': lidar2ego[:3,3].tolist(),
        'lidar2ego_rotation': R.from_matrix(lidar2ego[:3, :3]).as_quat().tolist(),
        'e2g_translation': ego2g[:3, 3].tolist(),
        'e2g_rotation': R.from_matrix(ego2g[:3, :3]).as_quat().tolist(),
        'lidar2g_translation': lidar2g_translation.tolist(),
        'lidar2g_rotation': lidar2g_rotation.tolist(),
        'lat_long_heading': lat_long_heading.tolist(),
        'prev': idx - 1 ,
        'next': idx + 1 ,
        'cams': cams,
        'token': unique_token,
        'group': group,
        'frame_number': frame_number
        #'lidar_path': os.path.join(stage00_dir, session_name, "lidar", f"{row.get('lidar')}.bin"),
    }


def collect_all_samples(
    input_dirs: List[str],
    csv_filename: str,
    lip_lir_calibration_dir: str,
    stage00_dir: str,
    group_output: bool = False,
) -> List[Dict[str, Any]]:
    """
    For each session, read its CSV and build sample dicts.
    Returns one flat list of all samples.
    """
    all_samples: List[Dict[str, Any]] = []
    idx = 0
    for session in input_dirs:
        if group_output:
            session_group_path = os.path.join(session, "grouped")
            # Iterate through all CSV files in session_group_path
            for file in os.listdir(session_group_path):
                if file.endswith(".csv"):
                    group = os.path.splitext(file)[0]
                    csv_path = os.path.join(session_group_path, file)
                    if not os.path.isfile(csv_path):
                        print(f"[WARN] Missing CSV for session {session}: {csv_path}")
                        continue

                    cams_calibration = get_calibration(os.path.join(session, "calibration_directory"))
                    # Load LIP and LIR calibration
                    lidar2ego = load_lip_lir(lip_lir_calibration_dir)
                    
                    rows = read_csv_rows(Path(csv_path))
                    for i, row in enumerate(rows):
                        sample = build_sample(session, row, idx, cams_calibration, lidar2ego, stage00_dir, group=group, frame_number=i + 1)
                        idx += 1
                        all_samples.append(sample)
            
            
        else:  
            csv_path = Path(os.path.join(session, csv_filename))
            if not csv_path.is_file():
                print(f"[WARN] Missing CSV for session {session}: {csv_path}")
                continue

            cams_calibration = get_calibration(os.path.join(session, "calibration_directory"))
            # Load LIP and LIR calibration
            lidar2ego = load_lip_lir(lip_lir_calibration_dir)
            #lidar2ego_rotation = R.from_matrix(load_lip_lir(lip_lir_calibration_dir)[:3,:3]).as_quat()
            
            
            rows = read_csv_rows(csv_path)
            for i, row in enumerate(rows):
                sample = build_sample(session, row, idx, cams_calibration, lidar2ego, stage00_dir, group=None)

                idx+=1
                all_samples.append(sample)

    return all_samples


def write_pickle(data: Any, out_path: Path) -> None:
    """Serialize `data` to `out_path` using highest pickle protocol."""
    with out_path.open('wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] Wrote {len(data) if isinstance(data, list) else '?'} items → {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the HRDX (RDX) sample pickle from stage-01 annotation batches.",
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=os.environ.get("RDX_INPUT_DIRS", "").split(":") if os.environ.get("RDX_INPUT_DIRS") else None,
        help=(
            "List of stage-01 annotation batch directories (e.g. "
            ".../annotation_data_aws/batch1 ...). Defaults to "
            "$RDX_INPUT_DIRS (colon-separated)."
        ),
    )
    parser.add_argument(
        "--lip-lir-calibration-dir",
        default=os.environ.get(
            "RDX_LIP_LIR_CALIBRATION_DIR",
            "datasets/RDX/lip_lir_calibration",
        ),
        help="Directory containing the LiDAR / camera calibration files.",
    )
    parser.add_argument(
        "--stage00-dir",
        default=os.environ.get(
            "RDX_STAGE00_DIR",
            "datasets/RDX/stage00",
        ),
        help=(
            "Root directory of raw stage-00 camera/lidar bin files. "
            "Per-sample camera paths are recorded as "
            "<stage00_dir>/<session>/<cam>/<frame>.bin. "
            "Defaults to $RDX_STAGE00_DIR or datasets/RDX/stage00."
        ),
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=Path(os.environ.get(
            "RDX_OUTPUT_PKL",
            "datasets/RDX/train_all_poses.pkl",
        )),
        help="Output pickle path.",
    )
    parser.add_argument(
        "--csv-filename",
        default="image_seq_nums_trimmed.csv",
        help="Per-session CSV filename containing image sequence numbers.",
    )
    parser.add_argument(
        "--no-group-output",
        action="store_true",
        help="Disable per-group output (groups are enabled by default).",
    )
    args = parser.parse_args()

    if not args.input_dirs:
        parser.error(
            "No --input-dirs provided. Pass them on the CLI or set $RDX_INPUT_DIRS "
            "(colon-separated)."
        )

    group_output = not args.no_group_output
    merged_dirs = []
    for base in args.input_dirs:
        for name in os.listdir(base):
            full = os.path.join(base, name)
            if os.path.isdir(full):
                merged_dirs.append(full)

    # Example split: 80% for training, 20% for validation.
    split_idx = int(len(merged_dirs) * 0.8)
    train_list = merged_dirs[:split_idx]
    val_list = merged_dirs[split_idx:]

    all_samples = collect_all_samples(
        merged_dirs,
        args.csv_filename,
        args.lip_lir_calibration_dir,
        args.stage00_dir,
        group_output=group_output,
    )
    write_pickle(all_samples, args.output_pkl)


"""
Nuscenes outptut example:
'lidar_path' =
'./datasets/nuscenes/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530949817.pcd.bin'
'token' =
'14d5adfe50bb4445bc3aa5fe607691a8'
'cams' =
{'CAM_FRONT': {'extrinsics': array([[ 5.68477868e-03, -9.99983517e-01,  8.05071338e-04,
         5.06031940e-03],
...0.00000000e+00,
         1.00000000e+00]]), 'intrinsics': [...], 'img_fpath': './datasets/nuscenes/samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530912460.jpg'}, 'CAM_FRONT_RIGHT': {'extrinsics': array([[-8.32929563e-01, -5.53304927e-01, -9.05540984e-03,
         1.03228825e+00],
...0.00000000e+00,
         1.00000000e+00]]), 'intrinsics': [...], 'img_fpath': './datasets/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-18-11-07-57+0800__CAM_FRONT_RIGHT__1531883530920339.jpg'}, 'CAM_FRONT_LEFT': {'extrinsics': array([[ 8.20758348e-01, -5.71271601e-01, -2.11945808e-03,
        -9.64967781e-01],
...0.00000000e+00,
         1.00000000e+00]]), 'intrinsics': [...], 'img_fpath': './datasets/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-07-18-11-07-57+0800__CAM_FRONT_LEFT__1531883530904844.jpg'}, 'CAM_BACK': {'extrinsics': array([[ 0.00242171,  0.99998907, -0.00400023,  0.00279685],
     ...
'lidar2ego_translation' =
[0.943713, 0.0, 1.84023]
'lidar2ego_rotation' =
[0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]
'e2g_translation' =
[1010.2614658861562, 612.8972020252967, 0.0]
'e2g_rotation' =
[-0.6947181373791192, -0.008149850208750354, 0.00881716422927033, -0.719181859582829]
'timestamp' =
1531883530949817
'location' =
'singapore-onenorth'
'scene_name' =
'scene-0001'
'sample_idx' =
1
'prev' =
0
'next' =
2
'modified_sample_idx' = 1
"""
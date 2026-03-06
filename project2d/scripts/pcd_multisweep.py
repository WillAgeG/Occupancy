"""Generates voxel and bev labels from our marked up data """
import argparse
import collections
import glob
import os
import pathlib
import shutil

import cv2
import numpy as np

from lib import visualization
from lib.common import label_lut
from lib.core import multisweep, projection, voxelizer
from lib.readers import pcd_reader

from .project_dataset import filter_moving_by_input


def generate_paths_for_scene(root_dir: str, scene_name: str) -> tuple[list[str], list[str]]:
    pcd_paths, pose_paths = [], []
    for num in range(41, 51):
        pcd_path = os.path.join(root_dir, scene_name, "lidar", f"frame_{num:06}.pcd")
        pose_path = os.path.join(root_dir, scene_name, "meta", f"frame_{num:06}.json")
        pcd_paths.append(pcd_path)
        pose_paths.append(pose_path)
    return pcd_paths, pose_paths


def pack(array):
    """ convert a boolean array into a bitwise array. """
    array = array.reshape(-1)

    # compressing bit flags.
    # yapf: disable
    compressed = array[::8] << 7 | array[1::8] << 6 | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
    # yapf: enable

    return np.array(compressed, dtype=np.uint8)


def priority_dilate_cv(labels: np.ndarray,
                       ksize: tuple[int, int] = (3, 3),
                       shape: int = cv2.MORPH_RECT,
                       iterations: int = 1,
                       border_type: int = cv2.BORDER_CONSTANT) -> np.ndarray:
    """
    Priority-aware dilation using OpenCV for label image with classes {0,1,2,3}.
    Higher label wins (3 > 2 > 1 > 0), exactly matching your overwrite rules.

    Args:
        labels: HxW array of dtype uint8/int containing {0,1,2,3}.
        ksize: kernel size (odd dims recommended).
        shape: cv2.MORPH_RECT / MORPH_ELLIPSE / MORPH_CROSS.
        iterations: number of dilation steps.
        border_type: OpenCV border handling (BORDER_CONSTANT keeps 0s at edges).

    Returns:
        Dilated label image (same shape, dtype as input).
    """
    if labels.dtype != np.uint8:
        labels = labels.astype(np.uint8, copy=False)

    kernel = cv2.getStructuringElement(shape, ksize)
    out = cv2.dilate(labels, kernel, iterations=iterations, borderType=border_type)
    return out


def save_bev(dense_map, save_dir_name, scene, index):
    bev_save_dir = result_dir / scene / save_dir_name
    bev_save_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Come up with a better way car height calculation
    bev = projection.project_to_bev(dense_map, car_height_m=(abs(args.grid[2]) + 3))

    bev_color = color_lut[bev]
    bev_color_save_dir = bev_save_dir.with_name(bev_save_dir.name + "_color")
    bev_color_save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(bev_color_save_dir / f"{index:06}_bev.png"), bev_color)
    bev = cv2.cvtColor(bev.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(bev_save_dir / f"{index:06}_bev.png"), bev)

    scales = [2, 4, 8]
    orig_size = bev.shape
    for scale in scales:
        save_path = bev_save_dir / f"{index:06}_bev_1_{scale}.png"
        bev_scaled = cv2.resize(bev, (orig_size[0] // scale, orig_size[1] // scale), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(save_path), bev_scaled)


def save_voxel(dense_map, save_dir_name, scene, index):
    voxel_save_dir = result_dir / scene / save_dir_name
    voxel_save_dir.mkdir(parents=True, exist_ok=True)
    dense_map = dense_map.astype(np.uint16)
    dense_map.tofile(voxel_save_dir / f"{index:06}.label")

    invalid = np.zeros_like(dense_map, dtype=np.uint8)
    invalid = pack(invalid)
    invalid.tofile(voxel_save_dir / f"{index:06}.invalid")

    occupancy = (dense_map != 0).astype(np.uint8)
    occupancy.tofile(voxel_save_dir / f"{index:06}.bin")


ap = argparse.ArgumentParser()
ap.add_argument("--data_root", type=str, default="/home/vldluchinskiy/Datasets/task_id_815", help="Path to a dataset")
ap.add_argument("--result_dir", type=str, default="output", help="Where to save resulting bevs")
ap.add_argument("--viz_voxel", action="store_true", help="Show 3D visualization")
ap.add_argument("--viz_seg", action="store_true", help="Show 3D visualization")
ap.add_argument("--save_voxels", action="store_true", help="Whether to save voxels")
ap.add_argument("--voxel", type=float, default=0.2, help="Open3D voxel size for display downsampling (e.g., 0.05)")
ap.add_argument("--grid", nargs="+", type=float,
                # default=(-51.2, -51.2, -1.6, 51.2, 51.2, 4.8),
                default=(0.0, -25.6, -0.8, 51.2, 25.6, 5.6),
                help="indicate voxel range. format: xyzxyz, minmax")
args = ap.parse_args()
assert len(args.grid) == 6, "Wrong grid length"

data_root = args.data_root
scenes = sorted(os.listdir(data_root))
json_ref = "frame_000046.json"
reference_index = 5
result_dir = pathlib.Path(args.result_dir)
result_dir.mkdir(parents=True, exist_ok=True)

label_to_color = {i: visualization.deterministic_color(lbl) for lbl, i in label_lut.LABEL2INT_extended.items()}
label_to_color[0] = (0.0, 0.0, 0.0)
color_lut = visualization.get_color_lut()

reader = pcd_reader.PCDReader()

for i, scene in enumerate(scenes):
    print(scene)
    label_dir = os.path.join(data_root, scene, "label")
    json_ref_path = glob.glob(f"{label_dir}/**/{json_ref}", recursive=True)[0]

    pcd_paths, pose_paths = generate_paths_for_scene(data_root, scene)
    pcd_ms_ref_path = pcd_paths[reference_index]
    points = reader.read_cloud(pcd_ms_ref_path, xyz=False)  # Already multisweep
    points_xyz = points[:, :3]
    points_single = points[points[:, -1] == reference_index]

    labels = reader.read_label(json_ref_path, points_xyz, default_label="unlabelled", return_type="object")
    float_labels = reader.map_labels(labels)

    if args.viz_seg:
        visualization.visualize_open3d(points_xyz, labels, voxel=(args.voxel if args.voxel > 0 else None))

    point_n_labels = np.concatenate([points_xyz, float_labels[:, np.newaxis]], axis=1)

    dense_map = voxelizer.voxelize(points_xyz, float_labels, voxel_size=(args.voxel, args.voxel, args.voxel),
                                coors_range=args.grid)
    dense_map = priority_dilate_cv(dense_map).astype(np.uint8)

    if args.viz_voxel:
        visualization.visualize_as_voxelgrid(dense_map, args.voxel, label_to_color)

    save_bev(dense_map, "bev_ms_10", scene, 5)

    pcds = [points_single[:, :3], reader.read_cloud(pcd_paths[reference_index + 1]), reader.read_cloud(pcd_paths[reference_index + 2])]
    poses = [reader.read_pose(p) for p in pose_paths[reference_index:reference_index + 3]]
    points = multisweep.merge_pointclouds_to_timestamp(pcds, poses, reference_index=0)[:, :3]
    voxel = voxelizer.voxelize_points(points, voxel_size=(args.voxel, args.voxel, args.voxel), coors_range=args.grid)
    dense_map_ms_3 = filter_moving_by_input(dense_map, voxel, labels=[3])
    save_bev(dense_map_ms_3, "bev_ms_3", scene, 5)

    if args.save_voxels:
        save_voxel(dense_map_ms_3, "voxels", scene, 5)

    voxel = voxelizer.voxelize_points(points_single[:, :3], voxel_size=(args.voxel, args.voxel, args.voxel), coors_range=args.grid)
    dense_map_single_sweep_dynamic = filter_moving_by_input(dense_map, voxel, labels=[3])
    save_bev(dense_map_single_sweep_dynamic, "bev", scene, 5)

    lidar_save_dir = result_dir / scene / "lidar"
    lidar_save_dir.mkdir(exist_ok=True, parents=True)
    reader.save_cloud(lidar_save_dir / f"{5:06}.pcd", points_single[:, :-1])
    shutil.copy(pcd_paths[reference_index + 1], lidar_save_dir / f"{6:06}.pcd")
    shutil.copy(pcd_paths[reference_index + 2], lidar_save_dir / f"{7:06}.pcd")

    point_labels_save_dir = result_dir / scene / "label"
    point_labels_save_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(json_ref_path, point_labels_save_dir / f"{5:06}.json")

    pose_save_dir = result_dir / scene / "meta"
    pose_save_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(pose_paths[reference_index], pose_save_dir / f"{5:06}.json")
    shutil.copy(pose_paths[reference_index + 1], pose_save_dir / f"{6:06}.json")
    shutil.copy(pose_paths[reference_index + 2], pose_save_dir / f"{7:06}.json")

    # Quick stats
    cnt = collections.Counter(labels.tolist())
    print("Label counts:", cnt)

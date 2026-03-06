"""Semantic kitti point cloud merging and voxelisation. Serves demonstration purposes"""
import argparse
import collections
import pathlib

import numpy as np

from lib import visualization
from lib.core import multisweep, voxelizer
from lib.readers import kitti_reader

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", type=str, default="/home/vldluchinskiy/Datasets/Occupancy/SemanticKitty/dataset/sequences", help="Path to semantic kitti sequences dir")
ap.add_argument("--seq", type=int, default=0, help="Number of the desired kitti sequence")
ap.add_argument("--viz_voxel", action="store_true", help="Show 3D visualization")
ap.add_argument("--viz_seg", action="store_true", help="Show 3D visualization")
ap.add_argument("--voxel", type=float, default=0.2, help="Open3D voxel size for display downsampling (e.g., 0.05)")
ap.add_argument("--grid", nargs="+", type=float,
                # default=(-51.2, -51.2, -1.6, 51.2, 51.2, 4.8),
                default=(0.0, -25.6, -2, 51.2, 25.6, 4.4),
                help="indicate voxel range. format: xyzxyz, minmax")
args = ap.parse_args()

data_root = pathlib.Path(args.data_root)
data_root = data_root / f"{args.seq:02d}"

result_dir = pathlib.Path("output")
result_dir.mkdir(parents=True, exist_ok=True)

pc_paths = [
    data_root / "velodyne/000003.bin",
    data_root / "velodyne/000004.bin",
    data_root / "velodyne/000005.bin",
]
label_paths = [
    data_root / "labels/000003.label",
    data_root / "labels/000004.label",
    data_root / "labels/000005.label",
]

reader = kitti_reader.KittiCloudReader()

clouds = [reader.read_cloud(pc) for pc in pc_paths]
labels = [reader.read_label(lb) for lb in label_paths]
poses = [reader.read_pose(pc) for pc in pc_paths]

clouds_n_labels = [np.concatenate([cl, lb[..., np.newaxis]], axis=1) for (cl, lb) in zip(clouds, labels)]

points = multisweep.merge_pointclouds_to_timestamp(clouds_n_labels, poses, reference_index=2)

# Quick stats
cnt = collections.Counter(labels[0].tolist())
print("Label counts:", cnt)

if args.viz_seg:
    visualization.visualize_open3d(points[:, :3], points[:, 3], None, voxel=(args.voxel if args.voxel > 0 else None))

label_to_color = {int(lbl): visualization.deterministic_color(lbl) for lbl in np.unique(points[:, 3])}
label_to_color[0] = (0.0, 0.0, 0.0)

dense_map = voxelizer.voxelize(points[:, :3], points[:, 3], voxel_size=(args.voxel, args.voxel, args.voxel),
                                coors_range=args.grid)

if args.viz_voxel:
    visualization.visualize_as_voxelgrid(dense_map, args.voxel, label_to_color)

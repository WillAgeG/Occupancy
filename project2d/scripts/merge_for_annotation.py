"""
Script for merging point cloud (in sustech format) in order to pass them for annotation.
"""

import argparse
import os
import pathlib

from tqdm import tqdm

from lib import visualization
from lib.common.file_utils import generate_paths_in_window
from lib.readers.pcd_reader import PCDReader
from lib.core import multisweep

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", type=str, default="/downloaded_data/datasets/occupancy/to_merge", help="Path to a dataset")
args = ap.parse_args()

data_root = args.data_root
scenes = list(os.listdir(data_root))
reference_index = 5
STEP_SIZE_FRAMES = 10

color_lut = visualization.get_color_lut()
reader = PCDReader()

for i, scene in tqdm(enumerate(scenes), total=len(scenes)):
    for start_interval in range(1, 201, STEP_SIZE_FRAMES):
        end_interval = start_interval + STEP_SIZE_FRAMES
        pcd_paths, pose_paths = generate_paths_in_window(os.path.join(data_root, scene), start_interval, end_interval)

        try:
            pcds = [reader.read_cloud(p, xyz=False) for p in pcd_paths]
        except:
            print("missing pcd for ", scene)
            continue
        poses = [reader.read_pose(p) for p in pose_paths]
        points = multisweep.merge_pointclouds_to_timestamp(pcds, poses, reference_index=reference_index)
        pathlib.Path(os.path.join("/", *data_root.split("/")[:-1], "squashed", scene)).mkdir(exist_ok=True, parents=True)
        reader.write_multisweep(pcd_paths[reference_index], points, os.path.join("/", *data_root.split("/")[:-1], "squashed", scene, pcd_paths[reference_index].split("/")[-1]))

# cloud = reader.read_cloud("/downloaded_data/datasets/occupancy/to_merge/squashed/kc2-091__logs__2025-01-31__15-52-02__situation__labeling__dev_test__74/frame_000046.pcd", xyz=False)
# visualization.visualize_open3d(cloud[:, :3], np.ones(len(cloud)))
# for ind in range(10):
#     visualization.visualize_open3d(cloud[cloud[:, 7] == ind][:, :3], np.ones(len(cloud)))

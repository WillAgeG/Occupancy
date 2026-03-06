import pathlib

import cv2
import numpy as np
import yaml

from project2d.lib.core import voxelizer
from project2d.lib.readers import kitti_reader
from project2d.scripts import check_projection, project_dataset
from src.data import augmentations, io_data

reader = kitti_reader.KittiCloudReader()
grid = (0.0, -25.6, -2, 51.2, 25.6, 4.4)
flipper = augmentations.RandomFlipper(grid)
assets_dir = pathlib.Path("tests/assets")
save_dir = assets_dir / "prod"
save_dir.mkdir(parents=True, exist_ok=True)

lidar = reader.read_cloud(assets_dir / "lidar_000000.bin")
lidar_label = reader.read_label(assets_dir / "lidar_000000.label")
voxel_label = io_data._read_label_SemKITTI(assets_dir / "000000.label").reshape((256, 256, 32))
bev = cv2.imread(str(assets_dir / "000000.png"))

with pathlib.Path("cfg/mappings/semantic-kitti.yaml").open() as f:
    config = yaml.safe_load(f)

remap_array, color_lut = project_dataset.build_mappings(config)

for flip in range(1, 4):

    lidar_flipped = flipper.flip_lidar(lidar, flip)
    voxel_flipped = flipper.flip_voxel(voxel_label, flip).astype(np.uint8)
    bev_flipped = flipper.flip_2d(bev, flip)

    remapped = project_dataset.remap_labels(voxel_flipped, remap_array)

    bev_rgb = cv2.cvtColor(bev_flipped, cv2.COLOR_BGR2RGB)

    check_projection.check_objects(remapped, bev_rgb, save_name=f"{save_dir!s}/voxel_check_flip_{flip}.png")

    voxel_labels = voxelizer.voxelize(lidar_flipped, lidar_label, coors_range=grid)
    remapped = project_dataset.remap_labels(voxel_labels, remap_array)
    check_projection.check_objects(remapped, bev_rgb, save_name=f"{save_dir!s}/lidar_check_flip_{flip}.png")

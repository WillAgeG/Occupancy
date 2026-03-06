import pathlib
from bisect import bisect_left

import numpy as np
from PIL import Image

from lib.common.box import points_in_obb
from lib.common.constants import DYNAMIC_LABEL, GRID, REF_SIZE_X, REF_SIZE_Y, VOXEL_SIZE
from lib.common.label_lut import label2int_minimal
from lib.core import projection, voxelizer
from lib.readers.pcd_reader import PCDReader


def crop_forward_from_center(img, scene_size_m=200.0, forward_m=51.2, lateral_m=25.6):
    w, h = img.size
    # pixels per meter along x/y
    px_per_m_x = w / scene_size_m
    px_per_m_y = h / scene_size_m

    # center
    cx = w // 2
    cy = h // 2

    # meters to pixels
    dx = int(round(forward_m * px_per_m_x))   # forward (+x) length
    dy = int(round(lateral_m * px_per_m_y))   # half-width in y

    # compute crop box
    left = max(0, cx)
    right = min(w, cx + dx)
    upper = max(0, cy - dy)
    lower = min(h, cy + dy)

    return img.crop((left, upper, right, lower))


def resize_to_256(img: Image.Image) -> Image.Image:
    return img.resize((256, 256), Image.NEAREST)


def find_nearest_file(ts: int, folder: str | pathlib.Path, return_previous: bool = False) -> str:
    folder_path = pathlib.Path(folder)

    paths = sorted(
        (p for p in folder_path.iterdir() if p.is_file()),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    timestamps = [int(p.stem.split("_")[-1]) for p in paths]
    index = bisect_left(timestamps, ts)

    if (
        abs(timestamps[index] - ts)
        > abs(timestamps[index + 1] - ts)
        or return_previous
    ):
        return str(paths[index])

    return str(paths[index + 1])


def get_bev(reader, cloud_path, anno_path, agg_static_path, frame_static_path, blind_zones_path, static_lut, color_lut):
    color_mapper = np.array([color_lut[k] for k in sorted(color_lut)])
    static_lut_mapper = np.zeros(256, dtype=np.uint8)
    for k, v in static_lut.items():
        static_lut_mapper[k] = v

    pc = reader.read_cloud(cloud_path)
    obbs = reader.get_boxes(anno_path)

    try:
        all_inds = []

        for obb in obbs:
            inside = np.where(points_in_obb(pc[:, :3], obb))
            all_inds.append(inside)

        pc = pc[np.hstack(all_inds).flatten()]

        dense_map = voxelizer.voxelize(pc, np.full((pc.shape[0],), fill_value=label2int_minimal["moving_now"]), voxel_size=(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE), coors_range=GRID, reduction="max")

        # project voxels on 2D map
        bev_moving = projection.project_to_bev(dense_map, car_height_m=(abs(GRID[-1])))  # in original meters from base link
        bev_moving = np.flip(bev_moving, axis=0)
    except ValueError as e:
        print(e)
        print(agg_static_path)
        bev_moving = np.zeros((REF_SIZE_X, REF_SIZE_Y), dtype=np.uint8)

    agg_static = Image.open(agg_static_path)  # 400pix = 200m
    single_static = Image.open(frame_static_path)  # 400pix = 200m
    shadows = Image.open(blind_zones_path)  # 300m diameter

    shadows = np.asarray(resize_to_256(crop_forward_from_center(shadows, 300, abs(GRID[3]) + abs(GRID[0]), abs(GRID[1])))) != 0
    single_static = np.asarray(resize_to_256(crop_forward_from_center(single_static, 200, abs(GRID[3]) + abs(GRID[0]), abs(GRID[1]))))
    agg_static = np.asarray(resize_to_256(crop_forward_from_center(agg_static, 200, abs(GRID[3]) + abs(GRID[0]), abs(GRID[1]))))

    agg_static_label = np.flip(np.maximum(bev_moving, static_lut_mapper[agg_static]) * shadows, axis=0).astype(np.uint8)
    single_static_label = np.flip(np.maximum(bev_moving, static_lut_mapper[single_static]) * shadows, axis=0).astype(np.uint8)

    agg_static_color = color_mapper[agg_static_label].astype(np.uint8)
    single_static_color = color_mapper[single_static_label].astype(np.uint8)

    return agg_static_color, agg_static_label, single_static_color, single_static_label

def cut_dynamic(cloud_path: str, labels_path: str):
    """
    Returns two clouds, first one is static, second one is dynamic

    :param cloud_path: Path to point cloud
    :type cloud_path: str
    :param labels_path: Path to labels
    :type labels_path: str

    Returns:

    Static points (N1, 4), np.ndarray, non-dynamic points of the scene. Last dimension is XYZ + sweep id
    Dynamic points: (N2, 4), np.ndarray dynamic points of the scene. Last dimension is XYZ + sweep id
    """

    reader = PCDReader()
    cloud = reader.read_cloud(cloud_path, xyz=False)
    cloud = cloud[:, [0, 1, 2, -1]]
    labels = reader.read_label(labels_path, cloud, return_type='int')

    cloud = np.concatenate((cloud, np.expand_dims(labels, 1)), axis=1)
    mask_dyn = labels == DYNAMIC_LABEL
    mask_stat = (labels != DYNAMIC_LABEL) * (labels != 0)

    return cloud[mask_dyn], cloud[mask_stat]

import numpy as np


def merge_pointclouds_to_timestamp(
    point_clouds,
    poses,
    reference_index: int = 0,
):
    if len(point_clouds) != len(poses):
        raise ValueError("pcd_paths and pose_paths must have the same length")

    Ts = poses

    T_ref = Ts[reference_index]

    merged_xyz = []

    for ind, (xyz_colors, T) in enumerate(zip(point_clouds, Ts)):

        xyz = xyz_colors[:, :3]
        colors = xyz_colors[:, 3:]

        xyz_map = T.inv().apply(xyz)
        xyz_ref = T_ref.apply(xyz_map)

        # add index of a point cloud for traceability
        xyz_ref = np.hstack((xyz_ref, colors, np.full((xyz_ref.shape[0], 1), ind)))

        merged_xyz.append(xyz_ref.T.astype(np.float32, copy=False))

    if not merged_xyz:
        return np.empty((0, 3), dtype=np.float32)

    out = np.hstack(merged_xyz).astype(np.float32, copy=False).T

    return out

import typing as tp

import numpy as np

from ..core import multisweep
from . import abstract_reader


class MultiSweepLidarReader:
    """
    A utility class that merges point clouds into one
    """
    def __init__(self, reader: abstract_reader.AbstractCloudReader):
        self.cloud_reader = reader

    @staticmethod
    def _process_timestamps(pcs, time_scalars_list: list[float] | tp.Literal["auto"]):
        if time_scalars_list == "auto":
            raise NotImplementedError("Reading timestamps from meta is not yet supported")
        times = np.array(time_scalars_list)
        for i, time_scalar in enumerate(times):
            pc = pcs[i]
            time_array = np.full(pc.shape[:-1], time_scalar, pc.dtype)
            time_array = time_array[..., np.newaxis]
            pc = np.concatenate([pc, time_array], axis=-1)
            pcs[i] = pc

        return pcs

    @staticmethod
    def _process_labels(pcs, labels):
        for i, pc in enumerate(pcs):
            pcs[i] = np.concatenate([pc, labels[i][..., np.newaxis]], axis=1)

    def read(
        self,
        pc_paths: list[str],
        pose_paths: list[str],
        timestamps: list[float] | None | tp.Literal["auto"] = None,
        label_paths: list[str] | None = None,
        reference_index: int = -1,
    ) -> np.ndarray:
        """
        Order of channels in the merged cloud:
        x, y, z, whatever was in the point cloud after cloud_reader.read_cloud(), time, label, source cloud index
        """

        assert len(pc_paths) == len(pose_paths), "Unequal amount of point cloud and pose paths"
        pcs = [self.cloud_reader.read_cloud(p) for p in pc_paths]
        poses = [self.cloud_reader.read_pose(p) for p in pose_paths]
        if timestamps is not None:
            pcs = self._process_timestamps(pcs, timestamps)
        if label_paths is not None:
            labels = [self.cloud_reader.read_label(p, pcs[j]) for j, p in enumerate(label_paths)]
            self._process_labels(pcs, labels)

        pc = multisweep.merge_pointclouds_to_timestamp(pcs, poses, reference_index=reference_index)

        return pc

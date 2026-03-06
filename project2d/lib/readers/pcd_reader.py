import json
import pathlib
import typing as tp

import numpy as np
import pypcd4
from scipy.spatial import transform as sc_transform

from ..common import box, label_lut
from ..common.geometry import RigidTransform
from . import abstract_reader


class PCDReader(abstract_reader.AbstractCloudReader):
    @classmethod
    def read_cloud(cls, file_path, xyz=True):
        if xyz:
            pc = pypcd4.PointCloud.from_path(file_path).numpy(("x", "y", "z"))
        else:
            pc = pypcd4.PointCloud.from_path(file_path).numpy()
        return pc

    @staticmethod
    def get_boxes(file_path: str | pathlib.Path):
        data = json.loads(pathlib.Path(file_path).read_text())
        if not isinstance(data, list):
            raise TypeError("Label file is not a box list")
        obbs = [box.OBB.from_json_entry(e) for e in data]
        return obbs

    @staticmethod
    def _get_label_from_entry(entry: dict, default_label: str) -> str:
        for key in ("class", "label", "class_name"):
            value = entry.get(key)
            if value is not None:
                return value
        return default_label

    @classmethod
    def _extract_pointwise_labels(cls, data: tp.Any, n_points: int, default_label: str) -> np.ndarray | None:
        labels = np.full(n_points, default_label, dtype=object)

        if isinstance(data, dict):
            if isinstance(data.get("labels"), list):
                data = data["labels"]
            elif isinstance(data.get("point_labels"), list):
                data = data["point_labels"]
            else:
                return None

        if not isinstance(data, list):
            return None

        if all(isinstance(x, (str, int)) for x in data):
            if len(data) != n_points:
                raise ValueError(f"Pointwise label count mismatch: {len(data)} != {n_points}")
            return np.asarray(data, dtype=object)

        if all(isinstance(x, dict) for x in data):
            for entry in data:
                if "point_id" in entry:
                    point_id = entry["point_id"]
                else:
                    point_id = entry.get("id")
                if point_id is None or point_id < 0 or point_id >= n_points:
                    continue
                labels[point_id] = cls._get_label_from_entry(entry, default_label)
            return labels

        return None

    @classmethod
    def read_label(
        cls,
        file_path: str | pathlib.Path,
        points: np.ndarray,
        return_type: tp.Literal["int", "object"] = "int",
        default_label: str = "unlabeled",
        class_priority: dict[str, int] | None = None,
    ):
        N = points.shape[0]
        data = json.loads(pathlib.Path(file_path).read_text())

        labels = cls._extract_pointwise_labels(data, N, default_label)

        if labels is None:
            obbs = cls.get_boxes(file_path)
            labels = np.full(N, default_label, dtype=object)
            best_pri = np.full(N, np.inf)

            for obb in obbs:
                inside = box.points_in_obb(points[:, :3], obb)
                if not np.any(inside):
                    continue
                pri = float(class_priority.get(obb.label, 0)) if class_priority else 0.0
                better = pri < best_pri
                update = inside & better
                labels[update] = obb.label
                best_pri[update] = pri

        if return_type == "int":
            labels = cls.map_labels(labels)

        return labels

    @staticmethod
    def save_cloud(file_path: str | pathlib.Path, points: np.ndarray):
        fields = ("x", "y", "z", "intensity", "lidar_id", "laser_id", "rgb")
        types = (points.dtype, ) * 7
        assert points.shape[1] == 7, f"Only support saving points with the 7 fields: {fields}"
        pc = pypcd4.PointCloud.from_points(points, fields, types)
        pc.save(file_path)

    @classmethod
    def map_labels(cls, labels, lut=label_lut.LABEL2INT_extended):
        return np.vectorize(lambda x: lut.get(str(x), 0))(labels)

    @classmethod
    def read_pose(cls, file_path, dtype=np.float64, scalar_first=True):
        with pathlib.Path(file_path).open() as f:
            json_meta = json.load(f)

        translation, rotation = cls._get_pose_from_meta(json_meta)

        pose = cls._from_translation_rotation(
            translation=[translation[key] for key in "xyz"],
            rotation=[rotation[key] for key in "wxyz"],
            dtype=dtype,
            scalar_first=scalar_first,
        )
        return pose

    @staticmethod
    def read_timestamp(file_path: str) -> int:
        with pathlib.Path(file_path).open() as f:
            json_meta = json.load(f)
        return json_meta["lidar"]["timestamp"]

    @staticmethod
    def read_pc_meta(file_path):
        return pypcd4.PointCloud.from_path(file_path).metadata

    @classmethod
    def write_multisweep(cls, reference_cloud_path: str, points: np.ndarray, save_path: str):
        """Writes a multisweep point cloud to a file.

        Args:
            reference_cloud_path (str): used to get original metadata
            points (np.ndarray): _description_
            save_path (str): _description_
        """
        meta = cls.read_pc_meta(reference_cloud_path)
        meta.fields = tuple(list(meta.fields) + ["sweep_id"])
        meta.size = tuple(list(meta.size) + [4])
        meta.type = tuple(list(meta.type) + ["F"])
        meta.count = tuple(list(meta.count) + [1])
        meta.points = points.shape[0]
        meta.width = points.shape[0]

        new_pc = pypcd4.PointCloud(meta, points)

        # Save to file
        new_pc.save(save_path)

    @classmethod
    def _get_pose_from_meta(cls, meta_json):
        translation = meta_json["lidar"]["translation"]
        rotation = meta_json["lidar"]["rotation"]
        return translation, rotation

    @staticmethod
    def _from_translation_rotation(
        translation, rotation, dtype=np.float64, scalar_first=True
    ):
        translation = np.array(translation, dtype=dtype)
        if scalar_first:
            rotation = np.roll(rotation, -1)
        rot_matrix = sc_transform.Rotation.from_quat(rotation).as_matrix()
        transform = np.eye(4, dtype=dtype)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = translation
        return RigidTransform(transform)

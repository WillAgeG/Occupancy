import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import abstract_reader
from ..common.geometry import RigidTransform

class KittiCloudReader(abstract_reader.AbstractCloudReader):
    @classmethod
    def read_cloud(cls, file_path, xyz=True, read_timestamp=False):
        pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        if xyz:
            pc = pc[:, :3]

        if read_timestamp:
            time_scalar = cls.read_timestamp(file_path)
            time_array = np.full(pc.shape[:-1], time_scalar, pc.dtype)
            time_array = time_array[..., np.newaxis]
            pc = np.concatenate([pc, time_array], axis=-1)

        return pc

    @staticmethod
    def read_label(file_path, *args, **kwargs):
        labels = np.fromfile(file_path, dtype=np.uint32)
        sem_label = labels & 0xFFFF  # семантическая метка

        return sem_label

    @classmethod
    def read_pose(cls, file_path, dtype=np.float64, scalar_first=True):
        """
        Возвращает позу как translation + quaternion (для совместимости с твоим meta-форматом).
        file_path: Путь к облаку, не к позе.
        """
        pose = cls._read_pose_orig(file_path)
        R = pose[:3, :3]
        t = pose[:3, 3]
        quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]

        translation = np.array([t[0], t[1], t[2]], dtype=dtype)
        rotation = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=dtype)  # [w, x, y, z]
        if scalar_first:
            rotation = np.roll(rotation, -1)  # [x, y, z, w]

        rot_matrix = Rotation.from_quat(rotation).as_matrix()
        transform = np.eye(4, dtype=dtype)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = translation

        return RigidTransform(transform).inv()

    @staticmethod
    def _get_sequence_dir(file_path: str) -> str:
        """
        Из пути к облаку возвращает путь к папке последовательности (например, sequences/00/).
        Проверяет, что структура пути корректна.
        """
        path = Path(file_path).expanduser().resolve()  # абсолютный путь без лишних '..'
        parts = path.parts

        # ищем индекс папки 'sequences'
        assert "sequences" in parts, f"Неверный путь: '{file_path}'. Ожидается, что он содержит подкаталог 'sequences/'."
        seq_idx = parts.index("sequences")

        # убеждаемся, что есть хотя бы 3 элемента после 'sequences'
        # .../sequences/<seq_id>/velodyne/<frame>.bin
        assert len(parts) > seq_idx + 3, (
            f"Неверный путь: '{file_path}'. Ожидается структура "
            f"'.../sequences/<seq_id>/velodyne/<frame>.bin'"
        )

        seq_id = parts[seq_idx + 1]
        velodyne_dir = parts[seq_idx + 2]

        assert seq_id.isdigit() and len(seq_id) == 2, (
            f"Название последовательности '{seq_id}' должно быть числовым, например '00', '01'."
        )

        # возвращаем путь до sequences/<seq_id>
        seq_dir = Path(*parts[: seq_idx + 2])
        return str(seq_dir)

    @classmethod
    def read_timestamp(cls, file_path: str) -> float:
        seq_dir = cls._get_sequence_dir(file_path)
        times_path = os.path.join(seq_dir, "times.txt")

        # читаем позу для текущего кадра (из камеры)
        frame_id = int(os.path.splitext(os.path.basename(file_path))[0])
        with Path(times_path).open() as f:
            for i, line in enumerate(f):
                if i == frame_id:
                    time_ = float(line)
                    break
            else:
                raise IndexError(f"Фрейм {frame_id} отсутствует в {times_path}")

        return time_

    @classmethod
    def _read_pose_orig(cls, file_path):
        """
        Читает позу для данного облака точек.
        Для KITTI 00 использует calib.txt для преобразования поз камеры в позы лидара.
        """
        seq_dir = cls._get_sequence_dir(file_path)
        poses_path = os.path.join(seq_dir, "poses.txt")
        calib_path = os.path.join(seq_dir, "calib.txt")

        if not Path(poses_path).exists():
            raise FileNotFoundError(f"Не найден файл поз: {poses_path}")
        if not Path(calib_path).exists():
            raise FileNotFoundError(f"Не найден калибровочный файл: {calib_path}")

        # читаем позу для текущего кадра (из камеры)
        frame_id = int(os.path.splitext(os.path.basename(file_path))[0])
        with Path(poses_path).open() as f:
            for i, line in enumerate(f):
                if i == frame_id:
                    vals = np.fromstring(line.strip(), sep=" ", dtype=np.float64)
                    T_cam = vals.reshape(3, 4)
                    T_cam = np.vstack((T_cam, [0, 0, 0, 1]))
                    break
            else:
                raise IndexError(f"Фрейм {frame_id} отсутствует в {poses_path}")

        # читаем калибровку: Tr_velo_to_cam или Tr
        with Path(calib_path).open() as f:
            lines = f.readlines()
        Tr = None
        for line in lines:
            if line.startswith("Tr_velo_to_cam:") or line.startswith("Tr:"):
                Tr = np.fromstring(line.split(":")[1], sep=" ", dtype=np.float64).reshape(3, 4)
                Tr = np.vstack((Tr, [0, 0, 0, 1]))
                break
        if Tr is None:
            raise ValueError(f"В {calib_path} не найдено поле Tr_velo_to_cam или Tr")

        # позы лидара = позы камеры * inv(Tr_velo_to_cam)
        T_lidar = T_cam @ Tr
        return T_lidar

import numpy as np

from src.common import enums

M = enums.Modality


class RandomFlipper:
    def __init__(self, grid: tuple[float]) -> None:
        self.grid = grid
        self.flip_functions = {
            M.LABEL_2D.value: self.flip_2d,
            M.LABEL_VOXELS.value: self.flip_voxel,
            M.LIDAR.value: self.flip_lidar,
            M.VOXELS.value: self.flip_voxel,
        }

    @staticmethod
    def map_key(key: str) -> str:
        if key.startswith(M.LABEL_2D.value):
            return M.LABEL_2D.value
        if key.startswith(M.LABEL_VOXELS.value):
            return M.LABEL_VOXELS.value
        return key

    def flip(self, data: dict, flip: int) -> None:
        for k, v in data.items():
            mapped_key = self.map_key(k)
            if mapped_key in self.flip_functions:
                data[k] = self.flip_functions[mapped_key](v, flip)

    @staticmethod
    def flip_2d(data: np.array, flip: int) -> np.ndarray:
        # Dims -> {HW}
        # Flipping around the X axis...
        if flip == 1:
            data = np.flip(data, axis=1).copy()

        # Flipping around the Y axis...
        if flip == 2:
            data = np.flip(data, axis=0).copy()

        # Flipping around the X and the Y axis...
        if flip == 3:
            data = np.flip(np.flip(data, axis=1), axis=0).copy()

        return data

    def flip_lidar(self, data: np.array, flip: int) -> np.ndarray:
        # Dims -> {XYZ...}
        x0, y0, _, x1, y1, _ = self.grid
        data = data.copy()
        # Flipping around the X axis...
        if flip == 1:
            data[:, 0] = x0 + x1 - data[:, 0]

        # Flipping around the Y axis...
        elif flip == 2:
            data[:, 1] = y0 + y1 - data[:, 1]

        # Flipping around the X and the Y axis...
        elif flip == 3:
            data[:, 0] = x0 + x1 - data[:, 0]
            data[:, 1] = y0 + y1 - data[:, 1]

        return data

    @staticmethod
    def flip_voxel(data: np.array, flip: int) -> np.ndarray:
        # Dims -> {XYZ}
        # Flipping around the X axis...
        if flip == 1:
            data = np.flip(data, axis=0).copy()

        # Flipping around the Y axis...
        elif flip == 2:
            data = np.flip(data, axis=1).copy()

        # Flipping around the X and the Y axis...
        elif flip == 3:
            data = np.flip(np.flip(data, axis=0), axis=1).copy()
        return data

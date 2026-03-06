import enum


class Modality(str, enum.Enum):
    LABEL_2D = "label_2d"
    LABEL_VOXELS = "label_voxels"
    LABEL_LIDAR = "label_lidar"
    LIDAR = "lidar"
    VOXELS = "voxels"
    INVALID = "invalid"
    POSE = "pose"

    @classmethod
    def from_key(cls, key: str) -> "Modality":
        for m in cls:
            if key.startswith(m.value):
                return m
        err = f"Unknown modality key: {key}"
        raise ValueError(err)


class DatasetType(str, enum.Enum):
    SEMANTIC_KITTI = "semantic_kitti"
    NAVIO = "navio"


class ModelType(str, enum.Enum):
    LMSCNET = "lmscnet"
    LMSCNET_2D = "lmscnet_2d"
    OCCRWKV = "occrwkv"
    OCCRWKV_2D = "occrwkv_2d"

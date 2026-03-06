import copy

from cfg.kitti import lmscnet
from src.common import base_config as base
from src.common import enums

M = enums.Modality

cfg = copy.deepcopy(lmscnet.cfg)

cfg.download = base.DownloadConfig(
    [
        base.DownloadEntry(
            "obs://selfdriving/datasets-public/semantic_kitti/SemanticKitty/",
            "/datasets/perception/occupancy/SemanticKitti",
            "s3",
        ),
        base.DownloadEntry(
            "lakefs://occupancy-datasets/main/SemanticKitty/voxel_labels_fixed/",
            "/datasets/perception/occupancy/SemanticKitti",
            "lakefs",
        ),
    ]
)
cfg.dataset.root_dir = "/datasets/perception/occupancy/SemanticKitti/dataset/sequences"
cfg.dataloader.train_batch_size = 40

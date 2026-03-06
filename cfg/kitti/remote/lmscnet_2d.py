import copy

from cfg.kitti import lmscnet_2d
from src.common import base_config as base
from src.common import enums

M = enums.Modality

cfg = copy.deepcopy(lmscnet_2d.cfg)

cfg.download = base.DownloadConfig(
    [
        base.DownloadEntry(
            "obs://selfdriving/datasets-public/semantic_kitti/SemanticKitty/",
            "/datasets/perception/occupancy/SemanticKitti",
            "s3",
        ),
        base.DownloadEntry(
            "lakefs://occupancy-datasets/main/SemanticKitty/bev_label_and_invalid_fix_moving/",
            "/datasets/perception/occupancy/SemanticKitti",
            "lakefs",
            overwrite=True,  # Перезаписываем, так как все bev лежат в lakefs по одному пути: dataset/sequences/??/bev
        ),
    ]
)
cfg.dataset.root_dir = "/datasets/perception/occupancy/SemanticKitti/dataset/sequences"
cfg.dataloader.train_batch_size = 40

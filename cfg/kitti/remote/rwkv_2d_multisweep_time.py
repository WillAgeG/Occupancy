import copy

from cfg.kitti import rwkv_2d_multisweep_time
from src.common import base_config as base
from src.common import enums

M = enums.Modality

cfg = copy.deepcopy(rwkv_2d_multisweep_time.cfg)

cfg.download = base.DownloadConfig(
    [
        base.DownloadEntry(
            "obs://selfdriving/datasets-public/semantic_kitti/SemanticKitty/",
            "/datasets/perception/occupancy/SemanticKitti",
            "s3",
        ),
        base.DownloadEntry(
            "lakefs://occupancy-datasets/main/SemanticKitty/bev_label_and_invalid_fix_moving_multisweep/",
            "/datasets/perception/occupancy/SemanticKitti",
            "lakefs",
            overwrite=True,  # Перезаписываем, так как все bev лежат в lakefs по одному пути: dataset/sequences/??/bev
        ),
    ]
)
cfg.dataset.root_dir = "/datasets/perception/occupancy/SemanticKitti/dataset/sequences"
cfg.dataloader.train_batch_size = 32

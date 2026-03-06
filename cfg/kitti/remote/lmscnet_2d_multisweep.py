import copy

from cfg.kitti import lmscnet_2d_multisweep
from src.common import base_config as base
from src.common import enums
from src.data import path_solvers

M = enums.Modality

cfg = copy.deepcopy(lmscnet_2d_multisweep.cfg)

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
cfg.dataset.path_solvers[M.LABEL_2D] = path_solvers.ScaledPathSolver(M.LABEL_2D.value, "bev", "png")
cfg.dataloader.train_batch_size = 40

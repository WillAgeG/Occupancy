import copy

from cfg.navio_markup import lmscnet_2d_multisweep
from src.common import base_config as base
from src.common import enums

M = enums.Modality

cfg = copy.deepcopy(lmscnet_2d_multisweep.cfg)

cfg.download = base.DownloadConfig(
    [
        base.DownloadEntry(
            "lakefs://occupancy-datasets/main/navio-markup/",
            "/datasets/perception/occupancy/navio-markup",
            "lakefs",
        ),
    ]
)
cfg.dataset.root_dir = "/datasets/perception/occupancy/navio-markup/"

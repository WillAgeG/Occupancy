import copy

from cfg.navio_markup import rwkv_2d
from src.common import base_config as base
from src.common import enums

M = enums.Modality

cfg = copy.deepcopy(rwkv_2d.cfg)

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

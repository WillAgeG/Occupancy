import dataclasses
import datetime
import pathlib
import typing as tp

from src.common import enums
from src.data import path_solvers


@dataclasses.dataclass
class MultisweepConfig:
    timestamps: tuple[float] | tp.Literal["auto"] | None = (0.8, 0.9, 1.0)
    num_sweeps: int = 3
    from_past: bool = False
    drop_n: int | tp.Literal["auto"] = "auto"

    def __post_init__(self) -> None:
        assert self.num_sweeps > 0, f"num_sweeps must be > 0, got {self.num_sweeps}"
        if self.timestamps is not None:
            assert len(self.timestamps) == self.num_sweeps


@dataclasses.dataclass
class AugmentationConfig:
    flip: bool = False
    lidar_shuffle_index: bool = False


@dataclasses.dataclass
class DatasetConfig:
    augmentation: AugmentationConfig
    path_solvers: dict[str, path_solvers.AbstractPathSolver]
    root_dir: str
    dataset_type: enums.DatasetType
    reference_length_modality: str | None = None
    multisweep: MultisweepConfig | None = None
    nbr_classes: int = 4
    grid: tuple[float] = (0.0, -25.6, -2, 51.2, 25.6, 4.4)
    voxel_size: float = 0.2
    grid_dims: tuple[int] = (256, 256, 32)
    H: int = 32
    mappings_path: str = "mappings/semantic-kitti.yaml"

    def __post_init__(self) -> None:
        assert enums.Modality.LIDAR in self.path_solvers, \
              "DatasetConfig is required to have a path solver for LIDAR modality"
        if self.multisweep is not None:
            assert enums.Modality.POSE in self.path_solvers


@dataclasses.dataclass
class TrainerConfig:
    model_type: enums.ModelType
    checkpoint_period: int = 15
    train_summary_period: int = 50
    val_summary_period: int = 100
    visualization_interval: int = 10
    num_epochs: int = 80


@dataclasses.dataclass
class DataloaderConfig:
    num_workers: int = 10
    train_batch_size: int = 12
    val_batch_size: int = 10


@dataclasses.dataclass
class OptimizerConfig:
    lr: int = 0.001
    betas: tuple[int] = (0.9, 0.999)
    momentum: tp.Any = None
    weight_decay: tp.Any = None
    optim_type: str = "AdamW"


@dataclasses.dataclass
class SchedulerConfig:
    frequency: str = "epoch"
    lr_power: int = 0.98
    scheduler_type: str = "power_iteration"


@dataclasses.dataclass
class DownloadEntry:
    source: str
    target: str
    source_type: tp.Literal["s3", "lakefs"]
    overwrite: bool = False


@dataclasses.dataclass
class DownloadConfig:
    entries: list[DownloadEntry]


@dataclasses.dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    trainer: TrainerConfig
    dataloader: DataloaderConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    download: DownloadConfig | None = None
    out_root: str = "../SSC_out/"
    out_path: pathlib.Path | None = None
    weights_path: str | None = None

    def __post_init__(self) -> None:
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        run_name = (f"{self.trainer.model_type.value}_{self.dataset.dataset_type.value}_{timestamp}")
        self.out_path = pathlib.Path(self.out_root) / run_name

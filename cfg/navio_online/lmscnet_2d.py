from src.common import base_config as base
from src.common import enums
from src.data import path_solvers

M = enums.Modality

cfg = base.ExperimentConfig(
    dataset=base.DatasetConfig(
        augmentation=base.AugmentationConfig(),
        path_solvers={
            M.VOXELS: path_solvers.SimplePathSolver(M.VOXELS.value, "lidar", "pcd"),
            M.LIDAR: path_solvers.SimplePathSolver(M.LIDAR.value, "lidar", "pcd"),
            M.LABEL_2D: path_solvers.ScaledPathSolver(M.LABEL_2D.value, "bev", "png"),
        },
        root_dir="/home/vldluchinskiy/Datasets/occupancy_dataset/",
        reference_length_modality=f"{M.LABEL_2D.value}_1_1",
        dataset_type=enums.DatasetType.NAVIO,
        grid=(0.0, -25.6, -0.8, 51.2, 25.6, 5.6),
        mappings_path="cfg/mappings/navio-online.yaml"
    ),
    trainer=base.TrainerConfig(model_type=enums.ModelType.LMSCNET_2D),
    dataloader=base.DataloaderConfig(train_batch_size=8),
    optimizer=base.OptimizerConfig(),
    scheduler=base.SchedulerConfig(),
    out_root="outputs",
)

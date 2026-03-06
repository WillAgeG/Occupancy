from src.common import base_config as base
from src.common import enums
from src.data import path_solvers

M = enums.Modality

cfg = base.ExperimentConfig(
    dataset=base.DatasetConfig(
        augmentation=base.AugmentationConfig(lidar_shuffle_index=True),
        path_solvers={
            M.VOXELS: path_solvers.SimplePathSolver(M.VOXELS.value, "voxels", "bin"),
            M.LIDAR: path_solvers.ReplacePathSolver(M.LIDAR.value, "voxels", "bin", 
                                                    replaces=[("voxels", "lidar"), (".bin", ".pcd")]),
            M.LABEL_2D: path_solvers.ScaledPathSolver(M.LABEL_2D.value, "bev_ms_3", "png"),
            M.POSE: path_solvers.ReplacePathSolver(M.POSE.value, "voxels", "bin",
                                                   replaces=[("voxels", "meta"), (".bin", ".json")]),
        },
        root_dir="/home/vldluchinskiy/Datasets/navio-markup/processed/",
        dataset_type=enums.DatasetType.NAVIO,
        grid=(0.0, -25.6, -0.8, 51.2, 25.6, 5.6),
        mappings_path="cfg/mappings/navio.yaml",
        multisweep=base.MultisweepConfig(timestamps=None),
    ),
    trainer=base.TrainerConfig(model_type=enums.ModelType.LMSCNET_2D),
    dataloader=base.DataloaderConfig(train_batch_size=8),
    optimizer=base.OptimizerConfig(),
    scheduler=base.SchedulerConfig(),
    out_root="outputs",
)

from src.common import base_config as base
from src.common import enums
from src.data import path_solvers

M = enums.Modality

cfg = base.ExperimentConfig(
    dataset=base.DatasetConfig(
        augmentation=base.AugmentationConfig(),
        path_solvers={
            M.VOXELS: path_solvers.SimplePathSolver(M.VOXELS.value, "voxels", "bin"),
            M.LIDAR: path_solvers.ReplacePathSolver(M.LIDAR.value, "voxels", "bin", replaces=[("voxels", "velodyne")]),
            M.LABEL_LIDAR: path_solvers.ReplacePathSolver(
                M.LABEL_LIDAR.value, "voxels", "bin", replaces=[("voxels", "labels"), (".bin", ".label")]),
            M.LABEL_2D: path_solvers.ScaledPathSolver(M.LABEL_2D.value, "bev", "png"),
            M.POSE: path_solvers.ReplacePathSolver(M.POSE.value, "voxels", "bin", replaces=[("voxels", "velodyne")]),
        },
        root_dir="/home/vldluchinskiy/Datasets/SemanticKitty/dataset/sequences",
        dataset_type=enums.DatasetType.SEMANTIC_KITTI,
        mappings_path="cfg/mappings/semantic-kitti-navio.yaml",
        multisweep=base.MultisweepConfig(timestamps=(0.5, 1.0, 0.0)),
    ),
    trainer=base.TrainerConfig(model_type=enums.ModelType.OCCRWKV_2D),
    dataloader=base.DataloaderConfig(train_batch_size=8),
    optimizer=base.OptimizerConfig(),
    scheduler=base.SchedulerConfig(),
    out_root="outputs",
)

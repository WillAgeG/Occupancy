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
            M.LABEL_VOXELS: path_solvers.ScaledPathSolver(M.LABEL_VOXELS.value, "voxel_labels_fixed", "label"),
            M.INVALID: path_solvers.ScaledPathSolver(M.INVALID.value, "voxel_labels_fixed", "invalid")
        },
        root_dir="/home/vldluchinskiy/Datasets/SemanticKitty/dataset/sequences",
        dataset_type=enums.DatasetType.SEMANTIC_KITTI,
        mappings_path="cfg/mappings/semantic-kitti.yaml",
        nbr_classes=20,
    ),
    trainer=base.TrainerConfig(model_type=enums.ModelType.LMSCNET),
    dataloader=base.DataloaderConfig(train_batch_size=8),
    optimizer=base.OptimizerConfig(),
    scheduler=base.SchedulerConfig(),
    out_root="outputs",
)

from src.common import base_config as base
from src.common import enums
from src.data import path_solvers

M = enums.Modality

cfg = base.ExperimentConfig(
    dataset=base.DatasetConfig(
        augmentation=base.AugmentationConfig(),
        path_solvers={
            M.VOXELS: path_solvers.SimplePathSolver(M.VOXELS.value, "voxels", "bin"),
            M.LIDAR: path_solvers.ReplacePathSolver(
                M.LIDAR.value,
                "voxels",
                "bin",
                replaces=[("voxels", "lidar"), (".bin", ".pcd")],
            ),
            M.LABEL_2D: path_solvers.ScaledPathSolver(M.LABEL_2D.value, "bev_ms_3", "png"),
            M.LABEL_LIDAR: path_solvers.ReplacePathSolver(
                M.LABEL_LIDAR.value, "voxels", "bin", replaces=[("voxels", "label"), (".bin", ".json")]
            ),
            M.POSE: path_solvers.ReplacePathSolver(
                M.POSE.value, "voxels", "bin", replaces=[("voxels", "meta"), (".bin", ".json")]
            ),
        },
        root_dir="/data/navio_occ/",
        dataset_type=enums.DatasetType.NAVIO,
        grid=(0.0, -25.6, -0.8, 51.2, 25.6, 5.6),
        mappings_path="cfg/mappings/navio.yaml",
        multisweep=base.MultisweepConfig(timestamps=(0.5, 1.0, 0.0)),
    ),
    trainer=base.TrainerConfig(model_type=enums.ModelType.OCCRWKV_2D),
    dataloader=base.DataloaderConfig(train_batch_size=4, val_batch_size=1),
    optimizer=base.OptimizerConfig(),
    scheduler=base.SchedulerConfig(),
    out_root="outputs",
    weights_path="weights/rwkv_navio_online.pth",
)

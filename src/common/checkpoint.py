import dataclasses
import os
import pathlib
import shutil
from glob import glob

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel

from src.common import clearml_utils, ddp_utils
from src.common.io_tools import create_directory


@dataclasses.dataclass
class BestStats:
    best_loss: float = float("inf")
    best_metric: float = float("-inf")


def load(model, optimizer, scheduler, resume, path, logger):
    """
    Load checkpoint file
    """

    # If not resume, initialize model and return everything as it is
    if not resume:
        logger.info("=> No checkpoint. Initializing model from scratch")
        model.weights_init()
        epoch = 1
        best_stats = BestStats()
        return model, optimizer, scheduler, epoch, best_stats

    # If resume, check that path exists and load everything to return
    file_path = os.path.join(path, "ckpt", "last.pth")
    assert pathlib.Path(file_path).is_file(), f"=> No checkpoint found at {path}"
    checkpoint = torch.load(file_path)
    epoch = checkpoint.pop("startEpoch")
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model.module.load_state_dict(checkpoint.pop("model"))
    else:
        model.load_state_dict(checkpoint.pop("model"))
    optimizer.load_state_dict(checkpoint.pop("optimizer"))
    scheduler.load_state_dict(checkpoint.pop("scheduler"))
    best_stats = BestStats(**checkpoint.pop("bestStats"))
    logger.info("=> Continuing training routine. Checkpoint loaded at %s", file_path)
    return model, optimizer, scheduler, epoch, best_stats


def load_model(model, filepath, logger):
    """
    Load checkpoint file safely, skipping mismatched keys.
    """
    assert pathlib.Path(filepath).is_file(), f"=> No file found at {filepath}"
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

    state = checkpoint.pop("model", checkpoint)

    # Handle wrapped models
    model_to_load = model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model
    model_state = model_to_load.state_dict()

    # Filter out incompatible keys
    filtered_state = {}
    for k, v in state.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                logger.warning(f"Skipping (shape mismatch): {k} {v.shape} vs {model_state[k].shape}")
        else:
            logger.warning("Skipping (missing key in model): %s", k)

    # Update compatible parameters only
    model_state.update(filtered_state)
    model_to_load.load_state_dict(model_state, strict=False)

    logger.info("=> Model loaded (compatible layers only) from %s", filepath)
    return model


def save(dir_path, filename: str, model, optimizer, scheduler, epoch, best_stats):
    """
    Save checkpoint file
    """

    create_directory(dir_path)

    assert filename.endswith((".pth", ".pt")), "Invalid suffix for checkpoint. .pth or .pt was expected."

    weights_fpath = os.path.join(dir_path, filename)

    torch.save(
        {
            "startEpoch": epoch + 1,  # To start on next epoch when loading the dict...
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "bestStats": dataclasses.asdict(best_stats),
        },
        weights_fpath,
    )

    return weights_fpath


def save_logits(overall_scores: dict[str : torch.Tensor], cfg_out_path: pathlib.Path):
    rank = dist.get_rank() if ddp_utils.is_ddp() else 0
    logits_dir = cfg_out_path / "logits" / f"rank{rank}"
    logits_dir.mkdir(parents=True, exist_ok=True)

    for filepath, tensor in overall_scores.items():
        filename = pathlib.Path(filepath).with_suffix(".npy").name
        path = logits_dir / filename
        tensor_np = tensor.cpu().numpy()
        np.save(path, tensor_np)

    if ddp_utils.is_ddp():
        dist.barrier()

    if ddp_utils.is_primary():

        task = clearml_utils.get_task()
        if task is not None:
            task.upload_artifact(
                name="logits",
                artifact_object=str(cfg_out_path / "logits")
            )

    if ddp_utils.is_ddp():
        dist.barrier()

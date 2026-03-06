import argparse
import logging
import os
import pathlib
import sys

import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from src.common import checkpoint, clearml_utils, config_utils, downloader, enums, visualization
from src.common.dataset import get_dataset
from src.common.ddp_utils import init_distributed, is_primary, unwrap
from src.common.io_tools import dict_to
from src.common.logger import DummySummaryWriter, get_logger
from src.common.metrics import Metrics
from src.common.model import get_model
from src.common.optimizer import build_optimizer, build_scheduler
from src.common.seed import seed_all

M = enums.Modality


def parse_args():
    parser = argparse.ArgumentParser(description="LMSCNet training")
    parser.add_argument(
        "--cfg",
        default="",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--dset_root",
        dest="dataset_root",
        default=None,
        metavar="DATASET",
        help="path to dataset root folder",
        type=str,
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Pass a path to a folder with artifacts, if you want to resume",
        type=str,
    )
    parser.add_argument(
        "--clearml",
        action="store_true",
        help="Whether to use clearml when training",
    )
    parser.add_argument(
        "--name",
        default="NavioOccupancy",
        type=str,
        help="Name of ClearML experiment",
    )
    return parser.parse_args()


def train(model, optimizer, scheduler, dataset, cfg, start_epoch, logger, cml_logger, tbwriter, best_stats):
    """
    Train a model using the PyTorch Module API.
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - scheduler: Scheduler for learning rate decay if used
    - dataset: The dataset to load files
    - _cfg: The configuration dictionary read from config file
    - start_epoch: The epoch at which start the training (checkpoint)
    - logger: The logger to save info
    - tbwriter: The tensorboard writer to save plots
    Returns: Nothing, but prints model accuracies during training.
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Moving optimizer and model to used device
    model = model.to(device=device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    dset = dataset["train"]

    nbr_epochs = cfg.trainer.num_epochs

    # Defining metrics class and initializing them..
    metrics = Metrics(cfg.dataset.nbr_classes, len(dset), unwrap(model).get_scales())
    metrics.reset_evaluator()
    metrics.losses_track.set_validation_losses(unwrap(model).get_validation_loss_keys())
    metrics.losses_track.set_train_losses(unwrap(model).get_train_loss_keys())

    for epoch in range(start_epoch, nbr_epochs + 1):
        logger.info("=> =============== Epoch [%s/%s] ===============", epoch, nbr_epochs)
        logger.info(f"=> Reminder - Output of routine on {cfg.out_path}")

        # Print learning rate
        # for param_group in optimizer.param_groups:
        logger.info(f"=> Learning rate: {scheduler.get_lr()[0]}")

        if hasattr(dataset["train"], "sampler") and isinstance(dataset["train"].sampler, DistributedSampler):
            dataset["train"].sampler.set_epoch(epoch)

        model.train()  # put model to training mode

        # for t, (data, indices) in enumerate(dataset['train']):
        for t, (data_raw, _indices) in enumerate(dset):
            data = dict_to(data_raw, device)

            scores, loss = model(data)

            # Zero out the gradients.
            optimizer.zero_grad()
            # Backward pass: gradient of loss wr. each model parameter.
            loss["total"].backward()
            # update parameters of model by gradients.
            optimizer.step()

            if cfg.scheduler.frequency == "iteration":
                scheduler.step()

            # Updating batch losses to then get mean for epoch loss
            metrics.losses_track.update_train_losses(loss)

            if (t + 1) % cfg.trainer.train_summary_period == 0:
                loss_print = (
                    f"=> Epoch [{epoch}/{nbr_epochs}], Iteration [{t + 1}/{len(dset)}], "
                    f"Learn Rate: {scheduler.get_lr()[0]}, Train Losses: "
                )
                for key in loss:
                    loss_print += f"{key} = {loss[key]:.6f},  "
                logger.info(loss_print[:-3])

            metrics.add_batch(prediction=scores, target=unwrap(model).get_target(data))

        train_iteration_counts = metrics.losses_track.train_iteration_counts
        for l_key in metrics.losses_track.train_losses:
            tbwriter.add_scalar(
                f"train_loss_epoch/{l_key}",
                metrics.losses_track.train_losses[l_key].item() / train_iteration_counts,
                epoch - 1,
            )
        tbwriter.add_scalar("lr/lr", scheduler.get_lr()[0], epoch - 1)

        epoch_loss = metrics.losses_track.train_losses["total"] / train_iteration_counts

        for scale in metrics.evaluator:
            tbwriter.add_scalar(f"train_performance/mIoU_{scale}", metrics.get_semantics_miou(scale).item(), epoch - 1)
            tbwriter.add_scalar(f"train_performance/IoU_{scale}", metrics.get_occupancy_iou(scale).item(), epoch - 1)

        logger.info("=> [Epoch %s - Total Train Loss = %s]", epoch, epoch_loss)
        for scale in metrics.evaluator:
            loss_scale = metrics.losses_track.train_losses[f"semantic_{scale}"].item() / train_iteration_counts
            logger.info(
                f"=> [Epoch {epoch} - Scale {scale}: Loss = {loss_scale:.6f}"
                f" - mIoU = {metrics.get_semantics_miou(scale).item():.6f}"
                f" - IoU = {metrics.get_occupancy_iou(scale).item():.6f}"
                f" - P = {metrics.get_occupancy_precision(scale).item():.6f}"
                f" - R = {metrics.get_occupancy_recall(scale).item():.6f}"
                f" - F1 = {metrics.get_occupancy_f1(scale).item():.6f}]"
            )

        logger.info("=> Epoch %s - Training set class-wise IoU:", epoch)
        for i in range(1, metrics.nbr_classes):
            class_name = dset.dataset.dataset_config["labels"][dset.dataset.dataset_config["learning_map_inv"][i]]
            class_score = metrics.evaluator["1_1"].get_iou()[1][i]
            logger.info(f"    => IoU {class_name}: {class_score:.6f}")

        # Reset evaluator for validation...
        metrics.reset_evaluator()

        checkpoint_info = validate(model, dataset["val"], cfg, epoch, logger, cml_logger, tbwriter, metrics, best_stats)

        # Reset evaluator and losses for next epoch...
        metrics.reset_evaluator()
        metrics.losses_track.restart_train_losses()
        metrics.losses_track.restart_validation_losses()

        if cfg.scheduler.frequency == "epoch":
            scheduler.step()

        if is_primary():
            # Save checkpoints
            for k in checkpoint_info:
                checkpoint_dir = os.path.join(cfg.out_path, "ckpt")
                checkpoint.save(checkpoint_dir, k + ".pth", unwrap(model), optimizer, scheduler, epoch, best_stats)

    return metrics.best_metric_record


def validate(model, dset, cfg, epoch, logger, cml_logger, tbwriter, metrics, best_stats):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    nbr_epochs = cfg.trainer.num_epochs

    logger.info("=> Passing the network on the validation set...")

    model.eval()

    cfg_path = cfg.dataset.mappings_path
    with pathlib.Path(cfg_path).open() as f:
        mappings = yaml.safe_load(f)

    seleceted_ids = set(range(0, len(dset.dataset), cfg.trainer.visualization_interval))

    with torch.no_grad():
        for t, (data_raw, indices) in enumerate(dset):
            data = dict_to(data_raw, device)

            scores, loss = model(data)
            batch_size = scores["pred_semantic_1_1"].shape[0]

            if f"{M.LABEL_2D.value}_1_1" in data:
                for j in range(batch_size):
                    global_idx = indices[j]
                    if global_idx in seleceted_ids:
                        target = data[f"{M.LABEL_2D.value}_1_1"].data.cpu().numpy()[j]
                        score = torch.argmax(scores["pred_semantic_1_1"], dim=1).data.cpu().numpy()[j]

                        viz = visualization.visualize([score, target], mappings, ["Pred", "GT"])
                        input_filepath = dset.dataset.filepaths[M.VOXELS.value][global_idx]
                        visualization.save_visualization(viz, input_filepath, "output", cml_logger, epoch)

            # Updating batch losses to then get mean for epoch loss
            metrics.losses_track.update_validaiton_losses(loss)

            if (t + 1) % cfg.trainer.val_summary_period == 0:
                loss_print = f"=> Epoch [{epoch}/{nbr_epochs}], Iteration [{t + 1}/{len(dset)}], Train Losses: "
                for key in loss:
                    loss_print += f"{key} = {loss[key]:.6f},  "
                logger.info(loss_print[:-3])

            metrics.add_batch(prediction=scores, target=unwrap(model).get_target(data))

        validation_iteration_counts = metrics.losses_track.validation_iteration_counts
        train_iteration_counts = metrics.losses_track.train_iteration_counts

        for l_key in metrics.losses_track.validation_losses:
            tbwriter.add_scalar(
                f"validation_loss_epoch/{l_key}",
                metrics.losses_track.validation_losses[l_key].item() / validation_iteration_counts,
                epoch - 1,
            )

        epoch_loss = metrics.losses_track.validation_losses["total"] / validation_iteration_counts

        for scale in metrics.evaluator:
            tbwriter.add_scalar(
                f"validation_performance/mIoU_{scale}", metrics.get_semantics_miou(scale).item(), epoch - 1
            )
            tbwriter.add_scalar(
                f"validation_performance/IoU_{scale}", metrics.get_occupancy_iou(scale).item(), epoch - 1
            )

        logger.info("=> [Epoch %s - Total Validation Loss = %s]", epoch, epoch_loss)
        for scale in metrics.evaluator:
            loss_scale = metrics.losses_track.validation_losses[f"semantic_{scale}"].item() / train_iteration_counts
            logger.info(
                f"=> [Epoch {epoch} - Scale {scale}: Loss = {loss_scale:.6f}"
                f" - mIoU = {metrics.get_semantics_miou(scale).item():.6f}"
                f" - IoU = {metrics.get_occupancy_iou(scale).item():.6f}"
                f" - P = {metrics.get_occupancy_precision(scale).item():.6f}"
                f" - R = {metrics.get_occupancy_recall(scale).item():.6f}"
                f" - F1 = {metrics.get_occupancy_f1(scale).item():.6f}]"
            )

        logger.info("=> Epoch %s - Validation set class-wise IoU:", epoch)
        for i in range(1, metrics.nbr_classes):
            class_name = dset.dataset.dataset_config["labels"][dset.dataset.dataset_config["learning_map_inv"][i]]
            class_score = metrics.evaluator["1_1"].get_iou()[1][i]
            logger.info(f"    => {class_name}: {class_score:.6f}")
            tbwriter.add_scalar(f"mIoU_per_class/{class_name}", class_score, epoch - 1)
        tbwriter.add_scalar("mIoU_per_class/overall", metrics.get_semantics_miou("1_1").item(), epoch - 1)

        checkpoint_info = {}

        if epoch_loss < best_stats.best_loss:
            logger.info(f"=> Best loss on validation set encountered: ({epoch_loss} < {best_stats.best_loss})")
            best_stats.best_loss = epoch_loss.item()
            checkpoint_info["best-loss"] = "BEST_LOSS"

        mIoU_1_1 = metrics.get_semantics_miou("1_1")
        IoU_1_1 = metrics.get_occupancy_iou("1_1")
        if mIoU_1_1 > best_stats.best_metric:
            logger.info(f"=> Best metric on validation set encountered: ({mIoU_1_1} > {best_stats.best_metric})")
            best_stats.best_metric = mIoU_1_1.item()
            checkpoint_info["best-metric"] = "BEST_METRIC"
            metrics.update_best_metric_record(mIoU_1_1, IoU_1_1, epoch_loss.item(), epoch)

        checkpoint_info["last"] = "LAST"

        return checkpoint_info


def main():
    # https://github.com/pytorch/pytorch/issues/27588
    torch.backends.cudnn.enabled = False
    is_ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    seed_all(0 + rank)

    args = parse_args()

    is_remote = clearml_utils.is_running_on_clearml_agent()
    use_clearml = args.clearml or is_remote
    if use_clearml:
        task, cml_logger = clearml_utils.init_task(name=args.name)
        cfg_f = task.connect_configuration(args.cfg)
    else:
        task, cml_logger = None, None
        cfg_f = args.cfg

    dataset_f = args.dataset_root

    cfg = config_utils.load_config(cfg_f)
    if args.resume is not None:
        resume = True
        out_path = args.resume
        cfg.out_path = pathlib.Path(out_path)
    else:
        if is_primary():
            cfg.out_path.mkdir(parents=True, exist_ok=False)
        resume = False

    if is_ddp:
        cfg.optimizer.lr *= world_size

    # Replace dataset path in config file by the one passed by argument
    if dataset_f is not None:
        cfg.dataset.root_dir = dataset_f

    if is_primary():
        # Create writer for Tensorboard
        tbwriter = SummaryWriter(log_dir=os.path.join(cfg.out_path, "metrics"))

        # Setting the logger to print statements and also save them into logs file
        logger = get_logger(cfg.out_path, "logs_train.log")

        # Copy cfg to the experiment dir
        config_utils.copy_config_to_experiment(cfg_f, cfg.out_path)

        # Copy full cfg as a json dict
        json_cfg_f = config_utils.copy_json_config_to_experiment(cfg_f, cfg.out_path)
        if use_clearml:
            task.connect_configuration(json_cfg_f, name="Json all")

        # Download data if needed
        downloader.download_data(cfg.download)
    else:
        logger = logging.getLogger("dummy")
        tbwriter = DummySummaryWriter()

    logger.info("============ Training routine: %s ============\n", cfg_f)
    dataset = get_dataset(cfg, is_ddp=is_ddp)

    logger.info("=> Loading network architecture...")
    base_model = get_model(cfg, dataset["train"].dataset)

    base_model = base_model.to(device)

    logger.info("=> Loading optimizer...")
    optimizer = build_optimizer(cfg.optimizer, base_model)
    scheduler = build_scheduler(cfg.scheduler, optimizer)

    base_model, optimizer, scheduler, epoch, best_stats = checkpoint.load(
        base_model, optimizer, scheduler, resume, cfg.out_path, logger
    )

    if (
        not args.resume
        and cfg.trainer.model_type.value.startswith("occrwkv")
        and is_primary()
        and cfg.weights_path is None
    ):
        checkpoint.load_model(base_model, "weights/OccRWKV_pretrain.pth", logger)
    elif not args.resume and is_primary() and cfg.weights_path is not None:
        checkpoint.load_model(base_model, cfg.weights_path, logger)

    if dist.is_initialized():
        dist.barrier()

    if is_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        model = base_model

    if dist.is_initialized():
        dist.barrier()

    if is_ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        model = base_model

    best_record = train(model, optimizer, scheduler, dataset, cfg, epoch, logger, cml_logger, tbwriter, best_stats)

    logger.info("=> ============ Network trained - all epochs passed... ============")

    logger.info(
        f"=> [Best performance: Epoch {best_record['epoch']} - mIoU = {best_record['mIoU']} - IoU {best_record['IoU']}]"
    )

    logger.info("=> Writing config file in output folder - deleting from config files folder")

    logger.info("=> Training routine completed...")

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

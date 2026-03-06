import argparse
import os
import pathlib
import sys

import clearml
import torch
import tqdm
import yaml
from torch import nn

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from src.common import checkpoint, clearml_utils, config_utils, downloader, enums, visualization
from src.common.dataset import get_dataset
from src.common.io_tools import dict_to
from src.common.logger import get_logger
from src.common.metrics import Metrics
from src.common.model import get_model
from src.common.seed import seed_all
from src.test import swap_channels

M = enums.Modality


def parse_args():
    parser = argparse.ArgumentParser(description="LMSCNet validating")
    parser.add_argument(
        "--weights",
        default="cac9ab1a30e74e79b7b78bce32502dce",
        help="path to a .pth checkpoint",
        type=str,
    )
    parser.add_argument(
        "--dset_root",
        default="",
        help="path to dataset root folder",
        type=str,
    )
    parser.add_argument(
        "--out_path",
        default="output",
        help="path to folder where predictions will be saved",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        default="",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--swap_c",
        action="store_true",
        help="Whether to switch road and object channels. For backward compatability purposes",
    )
    parser.add_argument(
        "--clearml",
        action="store_true",
        help="Whether to use clearml when training",
    )
    return parser.parse_args()


def validate(model, dset, cfg, logger, cml_logger, metrics, mappings, args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Moving optimizer and model to used device
    model = model.to(device=device)

    logger.info("=> Passing the network on the validation set...")

    model.eval()
    overall_scores = {}

    with torch.no_grad():

        for t, (data_raw, _indices) in tqdm.tqdm(enumerate(dset)):

            data = dict_to(data_raw, device)

            scores, loss = model(data)
            scores_copy = scores["pred_semantic_1_1"].clone()
            batch_size = scores["pred_semantic_1_1"].shape[0]

            if args.swap_c:
                swap_channels(scores, "pred_semantic_1_1")

            if f"{M.LABEL_2D.value}_1_1" in data:
                target = data[f"{M.LABEL_2D.value}_1_1"].data.cpu().numpy()[0]
                score = torch.argmax(scores["pred_semantic_1_1"], dim=1).data.cpu().numpy()[0]

                viz = visualization.visualize([score, target], mappings, ["Pred", "GT"])
                input_filepath = dset.dataset.filepaths[M.VOXELS.value][t * batch_size]
                visualization.save_visualization(viz, input_filepath, args.out_path, cml_logger)

            # Updating batch losses to then get mean for epoch loss
            metrics.losses_track.update_validaiton_losses(loss)

            if (t + 1) % cfg.trainer.val_summary_period == 0:
                loss_print = f"=> Iteration [{t + 1}/{len(dset)}], Train Losses: "
                for key in loss:
                    loss_print += f"{key} = {loss[key]:.6f},  "
                logger.info(loss_print[:-3])

            metrics.add_batch(prediction=scores, target=model.get_target(data))

            for j in range(batch_size):
                idx = t * batch_size + j
                filepath = dset.dataset.filepaths[M.VOXELS.value][idx]
                overall_scores[filepath] = scores_copy[j]

        iteration_counts = metrics.losses_track.validation_iteration_counts
        epoch_loss = metrics.losses_track.validation_losses["total"] / iteration_counts

        logger.info("=> [Total Validation Loss = %s]", epoch_loss)
        for scale in metrics.evaluator:
            loss_scale = metrics.losses_track.validation_losses[f"semantic_{scale}"].item() / iteration_counts
            logger.info(
                f"=> [Scale {scale}: Loss = {loss_scale:.6f} - mIoU = {metrics.get_semantics_miou(scale).item():.6f}"
                f" - IoU = {metrics.get_occupancy_iou(scale).item():.6f}"
                f" - P = {metrics.get_occupancy_precision(scale).item():.6f}"
                f" - R = {metrics.get_occupancy_recall(scale).item():.6f}"
                f" - F1 = {metrics.get_occupancy_f1(scale).item():.6f}]"
            )

        logger.info("=> Training set class-wise IoU:")
        for i in range(1, metrics.nbr_classes):
            class_name = dset.dataset.dataset_config["labels"][dset.dataset.dataset_config["learning_map_inv"][i]]
            class_score = metrics.evaluator["1_1"].get_iou()[1][i]
            logger.info(f"    => IoU {class_name}: {class_score:.6f}")

        checkpoint.save_logits(overall_scores, cfg.out_path)

        return


def main() -> None:

    # https://github.com/pytorch/pytorch/issues/27588
    torch.backends.cudnn.enabled = False

    seed_all(0)

    args = parse_args()

    is_remote = clearml_utils.is_running_on_clearml_agent()
    use_clearml = args.clearml or is_remote
    if use_clearml:
        task, cml_logger = clearml_utils.init_task(task_type=clearml.TaskTypes.testing, tags=["val"])
        cfg_f = task.connect_configuration(args.cfg)
    else:
        cfg_f = args.cfg
        cml_logger = None

    if is_remote:
        weights_f = clearml_utils.get_remote_weights(args.weights)
    else:
        weights_f = pathlib.Path(args.weights)

    dataset_f = args.dset_root
    cfg_f = args.cfg

    assert weights_f.is_file(), f"=> No file found at {weights_f}"

    cfg = config_utils.load_config(cfg_f)
    run_name = (f"{cfg.trainer.model_type}_{cfg.dataset.dataset_type.value}_{weights_f.stem}")
    cfg.out_path = pathlib.Path(cfg.out_root) / run_name
    cfg.out_path.mkdir(parents=True, exist_ok=True)

    if len(dataset_f) != 0:
        cfg.dataset.root_dir = dataset_f

    # Download data if needed
    downloader.download_data(cfg.download)

    cfg_path = cfg.dataset.mappings_path
    with pathlib.Path(cfg_path).open() as f:
        mappings = yaml.safe_load(f)

    # Setting the logger to print statements and also save them into logs file
    logger = get_logger(cfg.out_path, "logs_val.log")

    logger.info("============ Validation weights: %s ============\n", weights_f)
    dataset = get_dataset(cfg)

    logger.info("=> Loading network architecture...")
    model = get_model(cfg, dataset["train"].dataset)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.module

    logger.info("=> Loading network weights...")
    model = checkpoint.load_model(model, weights_f, logger)

    nbr_iterations = len(dataset["val"])
    metrics = Metrics(cfg.dataset.nbr_classes, nbr_iterations, model.get_scales())
    metrics.reset_evaluator()
    metrics.losses_track.set_validation_losses(model.get_validation_loss_keys())
    metrics.losses_track.set_train_losses(model.get_train_loss_keys())

    validate(model, dataset["val"], cfg, logger, cml_logger, metrics, mappings, args)

    if use_clearml:
        task.upload_artifact(
            name="validation_images",
            artifact_object=args.out_path,
        )

    logger.info("=> ============ Network Validation Done ============")


if __name__ == "__main__":
    main()

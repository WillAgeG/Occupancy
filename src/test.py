import argparse
import os
import pathlib
import sys

import torch
import tqdm
import yaml
from torch import nn

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from project2d.lib.core import projection
from project2d.scripts import project_dataset
from src.common import checkpoint, config_utils, enums, visualization
from src.common.dataset import get_dataset
from src.common.io_tools import dict_to
from src.common.logger import get_logger
from src.common.model import get_model
from src.common.seed import seed_all

M = enums.Modality


def parse_args():
    parser = argparse.ArgumentParser(description="LMSCNet validating")
    parser.add_argument(
        "--weights",
        dest="weights_file",
        default="",
        metavar="FILE",
        help="path to folder where model.pth file is",
        type=str,
    )
    parser.add_argument(
        "--dset_root",
        dest="dataset_root",
        default="",
        metavar="DATASET",
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
        "--title",
        default="",
        help="name of a title for every image",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--3d",
        action="store_true",
        dest="model_is_3d",
        help="Whether the model returns 3d output"
    )
    parser.add_argument(
        "--swap_c",
        action="store_true",
        help="Whether to switch road and object channels. For backward compatability purposes",

    )
    return parser.parse_args()


def swap_channels(scores, key):
    a1 = scores[key][:, 1].clone()
    a2 = scores[key][:, 2].clone()
    scores[key][:, 1], scores[key][:, 2] = a2, a1


def project(voxel_labels, color_lut, car_height_m=3.0, voxel_size_z=0.2, unlabeled_class=0):
    bev_classes = projection.project_to_bev(voxel_labels, car_height_m, voxel_size_z, unlabeled_class)
    bev_classes = projection.check_max_class(bev_classes, color_lut.shape[0] - 1, unlabeled_class)

    return bev_classes


def test(model, dset, logger, mappings, args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Moving optimizer and model to used device
    model = model.to(device=device)
    logger.info("=> Passing the network on the test set...")
    model.eval()

    _, color_lut = project_dataset.build_mappings(mappings)

    with torch.no_grad():

        for t, (data_raw, _indices) in tqdm.tqdm(enumerate(dset)):

            data = dict_to(data_raw, device)

            scores = model(data)
            batch_size = scores["pred_semantic_1_1"].shape[0]

            if args.swap_c:
                swap_channels(scores, "pred_semantic_1_1")

            score = torch.argmax(scores["pred_semantic_1_1"], dim=1).data.cpu().numpy()[0]

            if args.model_is_3d:
                score = project(score, color_lut)

            if len(args.title) > 0:
                viz = visualization.visualize([score], mappings, [args.title])
            else:
                viz = visualization.visualize([score], mappings)
            input_filepath = dset.dataset.filepaths[M.VOXELS.value][t * batch_size]
            visualization.save_visualization(viz, input_filepath, args.out_path)

        return


def main():

    # https://github.com/pytorch/pytorch/issues/27588
    torch.backends.cudnn.enabled = False

    seed_all(0)

    args = parse_args()

    weights_f = args.weights_file
    dataset_f = args.dataset_root
    out_path_root = args.out_path
    cfg_f = args.config_file

    assert pathlib.Path(weights_f).is_file(), "=> No file found at {}"

    cfg = config_utils.load_config(cfg_f)

    if len(dataset_f) != 0:
        cfg.dataset.root_dir = dataset_f

    # Setting the logger to print statements and also save them into logs file
    logger = get_logger(out_path_root, "logs_test.log")

    cfg_path = cfg.dataset.mappings_path
    with pathlib.Path(cfg_path).open() as f:
        mappings = yaml.safe_load(f)

    logger.info("============ Test weights: %s ============\n", weights_f)
    dataset = get_dataset(cfg, create_test=True)["test"]

    logger.info("=> Loading network architecture...")
    model = get_model(cfg, dataset.dataset)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.module

    logger.info("=> Loading network weights...")
    model = checkpoint.load_model(model, weights_f, logger)

    test(model, dataset, logger, mappings, args)

    logger.info("=> ============ Network Test Done ============")


if __name__ == "__main__":
    main()

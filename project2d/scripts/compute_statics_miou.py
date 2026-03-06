"""Computes miou between collected statics from sda and the one from human mark up"""
import argparse
import logging
import os
import pathlib
import re
import shutil

import cv2
import numpy as np
import torch

from lib.common import data_utils, file_utils, logging_utils
from lib.readers import pcd_reader

REFERENCE_FRAME_NAME = "000005"
# Prior knowledge regarding dataset structure. In our markup dataset there is only one frame with the occupancy markup
# for each scene. It is usually for frame 000046. In case of the dataset expected here it was processed to an index of
# 000005


class IoUEval:
    def __init__(self, n_classes: int, ignore: tuple[int, ...] = (), only_present_in_mean: bool = True) -> None:
        self.n_classes = n_classes
        self.ignore = np.array(ignore, dtype=np.int64).reshape(-1)
        self.include = np.isin(np.arange(self.n_classes),
                               np.array(list(set(range(self.n_classes)) - set(ignore)), dtype=np.int64))
        self.only_present_in_mean = only_present_in_mean
        self.reset()

    @property
    def num_classes(self) -> int:
        return self.n_classes

    def reset(self) -> None:
        self.conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def add_batch(self, preds: np.ndarray | torch.Tensor, targets: np.ndarray | torch.Tensor) -> None:
        # x: preds, y: targets
        preds = np.asarray(preds)
        targets = np.asarray(targets)
        assert preds.shape == targets.shape, (preds.shape, targets.shape)

        x_row = preds.reshape(-1).astype(np.int64)
        y_row = targets.reshape(-1).astype(np.int64)

        # Optional: clamp/validate indices
        valid = (x_row >= 0) & (x_row < self.n_classes) & (y_row >= 0) & (y_row < self.n_classes)
        if not np.all(valid):
            x_row = x_row[valid]
            y_row = y_row[valid]

        np.add.at(self.conf_matrix, (x_row, y_row), 1)

    def _conf_after_ignore(self):
        conf = self.get_confusion()
        if self.ignore.size > 0:
            conf[:, self.ignore] = 0  # drop pixels whose GT is ignored
        return conf

    def get_stats(self):
        conf = self._conf_after_ignore()
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def get_iou(self) -> tuple[np.float64, np.ndarray]:
        """
        Returns np.float64 mIoU value and an array with iou for each class
        """
        tp, fp, fn = self.get_stats()
        union = tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, tp / union, np.nan)

        # mean over include
        if self.only_present_in_mean:
            present = (union > 0)
            mean_mask = self.include & present
        else:
            mean_mask = self.include

        iou_mean = np.nanmean(iou[mean_mask]) if np.any(mean_mask) else float("nan")

        # return per-class IoU with NaNs for absent classes (more informative than hard 0)
        return iou_mean, iou

    def get_pixel_accuracy(self):
        # Overall pixel accuracy after ignoring GT columns
        conf = self._conf_after_ignore()
        correct = np.trace(conf)
        total = conf.sum()
        return float(correct) / (float(total) + 1e-15)

    def get_mean_class_accuracy(self):
        # Mean of per-class recall: tp / (tp + fn)
        tp, _, fn = self.get_stats()
        denom = tp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(denom > 0, tp / denom, np.nan)
        mask = self.include & (denom > 0)
        return np.nanmean(acc[mask]) if np.any(mask) else float("nan")

    def get_confusion(self):
        return self.conf_matrix.copy()


def process_statics_pred_for_kitti_grid(img_pred: np.ndarray) -> np.ndarray:
    h, w = img_pred.shape[:2]  # 512, 512
    img_pred = img_pred[h // 4:(3 * h) // 4, w // 2:]  # 256, 256, kitti grid
    img_pred = np.flip(img_pred, axis=0)  # flip for consistency with existing gt

    # Statics is black, make it white on black background, natural for prediction
    statics_pred = np.zeros_like(img_pred)
    statics_pred[img_pred == 0] = 1

    return statics_pred


def process_statics_gt(img_gt: np.ndarray) -> np.ndarray:
    # In GT static has a label of 2
    statics_gt = np.zeros_like(img_gt)
    statics_gt[img_gt == 2] = 1

    return statics_gt


def calculate_statics_miou(gt_root_dir: str, pred_root_dir: str) -> float:
    pred_root_dir = pathlib.Path(pred_root_dir)
    evaluator = IoUEval(n_classes=2, ignore=[0], only_present_in_mean=True)

    # Scene names should be the same, but there may be some missing in pred folder
    for scene in sorted(pred_root_dir.iterdir()):
        if not scene.is_dir():
            continue

        pred_dir = scene / "agg_static"
        png_files = [p for p in pred_dir.iterdir() if p.is_file() and p.suffix == ".png"]
        assert len(png_files) == 1, f"Expected 1 file in {pred_dir} but got {png_files}"

        pred_path = png_files[0]
        gt_path = pathlib.Path(gt_root_dir) / scene.name / "bev" / f"{REFERENCE_FRAME_NAME}_bev.png"

        img_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        statics_pred = process_statics_pred_for_kitti_grid(img_pred)
        statics_gt = process_statics_gt(img_gt)

        evaluator.add_batch(statics_pred, statics_gt)

    return float(evaluator.get_iou()[0])


def parse_scene_name(scene: str) -> str:
    scene = re.sub(r"^[0-9a-fA-F-]{36}_", "", scene)
    scene = scene.replace("__", "_")
    prefix = file_utils.scene_path_to_ride_id(scene)  # Extract kc and date
    return prefix


def filter_static_data(static_data_root: str, target_data_root: str, out_dir: str,  # noqa: PLR0914
                       logger: logging.Logger) -> None:
    """
    Due to specifics of our static (static objects) collection, there is a lot of info that is redundant
    for the miou computation. This functions search only viable frames based on the lidar timestamps in the gt marked
    up data
    """
    scenes = sorted(os.listdir(static_data_root))
    scene_map = {parse_scene_name(scene): scene for scene in scenes}

    folders_of_interest = ["meta", "blind_zones", "detects"]

    target_scenes = sorted(os.listdir(target_data_root))
    reader = pcd_reader.PCDReader()

    for scene in target_scenes:
        meta_path = pathlib.Path(target_data_root) / scene / "meta" / f"{REFERENCE_FRAME_NAME}.json"
        ts = reader.read_timestamp(meta_path)
        scene_prefix = parse_scene_name(scene)
        if scene_prefix not in scene_map:
            logger.info("Skipping scene %s", scene)
            continue
        source_scene = scene_map[scene_prefix]

        static_folder = pathlib.Path(static_data_root) / source_scene / "agg_static"
        static_path = data_utils.find_nearest_file(ts, static_folder, return_previous=True)

        dst_static_dir = pathlib.Path(out_dir) / scene / "agg_static"
        dst_static_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(static_path, dst_static_dir / pathlib.Path(static_path).name)

        frame_prefix = pathlib.Path(static_path).name[:10]  # frame_xxxx
        for folder in folders_of_interest:
            source_folder_path = pathlib.Path(static_data_root) / source_scene / folder
            matches = [p for p in source_folder_path.iterdir() if p.is_file() and p.name.startswith(frame_prefix)]
            assert len(matches) == 1, f"Expected exactly 1 file starting with {frame_prefix}, found {matches}"
            source_path = matches[0]

            dst_dir = pathlib.Path(out_dir) / scene / folder
            dst_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(source_path, dst_dir / source_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--static_data_root",
        type=str,
        default="/home/vldluchinskiy/Datasets/static_preds/new",
        help="Path to a folder with the raw statics data downloaded from ClearML task (S3)"
    )

    parser.add_argument(
        "--target_data_root",
        type=str,
        default="/home/vldluchinskiy/Datasets/navio-markup/processed/",
        help="Path to a folder with marked up data. "
             "Each subfolder is a scene containing one frame with base name 000005. "
             "See https://lakefs.cs.navio.auto/repositories/occupancy-datasets/objects?path=navio-markup%%2F&ref=main"
    )

    parser.add_argument(
        "--static_out_dir",
        type=str,
        default="/home/vldluchinskiy/Datasets/static_preds/temp",
        help="Path to a resulting folder with processed static. "
             "It will contain the same scenes as target_data_root, "
             "with the corresponding (closest timestamp) frame from static_data_root"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    data_root = args.static_data_root
    out_dir = args.static_out_dir
    target_data_root = args.target_data_root
    logger = logging_utils.get_logger()

    if pathlib.Path(out_dir).exists():
        shutil.rmtree(out_dir)

    filter_static_data(data_root, target_data_root, out_dir, logger)
    res = calculate_statics_miou(target_data_root, out_dir)
    logger.info(res)

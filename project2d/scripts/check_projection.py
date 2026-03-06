"""Checks the projection is correct for kitti data. Samples points directly from voxels and draws them on bev"""
# !/usr/bin/env python3
import matplotlib
import numpy as np

matplotlib.use("Agg")  # без GUI
import glob
import os
import pathlib

import cv2
import matplotlib.pyplot as plt
import yaml

from . import project_dataset


def check_objects(voxel_labels: np.array, bev_rgb: np.array, class_id: int = 1, num_samples: int = 200,
                  save_name: str = "bev_check.png") -> None:
    """
    Семплирует точки из данного класса и рисует их на проекции. Сделано это для проверки ориентации bev относительно
    вокселей
    """
    occ = np.argwhere(voxel_labels == class_id)  # (x,y,z) для объектов
    if len(occ) == 0:
        print(f"Нет вокселей класса {class_id} в кадре")
        return
    idx = np.random.choice(len(occ), size=min(num_samples, len(occ)), replace=False)
    chosen = occ[idx]
    xs, ys, zs = chosen[:, 0], chosen[:, 1], chosen[:, 2]

    plt.figure(figsize=(8, 8))
    plt.imshow(bev_rgb)
    plt.scatter(xs, ys, c="red", s=8, alpha=0.6, label="car voxels")
    plt.legend()
    plt.title("Соответствие автомобилей: voxels → BEV")
    plt.savefig(save_name, dpi=200)
    print(f"Saved {save_name}")


# ---------- MAIN ----------
if __name__ == "__main__":
    cfg_path = "config/semantic-kitti.yaml"
    with pathlib.Path(cfg_path).open() as f:
        config = yaml.safe_load(f)

    remap_array, color_lut = project_dataset.build_mappings(config)

    root_dir = "/home/vldluchinskiy/Datasets/Occupancy/SemanticKitty/dataset/sequences/00/voxels"
    files = sorted(glob.glob(os.path.join(root_dir, "*.label")))
    invalid_files = sorted(glob.glob(os.path.join(root_dir, "*.invalid")))
    if len(files) == 0:
        print("Нет .label файлов")
        exit()

    voxel_labels = project_dataset.load_voxel_label(files[0])
    voxel_occluded = project_dataset.load_invalid(invalid_files[0])

    # учитываем точки, которые ниоткуда не видны, возможно, этот шаг и не нужен.
    voxel_labels[np.isclose(voxel_occluded, 1)] = 251  # Специальный класс для invalid, 0 при проекции bev игнорируются

    remapped = project_dataset.remap_labels(voxel_labels, remap_array)

    _, bev_bgr = project_dataset.project(remapped, color_lut)
    bev_rgb = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2RGB)
    check_objects(remapped, bev_rgb)

"""Projects voxel labels of semantic kitti occupancy dataset onto 2d (bev)"""
# !/usr/bin/env python3
import argparse
import glob
import os
import pathlib

import cv2
import numpy as np
import tqdm
import yaml
from scipy.ndimage import maximum_filter

try:
    from project2d.lib import visualization
    from project2d.lib.core import multisweep, projection, voxelizer
    from project2d.lib.readers import kitti_reader
except ImportError:
    from lib.core import multisweep, projection, voxelizer
    from lib.readers import kitti_reader


def unpack(compressed):
  """ given a bit encoded voxel grid, make a normal voxel grid out of it.  """
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1

  return uncompressed


def load_voxel_label(path, shape=(256, 256, 32)):
    """Загрузка воксельной сетки SemanticKITTI (label или occupancy)."""
    arr = np.fromfile(path, dtype=np.uint16)
    return arr.reshape(shape)  # [W, D, H]


def load_invalid(path, shape=(256, 256, 32)):
    bin = np.fromfile(path, dtype=np.uint8)
    bin = unpack(bin)
    return bin.reshape(shape)


def load_input(path, shape=(256, 256, 32)):
    bin = np.fromfile(path, dtype=np.uint8)
    bin = unpack(bin)
    return bin.reshape(shape)


def save_voxel_label(path, voxel_label):
    voxel_label = voxel_label.astype(np.uint16)
    voxel_label.tofile(path)


def build_mappings(config):
    """
    Возвращает:
      - remap: numpy array, индексируемый оригинальным ID -> train-ID
      - color_lut: array [n_train_ids, 3] с цветами в RGB для train-ID
    """
    color_map = config["color_map"]
    learning_map = config["learning_map"]
    learning_map_inv = config["learning_map_inv"]

    # максимальный оригинальный id для безопасного индексирования
    max_orig = max(color_map.keys())

    # np.array для эффективного ремапа
    remap = np.zeros((max_orig + 1,), dtype=np.int32)
    for orig_id, train_id in learning_map.items():
        assert 0 <= orig_id <= max_orig, f"ID {orig_id} out of bounds"
        remap[orig_id] = train_id

    # максимальный train-id -- чтобы сформировать LUT
    max_train = max(learning_map_inv.keys())

    color_lut = np.zeros((max_train + 1, 3), dtype=np.uint8)
    for train_id, orig_id in learning_map_inv.items():
        bgr = color_map.get(orig_id, [0, 0, 0])
        color_lut[train_id] = bgr

    return remap, color_lut


def remap_labels(voxel_labels, remap_array):
    """
    Быстрое перемаппивание оригинальных ID в train-ID.
    Всё, что выходит за диапазон remap_array, считается 0.
    """
    max_orig = remap_array.shape[0] - 1
    # безопасно: значения > max_orig -> 0
    safe = np.where(voxel_labels <= max_orig, voxel_labels, 0)
    return remap_array[safe]


def project(voxel_labels, color_lut, car_height_m=3.0, voxel_size_z=0.2, unlabeled_class=0):
    bev_classes = projection.project_to_bev(voxel_labels, car_height_m, voxel_size_z, unlabeled_class)
    bev_classes = projection.check_max_class(bev_classes, color_lut.shape[0] - 1, unlabeled_class)

    bev = cv2.cvtColor(bev_classes.astype(np.uint8), cv2.COLOR_GRAY2BGR)  # для сохранения на диск
    bev_color = color_lut[bev_classes]

    return bev, bev_color


def filter_isolated_label(voxel_labels, target_label):
    """
    Зануляет все воксели с меткой target_label, если по оси Z
    (вверх/вниз) у соседей другая метка. Работает векторизованно.
    
    Аргументы:
      voxel_labels : np.ndarray — 3D массив меток
      target_label : int — метка, которую нужно фильтровать
    """
    mask = voxel_labels == target_label

    # Соседи по оси Z (предыдущий и следующий слой)
    prev_same = np.zeros_like(mask, dtype=bool)
    next_same = np.zeros_like(mask, dtype=bool)

    prev_same[:, :, 1:] = mask[:, :, :-1]   # слой ниже
    next_same[:, :, :-1] = mask[:, :, 1:]   # слой выше

    # Оставляем только те воксели, где есть хотя бы один сосед с той же меткой
    keep_mask = mask & (prev_same | next_same)

    # Все изолированные воксели с target_label зануляем
    voxel_labels[mask & (~keep_mask)] = 0

    return voxel_labels


def filter_moving_by_input(voxel_labels, voxel_input, labels=[252, 253, 254, 255, 256, 257, 258, 259], dilate_input=False):
    """
    Обнуляет воксели с moving-классами (252–259), если в input они отсутствуют.
    
    Аргументы:
      voxel_labels : np.ndarray — 3D массив меток (uint16)
      voxel_input  : np.ndarray — 3D массив входных вокселей (float32 или uint8)
      dilate_input : bool — флаг для расширения входной маски перед фильтрацией
    
    Возвращает:
      np.ndarray — обновлённый voxel_labels
    """
    # TODO: возможно стоит брать это из конфига
    voxel_labels_c = np.copy(voxel_labels)
    moving_classes = np.array(labels, dtype=np.uint16)
    mask_moving = np.isin(voxel_labels_c, moving_classes)

    if dilate_input:
        mask_input = maximum_filter(voxel_input > 0, size=(10, 2, 1))
        mask_no_input = ~mask_input
    else:
        mask_no_input = (voxel_input == 0)

    voxel_labels_c[mask_moving & mask_no_input] = 0
    return voxel_labels_c


def get_adjacent_filepaths(filepath: str | pathlib.Path, n: int = 2, prev=False) -> list[pathlib.Path]:
    """
    Возвращает список путей к n_prev предыдущим или последующим файлам.
    Сохраняет форматирование (ведущие нули) и расширение исходного файла.
    Пример:
        Path('/data/voxels/000005.bin') →
        [Path('/data/voxels/000003.bin'), Path('/data/voxels/000004.bin')]
    """
    path = pathlib.Path(filepath)
    name = path.stem      # '000005'
    ext = path.suffix     # '.bin'

    if not name.isdigit():
        raise ValueError(f"Имя файла '{name}' не является числовым.")

    num = int(name)
    width = len(name)

    if prev:
        paths = [
            path.with_name(f"{num - i:0{width}d}{ext}")
            for i in range(n, 0, -1)
            if num - i >= 0
        ]
    else:
        paths = [
        path.with_name(f"{num + i:0{width}d}{ext}")
            for i in range(1, n + 1)
        ]
    return paths


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/home/vldluchinskiy/Datasets/SemanticKitty/dataset/sequences", help="Path to semantic kitti sequences dir")
    ap.add_argument("--cfg", type=str, default="config/semantic-kitti-navio.yaml", help="Path to config file")
    ap.add_argument("--multisweep", action="store_true", help="Use multisweep clouds for dynamic objects filtering")
    ap.add_argument("--save_voxels", action="store_true", help="Whether to save voxels")

    args = ap.parse_args()

    cfg_path = args.cfg
    root_dir = args.data_root
    voxel_shape = (256, 256, 32)
    occluded_label = 251  # Специальный класс для occluded, 0 при проекции bev игнорируются

    with pathlib.Path(cfg_path).open() as f:
        config = yaml.safe_load(f)

    remap_array, color_lut = build_mappings(config)
    reader = kitti_reader.KittiCloudReader()

    # проход по всем sequence
    for seq in sorted(os.listdir(root_dir)):
        # Ищем файлы, создаём папки для результатов
        voxel_dir = os.path.join(root_dir, seq, "voxels")
        cloud_dir = os.path.join(root_dir, seq, "velodyne")
        if not pathlib.Path(voxel_dir).is_dir():
            continue
        save_seq_dir = os.path.join(root_dir, seq, "bev")
        pathlib.Path(save_seq_dir).mkdir(exist_ok=True, parents=True)
        save_seq_dir_color = os.path.join(root_dir, seq, "bev_color")
        pathlib.Path(save_seq_dir_color).mkdir(exist_ok=True, parents=True)
        save_seq_dir_voxel = os.path.join(root_dir, seq, "voxels")
        pathlib.Path(save_seq_dir_voxel).mkdir(exist_ok=True, parents=True)

        label_files = sorted(glob.glob(os.path.join(voxel_dir, "*.label")))
        input_files = sorted(glob.glob(os.path.join(voxel_dir, "*.bin")))
        cloud_files = [fp.replace("voxels", "velodyne") for fp in input_files]
        occluded_files = sorted(glob.glob(os.path.join(voxel_dir, "*.invalid")))
        print(f"Processing sequence {seq}: {len(label_files)} files -> {save_seq_dir}")

        for i, path in tqdm.tqdm(enumerate(label_files), desc=f"seq {seq}", unit="file"):
            if args.multisweep and i < len(label_files) - 1:
                cloud_path = cloud_files[i]
                pc_paths = get_adjacent_filepaths(cloud_path)
                pc_paths.append(cloud_path)

                clouds = [reader.read_cloud(p) for p in pc_paths]
                poses = [reader.read_pose(p) for p in pc_paths]

                cloud = multisweep.merge_pointclouds_to_timestamp(clouds, poses, reference_index=2)[:, :3]
                voxel_input = voxelizer.voxelize_points(cloud, coors_range=(0.0, -25.6, -2, 51.2, 25.6, 4.4)).astype(np.uint8)
                voxel_input_orig = load_input(input_files[i])
                voxel_input = np.logical_or(voxel_input, voxel_input_orig)

            else:
                voxel_input = load_input(input_files[i])

            fname = os.path.splitext(os.path.basename(path))[0]

            # загружаем
            voxel_labels = load_voxel_label(path, shape=voxel_shape)
            voxel_occluded = load_invalid(occluded_files[i])

            # убираем из лейблов движущиеся объекты, которых нет в текущем свипе
            voxel_labels = filter_moving_by_input(voxel_labels, voxel_input)

            # сохраняем воксели с отфильтрованной динамикой, если надо
            if args.save_voxels:
                save_path_voxel = os.path.join(save_seq_dir_voxel, f"{fname}.label")
                save_voxel_label(save_path_voxel, voxel_labels)

            # учитываем точки, которые не видны из текущего свипа, возможно, этот шаг и не нужен.
            mask = (voxel_labels == 0)
            voxel_labels[mask] = voxel_occluded[mask] * occluded_label
            voxel_labels = filter_isolated_label(voxel_labels, occluded_label)

            # remap original IDs -> train-IDs
            remapped = remap_labels(voxel_labels, remap_array)

            # проектирование в BEV (возвращает uint8 RGB)
            bev, bev_color = project(remapped, color_lut, car_height_m=2.0, voxel_size_z=0.2, unlabeled_class=0)

            # Сохраняем
            save_path = os.path.join(save_seq_dir, f"{fname}.png")
            cv2.imwrite(save_path, bev)

            save_path_color = os.path.join(save_seq_dir_color, f"{fname}.png")
            cv2.imwrite(save_path_color, bev_color)

            # Сохраняем для меньших масштабов
            scales = [2, 4, 8]
            orig_size = bev.shape
            for scale in scales:
                save_path = os.path.join(save_seq_dir, f"{fname}_1_{scale}.png")
                bev_scaled = cv2.resize(bev, (orig_size[0] // scale, orig_size[1] // scale), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(save_path, bev_scaled)

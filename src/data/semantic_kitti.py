import collections
import os
import pathlib
import random
from glob import glob

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

import src.data.io_data as SemanticKittiIO
from project2d.lib.core import voxelizer
from project2d.lib.readers import kitti_reader, multisweep_reader
from src.common import base_config, enums
from src.data import augmentations, path_solvers

M = enums.Modality


class SemanticKITTIDataset(Dataset):

    def __init__(self, dataset: base_config.DatasetConfig, phase, split=None):
        """

        :param dataset: The dataset configuration (data augmentation, input encoding, etc)
        :param phase: To differentiate between training, validation and test phase
        """
        self.cfg = dataset
        yaml_path = pathlib.Path(__file__).resolve().parent.parent.parent
        with pathlib.Path(os.path.join(yaml_path, self.cfg.mappings_path)).open() as f:
            self.dataset_config = yaml.safe_load(f)
        self.remap_lut = self.get_remap_lut()

        self.root_dir = self.cfg.root_dir
        self.reader = kitti_reader.KittiCloudReader()
        self.class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                            6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                            2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                            2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                            2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])

        self.filepaths = {}
        self.phase = phase

        self.flipper = augmentations.RandomFlipper(self.cfg.grid)

        if split is None:
            self.dataset_config["split"]
            self.split = {
                "train":  self.dataset_config["split"]["train"],
                "val": self.dataset_config["split"]["valid"],
                "test": self.dataset_config["split"]["test"],
            }
        else:
            self.split = split

        self.multisweep = self.cfg.multisweep is not None
        if self.multisweep:
            self.multisweep_reader = multisweep_reader.MultiSweepLidarReader(self.reader)
            if self.cfg.multisweep.drop_n == "auto":
                self.multisweep_strip_n = 1 + (self.cfg.multisweep.num_sweeps - 1) // 6
            else:
                self.multisweep_strip_n = self.cfg.multisweep.drop_n
            self.multisweep_strip_front = self.cfg.multisweep.from_past

        self.get_filepaths()

        self.nbr_files = len(self.filepaths[enums.Modality.VOXELS.value])

    def get_filepaths(self):
        filepaths = collections.defaultdict(lambda: collections.defaultdict(list))
        sequences = [sorted(glob(os.path.join(self.root_dir, "*")))[i] for i in self.split[self.phase]]

        for k in self.cfg.path_solvers:
            self.cfg.path_solvers[k].get_filepaths(filepaths, sequences)

        if self.multisweep:
            path_solvers.strip_n(filepaths, self.multisweep_strip_n, self.multisweep_strip_front)
        self.filepaths = path_solvers.merge_sequences(filepaths)

        self.check_filepaths()

    def check_filepaths(self):
        lengths = set()
        lengths.update(len(v) for v in self.filepaths.values())
        assert len(lengths) == 1, "Mismatch in filepaths size"

        for i, path_ in enumerate(self.filepaths[enums.Modality.VOXELS.value]):
            path = pathlib.Path(path_)
            for path_list in self.filepaths.values():
                another_path = pathlib.Path(path_list[i])
                assert self.get_frame_num_from_filepath(path) == self.get_frame_num_from_filepath(another_path), \
                    f"Mismatch for file names on with the same index. Index {i}," \
                    f"file 1 {path}, file 2 {another_path}, stem 1 {path.stem}, stem 2 {another_path.stem}"

    def __getitem__(self, idx):
        data = {}

        for modality in self.filepaths:
            self.get_data_modality(modality, idx, data)

        if self.cfg.augmentation.flip and self.phase == "train":
            do_flip = random.randint(0, 3)
            if do_flip > 0:
                do_flip = 2
            self.flipper.flip(data, do_flip)

        for k, v in data.items():
            data[k] = torch.from_numpy(v)

        return data, idx

    @staticmethod
    def get_frame_num_from_filepath(filepath: str | pathlib.Path) -> str:
        path = pathlib.Path(filepath)
        name = path.stem      # '000005*'
        return name[:6]

    @staticmethod
    def get_adjacent_filepaths(filepath: str | pathlib.Path, n: int = 2, prev=False) -> list[pathlib.Path]:
        """
        Возвращает список путей к n предыдущим или последующим файлам.
        Сохраняет форматирование (ведущие нули) и расширение исходного файла.
        Пример:
            Path('/data/voxels/000005.bin') →
            [Path('/data/voxels/000003.bin'), Path('/data/voxels/000004.bin')]
        """
        path = pathlib.Path(filepath)
        name = path.stem      # '000005'
        ext = path.suffix     # '.bin'

        if not name.isdigit():
            err = f"Имя файла '{name}' не является числовым."
            raise ValueError(err)

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

    def get_adjacent_lidar_label_paths(self, label_path):
        label_paths = self.get_adjacent_filepaths(label_path, self.cfg.multisweep.num_sweeps - 1,
                                                  self.cfg.multisweep.from_past)
        label_paths.append(label_path)
        return label_paths

    def read_lidar(self, idx, get_labels=False):
        points_path = self.filepaths[M.LIDAR.value][idx]
        points_labels = None
        if not self.multisweep:
            points = self.reader.read_cloud(points_path)
            if get_labels:
                label_path = self.filepaths[M.LABEL_LIDAR.value][idx]
                points_labels = self.reader.read_label(label_path, points)
                ss_remap_lut = self.get_remap_lut(completion=False)
                points_labels = ss_remap_lut[points_labels]
        else:
            pc_paths = self.get_adjacent_filepaths(points_path, self.cfg.multisweep.num_sweeps - 1,
                                                   self.cfg.multisweep.from_past)
            pc_paths.append(points_path)

            pose_path = self.filepaths[M.POSE.value][idx]
            pose_paths = self.get_adjacent_filepaths(pose_path, self.cfg.multisweep.num_sweeps - 1,
                                                     self.cfg.multisweep.from_past)
            pose_paths.append(pose_path)

            label_paths = None
            if get_labels:
                label_path = self.filepaths[M.LABEL_LIDAR.value][idx]
                label_paths = self.get_adjacent_lidar_label_paths(label_path)

            points = self.multisweep_reader.read(pc_paths, pose_paths, self.cfg.multisweep.timestamps, label_paths,
                                                 reference_index=-1)
            points = points[:, :-1]
            if get_labels:
                points_labels = points[:, -1].astype(np.uint32)
                points = points[:, :-1]
                ss_remap_lut = self.get_remap_lut(completion=False)
                points_labels = ss_remap_lut[points_labels]

        filter_mask = SemanticKittiIO.get_mask(points, self.cfg.grid)
        points = points[filter_mask]
        if get_labels:
            points_labels = points_labels[filter_mask]

        if self.cfg.augmentation.lidar_shuffle_index:
            pt_idx = np.random.permutation(np.arange(0, points.shape[0]))
            points = points[pt_idx]
            if get_labels:
                points_labels = points_labels[pt_idx]

        return points, points_labels

    def get_data_modality(self, modality, idx, data):

        if modality in {enums.Modality.VOXELS.value, enums.Modality.LIDAR.value}:
            get_labels_bool = self.phase != "test" and M.LABEL_LIDAR in self.cfg.path_solvers
            points, points_labels = self.read_lidar(idx, get_labels_bool)

            data[M.LIDAR.value] = points

            if modality == enums.Modality.VOXELS.value:
                voxels = voxelizer.voxelize_points(points[..., :3], coors_range=self.cfg.grid)
                data[M.VOXELS.value] = voxels

            if get_labels_bool:
                data[M.LABEL_LIDAR.value] = points_labels

        elif modality.startswith(enums.Modality.LABEL_VOXELS.value):
            LABEL = self.get_label_voxels(modality, idx)
            data[modality] = LABEL

        elif modality.startswith(enums.Modality.LABEL_2D.value):
            LABEL_2D = self.get_label_2d(modality, idx)
            data[modality] = LABEL_2D

    def get_label_voxels(self, modality, idx):

        scale_divide = int(modality[-1])
        LABEL = SemanticKittiIO._read_label_SemKITTI(self.filepaths[modality][idx])
        INVALID = SemanticKittiIO._read_invalid_SemKITTI(
            self.filepaths[f"{enums.Modality.INVALID.value}_1_{scale_divide}"][idx])

        # if scale == '1_1':
        LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
        LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
        LABEL = LABEL.reshape(np.array(self.cfg.grid_dims) // scale_divide)

        return LABEL

    def get_label_2d(self, modality, idx):
        label = cv2.imread(self.filepaths[modality][idx])
        label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
        label[label == 4] = 255

        return label

    def get_inv_remap_lut(self):
        """
        remap_lut to remap classes of semantic kitti for training...
        :return:
        """

        # make lookup table for mapping
        maxkey = max(self.dataset_config["learning_map_inv"].keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
        remap_lut[list(self.dataset_config["learning_map_inv"].keys())] = list(
            self.dataset_config["learning_map_inv"].values())

        return remap_lut

    def get_remap_lut(self, completion=True):
        """
        remap_lut to remap classes of semantic kitti for training...
        :return:
        """

        # make lookup table for mapping
        maxkey = max(self.dataset_config["learning_map"].keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(self.dataset_config["learning_map"].keys())] = list(self.dataset_config["learning_map"].values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        if completion:
            remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
            remap_lut[0] = 0  # only 'empty' stays 'empty'.

        return remap_lut

    def __len__(self):
        """
        Returns the length of the dataset
        """
        # Return the number of elements in the dataset
        return self.nbr_files

import collections
import os

from project2d.lib.readers import multisweep_reader, pcd_reader
from src.common import base_config, enums
from src.data import path_solvers, semantic_kitti

M = enums.Modality


class NavioDataset(semantic_kitti.SemanticKITTIDataset):

    def __init__(self, dataset: base_config.DatasetConfig, phase, split=None):
        super().__init__(dataset, phase, split)
        self.reader = pcd_reader.PCDReader()
        if self.multisweep:
            self.multisweep_reader = multisweep_reader.MultiSweepLidarReader(self.reader)

    def get_filepaths(self):
        filepaths = collections.defaultdict(lambda: collections.defaultdict(list))
        sequences = []
        for seq in self.split[self.phase]:
            sequences.append(os.path.join(self.root_dir, seq))
        sequences = sorted(sequences)

        for k in self.cfg.path_solvers:
            self.cfg.path_solvers[k].get_filepaths(filepaths, sequences)

        if self.cfg.reference_length_modality is not None:
            path_solvers.strip_using_ref(filepaths, self.cfg.reference_length_modality)

        self.filepaths = path_solvers.merge_sequences(filepaths)

        self.check_filepaths()

    def get_adjacent_lidar_label_paths(self, label_path):
        return [label_path] * self.cfg.multisweep.num_sweeps

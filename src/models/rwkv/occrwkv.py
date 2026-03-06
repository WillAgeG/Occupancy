import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.common import base_config, enums

from .bev_net import BEVUNetv1
from .completion import CompletionBranch, CompletionBranch2D, CompletionBranchMultiD
from .lovasz_losses import lovasz_softmax
from .preprocess import PcPreprocessor
from .semantic_segmentation import SemanticBranch, SemanticBranch2D

M = enums.Modality


class OccRWKV(nn.Module):
    def __init__(self, cfg: base_config.ExperimentConfig, phase="train",
                 sc_freq: list[int] | tp.Literal["kitti_default"] = "kitti_default",
                 ss_freq: list[int] | tp.Literal["kitti_default"] = "kitti_default"):
        super().__init__()
        self.phase = phase
        nbr_classes = cfg.dataset.nbr_classes
        self.nbr_classes = nbr_classes
        try:
            self.timestamp = cfg.dataset.multisweep.timestamps
        except AttributeError:
            self.timestamp = False

        if sc_freq == "kitti_default":
            self.sc_freq = [7632350044, 15783539, 125136, 118809, 646799, 821951, 262978, 283696, 204750,
                            61688703, 4502961, 44883650, 2269923, 56840218, 15719652, 158442623, 2061623,
                            36970522, 1151988, 334146]
        else:
            self.sc_freq = sc_freq

        if ss_freq == "kitti_default":
            self.ss_freq = [55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858, 240942562, 17294618, 
                            170599734, 6369672, 230413074, 101130274, 476491114, 9833174, 129609852, 4506626, 1168181]
        else:
            self.ss_freq = ss_freq

        g = cfg.dataset.grid
        self.lims = [[g[0], g[3]], [g[1], g[4]], [g[2], g[5]]]  # [[0, 51.2], [-25.6, 25.6], [-2, 4.4]]
        self.sizes = cfg.dataset.grid_dims  # [256, 256, 32]  # W, H, D (x, y, z)
        v_s = cfg.dataset.voxel_size
        self.grid_meters = [v_s, v_s, v_s]  # [0.2, 0.2, 0.2]

        self.n_height = cfg.dataset.H  # 32
        self.dilation = 1
        self.bilinear = True
        self.group_conv = False
        self.input_batch_norm = True
        self.dropout = 0.5
        self.circular_padding = False
        self.dropblock = False

        self.preprocess = PcPreprocessor(lims=self.lims, sizes=self.sizes, grid_meters=self.grid_meters,
                                         init_size=self.n_height, timestamp=self.timestamp)
        self.sem_branch = SemanticBranch(sizes=self.sizes, nbr_class=nbr_classes - 1, init_size=self.n_height,
                                         class_frequencies=self.ss_freq, phase=phase)
        self.com_branch = CompletionBranch(_init_size=self.n_height, nbr_class=nbr_classes, phase=phase)
        self.bev_model = BEVUNetv1(
            self.nbr_classes * self.n_height, self.n_height, self.dilation, self.bilinear, self.group_conv,
            self.input_batch_norm, self.dropout, self.circular_padding, self.dropblock
        )

    def forward(self, example):
        batch_size = len(example[M.LIDAR.value])
        with torch.no_grad():
            indicator = [0]
            pc_ibatch = []
            for i in range(batch_size):
                pc_i = example[M.LIDAR.value][i]
                pc_ibatch.append(pc_i)
                indicator.append(pc_i.size(0) + indicator[-1])
            pc = torch.cat(pc_ibatch, dim=0)
        vw_feature, coord_ind, full_coord, info = self.preprocess(pc, indicator)  # N, C; B, C, W, H, D
        coord = torch.cat([coord_ind[:, 0].reshape(-1, 1), torch.flip(coord_ind, dims=[1])[:, :3]], dim=1)
        bev_dense = self.sem_branch.bev_projection(
            vw_feature, coord, np.array(self.sizes, np.int32)[::-1], batch_size)  # B, C, H, W
        torch.cuda.empty_cache()

        ss_data_dict = {}
        ss_data_dict["vw_features"] = vw_feature
        ss_data_dict["coord_ind"] = coord_ind
        ss_data_dict["full_coord"] = full_coord
        ss_data_dict["info"] = info
        ss_out_dict = self.sem_branch(ss_data_dict, example)  # B, C, D, H, W

        sc_data_dict = {}
        occupancy = example[M.VOXELS.value].permute(0, 3, 2, 1)  # B, D, H, W
        sc_data_dict["vw_dense"] = occupancy.unsqueeze(1)
        sc_out_dict = self.com_branch(sc_data_dict, example)

        inputs = torch.cat([occupancy, bev_dense], dim=1)  # B, C, H, W
        x = self.bev_model(inputs, ss_out_dict["mss_bev_dense"], sc_out_dict["mss_bev_dense"])  # [B, 640, 256, 256]
        out_scale_1_1 = self.process_output(x)  # [B, 20, 256, 256]

        if self.phase in {"train", "val"}:
            loss_1_1 = self.compute_loss(out_scale_1_1, self.get_target(example)["1_1"])
            loss_dict = self.process_loss(loss_1_1, ss_out_dict["loss"], sc_out_dict["loss"])

            return {"pred_semantic_1_1": out_scale_1_1}, loss_dict

        return {"pred_semantic_1_1": out_scale_1_1}

    def process_output(self, x):
        new_shape = [x.shape[0], self.nbr_classes, self.n_height, *x.shape[-2:]]    # [B, 20, 32, 256, 256]
        x = x.view(new_shape)
        out_scale_1_1 = x.permute(0, 1, 4, 3, 2)   # [B,20,256,256,32]
        return out_scale_1_1

    def compute_loss(self, scores, labels):
        """
        :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
        """
        device = scores.device
        class_weights = self.get_class_weights(device, scores.dtype)
        loss_1_1 = F.cross_entropy(scores, labels.long(), weight=class_weights, ignore_index=255)
        loss_1_1 += lovasz_softmax(torch.nn.functional.softmax(scores, dim=1), labels.long(), ignore=255)
        loss_1_1 *= 3

        return loss_1_1

    @staticmethod
    def process_loss(loss_1_1, ss_loss_dict, sc_loss_dict) -> dict:
        loss_seg = sum(ss_loss_dict.values())
        loss_com = sum(sc_loss_dict.values())
        loss_total = loss_1_1 + loss_seg + loss_com
        loss = {"total": loss_total, "semantic_1_1": loss_1_1, "semantic_seg": loss_seg, "scene_completion": loss_com}

        return loss

    @staticmethod
    def weights_initializer(m) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def weights_init(self):
        self.apply(self.weights_initializer)

    def get_parameters(self):
        return self.parameters()

    def get_class_weights(self, device, dtype):
        """
        Class weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
        """
        epsilon_w = 0.001  # eps to avoid zero division
        weights = torch.from_numpy(1 / np.log(np.array(self.sc_freq) + epsilon_w))
        weights = weights.to(device=device, dtype=dtype, non_blocking=True)

        return weights

    @staticmethod
    def get_target(data) -> dict:
        """
        Return the target to use for evaluation of the model
        """
        label_copy = data[f"{M.LABEL_VOXELS.value}_1_1"].clone()
        return {"1_1": label_copy}

    @staticmethod
    def get_scales() -> list[str]:
        """
        Return scales needed to train the model
        """
        scales = ["1_1"]
        return scales

    @staticmethod
    def get_validation_loss_keys() -> list[str]:
        return ["total", "semantic_1_1", "semantic_seg", "scene_completion"]

    @staticmethod
    def get_train_loss_keys() -> list[str]:
        return ["total", "semantic_1_1", "semantic_seg", "scene_completion"]


class OccRWKV2D(OccRWKV):
    def __init__(self, cfg, phase="train"):
        super().__init__(cfg, phase)
        self.out_head = nn.Sequential(
                nn.Conv2d(128, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(32, self.nbr_classes, kernel_size=1))
        self.com_branch = CompletionBranch2D(_init_size=self.n_height, nbr_class=self.nbr_classes, phase=phase)
        self.sem_branch = SemanticBranch2D(sizes=self.sizes, nbr_class=self.nbr_classes - 1, init_size=self.n_height,
                                         class_frequencies=None, phase=phase)

    @staticmethod
    def get_class_weights(device, dtype) -> torch.Tensor:
        """
        Eye-ball class weights for now
        Classes: unlabeled, road, static, dynamic
        """

        return torch.tensor([1.0, 2.0, 2.0, 3.0], device=device, dtype=dtype)

    @staticmethod
    def get_target(data) -> dict:
        """
        Return the target to use for evaluation of the model
        """
        invalid_class = 4
        label_copy = data[f"{M.LABEL_2D.value}_1_1"].clone()
        label_copy[data[f"{M.LABEL_2D.value}_1_1"] == invalid_class] = 255
        return {"1_1": label_copy}

    def process_output(self, x):
        out_scale_1_1 = self.out_head(x)
        return out_scale_1_1


class OccRWKVMultiD(OccRWKV):
    def __init__(self, cfg, phase="train"):
        super().__init__(cfg, phase)
        self.out_head = nn.Sequential(
                nn.Conv2d(128, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(32, self.nbr_classes, kernel_size=1))
        self.com_branch = CompletionBranchMultiD(init_size=self.n_height, nbr_class=self.nbr_classes, phase=phase)
        self.sem_branch = SemanticBranch2D(sizes=self.sizes, nbr_class=self.nbr_classes - 1, init_size=self.n_height,
                                         class_frequencies=None, phase=phase)

    @staticmethod
    def get_class_weights(device, dtype) -> torch.Tensor:
        """
        Eye-ball class weights for now
        Classes: unlabeled, road, static, dynamic
        """

        return torch.tensor([1.0, 2.0, 2.0, 3.0], device=device, dtype=dtype)

    @staticmethod
    def get_target(data) -> dict:
        """
        Return the target to use for evaluation of the model
        """
        invalid_class = 4
        label_copy_2d = data["label_1_1_2d"].clone()
        label_copy_2d[data["label_1_1_2d"] == invalid_class] = 255

        label_copy = data["label_1_1"].clone()
        label_copy[data["invalid_1_1"] == 1] = 255

        return {"1_1": label_copy, "1_1_2d": label_copy_2d}

    def process_output_2d(self, x):
        out_scale_1_1 = self.out_head(x)
        return out_scale_1_1

    def process_output(self, x):
        new_shape = [x.shape[0], self.nbr_classes, self.n_height, *x.shape[-2:]]    # [B, 20, 32, 256, 256]
        x = x.view(new_shape)
        out_scale_1_1 = x.permute(0, 1, 4, 3, 2)   # [B,20,256,256,32]
        return out_scale_1_1

    def forward(self, example):
        batch_size = len(example["points"])
        with torch.no_grad():
            indicator = [0]
            pc_ibatch = []
            for i in range(batch_size):
                pc_i = example["points"][i]
                pc_ibatch.append(pc_i)
                indicator.append(pc_i.size(0) + indicator[-1])
            pc = torch.cat(pc_ibatch, dim=0)
        vw_feature, coord_ind, full_coord, info = self.preprocess(pc, indicator)  # N, C; B, C, W, H, D
        coord = torch.cat([coord_ind[:, 0].reshape(-1, 1), torch.flip(coord_ind, dims=[1])[:, :3]], dim=1)
        bev_dense = self.sem_branch.bev_projection(
            vw_feature, coord, np.array(self.sizes, np.int32)[::-1], batch_size)  # B, C, H, W
        torch.cuda.empty_cache()

        ss_data_dict = {}
        ss_data_dict["vw_features"] = vw_feature
        ss_data_dict["coord_ind"] = coord_ind
        ss_data_dict["full_coord"] = full_coord
        ss_data_dict["info"] = info
        ss_out_dict = self.sem_branch(ss_data_dict, example)  # B, C, D, H, W

        sc_data_dict = {}
        occupancy = example["occupancy"].permute(0, 3, 2, 1)  # B, D, H, W
        sc_data_dict["vw_dense"] = occupancy.unsqueeze(1)
        sc_out_dict = self.com_branch(sc_data_dict, example)

        inputs = torch.cat([occupancy, bev_dense], dim=1)  # B, C, H, W
        x = self.bev_model(inputs, ss_out_dict["mss_bev_dense"], sc_out_dict["mss_bev_dense"])  # [B, 640, 256, 256]
        out_scale_1_1 = self.process_output(x)  # [B, 20, 256, 256]
        out_scale_1_1_2d = self.process_output_2d(x)

        if self.phase in {"train", "val"}:
            loss_1_1 = self.compute_loss(out_scale_1_1, self.get_target(example)["1_1"])
            loss_1_1_2d = self.compute_loss(out_scale_1_1_2d, self.get_target(example)["1_1_2d"])
            loss_dict = self.process_loss(loss_1_1 + loss_1_1_2d, ss_out_dict["loss"], sc_out_dict["loss"])
            return {"pred_semantic_1_1": out_scale_1_1, "pred_semantic_1_1_2d": out_scale_1_1_2d}, loss_dict

        return {"pred_semantic_1_1": out_scale_1_1, "pred_semantic_1_1_2d": out_scale_1_1_2d}

    @staticmethod
    def get_scales() -> list[str]:
        """
        Return scales needed to train the model
        """
        scales = ["1_1", "1_1_2d"]
        return scales

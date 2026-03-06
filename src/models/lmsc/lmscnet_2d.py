import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common.enums import Modality
from models.lmsc import lmscnet

M = Modality


class SegmentationHead(nn.Module):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    """
    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(planes, planes, kernel_size=3, padding=dil, dilation=dil) for dil in dilations_conv_list])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes) for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList(
            [nn.Conv2d(planes, planes, kernel_size=3, padding=dil, dilation=dil) for dil in dilations_conv_list])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes) for dil in dilations_conv_list])
        self.relu = nn.ReLU(inplace=True)

        # Convolution for output
        self.conv_classes = nn.Conv2d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, x_in):
        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        x_in = self.conv_classes(x_in)

        return x_in


class LMSCNet2d(lmscnet.LMSCNet):

    def __init__(self, class_num, input_height, class_frequencies, phase="train"):
        super().__init__(class_num, input_height, class_frequencies, phase)

        self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

        # Treatment output 1:8
        self.seg_head_1_8 = SegmentationHead(4, 8, self.nbr_classes, [1, 2, 3])

        # Treatment output 1:4
        self.seg_head_1_4 = SegmentationHead(8, 8, self.nbr_classes, [1, 2, 3])

        # Treatment output 1:2
        self.seg_head_1_2 = SegmentationHead(16, 8, self.nbr_classes, [1, 2, 3])

        # Treatment output 1:1
        self.seg_head_1_1 = SegmentationHead(32, 8, self.nbr_classes, [1, 2, 3])

    @staticmethod
    def final_permutation(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2)

    def compute_loss(self, scores, data):
        """
        :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
        """

        target = data[f"{M.LABEL_2D.value}_1_1"]
        device = target.device
        class_weights = self.get_class_weights(device, torch.float32)

        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction="mean").to(device=device)

        loss_1_1 = criterion(scores["pred_semantic_1_1"], data[f"{M.LABEL_2D.value}_1_1"].long())
        loss_1_2 = criterion(scores["pred_semantic_1_2"], data[f"{M.LABEL_2D.value}_1_2"].long())
        loss_1_4 = criterion(scores["pred_semantic_1_4"], data[f"{M.LABEL_2D.value}_1_4"].long())
        loss_1_8 = criterion(scores["pred_semantic_1_8"], data[f"{M.LABEL_2D.value}_1_8"].long())

        loss_total = (loss_1_1 + loss_1_2 + loss_1_4 + loss_1_8) / 4

        loss = {"total": loss_total, "semantic_1_1": loss_1_1, "semantic_1_2": loss_1_2, "semantic_1_4": loss_1_4,
                "semantic_1_8": loss_1_8}

        return loss

    @staticmethod
    def get_class_weights(device, dtype):
        # Currently for the classes unlabeled, road, static, dynamic
        return torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=dtype, device=device)

    @staticmethod
    def get_target(data):
        """
        Return the target to use for evaluation of the model
        """
        return {"1_1": data[f"{M.LABEL_2D.value}_1_1"], "1_2": data[f"{M.LABEL_2D.value}_1_2"],
                "1_4": data[f"{M.LABEL_2D.value}_1_4"], "1_8": data[f"{M.LABEL_2D.value}_1_8"]}


class LMSCNet2dForAdversarial(LMSCNet2d):
    @staticmethod
    def get_train_loss_keys():
        return ["total", "semantic_1_1", "semantic_1_2", "semantic_1_4", "semantic_1_8", "dataset"]

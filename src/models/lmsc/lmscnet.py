import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common.enums import Modality

M = Modality


class SegmentationHead(nn.Module):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    """
    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False)
             for dil in dilations_conv_list])
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList(
            [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False)
             for dil in dilations_conv_list])
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.relu = nn.ReLU(inplace=True)

        # Convolution for output
        self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, x_in):

        # Dimension exapension
        x_in = x_in[:, None, :, :, :]

        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)    # modified

        x_in = self.conv_classes(x_in)

        return x_in


class LMSCNet(nn.Module):

    def __init__(self, class_num, input_height, class_frequencies, phase="train"):
        """
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        """

        super().__init__()
        self.nbr_classes = class_num
        self.class_frequencies = class_frequencies
        f = input_height    # H or z or height
        self.phase = phase

        self.pool = nn.MaxPool2d(2)    # [F=2; S=2; P=0; D=1]

        self.Encoder_block1 = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.Encoder_block2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(f, int(f * 1.5), kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(int(f * 1.5), int(f * 1.5), kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.Encoder_block3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(f * 1.5), int(f * 2), kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(int(f * 2), int(f * 2), kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.Encoder_block4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(f * 2), int(f * 2.5), kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(int(f * 2.5), int(f * 2.5), kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        # Treatment output 1:8
        self.conv_out_scale_1_8 = nn.Conv2d(int(f * 2.5), int(f / 8), kernel_size=3, padding=1, stride=1)
        self.seg_head_1_8 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
        self.deconv_1_8__1_2 = nn.ConvTranspose2d(int(f / 8), int(f / 8), kernel_size=4, padding=0, stride=4)
        self.deconv_1_8__1_1 = nn.ConvTranspose2d(int(f / 8), int(f / 8), kernel_size=8, padding=0, stride=8)

        # Treatment output 1:4
        self.deconv1_8 = nn.ConvTranspose2d(int(f / 8), int(f / 8), kernel_size=6, padding=2, stride=2)
        self.conv1_4 = nn.Conv2d(int(f * 2) + int(f / 8), int(f * 2), kernel_size=3, padding=1, stride=1)
        self.conv_out_scale_1_4 = nn.Conv2d(int(f * 2), int(f / 4), kernel_size=3, padding=1, stride=1)
        self.seg_head_1_4 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
        self.deconv_1_4__1_1 = nn.ConvTranspose2d(int(f / 4), int(f / 4), kernel_size=4, padding=0, stride=4)

        # Treatment output 1:2
        self.deconv1_4 = nn.ConvTranspose2d(int(f / 4), int(f / 4), kernel_size=6, padding=2, stride=2)
        self.conv1_2 = nn.Conv2d(int(f * 1.5) + int(f / 4) + int(f / 8), int(f * 1.5), kernel_size=3,
                                 padding=1, stride=1)
        self.conv_out_scale_1_2 = nn.Conv2d(int(f * 1.5), int(f / 2), kernel_size=3, padding=1, stride=1)
        self.seg_head_1_2 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])

        # Treatment output 1:1
        self.deconv1_2 = nn.ConvTranspose2d(int(f / 2), int(f / 2), kernel_size=6, padding=2, stride=2)
        self.conv1_1 = nn.Conv2d(int(f / 8) + int(f / 4) + int(f / 2) + int(f), f, kernel_size=3, padding=1, stride=1)
        self.seg_head_1_1 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])

    @staticmethod
    def final_permutation(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 4, 2)

    def forward(self, x):

        input_ = x[M.VOXELS.value]  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, D, H]
        input_ = input_.permute(0, 3, 1, 2)  # Reshaping to the right way for 2D convs [bs, H, W, D]

        # Encoder block
        skip_1_1 = self.Encoder_block1(input_)
        skip_1_2 = self.Encoder_block2(skip_1_1)
        skip_1_4 = self.Encoder_block3(skip_1_2)
        skip_1_8 = self.Encoder_block4(skip_1_4)

        # Out 1_8
        out_scale_1_8__2D = self.conv_out_scale_1_8(skip_1_8)
        out_scale_1_8__3D = self.seg_head_1_8(out_scale_1_8__2D)

        # Out 1_4
        out = self.deconv1_8(out_scale_1_8__2D)
        out = torch.cat((out, skip_1_4), 1)
        out = F.relu(self.conv1_4(out))
        out_scale_1_4__2D = self.conv_out_scale_1_4(out)
        out_scale_1_4__3D = self.seg_head_1_4(out_scale_1_4__2D)

        # Out 1_2
        out = self.deconv1_4(out_scale_1_4__2D)
        out = torch.cat((out, skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)
        out = F.relu(self.conv1_2(out))
        out_scale_1_2__2D = self.conv_out_scale_1_2(out)
        out_scale_1_2__3D = self.seg_head_1_2(out_scale_1_2__2D)

        # Out 1_1
        out = self.deconv1_2(out_scale_1_2__2D)
        out = torch.cat((out, skip_1_1, self.deconv_1_4__1_1(out_scale_1_4__2D),
                         self.deconv_1_8__1_1(out_scale_1_8__2D)), 1)
        out_scale_1_1__2D = F.relu(self.conv1_1(out))
        out_scale_1_1__3D = self.seg_head_1_1(out_scale_1_1__2D)

        # Take back to [W, H, D] axis order
        out_scale_1_8__3D = self.final_permutation(out_scale_1_8__3D)  # [bs, C, H, W, D] -> [bs, C, W, D, H]
        out_scale_1_4__3D = self.final_permutation(out_scale_1_4__3D)  # [bs, C, H, W, D] -> [bs, C, W, D, H]
        out_scale_1_2__3D = self.final_permutation(out_scale_1_2__3D)  # [bs, C, H, W, D] -> [bs, C, W, D, H]
        out_scale_1_1__3D = self.final_permutation(out_scale_1_1__3D)  # [bs, C, H, W, D] -> [bs, C, W, D, H]

        scores = {
                    "pred_semantic_1_1": out_scale_1_1__3D,
                    "pred_semantic_1_2": out_scale_1_2__3D,
                    "pred_semantic_1_4": out_scale_1_4__3D,
                    "pred_semantic_1_8": out_scale_1_8__3D,
                }

        if self.phase in {"train", "val"}:
            loss = self.compute_loss(scores, x)
            return scores, loss

        return scores

    @staticmethod
    def weights_initializer(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def weights_init(self):
        self.apply(self.weights_initializer)

    def get_parameters(self):
        return self.parameters()

    def compute_loss(self, scores, data):
        """
        :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
        """

        target = data[f"{M.LABEL_VOXELS.value}_1_1"]
        device = target.device
        class_weights = self.get_class_weights(target.device, target.dtype)

        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction="mean").to(device=device)

        loss_1_1 = criterion(scores["pred_semantic_1_1"], data[f"{M.LABEL_VOXELS.value}_1_1"].long())
        loss_1_2 = criterion(scores["pred_semantic_1_2"], data[f"{M.LABEL_VOXELS.value}_1_2"].long())
        loss_1_4 = criterion(scores["pred_semantic_1_4"], data[f"{M.LABEL_VOXELS.value}_1_4"].long())
        loss_1_8 = criterion(scores["pred_semantic_1_8"], data[f"{M.LABEL_VOXELS.value}_1_8"].long())

        loss_total = (loss_1_1 + loss_1_2 + loss_1_4 + loss_1_8) / 4

        loss = {"total": loss_total, "semantic_1_1": loss_1_1, "semantic_1_2": loss_1_2, "semantic_1_4": loss_1_4,
                        "semantic_1_8": loss_1_8}

        return loss

    def get_class_weights(self, device, dtype):
        """
        Cless weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
        """
        epsilon_w = 0.001    # eps to avoid zero division
        weights = torch.from_numpy(1 / np.log(self.class_frequencies + epsilon_w))
        weights = weights.to(device=device, dtype=dtype, non_blocking=True)

        return weights

    @staticmethod
    def get_target(data):
        """
        Return the target to use for evaluation of the model
        """
        return {"1_1": data[f"{M.LABEL_VOXELS.value}_1_1"], "1_2": data[f"{M.LABEL_VOXELS.value}_1_2"],
                        "1_4": data[f"{M.LABEL_VOXELS.value}_1_4"], "1_8": data[f"{M.LABEL_VOXELS.value}_1_8"]}

    @staticmethod
    def get_scales():
        """
        Return scales needed to train the model
        """
        scales = ["1_1", "1_2", "1_4", "1_8"]
        return scales

    @staticmethod
    def get_validation_loss_keys():
        return ["total", "semantic_1_1", "semantic_1_2", "semantic_1_4", "semantic_1_8"]

    @staticmethod
    def get_train_loss_keys():
        return ["total", "semantic_1_1", "semantic_1_2", "semantic_1_4", "semantic_1_8"]

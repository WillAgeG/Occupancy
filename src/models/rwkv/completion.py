import torch.nn.functional as F
from torch import nn

from src.common import enums

from .lovasz_losses import lovasz_softmax
from .vrwkv import Block as RWKVBlock

M = enums.Modality


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding, stride, dilation=1):
        super().__init__()
        self.reduction = nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size,
                                   padding=padding, stride=stride, dilation=dilation)
        self.layer = nn.Conv3d(out_dim, out_dim, kernel_size=kernel_size,
                               padding=padding, stride=stride, dilation=dilation)

    def forward(self, x):
        add = self.reduction(x)
        out = self.layer(F.relu(add))
        out_res = F.relu(add + out)
        return out_res


def make_layers(in_dim, out_dim, kernel_size=3, padding=1, stride=1, dilation=1, downsample=False, blocks=2):
    layers = []
    if downsample:
        layers.append(nn.MaxPool3d(2))
    layers.append(ResBlock(in_dim, out_dim, kernel_size, padding, stride, dilation))
    for _ in range(1, blocks):
        layers.append(ResBlock(out_dim, out_dim, kernel_size, padding, stride, dilation))
    return nn.Sequential(*layers)


class CompletionBranch(nn.Module):
    def __init__(self, _init_size=32, nbr_class=20, phase="train"):
        super().__init__()
        self.nclass = nbr_class
        self.in_layer = nn.Conv3d(1, 16, kernel_size=7, padding=3, stride=2, dilation=1)  # 1/2, 16
        self.block_1 = make_layers(16, 16, kernel_size=3, padding=1, stride=1, dilation=1, blocks=1)  # 1/2, 16
        self.block_2 = make_layers(16, 32, kernel_size=3, padding=1, stride=1, dilation=1,
                                    downsample=True, blocks=1)  # 1/4, 32
        self.block_3 = make_layers(32, 64, kernel_size=3, padding=2, stride=1, dilation=2,
                                    downsample=True, blocks=1)  # 1/8, 64

        # RWKV blocks
        self.rwkv_block1 = RWKVBlock(n_embd=64, n_layer=18, layer_id=0)
        self.rwkv_block2 = RWKVBlock(n_embd=128, n_layer=18, layer_id=0)
        self.rwkv_block3 = RWKVBlock(n_embd=256, n_layer=18, layer_id=0)

        self.reduction_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU()
        )
        self.reduction_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
        )

        self.phase = phase
        if self.phase in {"train", "val"}:
            self.out2 = nn.Sequential(
                nn.Conv3d(16, 16, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(16, 2, kernel_size=1))
            self.out4 = nn.Sequential(
                nn.Conv3d(32, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(32, 2, kernel_size=1))
            self.out8 = nn.Sequential(
                nn.Conv3d(64, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(32, 2, kernel_size=1))

    def forward_once(self, inputs):
        out = F.relu(self.in_layer(inputs))
        res1 = self.block_1(out)  # B, 16, 16, 128, 128
        res2 = self.block_2(res1)  # B, 32, 8, 64, 64
        res3 = self.block_3(res2)  # B, 64, 4, 32, 32

        bev_1 = self.reduction_1(res1.flatten(1, 2))  # B, 64, 128, 128
        bev_2 = self.reduction_2(res2.flatten(1, 2))  # B, 128, 64, 64
        bev_3 = res3.flatten(1, 2)  # B, 256, 32, 32

        B, C, H, W = bev_1.shape
        patch_resolution = (H, W)
        bev_1 = self.rwkv_block1(bev_1.permute(0, 2, 3, 1).reshape(B, H * W, C),
                                  patch_resolution=patch_resolution).view(B, H, W, C).permute(0, 3, 1, 2)

        B, C, H, W = bev_2.shape
        patch_resolution = (H, W)
        bev_2 = self.rwkv_block2(bev_2.permute(0, 2, 3, 1).reshape(B, H * W, C),
                                  patch_resolution=patch_resolution).view(B, H, W, C).permute(0, 3, 1, 2)

        B, C, H, W = bev_3.shape
        patch_resolution = (H, W)
        bev_3 = self.rwkv_block3(bev_3.permute(0, 2, 3, 1).reshape(B, H * W, C),
                                  patch_resolution=patch_resolution).view(B, H, W, C).permute(0, 3, 1, 2)

        if self.phase in {"train", "val"}:
            logits_2 = self.out2(res1)
            logits_4 = self.out4(res2)
            logits_8 = self.out8(res3)

            return {
                    "mss_bev_dense": [bev_1, bev_2, bev_3],
                    "mss_logits_list": [logits_2, logits_4, logits_8]
            }

        return {
            "mss_bev_dense": [bev_1, bev_2, bev_3]
        }

    def forward(self, data_dict, example):
        if self.phase in {"train", "val"}:
            out_dict = self.forward_once(data_dict["vw_dense"])
            teacher_2, teacher_4, teacher_8 = out_dict["mss_logits_list"]
            teacher_2 = teacher_2.permute(0, 1, 4, 3, 2)
            teacher_4 = teacher_4.permute(0, 1, 4, 3, 2)
            teacher_8 = teacher_8.permute(0, 1, 4, 3, 2)
            device = teacher_2.device

            sc_label_1_2_copy = example[f"{M.LABEL_VOXELS.value}_1_2"].clone()
            sc_label_1_2_copy = ((sc_label_1_2_copy > 0) & (sc_label_1_2_copy < self.nclass)).long()
            scale_loss_1_2 = lovasz_softmax(F.softmax(teacher_2, dim=1), sc_label_1_2_copy, ignore=255)
            focal_loss_1_2 = F.cross_entropy(teacher_2.cpu(), sc_label_1_2_copy.cpu(), ignore_index=255).to(device)
            loss = {"1_2_lovasz_loss": scale_loss_1_2, "1_2_ce_loss": focal_loss_1_2}

            sc_label_1_4_copy = example[f"{M.LABEL_VOXELS.value}_1_4"].clone()
            sc_label_1_4_copy = ((sc_label_1_4_copy > 0) & (sc_label_1_4_copy < self.nclass)).long()
            scale_loss_1_4 = lovasz_softmax(F.softmax(teacher_4, dim=1), sc_label_1_4_copy, ignore=255)
            focal_loss_1_4 = F.cross_entropy(teacher_4.cpu(), sc_label_1_4_copy.cpu(), ignore_index=255).to(device)
            loss.update({"1_4_lovasz_loss": scale_loss_1_4, "1_4_ce_loss": focal_loss_1_4})

            sc_label_1_8_copy = example[f"{M.LABEL_VOXELS.value}_1_8"].clone()
            sc_label_1_8_copy = ((sc_label_1_8_copy > 0) & (sc_label_1_8_copy < self.nclass)).long()
            scale_loss_1_8 = lovasz_softmax(F.softmax(teacher_8, dim=1), sc_label_1_8_copy, ignore=255)
            focal_loss_1_8 = F.cross_entropy(teacher_8.cpu(), sc_label_1_8_copy.cpu(), ignore_index=255).to(device)
            loss.update({"1_8_lovasz_loss": scale_loss_1_8, "1_8_ce_loss": focal_loss_1_8})

            return {
                "mss_bev_dense": out_dict["mss_bev_dense"],
                "loss": loss
            }
        out_dict = self.forward_once(data_dict["vw_dense"])
        return out_dict


class CompletionBranch2D(nn.Module):
    def __init__(self, _init_size=32, nbr_class=20, phase="train"):
        super().__init__()
        self.nclass = nbr_class
        self.in_layer = nn.Conv3d(1, 16, kernel_size=7, padding=3, stride=2, dilation=1)  # 1/2, 16
        self.block_1 = make_layers(16, 16, kernel_size=3, padding=1, stride=1, dilation=1, blocks=1)  # 1/2, 16
        self.block_2 = make_layers(16, 32, kernel_size=3, padding=1, stride=1, dilation=1,
                                    downsample=True, blocks=1)  # 1/4, 32
        self.block_3 = make_layers(32, 64, kernel_size=3, padding=2, stride=1, dilation=2,
                                    downsample=True, blocks=1)  # 1/8, 64

        # RWKV blocks
        self.rwkv_block1 = RWKVBlock(n_embd=64, n_layer=18, layer_id=0)
        self.rwkv_block2 = RWKVBlock(n_embd=128, n_layer=18, layer_id=0)
        self.rwkv_block3 = RWKVBlock(n_embd=256, n_layer=18, layer_id=0)

        self.reduction_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU()
        )
        self.reduction_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
        )

        self.phase = phase
        if self.phase in {"train", "val"}:
            self.out2_2d = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(32, 2, kernel_size=1))
            self.out4_2d = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64, 2, kernel_size=1))
            self.out8_2d = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64, 2, kernel_size=1))

    def forward_once(self, inputs):
        out = F.relu(self.in_layer(inputs))
        res1 = self.block_1(out)  # B, 16, 16, 128, 128
        res2 = self.block_2(res1)  # B, 32, 8, 64, 64
        res3 = self.block_3(res2)  # B, 64, 4, 32, 32

        bev_1 = self.reduction_1(res1.flatten(1, 2))  # B, 64, 128, 128
        bev_2 = self.reduction_2(res2.flatten(1, 2))  # B, 128, 64, 64
        bev_3 = res3.flatten(1, 2)  # B, 256, 32, 32
        bev_1_out = bev_1.clone()
        bev_2_out = bev_2.clone()
        bev_3_out = bev_3.clone()

        B, C, H, W = bev_1.shape
        patch_resolution = (H, W)
        bev_1 = self.rwkv_block1(bev_1.permute(0, 2, 3, 1).reshape(B, H * W, C),
                                 patch_resolution=patch_resolution).view(B, H, W, C).permute(0, 3, 1, 2)

        B, C, H, W = bev_2.shape
        patch_resolution = (H, W)
        bev_2 = self.rwkv_block2(bev_2.permute(0, 2, 3, 1).reshape(B, H * W, C),
                                 patch_resolution=patch_resolution).view(B, H, W, C).permute(0, 3, 1, 2)

        B, C, H, W = bev_3.shape
        patch_resolution = (H, W)
        bev_3 = self.rwkv_block3(bev_3.permute(0, 2, 3, 1).reshape(B, H * W, C),
                                 patch_resolution=patch_resolution).view(B, H, W, C).permute(0, 3, 1, 2)

        if self.phase in {"train", "val"}:
            logits_2_2d = self.out2_2d(bev_1_out)
            logits_4_2d = self.out4_2d(bev_2_out)
            logits_8_2d = self.out8_2d(bev_3_out)

            return {
                "mss_bev_dense": [bev_1, bev_2, bev_3],
                "mss_logits_list": [logits_2_2d, logits_4_2d, logits_8_2d]
            }

        return {
            "mss_bev_dense": [bev_1, bev_2, bev_3]
        }

    def forward(self, data_dict, example):
        if self.phase in {"train", "val"}:
            out_dict = self.forward_once(data_dict["vw_dense"])
            teacher_2_2d, teacher_4_2d, teacher_8_2d = out_dict["mss_logits_list"]
            teacher_2_2d = teacher_2_2d.permute(0, 1, 3, 2)
            teacher_4_2d = teacher_4_2d.permute(0, 1, 3, 2)
            teacher_8_2d = teacher_8_2d.permute(0, 1, 3, 2)
            device = teacher_2_2d.device

            loss = {}

            invalid_class = 4
            sc_label_1_2_2d_copy = example[f"{M.LABEL_2D.value}_1_2"].clone()
            invalid_1_2 = sc_label_1_2_2d_copy == invalid_class
            sc_label_1_2_2d_copy = ((sc_label_1_2_2d_copy > 0) & (sc_label_1_2_2d_copy < self.nclass)).long()
            sc_label_1_2_2d_copy[invalid_1_2] = 0
            scale_loss_1_2_2d = lovasz_softmax(F.softmax(teacher_2_2d, dim=1), sc_label_1_2_2d_copy, ignore=255)
            focal_loss_1_2_2d = F.cross_entropy(
                teacher_2_2d.cpu(), sc_label_1_2_2d_copy.cpu(), ignore_index=255).to(device)
            loss.update({"1_2_lovasz_loss": scale_loss_1_2_2d, "1_2_ce_loss": focal_loss_1_2_2d})

            sc_label_1_4_2d_copy = example[f"{M.LABEL_2D.value}_1_4"].clone()
            invalid_1_4 = sc_label_1_4_2d_copy == invalid_class
            sc_label_1_4_2d_copy = ((sc_label_1_4_2d_copy > 0) & (sc_label_1_4_2d_copy < self.nclass)).long()
            sc_label_1_4_2d_copy[invalid_1_4] = 0
            scale_loss_1_4_2d = lovasz_softmax(F.softmax(teacher_4_2d, dim=1), sc_label_1_4_2d_copy, ignore=255)
            focal_loss_1_4_2d = F.cross_entropy(
                teacher_4_2d.cpu(), sc_label_1_4_2d_copy.cpu(), ignore_index=255).to(device)
            loss.update({"1_4_lovasz_loss": scale_loss_1_4_2d, "1_4_ce_loss": focal_loss_1_4_2d})

            sc_label_1_8_2d_copy = example[f"{M.LABEL_2D.value}_1_8"].clone()
            invalid_1_8 = sc_label_1_8_2d_copy == invalid_class
            sc_label_1_8_2d_copy = ((sc_label_1_8_2d_copy > 0) & (sc_label_1_8_2d_copy < self.nclass)).long()
            sc_label_1_8_2d_copy[invalid_1_8] = 0
            scale_loss_1_8_2d = lovasz_softmax(F.softmax(teacher_8_2d, dim=1), sc_label_1_8_2d_copy, ignore=255)
            focal_loss_1_8_2d = F.cross_entropy(
                teacher_8_2d.cpu(), sc_label_1_8_2d_copy.cpu(), ignore_index=255).to(device)
            loss.update({"1_8_lovasz_loss": scale_loss_1_8_2d, "1_8_ce_loss": focal_loss_1_8_2d})

            return {
                "mss_bev_dense": out_dict["mss_bev_dense"],
                "loss": loss
            }
        out_dict = self.forward_once(data_dict["vw_dense"])
        return out_dict


class CompletionBranchMultiD(nn.Module):
    def __init__(self, _init_size=32, nbr_class=20, phase="train"):
        super().__init__()
        self.nclass = nbr_class
        self.in_layer = nn.Conv3d(1, 16, kernel_size=7, padding=3, stride=2, dilation=1)  # 1/2, 16
        self.block_1 = make_layers(16, 16, kernel_size=3, padding=1, stride=1, dilation=1, blocks=1)  # 1/2, 16
        self.block_2 = make_layers(16, 32, kernel_size=3, padding=1, stride=1, dilation=1,
                                   downsample=True, blocks=1)  # 1/4, 32
        self.block_3 = make_layers(32, 64, kernel_size=3, padding=2, stride=1, dilation=2,
                                   downsample=True, blocks=1)  # 1/8, 64

        # RWKV blocks
        self.rwkv_block1 = RWKVBlock(n_embd=64, n_layer=18, layer_id=0)
        self.rwkv_block2 = RWKVBlock(n_embd=128, n_layer=18, layer_id=0)
        self.rwkv_block3 = RWKVBlock(n_embd=256, n_layer=18, layer_id=0)

        self.reduction_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU()
        )
        self.reduction_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
        )

        self.phase = phase
        if self.phase in {"train", "val"}:
            self.out2 = nn.Sequential(
                nn.Conv3d(16, 16, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(16, 2, kernel_size=1))
            self.out4 = nn.Sequential(
                nn.Conv3d(32, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(32, 2, kernel_size=1))
            self.out8 = nn.Sequential(
                nn.Conv3d(64, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(32, 2, kernel_size=1))
            self.out2_2d = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(32, 2, kernel_size=1))
            self.out4_2d = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64, 2, kernel_size=1))
            self.out8_2d = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64, 2, kernel_size=1))

    def forward_once(self, inputs):
        out = F.relu(self.in_layer(inputs))
        res1 = self.block_1(out)  # B, 16, 16, 128, 128
        res2 = self.block_2(res1)  # B, 32, 8, 64, 64
        res3 = self.block_3(res2)  # B, 64, 4, 32, 32

        bev_1 = self.reduction_1(res1.flatten(1, 2))  # B, 64, 128, 128
        bev_2 = self.reduction_2(res2.flatten(1, 2))  # B, 128, 64, 64
        bev_3 = res3.flatten(1, 2)  # B, 256, 32, 32
        bev_1_out = bev_1.clone()
        bev_2_out = bev_2.clone()
        bev_3_out = bev_3.clone()

        B, C, H, W = bev_1.shape
        patch_resolution = (H, W)
        bev_1 = self.rwkv_block1(bev_1.permute(0, 2, 3, 1).reshape(B, H * W, C),
                                 patch_resolution=patch_resolution).view(B, H, W, C).permute(0, 3, 1, 2)

        B, C, H, W = bev_2.shape
        patch_resolution = (H, W)
        bev_2 = self.rwkv_block2(bev_2.permute(0, 2, 3, 1).reshape(B, H * W, C),
                                 patch_resolution=patch_resolution).view(B, H, W, C).permute(0, 3, 1, 2)

        B, C, H, W = bev_3.shape
        patch_resolution = (H, W)
        bev_3 = self.rwkv_block3(bev_3.permute(0, 2, 3, 1).reshape(B, H * W, C),
                                 patch_resolution=patch_resolution).view(B, H, W, C).permute(0, 3, 1, 2)

        if self.phase in {"train", "val"}:
            logits_2 = self.out2(res1)
            logits_4 = self.out4(res2)
            logits_8 = self.out8(res3)
            logits_2_2d = self.out2_2d(bev_1_out)
            logits_4_2d = self.out4_2d(bev_2_out)
            logits_8_2d = self.out8_2d(bev_3_out)

            return {
                "mss_bev_dense": [bev_1, bev_2, bev_3],
                "mss_logits_list": [logits_2, logits_4, logits_8, logits_2_2d, logits_4_2d, logits_8_2d]
            }

        return {
            "mss_bev_dense": [bev_1, bev_2, bev_3]
        }

    def forward(self, data_dict, example):
        if self.phase in {"train", "val"}:
            out_dict = self.forward_once(data_dict["vw_dense"])
            teacher_2, teacher_4, teacher_8, teacher_2_2d, teacher_4_2d, teacher_8_2d = out_dict["mss_logits_list"]
            teacher_2 = teacher_2.permute(0, 1, 4, 3, 2)
            teacher_4 = teacher_4.permute(0, 1, 4, 3, 2)
            teacher_8 = teacher_8.permute(0, 1, 4, 3, 2)
            teacher_2_2d = teacher_2_2d.permute(0, 1, 3, 2)
            teacher_4_2d = teacher_4_2d.permute(0, 1, 3, 2)
            teacher_8_2d = teacher_8_2d.permute(0, 1, 3, 2)

            loss = {}
            invalid_class = 4

            sc_label_1_2_copy = example["label_1_2"].clone()
            sc_label_1_2_copy = ((sc_label_1_2_copy > 0) & (sc_label_1_2_copy < self.nclass)).long()
            sc_label_1_2_copy[example["invalid_1_2"] == 1] = 255
            scale_loss_1_2 = lovasz_softmax(F.softmax(teacher_2, dim=1), sc_label_1_2_copy, ignore=255)
            focal_loss_1_2 = F.cross_entropy(teacher_2, sc_label_1_2_copy, ignore_index=255)
            loss.update({"1_2_lovasz_loss": scale_loss_1_2, "1_2_ce_loss": focal_loss_1_2})

            sc_label_1_4_copy = example["label_1_4"].clone()
            sc_label_1_4_copy = ((sc_label_1_4_copy > 0) & (sc_label_1_4_copy < self.nclass)).long()
            sc_label_1_4_copy[example["invalid_1_4"] == 1] = 255
            scale_loss_1_4 = lovasz_softmax(F.softmax(teacher_4, dim=1), sc_label_1_4_copy, ignore=255)
            focal_loss_1_4 = F.cross_entropy(teacher_4, sc_label_1_4_copy, ignore_index=255)
            loss.update({"1_4_lovasz_loss": scale_loss_1_4, "1_4_ce_loss": focal_loss_1_4})

            sc_label_1_8_copy = example["label_1_8"].clone()
            sc_label_1_8_copy = ((sc_label_1_8_copy > 0) & (sc_label_1_8_copy < self.nclass)).long()
            sc_label_1_8_copy[example["invalid_1_8"] == 1] = 255
            scale_loss_1_8 = lovasz_softmax(F.softmax(teacher_8, dim=1), sc_label_1_8_copy, ignore=255)
            focal_loss_1_8 = F.cross_entropy(teacher_8, sc_label_1_8_copy, ignore_index=255)
            loss.update({"1_8_lovasz_loss": scale_loss_1_8, "1_8_ce_loss": focal_loss_1_8})

            sc_label_1_2_2d_copy = example["label_1_2_2d"].clone()
            invalid_1_2 = sc_label_1_2_2d_copy == invalid_class
            sc_label_1_2_2d_copy = ((sc_label_1_2_2d_copy > 0) & (sc_label_1_2_2d_copy < self.nclass)).long()
            sc_label_1_2_2d_copy[invalid_1_2] = 0
            scale_loss_1_2_2d = lovasz_softmax(F.softmax(teacher_2_2d, dim=1), sc_label_1_2_2d_copy, ignore=255)
            focal_loss_1_2_2d = F.cross_entropy(teacher_2_2d, sc_label_1_2_2d_copy, ignore_index=255)
            loss.update({"1_2_lovasz_loss_2d": scale_loss_1_2_2d, "1_2_ce_loss_2d": focal_loss_1_2_2d})

            sc_label_1_4_2d_copy = example["label_1_4_2d"].clone()
            invalid_1_4 = sc_label_1_4_2d_copy == invalid_class
            sc_label_1_4_2d_copy = ((sc_label_1_4_2d_copy > 0) & (sc_label_1_4_2d_copy < self.nclass)).long()
            sc_label_1_4_2d_copy[invalid_1_4] = 0
            scale_loss_1_4_2d = lovasz_softmax(F.softmax(teacher_4_2d, dim=1), sc_label_1_4_2d_copy, ignore=255)
            focal_loss_1_4_2d = F.cross_entropy(teacher_4_2d, sc_label_1_4_2d_copy, ignore_index=255)
            loss.update({"1_4_lovasz_loss_2d": scale_loss_1_4_2d, "1_4_ce_loss_2d": focal_loss_1_4_2d})

            sc_label_1_8_2d_copy = example["label_1_8_2d"].clone()
            invalid_1_8 = sc_label_1_8_2d_copy == invalid_class
            sc_label_1_8_2d_copy = ((sc_label_1_8_2d_copy > 0) & (sc_label_1_8_2d_copy < self.nclass)).long()
            sc_label_1_8_2d_copy[invalid_1_8] = 0
            scale_loss_1_8_2d = lovasz_softmax(F.softmax(teacher_8_2d, dim=1), sc_label_1_8_2d_copy, ignore=255)
            focal_loss_1_8_2d = F.cross_entropy(teacher_8_2d, sc_label_1_8_2d_copy, ignore_index=255)
            loss.update({"1_8_lovasz_loss_2d": scale_loss_1_8_2d, "1_8_ce_loss_2d": focal_loss_1_8_2d})

            return {
                "mss_bev_dense": out_dict["mss_bev_dense"],
                "loss": loss
            }
        out_dict = self.forward_once(data_dict["vw_dense"])
        return out_dict

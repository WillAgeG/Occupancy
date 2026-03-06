import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.common import base_config
from src.common.enums import DatasetType, Modality
from src.data.navio import NavioDataset
from src.data.semantic_kitti import SemanticKITTIDataset


def collate_fn(data):
    keys = data[0][0].keys()
    out_dict = {}
    for key in keys:
        # if key == Modality.LIDAR.value:
        if key in {Modality.LIDAR.value, Modality.LABEL_LIDAR.value}:
            out_dict[key] = [d[0][key] for d in data]
        else:
            out_dict[key] = torch.stack([d[0][key] for d in data])
    idx = [d[1] for d in data]
    return out_dict, idx


def get_dataset(cfg: base_config.ExperimentConfig, create_test=False, is_ddp=False):

    if cfg.dataset.dataset_type == DatasetType.SEMANTIC_KITTI:
        ds_train = SemanticKITTIDataset(cfg.dataset, "train")
        ds_val = SemanticKITTIDataset(cfg.dataset, "val")
        ds_test = SemanticKITTIDataset(cfg.dataset, "test") if create_test else None
    elif cfg.dataset.dataset_type == DatasetType.NAVIO:
        ds_train = NavioDataset(cfg.dataset, "train")
        ds_val = NavioDataset(cfg.dataset, "val")
        ds_test = NavioDataset(cfg.dataset, "test") if create_test else None

    dataset = {}

    train_batch_size = cfg.dataloader.train_batch_size
    val_batch_size = cfg.dataloader.val_batch_size
    num_workers = cfg.dataloader.num_workers

    train_sampler = DistributedSampler(ds_train, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(ds_val, shuffle=False) if is_ddp else None
    test_sampler = DistributedSampler(ds_test, shuffle=False) if is_ddp and create_test else None

    dataset["train"] = DataLoader(ds_train, batch_size=train_batch_size, num_workers=num_workers, sampler=train_sampler,
                                  shuffle=(train_sampler is None), collate_fn=collate_fn, drop_last=True,
                                  pin_memory=True, persistent_workers=cfg.dataloader.num_workers > 0,
                                  prefetch_factor=2)
    dataset["val"] = DataLoader(ds_val, batch_size=val_batch_size, num_workers=num_workers, sampler=val_sampler,
                                shuffle=False, collate_fn=collate_fn, pin_memory=True,
                                persistent_workers=cfg.dataloader.num_workers > 0, prefetch_factor=2)
    dataset["test"] = DataLoader(ds_test, batch_size=val_batch_size, num_workers=num_workers, sampler=test_sampler,
                                 shuffle=False, collate_fn=collate_fn, pin_memory=True, prefetch_factor=2,
                                 persistent_workers=cfg.dataloader.num_workers > 0) if create_test else None

    return dataset

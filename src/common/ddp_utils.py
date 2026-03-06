import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def is_primary() -> bool:
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()

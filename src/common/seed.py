import os
import random

import numpy as np
import torch


def seed_all(seed):
    """
    Set seeds for training reproducibility
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

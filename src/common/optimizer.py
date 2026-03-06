from torch import optim

from src.common import base_config


def build_optimizer(cfg: base_config.OptimizerConfig, model):

    opt = cfg.optim_type
    lr = cfg.lr
    momentum = cfg.momentum
    weight_decay = cfg.weight_decay
    betas = cfg.betas

    if opt == "Adam":
        optimizer = optim.Adam(model.get_parameters(), lr=lr, betas=betas)

    elif opt == "AdamW":
        optimizer = optim.AdamW(model.get_parameters(), lr=lr, betas=betas)

    elif opt == "SGD":
        optimizer = optim.SGD(
            model.get_parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    return optimizer


def build_scheduler(cfg: base_config.SchedulerConfig, optimizer):

    # Constant learning rate
    if cfg.scheduler_type == "constant":
        lambda1 = lambda _epoch: 1
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # Learning rate scaled by 0.98^(epoch)
    if cfg.scheduler_type == "power_iteration":
        lambda1 = lambda epoch: (cfg.lr_power) ** (epoch)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    return scheduler

"""Runtime helpers for DataParallel wrapping."""
import torch
import torch.nn as nn


def parallelize(model):
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        return nn.DataParallel(model)
    return model

from contextlib import contextmanager

from torch import nn


@contextmanager
def freeze(model: nn.Module) -> nn.Module:
    original_state = {}

    for name, param in model.named_parameters():
        original_state[name] = param.requires_grad
        param.requires_grad = False

    yield model

    for name, param in model.named_parameters():
        param.requires_grad = original_state[name]

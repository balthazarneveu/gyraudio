import torch
from gyraudio.audio_separation.properties import (
    NAME, ANNOTATIONS, NB_PARAMS
)
REGISTERED_EXPERIMENTS_LIST = {}


def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters

    Args:
        model (torch.nn.Module): Pytorch model

    Returns:
        int: Number of trainable elements
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def registered_experiment(major=None):
    def decorator(func):
        assert (major) not in REGISTERED_EXPERIMENTS_LIST, f"Experiment {major} already registered"
        # REGISTERED_EXPERIMENTS_LIST[major] = func

        def wrapper(config, minor=None, no_model=False, model=torch.nn.Module()):
            config, model = func(config, model=None if not no_model else model, minor=minor)
            config[NB_PARAMS] = count_parameters(model)
            assert NAME in config, "NAME not defined"
            assert ANNOTATIONS in config, "ANNOTATIONS not defined"
            return model, config
        REGISTERED_EXPERIMENTS_LIST[major] = wrapper
        return wrapper

    return decorator

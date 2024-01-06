from gyraudio.audio_separation.architecture.flat_conv import FlatConvolutional
from gyraudio.audio_separation.architecture.unet import ResUNet
from gyraudio.audio_separation.properties import (
    NAME, ANNOTATIONS, MAX_STEPS_PER_EPOCH, EPOCHS, BATCH_SIZE
)
from gyraudio.audio_separation.experiment_tracking.experiments_decorator import (
    registered_experiment, REGISTERED_EXPERIMENTS_LIST

)


@registered_experiment(major=0)
def exp_unit_test(config, model: bool = None, minor=None):
    config[MAX_STEPS_PER_EPOCH] = 2
    config[BATCH_SIZE] = [4, 4, 4]
    config[EPOCHS] = 2
    config[NAME] = "Unit Test - Flat Convolutional"
    config[ANNOTATIONS] = "Baseline"
    if model is None:
        model = FlatConvolutional()
    return config, model


@registered_experiment(major=1)
def exp_1(config, model: bool = None, minor=None):
    config[EPOCHS] = 50
    config[NAME] = "Flat Convolutional"
    config[ANNOTATIONS] = "Baseline"
    if model is None:
        model = FlatConvolutional()
    return config, model


def exp_unet(config, h_dim=16, model=None):
    config[NAME] = "Res-UNet"
    config[ANNOTATIONS] = "Res-UNet-4 scales"
    if model is None:
        model = ResUNet(h_dim=h_dim)
    return config, model


@registered_experiment(major=2)
def exp_2_unet(config, model: bool = None, minor=None):
    config[BATCH_SIZE] = [16, 32, 32]
    config[EPOCHS] = 200
    config, model = exp_unet(config, model=model)
    return config, model


@registered_experiment(major=3)
def exp_3_unet(config, model: bool = None, minor=None):
    config[EPOCHS] = 200
    config[BATCH_SIZE] = [32, 64, 64]
    config, model = exp_unet(config, model=model)
    return config, model


def get_experiment_generator(exp_major: int):
    assert exp_major in REGISTERED_EXPERIMENTS_LIST, f"Experiment {exp_major} not registered"
    exp_generator = REGISTERED_EXPERIMENTS_LIST[exp_major]
    return exp_generator


if __name__ == "__main__":
    print(f"Available experiments: {list(REGISTERED_EXPERIMENTS_LIST.keys())}")

from gyraudio.audio_separation.architecture.flat_conv import FlatConvolutional
from gyraudio.audio_separation.architecture.unet import ResUNet
from gyraudio.audio_separation.architecture.wave_unet import WaveUNet
from gyraudio.audio_separation.architecture.transformer import TransformerModel
from gyraudio.audio_separation.properties import (
    NAME, ANNOTATIONS, MAX_STEPS_PER_EPOCH, EPOCHS, BATCH_SIZE,
    OPTIMIZER, LEARNING_RATE,
    DATALOADER,
    WEIGHT_DECAY,
    AUGMENTATION, AUG_TRIM, AUG_AWGN, AUG_RESCALE,
    LENGTHS, LENGTH_DIVIDER, TRIM_PROB
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

# ------------------ Res U-Net ------------------


def exp_resunet(config, h_dim=16, k_size=5, model=None):
    config[NAME] = "Res-UNet"
    scales = 4
    config[ANNOTATIONS] = f"Res-UNet-{scales}scales_h={h_dim}_k={k_size}"
    config["Architecture"] = {
        "name": "Res-UNet",
        "h_dim": h_dim,
        "scales": scales,
        "k_size": k_size,
    }
    if model is None:
        model = ResUNet(h_dim=h_dim, k_size=k_size)
    return config, model

# ------------------ Wave U-Net ------------------


def exp_wave_unet(config: dict,
                  channels_extension: int = 24,
                  k_conv_ds: int = 15,
                  k_conv_us: int = 5,
                  num_layers: int = 4,
                  dropout: float = 0.0,
                  model=None):
    config[NAME] = "Wave-UNet"
    config[ANNOTATIONS] = f"Wave-UNet-{num_layers}scales_h_ext={channels_extension}_k={k_conv_ds}ds-{k_conv_us}us"
    if dropout > 0:
        config[ANNOTATIONS] += f"-dr{dropout:.1e}"
    config["Architecture"] = {
        "k_conv_us": k_conv_us,
        "k_conv_ds": k_conv_ds,
        "num_layers": num_layers,
        "channels_extension": channels_extension,
        "dropout": dropout
    }
    if model is None:
        model = WaveUNet(
            **config["Architecture"]
        )
    config["Architecture"][NAME] = "Wave-UNet"
    return config, model


def get_experiment_generator(exp_major: int):
    assert exp_major in REGISTERED_EXPERIMENTS_LIST, f"Experiment {exp_major} not registered"
    exp_generator = REGISTERED_EXPERIMENTS_LIST[exp_major]
    return exp_generator


if __name__ == "__main__":
    print(f"Available experiments: {list(REGISTERED_EXPERIMENTS_LIST.keys())}")

from gyraudio.audio_separation.architecture.flat_conv import FlatConvolutional
from gyraudio.audio_separation.architecture.unet import ResUNet
from gyraudio.audio_separation.architecture.wave_unet import WaveUNet
from gyraudio.audio_separation.architecture.transformer import TransformerModel
from gyraudio.audio_separation.properties import (
    NAME, ANNOTATIONS, MAX_STEPS_PER_EPOCH, EPOCHS, BATCH_SIZE, OPTIMIZER, LEARNING_RATE,
    WEIGHT_DECAY
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


@registered_experiment(major=2)
def exp_2_resunet(config, model: bool = None, minor=None):
    config[BATCH_SIZE] = [16, 16, 16]
    config[EPOCHS] = 200
    config, model = exp_resunet(config, model=model)
    return config, model


@registered_experiment(major=3)
def exp_3_resunet(config, model: bool = None, minor=None):
    config[EPOCHS] = 200
    config[BATCH_SIZE] = [32, 32, 32]
    config, model = exp_resunet(config, model=model)
    return config, model


@registered_experiment(major=4)
def exp_4_resunet(config, model: bool = None, minor=None):
    config[EPOCHS] = 200
    config[BATCH_SIZE] = [16, 16, 16]
    config, model = exp_resunet(config, model=model, k_size=7)
    return config, model


@registered_experiment(major=5)
def exp_5_resunet(config, model: bool = None, minor=None):
    config[EPOCHS] = 200
    config[BATCH_SIZE] = [8, 8, 8]
    config, model = exp_resunet(config, model=model, h_dim=32)
    return config, model

# ------------------ Wave U-Net ------------------


def exp_wave_unet(config: dict,
                  channels_extension: int = 24,
                  k_conv_ds: int = 15,
                  k_conv_us: int = 5,
                  num_layers: int = 4,
                  model=None):
    config[NAME] = "Wave-UNet"
    config[ANNOTATIONS] = f"Wave-UNet-{num_layers}scales_h_ext={channels_extension}_k={k_conv_ds}ds-{k_conv_us}us"
    config["Architecture"] = {
        "k_conv_us": k_conv_us,
        "k_conv_ds": k_conv_ds,
        "num_layers": num_layers,
        "channels_extension": channels_extension
    }
    if model is None:
        model = WaveUNet(
            **config["Architecture"]
        )
    config["Architecture"][NAME] = "Wave-UNet"
    return config, model


@registered_experiment(major=300)
def exp_300_waveunet(config, model: bool = None, minor=None):
    config[BATCH_SIZE] = [16, 16, 16]
    config[EPOCHS] = 60
    config, model = exp_wave_unet(config, model=model, num_layers=4, channels_extension=24)
    # 4 layers, ext +24 - Nvidia T500 4Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=301)
def exp_301_waveunet(config, model: bool = None, minor=None):
    config[BATCH_SIZE] = [16, 16, 16]
    config[EPOCHS] = 60
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=16)
    # 7 layers, ext +16 - Nvidia T500 4Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=302)
def exp_302_waveunet(config, model: bool = None, minor=None):
    config[BATCH_SIZE] = [16, 16, 16]
    config[EPOCHS] = 60
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=24)
    # 7 layers, ext +24 - Nvidia RTX3060 6Gb RAM - 16 batch size
    return config, model

@registered_experiment(major=303)
def exp_303_waveunet(config, model: bool = None, minor=None):
    config[BATCH_SIZE] = [16, 16, 16]
    config[EPOCHS] = 100
    # config[OPTIMIZER][LEARNING_RATE] = 0.0005 # DIVERGE
    config[OPTIMIZER][WEIGHT_DECAY] = 0.01
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=24)
    # 7 layers, ext +24 - Nvidia RTX3060 6Gb RAM - 16 batch size
    return config, model

# ------------------ TRANSFORMER ------------------


def exp_transformer(
    config,
    model=None,
    nhead: int = 4,  # H
    nlayers: int = 4,  # L
    dropout: float = 0.,  # dr
    embedding_dim: int = 64,  # D
    ch_in: int = 1,
    ch_out: int = 1,
    k_size=5,
    positional_encoding: str = None,
):
    config[NAME] = "Transformer"

    config["Architecture"] = dict(
        nhead=nhead,
        nlayers=nlayers,
        dropout=dropout,
        embedding_dim=embedding_dim,
        ch_in=ch_in,
        ch_out=ch_out,
        k_size=k_size,
        positional_encoding=positional_encoding
    )
    config[ANNOTATIONS] = f"Transformer-{nhead}H-{nlayers}L-D={embedding_dim}-k={k_size}"
    if dropout > 0:
        config[ANNOTATIONS] += f"-dropout={dropout:.3f}"
    if model is None:
        model = TransformerModel(
            **config["Architecture"]
        )
    config["Architecture"][NAME] = "Transformer"
    return config, model


@registered_experiment(major=200)
def exp_200_transformer_baseline(config, model: bool = None, minor=None):
    config[EPOCHS] = 200
    config[BATCH_SIZE] = [8, 8, 8]
    config, model = exp_transformer(config, model=model, nhead=2, nlayers=2, dropout=0., embedding_dim=16)
    return config, model


def get_experiment_generator(exp_major: int):
    assert exp_major in REGISTERED_EXPERIMENTS_LIST, f"Experiment {exp_major} not registered"
    exp_generator = REGISTERED_EXPERIMENTS_LIST[exp_major]
    return exp_generator


if __name__ == "__main__":
    print(f"Available experiments: {list(REGISTERED_EXPERIMENTS_LIST.keys())}")

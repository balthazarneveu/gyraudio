from gyraudio.audio_separation.architecture.flat_conv import FlatConvolutional
from gyraudio.audio_separation.architecture.unet import ResUNet
from gyraudio.audio_separation.architecture.wave_unet import WaveUNet
from gyraudio.audio_separation.architecture.neutral import NeutralModel
from gyraudio.audio_separation.architecture.transformer import TransformerModel
from gyraudio.audio_separation.properties import (
    NAME, ANNOTATIONS, MAX_STEPS_PER_EPOCH, EPOCHS, BATCH_SIZE,
    OPTIMIZER, LEARNING_RATE,
    DATALOADER,
    WEIGHT_DECAY,
    LOSS, LOSS_L1,
    AUGMENTATION, AUG_TRIM, AUG_AWGN, AUG_RESCALE,
    LENGTHS, LENGTH_DIVIDER, TRIM_PROB,
    SCHEDULER, SCHEDULER_CONFIGURATION
)
from gyraudio.audio_separation.experiment_tracking.experiments_decorator import (
    registered_experiment, REGISTERED_EXPERIMENTS_LIST
)


@registered_experiment(major=9999)
def neutral(config, model: bool = None, minor=None):
    config[BATCH_SIZE] = [4, 4, 4]
    config[EPOCHS] = 1
    config[NAME] = "Neutral"
    config[ANNOTATIONS] = "Neutral"
    if model is None:
        model = NeutralModel()
    config[NAME] = "Neutral"
    return config, model


@registered_experiment(major=0)
def exp_unit_test(config, model: bool = None, minor=None):
    config[MAX_STEPS_PER_EPOCH] = 2
    config[BATCH_SIZE] = [4, 4, 4]
    config[EPOCHS] = 2
    config[NAME] = "Unit Test - Flat Convolutional"
    config[ANNOTATIONS] = "Baseline"
    config[SCHEDULER] = "ReduceLROnPlateau"
    config[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.8)
    if model is None:
        model = FlatConvolutional()
    return config, model

# ---------------- Low Baseline -----------------


def exp_low_baseline(
    config: dict,
    batch_size: int = 16,
    h_dim: int = 16,
    k_size: int = 9,
    dilation: int = 0,
    model: bool = None,
    minor=None
):
    config[BATCH_SIZE] = [batch_size, batch_size, batch_size]
    config[NAME] = "Flat Convolutional"
    config[ANNOTATIONS] = f"Baseline H={h_dim}_K={k_size}"
    if dilation > 1:
        config[ANNOTATIONS] += f"_dil={dilation}"
    config["Architecture"] = {
        "name": "Flat-Conv",
        "h_dim": h_dim,
        "scales": 1,
        "k_size": k_size,
        "dilation": dilation
    }
    if model is None:
        model = FlatConvolutional(k_size=k_size, h_dim=h_dim)
    return config, model


@registered_experiment(major=1)
def exp_1(config, model: bool = None, minor=None):
    config, model = exp_low_baseline(config, batch_size=32, k_size=5)
    return config, model


@registered_experiment(major=2)
def exp_2(config, model: bool = None, minor=None):
    config, model = exp_low_baseline(config, batch_size=32, k_size=9)
    return config, model


@registered_experiment(major=3)
def exp_3(config, model: bool = None, minor=None):
    config, model = exp_low_baseline(config, batch_size=32, k_size=9, dilation=2)
    return config, model


@registered_experiment(major=4)
def exp_4(config, model: bool = None, minor=None):
    config, model = exp_low_baseline(config, batch_size=16, k_size=9)
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


@registered_experiment(major=2000)
def exp_2000_waveunet(config, model: bool = None, minor=None):
    config[EPOCHS] = 60
    config, model = exp_resunet(config)
    return config, model


@registered_experiment(major=2001)
def exp_2001_waveunet(config, model: bool = None, minor=None):
    config[EPOCHS] = 60
    config, model = exp_resunet(config, h_dim=32, k_size=5)
    return config, model

# ------------------ Wave U-Net ------------------


def exp_wave_unet(config: dict,
                  channels_extension: int = 24,
                  k_conv_ds: int = 15,
                  k_conv_us: int = 5,
                  num_layers: int = 4,
                  dropout: float = 0.0,
                  bias: bool = True,
                  model=None):
    config[NAME] = "Wave-UNet"
    config[ANNOTATIONS] = f"Wave-UNet-{num_layers}scales_h_ext={channels_extension}_k={k_conv_ds}ds-{k_conv_us}us"
    if dropout > 0:
        config[ANNOTATIONS] += f"-dr{dropout:.1e}"
    if not bias:
        config[ANNOTATIONS] += "-BiasFree"
    config["Architecture"] = {
        "k_conv_us": k_conv_us,
        "k_conv_ds": k_conv_ds,
        "num_layers": num_layers,
        "channels_extension": channels_extension,
        "dropout": dropout,
        "bias": bias
    }
    if model is None:
        model = WaveUNet(
            **config["Architecture"]
        )
    config["Architecture"][NAME] = "Wave-UNet"
    return config, model


@registered_experiment(major=1000)
def exp_1000_waveunet(config, model: bool = None, minor=None):
    config[EPOCHS] = 60
    config, model = exp_wave_unet(config, model=model, num_layers=4, channels_extension=24)
    # 4 layers, ext +24 - Nvidia T500 4Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=1001)
def exp_1001_waveunet(config, model: bool = None, minor=None):
    # OVERFIT 1M param ?
    config[EPOCHS] = 60
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=16)
    # 7 layers, ext +16 - Nvidia T500 4Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=1002)
def exp_1002_waveunet(config, model: bool = None, minor=None):
    # OVERFIT 1M param ?
    config[EPOCHS] = 60
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=16)
    config[DATALOADER][AUGMENTATION] = {
        AUG_TRIM: {LENGTHS: [8192, 80000], LENGTH_DIVIDER: 1024, TRIM_PROB: 0.8},
        AUG_RESCALE: True
    }
    # 7 layers, ext +16 - Nvidia T500 4Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=1003)
def exp_1003_waveunet(config, model: bool = None, minor=None):
    # OVERFIT 2.3M params
    config[EPOCHS] = 60
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=24)
    # 7 layers, ext +24 - Nvidia RTX3060 6Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=1004)
def exp_1004_waveunet(config, model: bool = None, minor=None):
    config[EPOCHS] = 120
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=28)
    # 7 layers, ext +28 - Nvidia RTX3060 6Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=1014)
def exp_1014_waveunet(config, model: bool = None, minor=None):
    # trained with min and max mixing snr hard coded between -2 and -1
    config[EPOCHS] = 50
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=28)
    # 7 layers, ext +28 - Nvidia RTX3060 6Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=1005)
def exp_1005_waveunet(config, model: bool = None, minor=None):
    config[EPOCHS] = 150
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=16)
    config[DATALOADER][AUGMENTATION] = {
        AUG_TRIM: {LENGTHS: [8192, 80000], LENGTH_DIVIDER: 1024, TRIM_PROB: 0.8},
        AUG_RESCALE: True
    }
    # 7 layers, ext +16 - Nvidia T500 4Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=1006)
def exp_1006_waveunet(config, model: bool = None, minor=None):
    config[EPOCHS] = 150
    config, model = exp_wave_unet(config, model=model, num_layers=11, channels_extension=16)
    config[DATALOADER][AUGMENTATION] = {
        AUG_TRIM: {LENGTHS: [8192, 80000], LENGTH_DIVIDER: 4096, TRIM_PROB: 0.8},
        AUG_RESCALE: True
    }
    # 11 layers, ext +16 - Nvidia T500 4Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=1007)
def exp_1007_waveunet(config, model: bool = None, minor=None):
    config[EPOCHS] = 150
    config, model = exp_wave_unet(config, model=model, num_layers=9, channels_extension=16)
    config[DATALOADER][AUGMENTATION] = {
        AUG_TRIM: {LENGTHS: [8192, 80000], LENGTH_DIVIDER: 4096, TRIM_PROB: 0.8},
        AUG_RESCALE: True
    }
    # 11 layers, ext +16 - Nvidia T500 4Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=1008)
def exp_1008_waveunet(config, model: bool = None, minor=None):
    # CHEAP BASELINE
    config[EPOCHS] = 150
    config, model = exp_wave_unet(config, model=model, num_layers=4, channels_extension=16)
    config[DATALOADER][AUGMENTATION] = {
        AUG_TRIM: {LENGTHS: [8192, 80000], LENGTH_DIVIDER: 1024, TRIM_PROB: 0.8},
        AUG_RESCALE: True
    }
    # 4 layers, ext +16 - Nvidia T500 4Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=3000)
def exp_3000_waveunet(config, model: bool = None, minor=None):
    config[EPOCHS] = 120
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=28, bias=False)
    # 7 layers, ext +28 - Nvidia RTX3060 6Gb RAM - 16 batch size
    return config, model


@registered_experiment(major=3001)
def exp_3001_waveunet(config, model: bool = None, minor=None):
    config[EPOCHS] = 200
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=28, bias=False)
    # 7 layers, ext +28 - Nvidia RTX3060 6Gb RAM - 16 batch size
    config[SCHEDULER] = "ReduceLROnPlateau"
    config[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.8)
    config[OPTIMIZER][LEARNING_RATE] = 0.002
    return config, model


@registered_experiment(major=3002)
def exp_3002_waveunet(config, model: bool = None, minor=None):
    # TRAINED WITH SNR -12db +12db (code changed manually!)
    # See f910c6da3123e3d35cc0ce588bb5a72ce4a8c422
    config[EPOCHS] = 200
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=28, bias=False)
    # 7 layers, ext +28 - Nvidia RTX3060 6Gb RAM - 16 batch size
    config[SCHEDULER] = "ReduceLROnPlateau"
    config[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.8)
    config[OPTIMIZER][LEARNING_RATE] = 0.002
    return config, model


@registered_experiment(major=4000)
def exp_4000_bias_free_waveunet_l1(config, model: bool = None, minor=None):
    # config[MAX_STEPS_PER_EPOCH] = 2
    # config[BATCH_SIZE] = [2, 2, 2]
    config[EPOCHS] = 200
    config[LOSS] = LOSS_L1
    config, model = exp_wave_unet(config, model=model, num_layers=7, channels_extension=28, bias=False)
    # 7 layers, ext +28 - Nvidia RTX3060 6Gb RAM - 16 batch size
    config[SCHEDULER] = "ReduceLROnPlateau"
    config[SCHEDULER_CONFIGURATION] = dict(patience=5, factor=0.8)
    config[OPTIMIZER][LEARNING_RATE] = 0.002
    return config, model


def get_experiment_generator(exp_major: int):
    assert exp_major in REGISTERED_EXPERIMENTS_LIST, f"Experiment {exp_major} not registered"
    exp_generator = REGISTERED_EXPERIMENTS_LIST[exp_major]
    return exp_generator


if __name__ == "__main__":
    print(f"Available experiments: {list(REGISTERED_EXPERIMENTS_LIST.keys())}")

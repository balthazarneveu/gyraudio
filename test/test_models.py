from gyraudio.audio_separation.architecture.flat_conv import FlatConvolutional
from gyraudio.audio_separation.architecture.unet import ResUNet
from gyraudio.audio_separation.architecture.wave_unet import WaveUNet


def test_rf():
    for model in [FlatConvolutional(), WaveUNet(), ResUNet()]:
        rf = model.receptive_field()
        print(f"RF={rf}")


def test_analytic_rf():
    for k_size in [3, 5, 7, 9, 11]:
        model = FlatConvolutional(k_size=k_size)
        rf = model.receptive_field()
        rf_analytic = 4*(k_size-1) + 1
        rf_analytic = rf_analytic
        assert rf == rf_analytic
        print(f"RF={rf}")

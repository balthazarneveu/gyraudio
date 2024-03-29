from gyraudio.audio_separation.architecture.flat_conv import FlatConvolutional
from gyraudio.audio_separation.architecture.unet import ResUNet
from gyraudio.audio_separation.architecture.wave_unet import WaveUNet
import torch


def test_rf():
    for model in [FlatConvolutional(), WaveUNet(), ResUNet()]:
        rf = model.receptive_field()
        print(f"RF={rf}")


def test_analytic_rf():
    for k_size in [3, 5, 7, 9, 11]:
        model = FlatConvolutional(k_size=k_size)
        rf = model.receptive_field()
        rf_analytic = 4*(k_size-1) + 1
        assert rf == rf_analytic
        print(f"RF={rf}")


def test_inference_shapes_flat_conv():
    # FIXME! This test is not working because the receptive field is not computed correctly
    for dilation in [1, 2, 4, 8]:
        model = FlatConvolutional(k_size=9, dilation=dilation)
        rf = model.receptive_field()
        print(f"{dilation=:} {rf=:}")
        inp = torch.randn(1, 1, 2048)
        outs, outn = model(inp)
        assert outn.shape == outs.shape == inp.shape

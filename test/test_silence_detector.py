
from gyraudio.audio_separation.data.silence_detector import get_silence_mask
from gyraudio.default_locations import SAMPLE_ROOT
from gyraudio.audio_separation.visualization.pre_load_audio import audio_loading
from gyraudio.audio_separation.properties import CLEAN, BUFFERS
import torch


def test_silence_detection():
    sample_folder = SAMPLE_ROOT/"0009"
    signals = audio_loading(sample_folder, preload=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sig_in = signals[BUFFERS][CLEAN].to(device)
    silence_mask = get_silence_mask(
        sig_in,
        morph_kernel_size=499, k_smooth=21, thresh=0.0001
    )
    assert silence_mask.bool().sum() == 31657
    assert silence_mask.shape == sig_in.shape

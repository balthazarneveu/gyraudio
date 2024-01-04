import scipy
import numpy as np
import torchaudio
import torch
from pathlib import Path
from typing import Tuple


def load_raw_audio(path: str) -> np.array:
    assert path.exists(), f"Audio path {path} does not exist"
    rate, signal = scipy.io.wavfile.read(path)
    return rate, signal


def load_audio_tensor(path: Path, device=None) -> Tuple[torch.Tensor, int]:
    assert path.exists(), f"Audio path {path} does not exist"
    signal, rate = torchaudio.load(str(path))
    if device is not None:
        signal = signal.to(device)
    return signal, rate


def save_audio_tensor(path: Path, signal: torch.Tensor, sampling_rate: int):
    torchaudio.save(
        str(path),
        signal.detach().cpu(),
        sample_rate=sampling_rate,
        channels_first=True
    )

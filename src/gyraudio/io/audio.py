import scipy
import numpy as np


def load_raw_audio(path: str) -> np.array:
    assert path.exists(), f"Audio path {path} does not exist"
    rate, signal = scipy.io.wavfile.read(path)
    return rate, signal

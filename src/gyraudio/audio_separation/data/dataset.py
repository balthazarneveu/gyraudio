from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import default_collate
from typing import Tuple
from gyraudio.audio_separation.properties import (
    AUG_AWGN, AUG_RESCALE
)


class AudioDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        augmentation_config: dict = {},
        snr_filter: Optional[float] = None,
        debug: bool = False
    ):
        self.debug = debug
        self.data_path = data_path
        self.augmentation_config = augmentation_config
        self.snr_filter = snr_filter
        self.load_data()
        self.length = len(self.file_list)

    def filter_data(self, snr):
        if self.snr_filter is None:
            return True
        if snr in self.snr_filter:
            return True
        else:
            return False

    def load_data(self):
        raise NotImplementedError("load_data method must be implemented")
    
    def augment_data(self, mixed_audio_signal, noise_audio_signal,  clean_audio_signal) :
        if AUG_RESCALE in self.augmentation_config:
            current_amplitude = 0.5 + 1.5*torch.rand(1, device=mixed_audio_signal.device)
            # logging.debug(current_amplitude)
            mixed_audio_signal *= current_amplitude
            noise_audio_signal *= current_amplitude
            clean_audio_signal *= current_amplitude
        if AUG_AWGN in self.augmentation_config:
            # noise_std = self.augmentation_config[AUG_AWGN]["noise_std"]
            noise_std = 0.01
            current_noise_std = torch.randn(1) * noise_std
            # logging.debug(current_noise_std)
            extra_awgn = torch.randn(mixed_audio_signal.shape, device=mixed_audio_signal.device) * current_noise_std
            mixed_audio_signal = mixed_audio_signal+extra_awgn
            # Open question: should we add noise to the noise signal aswell?

        return mixed_audio_signal, clean_audio_signal, noise_audio_signal

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        raise NotImplementedError("__getitem__ method must be implemented")


def trim_collate_mix(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function to allow trimming (=crop the time dimension) of the signals in a batch.

    Args:
    batch (list): A list of tuples (triplets), where each tuple contain:
    - mixed_audio_signal
    - clean_audio_signal
    - noise_audio_signal

    Returns:
    - Tensor: A batch of mixed_audio_signal, trimmed to the same length.
    - Tensor: A batch of clean_audio_signal
    - Tensor: A batch of noise_audio_signal
    """

    # Find the length of the shortest signal in the batch
    mixed_audio_signal, clean_audio_signal, noise_audio_signal = default_collate(batch)
    length = mixed_audio_signal[0].shape[-1]
    take_full_signal = torch.rand(1) > 0.5
    if not take_full_signal:
        trim_length = torch.randint(2048, length-1, (1,))
        trim_length = trim_length-trim_length % 1024
        start = torch.randint(0, length-trim_length, (1,))
        end = start + trim_length
        mixed_audio_signal = mixed_audio_signal[..., start:end]
        clean_audio_signal = clean_audio_signal[..., start:end]
        noise_audio_signal = noise_audio_signal[..., start:end]
    return mixed_audio_signal, clean_audio_signal, noise_audio_signal

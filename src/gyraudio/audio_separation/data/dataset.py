from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import default_collate
from typing import Tuple
from functools import partial
from gyraudio.audio_separation.properties import (
    AUG_AWGN, AUG_RESCALE, AUG_TRIM, LENGTHS, LENGTH_DIVIDER, TRIM_PROB
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
        self.collate_fn = None
        if AUG_TRIM in self.augmentation_config:
            self.collate_fn = partial(collate_fn_generic, 
                                      lengths_lim = self.augmentation_config[AUG_TRIM][LENGTHS],
                                        length_divider = self.augmentation_config[AUG_TRIM][LENGTH_DIVIDER],
                                         trim_prob = self.augmentation_config[AUG_TRIM][TRIM_PROB])
    def filter_data(self, snr):
        if self.snr_filter is None:
            return True
        if snr in self.snr_filter:
            return True
        else:
            return False

    def load_data(self):
        raise NotImplementedError("load_data method must be implemented")
    
    def augment_data(self, mixed_audio_signal, clean_audio_signal, noise_audio_signal) :
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


def collate_fn_generic(batch, lengths_lim, length_divider = 1024, trim_prob = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function to allow trimming (=crop the time dimension) of the signals in a batch.

    Args:
    batch (list): A list of tuples (triplets), where each tuple contain:
    - mixed_audio_signal
    - clean_audio_signal
    - noise_audio_signal
    lengths_lim (list) : A list of containing a minimum length (0) and a maximum length (1)
    length_divider (int) : has to be a trimmed length divider
    trim_prob (float) : trimming probability

    Returns:
    - Tensor: A batch of mixed_audio_signal, trimmed to the same length.
    - Tensor: A batch of clean_audio_signal
    - Tensor: A batch of noise_audio_signal
    """

    # Find the length of the shortest signal in the batch
    mixed_audio_signal, clean_audio_signal, noise_audio_signal = default_collate(batch)
    length = mixed_audio_signal[0].shape[-1]
    min_length, max_length = lengths_lim
    take_full_signal = torch.rand(1) > trim_prob
    if not take_full_signal:
        start = torch.randint(0, length-min_length, (1,))
        trim_length = torch.randint(min_length, min(max_length, length-start-1), (1,))
        trim_length = trim_length-trim_length % length_divider
        end = start + trim_length
        mixed_audio_signal = mixed_audio_signal[..., start:end]
        clean_audio_signal = clean_audio_signal[..., start:end]
        noise_audio_signal = noise_audio_signal[..., start:end]
    return mixed_audio_signal, clean_audio_signal, noise_audio_signal

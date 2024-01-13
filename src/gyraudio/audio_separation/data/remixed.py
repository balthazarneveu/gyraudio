from gyraudio.audio_separation.data.dataset import AudioDataset
from typing import Tuple
import logging
from torch import Tensor
import torch
import torchaudio
from random import randint
from gyraudio.audio_separation.properties import (
    AUG_AWGN, AUG_RESCALE
)


class RemixedAudioDataset(AudioDataset):
    def load_data(self):
        self.folder_list = sorted(list(self.data_path.iterdir()))
        self.file_list = [
            [
                folder/"voice.wav",
                folder/"noise.wav"
            ] for folder in self.folder_list
        ]
        if self.debug:
            print("Filtered", len(self.file_list), self.snr_filter)
        self.sampling_rate = None

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        signal_path = self.file_list[idx][0]
        idx_noise = randint(0, len(self.file_list)-1)
        noise_path = self.file_list[idx_noise][1]

        assert signal_path.exists()
        assert noise_path.exists()
        clean_audio_signal, sampling_rate = torchaudio.load(str(signal_path))
        noise_audio_signal, sampling_rate = torchaudio.load(str(noise_path))
        min_snr, max_snr = -4, 4
        snr = min_snr + (max_snr-min_snr)*torch.rand(1)
        alpha = 10 ** (-snr / 20) * torch.norm(clean_audio_signal) / torch.norm(noise_audio_signal)
        mixed_audio_signal = clean_audio_signal + alpha*noise_audio_signal
        power_target_sqrt = 16.
        mixed_audio_signal = mixed_audio_signal * power_target_sqrt / torch.norm(mixed_audio_signal)
        self.sampling_rate = sampling_rate
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
        if self.debug:
            logging.debug(f"{mixed_audio_signal.shape}")
            logging.debug(f"{clean_audio_signal.shape}")
            logging.debug(f"{noise_audio_signal.shape}")
        return mixed_audio_signal, clean_audio_signal, noise_audio_signal

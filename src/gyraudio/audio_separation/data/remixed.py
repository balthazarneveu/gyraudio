from gyraudio.audio_separation.data.dataset import AudioDataset
from typing import Tuple
import logging
from torch import Tensor
import torch
import torchaudio
from random import randint


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
        if self.debug:
            logging.debug(f"{mixed_audio_signal.shape}")
            logging.debug(f"{clean_audio_signal.shape}")
            logging.debug(f"{noise_audio_signal.shape}")
        return mixed_audio_signal, clean_audio_signal, noise_audio_signal

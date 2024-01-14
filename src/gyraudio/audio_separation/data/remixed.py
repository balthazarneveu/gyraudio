from gyraudio.audio_separation.data.dataset import AudioDataset
from typing import Tuple
import logging
from torch import Tensor
import torch
import torchaudio


class RemixedAudioDataset(AudioDataset):
    def generate_snr_list(self) :
        self.snr_list = None

    def load_data(self):
        self.folder_list = sorted(list(self.data_path.iterdir()))
        self.file_list = [
            [
                folder/"voice.wav",
                folder/"noise.wav"
            ] for folder in self.folder_list
        ]
        self.sampling_rate = None
        self.min_snr, self.max_snr = -4, 4
        self.generate_snr_list()
        if self.debug:
            print("Not filtered", len(self.file_list), self.snr_filter)
            print(self.snr_list)


    def get_idx_noise(self, idx) :
        raise NotImplementedError("get_idx_noise method must be implemented")

    def get_snr(self, idx) :
        raise NotImplementedError("get_snr method must be implemented")

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        signal_path = self.file_list[idx][0]
        idx_noise = self.get_idx_noise(idx)
        noise_path = self.file_list[idx_noise][1]

        assert signal_path.exists()
        assert noise_path.exists()
        clean_audio_signal, sampling_rate = torchaudio.load(str(signal_path))
        noise_audio_signal, sampling_rate = torchaudio.load(str(noise_path))
        snr = self.get_snr(idx)
        alpha = 10 ** (-snr / 20) * torch.norm(clean_audio_signal) / torch.norm(noise_audio_signal)
        mixed_audio_signal = clean_audio_signal + alpha*noise_audio_signal
        self.sampling_rate = sampling_rate
        mixed_audio_signal, clean_audio_signal, noise_audio_signal = self.augment_data(mixed_audio_signal, clean_audio_signal, noise_audio_signal)
        if self.debug:
            logging.debug(f"{mixed_audio_signal.shape}")
            logging.debug(f"{clean_audio_signal.shape}")
            logging.debug(f"{noise_audio_signal.shape}")
        return mixed_audio_signal, clean_audio_signal, noise_audio_signal

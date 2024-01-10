from gyraudio.audio_separation.data.dataset import AudioDataset
from gyraudio.audio_separation.properties import AUG_AWGN, AUG_RESCALE
import logging
import torch
import torchaudio
from typing import Tuple


class MixedAudioDataset(AudioDataset):
    def load_data(self):
        self.folder_list = sorted(list(self.data_path.iterdir()))
        self.file_list = [
            [
                list(folder.glob("mix*.wav"))[0],
                folder/"voice.wav",
                folder/"noise.wav"
            ] for folder in self.folder_list
        ]
        snr_list = [float(file[0].stem.split("_")[-1]) for file in self.file_list]
        self.file_list = [files for snr, files in zip(snr_list, self.file_list) if self.filter_data(snr)]
        if self.debug:
            logging.info(f"Available SNR {set(snr_list)}")
            print(f"Available SNR {set(snr_list)}")
            print("Filtered", len(self.file_list), self.snr_filter)
        self.sampling_rate = None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed_audio_path, signal_path, noise_path = self.file_list[idx]
        assert mixed_audio_path.exists()
        assert signal_path.exists()
        assert noise_path.exists()
        mixed_audio_signal, sampling_rate = torchaudio.load(str(mixed_audio_path))
        clean_audio_signal, sampling_rate = torchaudio.load(str(signal_path))
        noise_audio_signal, sampling_rate = torchaudio.load(str(noise_path))
        self.sampling_rate = sampling_rate
        if AUG_RESCALE in self.augmentation_config:
            noise_std = self.augmentation_config[AUG_RESCALE]
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
            clean_audio_signal = clean_audio_signal+extra_awgn
            # Open question: should we add noise to the noise signal aswell?
        if self.debug:
            logging.debug(f"{mixed_audio_signal.shape}")
            logging.debug(f"{clean_audio_signal.shape}")
            logging.debug(f"{noise_audio_signal.shape}")
        return mixed_audio_signal, clean_audio_signal, noise_audio_signal

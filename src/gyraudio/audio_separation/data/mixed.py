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
        mixed_audio_signal, clean_audio_signal, noise_audio_signal = self.augment_data(mixed_audio_signal, clean_audio_signal, noise_audio_signal)
        if self.debug:
            logging.debug(f"{mixed_audio_signal.shape}")
            logging.debug(f"{clean_audio_signal.shape}")
            logging.debug(f"{noise_audio_signal.shape}")
        return mixed_audio_signal, clean_audio_signal, noise_audio_signal

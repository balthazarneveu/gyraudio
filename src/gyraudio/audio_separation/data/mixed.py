from gyraudio.audio_separation.data.dataset import AudioDataset
from typing import Tuple
import logging
from torch import Tensor
import torchaudio


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

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        mixed_audio_path, signal_path, noise_path = self.file_list[idx]
        assert mixed_audio_path.exists()
        assert signal_path.exists()
        assert noise_path.exists()
        mixed_audio_signal, sampling_rate = torchaudio.load(str(mixed_audio_path))
        clean_audio_signal, sampling_rate = torchaudio.load(str(signal_path))
        noise_audio_signal, sampling_rate = torchaudio.load(str(noise_path))
        logging.debug(f"{mixed_audio_signal.shape}")
        logging.debug(f"{clean_audio_signal.shape}")
        logging.debug(f"{noise_audio_signal.shape}")
        return mixed_audio_signal, clean_audio_signal, noise_audio_signal

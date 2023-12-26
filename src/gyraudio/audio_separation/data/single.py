from gyraudio.audio_separation.data.dataset import AudioDataset
import logging
import torchaudio


class SingleAudioDataset(AudioDataset):
    def load_data(self):
        self.file_list = sorted(list(self.data_path.glob("*.wav")))

    def __getitem__(self, idx: int):
        audio_path = self.file_list[idx]
        assert audio_path.exists()
        audio_signal, sampling_rate = torchaudio.load(str(audio_path))
        logging.debug(f"{audio_signal.shape}")
        return audio_signal

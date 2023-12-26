from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional, List
from gyraudio.audio_separation.properties import (
    DATA_PATH, AUGMENTATION, SNR_FILTER, SHUFFLE, BATCH_SIZE, TRAIN, VALID, TEST
)
import logging
from gyraudio import root_dir
import torchaudio


class MixedAudioDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        augmentation_config: dict = {},
        snr_filter: Optional[float] = None
    ):
        self.data_path = data_path
        self.augmentation_config = augmentation_config
        self.snr_filter = snr_filter
        self.load_data()
        self.length = len(self.file_list)

    def load_data(self):
        self.folder_list = sorted(list(self.data_path.iterdir()))
        self.file_list = [
            [
                list(folder.glob("mix*.wav"))[0],
                folder/"voice.wav",
                folder/"noise.wav"
            ] for folder in self.folder_list
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        mixed_audio_path, signal_path, noise_path = self.file_list[idx]
        assert mixed_audio_path.exists()
        assert signal_path.exists()
        assert noise_path.exists()
        mixed_audio_signal, sampling_rate = torchaudio.load(str(mixed_audio_path))
        clean_audio_signal, sampling_rate = torchaudio.load(str(signal_path))
        noise_audio_signal, sampling_rate = torchaudio.load(str(noise_path))
        logging.info(f"{mixed_audio_signal.shape}")
        logging.info(f"{mixed_audio_signal.shape}")
        logging.info(f"{mixed_audio_signal.shape}")
        return mixed_audio_signal, clean_audio_signal, noise_audio_signal


class SingleAudioDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        augmentation_config: dict = {},
        snr_filter: Optional[float] = None
    ):
        self.data_path = data_path
        self.augmentation_config = augmentation_config
        self.snr_filter = snr_filter
        self.load_data()
        self.length = len(self.file_list)

    def load_data(self):
        self.file_list = sorted(list(self.data_path.glob("*.wav")))

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        audio_path = self.file_list[idx]
        assert audio_path.exists()
        audio_signal, sampling_rate = torchaudio.load(str(audio_path))
        logging.debug(f"{audio_signal.shape}")
        return audio_signal


def get_dataloader(configurations: dict, audio_dataset=MixedAudioDataset):
    dataloaders = {}
    for mode, configuration in configurations.items():
        dataset = audio_dataset(
            configuration[DATA_PATH],
            augmentation_config=configuration[AUGMENTATION],
            snr_filter=configuration[SNR_FILTER]
        )
        dl = DataLoader(
            dataset,
            shuffle=configuration[SHUFFLE],
            batch_size=configuration[BATCH_SIZE]
        )
        dataloaders[mode] = dl
    return dataloaders


RAW_AUDIO_ROOT = root_dir/"__data_source_separation"/"voice_origin"
MIXED_AUDIO_ROOT = root_dir/"__data_source_separation"/"source_separation"


def get_config_dataloader(
        audio_root=MIXED_AUDIO_ROOT,
        mode: str = TRAIN,
        shuffle: Optional[bool] = None,
        batch_size: Optional[int] = 16,
        augmentation: List[str] = []):
    audio_folder = audio_root/mode
    assert mode in [TRAIN, VALID, TEST]
    assert audio_folder.exists()
    augmentation_dict = {}
    for augmentation_str in augmentation:
        augmentation_dict[augmentation_str] = True
    config = {
        DATA_PATH: audio_folder,
        SHUFFLE: shuffle if shuffle is not None else (True if mode == TRAIN else False),
        AUGMENTATION: augmentation_dict,
        SNR_FILTER: None,
        BATCH_SIZE: batch_size
    }
    return config


def check_single_audio():
    train_set = get_config_dataloader(audio_root=RAW_AUDIO_ROOT, mode=TRAIN)
    test_set = get_config_dataloader(audio_root=RAW_AUDIO_ROOT, mode=TEST)
    dataloaders = get_dataloader({TRAIN: train_set, TEST: test_set}, audio_dataset=SingleAudioDataset)
    print(len(dataloaders[TRAIN]), len(dataloaders[TEST]))
    batch_signal = next(iter(dataloaders[TRAIN]))
    print(batch_signal.shape)


def check_mixed_audio():
    train_set = get_config_dataloader(audio_root=MIXED_AUDIO_ROOT, mode=TRAIN)
    test_set = get_config_dataloader(audio_root=MIXED_AUDIO_ROOT, mode=TEST)
    dataloaders = get_dataloader({TRAIN: train_set, TEST: test_set})
    batch_mix, batch_signal, batch_noise = next(iter(dataloaders[TRAIN]))
    print(batch_mix.shape, batch_signal.shape, batch_noise.shape)


if __name__ == '__main__':
    check_mixed_audio()
    # from copy import deepcopy
    # config_data_paths = deepcopy(CONFIG_DATALOADER)
    # config_data_paths[TRAIN][SNR_FILTER] = [
    #     0,]  # [10, 20, 30]  # Select some SNR
    # dl_dict = get_dataloaders(config_data_paths)
    # print(len(dl_dict[TRAIN].dataset))
    # batch_signal, batch_labels = next(iter(dl_dict[TRAIN]))

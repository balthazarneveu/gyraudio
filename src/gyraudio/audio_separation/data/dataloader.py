from torch.utils.data import DataLoader
from gyraudio.audio_separation.data.mixed import MixedAudioDataset
from typing import Optional, List
from gyraudio.audio_separation.properties import (
    DATA_PATH, AUGMENTATION, SNR_FILTER, SHUFFLE, BATCH_SIZE, TRAIN, VALID, TEST, AUG_TRIM
)
from gyraudio import root_dir
RAW_AUDIO_ROOT = root_dir/"__data_source_separation"/"voice_origin"
MIXED_AUDIO_ROOT = root_dir/"__data_source_separation"/"source_separation"


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
            batch_size=configuration[BATCH_SIZE],
            collate_fn=dataset.collate_fn
        )
        dataloaders[mode] = dl
    return dataloaders


def get_config_dataloader(
        audio_root=MIXED_AUDIO_ROOT,
        mode: str = TRAIN,
        shuffle: Optional[bool] = None,
        batch_size: Optional[int] = 16,
        snr_filter: Optional[List[float]] = None,
        augmentation: dict = {}):
    audio_folder = audio_root/mode
    assert mode in [TRAIN, VALID, TEST]
    assert audio_folder.exists()
    config = {
        DATA_PATH: audio_folder,
        SHUFFLE: shuffle if shuffle is not None else (True if mode == TRAIN else False),
        AUGMENTATION: augmentation,
        SNR_FILTER: snr_filter,
        BATCH_SIZE: batch_size
    }
    return config

from gyraudio.audio_separation.properties import TRAIN, TEST
from gyraudio.audio_separation.data import get_dataloader, get_config_dataloader, SingleAudioDataset, MixedAudioDataset
from gyraudio.default_locations import RAW_AUDIO_ROOT, MIXED_AUDIO_ROOT
import logging


def test_single_audio():
    train_set = get_config_dataloader(audio_root=RAW_AUDIO_ROOT, mode=TRAIN)
    test_set = get_config_dataloader(audio_root=RAW_AUDIO_ROOT, mode=TEST)
    dataloaders = get_dataloader({TRAIN: train_set, TEST: test_set}, audio_dataset=SingleAudioDataset)
    logging.info(f'{len(dataloaders[TRAIN])}, {len(dataloaders[TEST])}')
    batch_signal = next(iter(dataloaders[TRAIN]))
    logging.info(batch_signal.shape)


def test_mixed_audio():
    train_set = get_config_dataloader(audio_root=MIXED_AUDIO_ROOT, mode=TRAIN)
    test_set = get_config_dataloader(audio_root=MIXED_AUDIO_ROOT, mode=TEST)
    dataloaders = get_dataloader({TRAIN: train_set, TEST: test_set}, audio_dataset=MixedAudioDataset)
    batch_mix, batch_signal, batch_noise = next(iter(dataloaders[TRAIN]))
    assert batch_mix.shape == batch_signal.shape == batch_noise.shape

from gyraudio.audio_separation.properties import TRAIN, TEST, AUG_TRIM
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
    total_size = 0
    for snr in [0, 1, 2, 3, 4, -1, -2, -4, -3]:
        train_set = get_config_dataloader(audio_root=MIXED_AUDIO_ROOT, mode=TRAIN, snr_filter=[snr])
        test_set = get_config_dataloader(audio_root=MIXED_AUDIO_ROOT, mode=TEST, snr_filter=[snr])
        dataloaders = get_dataloader({
            TRAIN: train_set,
            TEST: test_set
        },
            audio_dataset=MixedAudioDataset
        )
        batch_mix, batch_signal, batch_noise = next(iter(dataloaders[TRAIN]))
        assert batch_mix.shape == batch_signal.shape == batch_noise.shape
        train_size = len(dataloaders[TRAIN].dataset)
        test_size = len(dataloaders[TEST].dataset)
        total_size += train_size
        print(f"SNR: {snr} - TRAIN SIZE {train_size} TEST SIZE {test_size}")
    total_expected = len(get_dataloader({"full": get_config_dataloader(audio_root=MIXED_AUDIO_ROOT)})["full"].dataset)
    # No filter
    assert total_size == total_expected == 5000


def test_mixed_audio_trim():
    train_set = get_config_dataloader(audio_root=MIXED_AUDIO_ROOT, mode=TRAIN, augmentation=[AUG_TRIM])
    test_set = get_config_dataloader(audio_root=MIXED_AUDIO_ROOT, mode=TEST)
    dataloaders = get_dataloader({
        TRAIN: train_set,
        TEST: test_set
    },
        audio_dataset=MixedAudioDataset
    )
    for u in range(20):
        batch_mix, batch_signal, batch_noise = next(iter(dataloaders[TRAIN]))
        print(batch_mix.shape)
        assert batch_mix.shape == batch_signal.shape == batch_noise.shape
    train_size = len(dataloaders[TRAIN].dataset)
    test_size = len(dataloaders[TEST].dataset)
    print(f"TRAIN SIZE {train_size} TEST SIZE {test_size}")
    total_expected = len(get_dataloader({"full": get_config_dataloader(audio_root=MIXED_AUDIO_ROOT)})["full"].dataset)

    assert total_expected == 5000

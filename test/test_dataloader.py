from gyraudio.audio_separation.properties import TRAIN, TEST, AUG_TRIM, LENGTHS, LENGTH_DIVIDER, TRIM_PROB
from gyraudio.audio_separation.data import get_dataloader, get_config_dataloader, SingleAudioDataset, MixedAudioDataset, RemixedFixedAudioDataset, RemixedRandomAudioDataset
from gyraudio.default_locations import RAW_AUDIO_ROOT, MIXED_AUDIO_ROOT
import logging
import torch


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
    train_set = get_config_dataloader(audio_root=MIXED_AUDIO_ROOT, mode=TRAIN, augmentation={AUG_TRIM : {LENGTHS : [2048, 8000], LENGTH_DIVIDER : 1024, TRIM_PROB : 0.5}})
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


def test_remixed_audio():
    total_size = 0
    dl_train = get_dataloader(
        {
            TRAIN: get_config_dataloader(
                audio_root=MIXED_AUDIO_ROOT,
                mode=TRAIN,
            )
        },
        audio_dataset=RemixedRandomAudioDataset
    )[TRAIN]
    dl_test = get_dataloader(
        {
            TEST: get_config_dataloader(
                audio_root=MIXED_AUDIO_ROOT,
                mode=TEST,
            )
        },
        audio_dataset=RemixedFixedAudioDataset
    )[TEST]
    dataloaders = {
        TRAIN: dl_train,
        TEST: dl_test
    }
    batch_mix, batch_signal, batch_noise = next(iter(dataloaders[TRAIN]))
    assert batch_mix.shape == batch_signal.shape == batch_noise.shape
    train_size = len(dataloaders[TRAIN].dataset)
    test_size = len(dataloaders[TEST].dataset)
    total_size += train_size
    print(f"TRAIN SIZE {train_size} TEST SIZE {test_size}")
    total_expected = len(get_dataloader({"full": get_config_dataloader(audio_root=MIXED_AUDIO_ROOT)})["full"].dataset)
    # No filter
    assert total_size == total_expected == 5000

    # Test 2 : fixed snr and noise
    fixed_dataset_1 = RemixedFixedAudioDataset(MIXED_AUDIO_ROOT/"test")
    fixed_dataset_2 = RemixedFixedAudioDataset(MIXED_AUDIO_ROOT/"test")
    mixed_audio_signal, clean_audio_signal, noise_audio_signal = fixed_dataset_1[0]
    mixed_audio_signal_2, clean_audio_signal_2, noise_audio_signal_2 = fixed_dataset_2[0]
    fixed_snr = torch.norm(clean_audio_signal) / torch.norm(noise_audio_signal)
    assert torch.equal(fixed_snr, torch.norm(clean_audio_signal_2) / torch.norm(noise_audio_signal_2))
    assert torch.equal(noise_audio_signal, noise_audio_signal_2)

    # Test 3 : random snr and noise
    rnd_dataset_1 = RemixedRandomAudioDataset(MIXED_AUDIO_ROOT/"test")
    rnd_dataset_2 = RemixedRandomAudioDataset(MIXED_AUDIO_ROOT/"test")
    mixed_audio_signal, clean_audio_signal, noise_audio_signal = rnd_dataset_1[0]
    mixed_audio_signal_2, clean_audio_signal_2, noise_audio_signal_2 = rnd_dataset_2[0]
    assert torch.norm(clean_audio_signal) / torch.norm(noise_audio_signal) != torch.norm(clean_audio_signal_2) / torch.norm(noise_audio_signal_2)
    assert fixed_snr != torch.norm(clean_audio_signal) / torch.norm(noise_audio_signal)
    assert not(torch.equal(noise_audio_signal, noise_audio_signal_2))
    assert torch.equal(clean_audio_signal, clean_audio_signal_2)

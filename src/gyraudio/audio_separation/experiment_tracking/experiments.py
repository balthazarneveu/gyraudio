from gyraudio.default_locations import MIXED_AUDIO_ROOT
from gyraudio.audio_separation.properties import (
    TRAIN, TEST, VALID, NAME, EPOCHS, LEARNING_RATE,
    OPTIMIZER, BATCH_SIZE, DATALOADER, AUGMENTATION,
    SHORT_NAME
)
from gyraudio.audio_separation.data.remixed_fixed import RemixedFixedAudioDataset
from gyraudio.audio_separation.data.remixed_rnd import RemixedRandomAudioDataset
from gyraudio.audio_separation.data import get_dataloader, get_config_dataloader
from gyraudio.audio_separation.experiment_tracking.experiments_definition import get_experiment_generator
import torch
from typing import Tuple


def get_experience(exp_major: int, exp_minor: int = 0, dry_run=False) -> Tuple[str, torch.nn.Module, dict, dict]:
    """Get all experience details

    Args:
        exp_major (int): Major experience number
        exp_minor (int, optional): Used for HP search. Defaults to 0.


    Returns:
        Tuple[str, torch.nn.Module, dict, dict]: short_name, model, config, dataloaders
    """
    model = None
    config = {}
    dataloader_name = "remix"
    config = {
        NAME: None,
        OPTIMIZER: {
            NAME: "adam",
            LEARNING_RATE: 0.001
        },
        EPOCHS: 60,
        DATALOADER: {
            NAME: dataloader_name,
        },
        BATCH_SIZE: [16, 16, 16]
    }

    model, config = get_experiment_generator(exp_major=exp_major)(config, no_model=dry_run, minor=exp_minor)
    # POST PROCESSING
    if isinstance(config[BATCH_SIZE], list) or isinstance(config[BATCH_SIZE], tuple):
        config[BATCH_SIZE] = {
            TRAIN: config[BATCH_SIZE][0],
            TEST:  config[BATCH_SIZE][1],
            VALID:  config[BATCH_SIZE][2],
        }

    if config[DATALOADER][NAME] == "premix":
        mixed_audio_root = MIXED_AUDIO_ROOT
        dataloaders = get_dataloader({
            TRAIN: get_config_dataloader(
                audio_root=mixed_audio_root,
                mode=TRAIN,
                shuffle=True,
                batch_size=config[BATCH_SIZE][TRAIN],
                augmentation=config[DATALOADER].get(AUGMENTATION, {})
            ),
            TEST: get_config_dataloader(
                audio_root=mixed_audio_root,
                mode=TEST,
                shuffle=False,
                batch_size=config[BATCH_SIZE][TEST]
            )
        })
    elif config[DATALOADER][NAME] == "remix":
        mixed_audio_root = MIXED_AUDIO_ROOT
        dl_train = get_dataloader(
            {
                TRAIN: get_config_dataloader(
                    audio_root=mixed_audio_root,
                    mode=TRAIN,
                    shuffle=True,
                    batch_size=config[BATCH_SIZE][TRAIN],
                    augmentation=config[DATALOADER].get(AUGMENTATION, {})
                )
            },
            audio_dataset=RemixedRandomAudioDataset
        )[TRAIN]
        dl_test = get_dataloader(
            {
                TEST: get_config_dataloader(
                    audio_root=mixed_audio_root,
                    mode=TEST,
                    shuffle=False,
                    batch_size=config[BATCH_SIZE][TEST]
                )
            },
            audio_dataset=RemixedFixedAudioDataset
        )[TEST]
        dataloaders = {
            TRAIN: dl_train,
            TEST: dl_test
        }
    else:
        raise NotImplementedError(f"Unknown dataloader {dataloader_name}")
    assert config[NAME] is not None

    short_name = f"{exp_major:04d}_{exp_minor:04d}"
    config[SHORT_NAME] = short_name
    return short_name, model, config, dataloaders


if __name__ == "__main__":
    from gyraudio.audio_separation.parser import shared_parser
    parser_def = shared_parser()
    args = parser_def.parse_args()

    for exp in args.experiments:
        short_name, model, config, dl = get_experience(exp)
        print(short_name)
        print(config)

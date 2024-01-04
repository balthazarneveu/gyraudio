from gyraudio.default_locations import MIXED_AUDIO_ROOT
from gyraudio.audio_separation.properties import (
    TRAIN, TEST, VALID, NAME, EPOCHS, LEARNING_RATE,
    OPTIMIZER, BATCH_SIZE, NB_PARAMS, ANNOTATIONS,
    SHORT_NAME
)
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
    dataloader_name = "premix"
    batch_sizes = [16, 128, 128]
    config = {
        NAME: None,
        OPTIMIZER: {
            NAME: "adam",
            LEARNING_RATE: 0.001
        },
        EPOCHS: 10,
    }

    model, config = get_experiment_generator(exp_major=exp_major)(config, no_model=dry_run, minor=exp_minor)
    # POST PROCESSING
    config[BATCH_SIZE] = {
        TRAIN: batch_sizes[0],
        TEST:  batch_sizes[1],
        VALID:  batch_sizes[2],
    }

    if dataloader_name == "premix":
        mixed_audio_root = MIXED_AUDIO_ROOT
        dataloaders = get_dataloader({
            TRAIN: get_config_dataloader(
                audio_root=mixed_audio_root,
                mode=TRAIN,
                shuffle=True,
                batch_size=config[BATCH_SIZE][TRAIN]
            ),
            TEST: get_config_dataloader(
                audio_root=mixed_audio_root,
                mode=TEST,
                shuffle=False,
                batch_size=config[BATCH_SIZE][TEST]
            )
        })
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

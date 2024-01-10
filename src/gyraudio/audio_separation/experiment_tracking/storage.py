from gyraudio.audio_separation.properties import SHORT_NAME, MODEL, OPTIMIZER, CURRENT_EPOCH, CONFIGURATION
from pathlib import Path
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
import logging
import torch


def get_output_folder(config: dict, root_dir: Path = EXPERIMENT_STORAGE_ROOT, override: bool = False) -> Path:
    output_folder = root_dir/config["short_name"]
    exists = False
    if output_folder.exists():
        if not override:
            logging.info(f"Experiment {config[SHORT_NAME]} already exists. Override is set to False. Skipping.")
        if override:
            logging.warning(f"Experiment {config[SHORT_NAME]} will be OVERRIDDEN")
            exists = True
    else:
        output_folder.mkdir(parents=True, exist_ok=True)
        exists = True
    return exists, output_folder


def checkpoint_paths(exp_dir: Path, epoch=None):
    if epoch is None:
        checkpoints = sorted(exp_dir.glob("model_*.pt"))
        assert len(checkpoints) > 0, f"No checkpoints found in {exp_dir}"
        model_checkpoint = checkpoints[-1]
        epoch = int(model_checkpoint.stem.split("_")[-1])
        optimizer_checkpoint = exp_dir/model_checkpoint.stem.replace("model", "optimizer")
    else:
        model_checkpoint = exp_dir/f"model_{epoch:04d}.pt"
        optimizer_checkpoint = exp_dir/f"optimizer_{epoch:04d}.pt"
    return model_checkpoint, optimizer_checkpoint, epoch


def load_checkpoint(model, exp_dir: Path, optimizer=None, epoch: int = None, device="cuda" if torch.cuda.is_available() else "cpu"):
    config = {}
    model_checkpoint, optimizer_checkpoint, epoch = checkpoint_paths(exp_dir, epoch=epoch)
    model_state_dict = torch.load(model_checkpoint, map_location=torch.device(device))
    model.load_state_dict(model_state_dict[MODEL])
    if optimizer is not None:
        optimizer_state_dict = torch.load(optimizer_checkpoint, map_location=torch.device(device))
        optimizer.load_state_dict(optimizer_state_dict[OPTIMIZER])
        config = optimizer_state_dict[CONFIGURATION]
    return model, optimizer, epoch, config


def save_checkpoint(model, exp_dir: Path, optimizer=None, config: dict = {}, epoch: int = None):
    model_checkpoint, optimizer_checkpoint, epoch = checkpoint_paths(exp_dir, epoch=epoch)
    torch.save(
        {
            MODEL: model.state_dict(),
        },
        model_checkpoint
    )
    torch.save(
        {
            CURRENT_EPOCH: epoch,
            CONFIGURATION: config,
            OPTIMIZER: optimizer.state_dict()
        },
        optimizer_checkpoint
    )
    print(f"Checkpoint saved:\n   - model: {model_checkpoint}\n   - checkpoint: {optimizer_checkpoint}")

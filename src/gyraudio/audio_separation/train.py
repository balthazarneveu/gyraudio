from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.parser import shared_parser
from gyraudio.audio_separation.properties import (
    TRAIN, TEST, EPOCHS, OPTIMIZER, NAME, MAX_STEPS_PER_EPOCH, LOSS_L2
)
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder, save_checkpoint
# from gyraudio.audio_separation.experiment_tracking.storage import load_checkpoint
from pathlib import Path
from gyraudio.io.dump import Dump
import sys
import torch
from tqdm import tqdm
from copy import deepcopy
import wandb
import logging


def launch_training(exp: int, wandb_flag: bool = True, device: str = "cuda", save_dir: Path = None, override=False):

    short_name, model, config, dl = get_experience(exp)
    exists, output_folder = get_output_folder(config, root_dir=save_dir, override=override)
    if not exists:
        logging.warning(f"Skipping experiment {short_name}")
        return False
    else:
        logging.info(f"Experiment {short_name} saved in {output_folder}")

    print(short_name)
    print(config)
    logging.info(f"Starting training for {short_name}")
    logging.info(f"Config: {config}")
    if wandb_flag:
        wandb.init(
            project="audio-separation",
            entity="teammd",
            name=short_name,
            tags=["debug"],
            config=config
        )
    training_loop(model, config, dl, wandb_flag=wandb_flag, device=device, exp_dir=output_folder)
    if wandb_flag:
        wandb.finish()
    return True


def training_loop(model: torch.nn.Module, config: dict, dl, device: str = "cuda", wandb_flag: bool = False,
                  exp_dir: Path = None):
    optim_params = deepcopy(config[OPTIMIZER])
    optim_name = optim_params[NAME]
    optim_params.pop(NAME)
    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optim_params)
    max_steps = config.get(MAX_STEPS_PER_EPOCH, None)

    for epoch in range(config[EPOCHS]):
        model.to(device)
        # model, optimizer, epoch, config = load_checkpoint(model, exp_dir, optimizer, epoch=epoch)
        # Training loop
        # -----------------------------------------------------------

        metrics = {TRAIN: {}, TEST: {}}
        training_loss = 0.
        for step_index, (batch_mix, batch_signal, batch_noise) in tqdm(
                enumerate(dl[TRAIN]), desc=f"Epoch {epoch}", total=len(dl[TRAIN])):
            if max_steps is not None and step_index >= max_steps:
                break
            batch_mix, batch_signal, batch_noise = batch_mix.to(device), batch_signal.to(device), batch_noise.to(device)
            model.zero_grad()
            batch_output_signal, _batch_output_noise = model(batch_mix)
            loss = torch.nn.functional.mse_loss(batch_output_signal, batch_signal)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss = training_loss/len(dl[TRAIN])
        # Validation loop
        # -----------------------------------------------------------
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            test_loss = 0.
            for step_index, (batch_mix, batch_signal, batch_noise) in tqdm(
                    enumerate(dl[TEST]), desc=f"Epoch {epoch}", total=len(dl[TEST])):
                if max_steps is not None and step_index >= max_steps:
                    break
                batch_mix, batch_signal, batch_noise = batch_mix.to(
                    device), batch_signal.to(device), batch_noise.to(device)
                batch_output_signal, _batch_output_noise = model(batch_mix)
                loss = torch.nn.functional.mse_loss(batch_output_signal, batch_signal)
                test_loss += loss.item()
        test_loss = test_loss/len(dl[TEST])
        logging.info(f"{training_loss:.3e} | {test_loss:.3e}")
        if wandb_flag:
            wandb.log({"train/loss": training_loss, "test/loss": test_loss})
        metrics[TRAIN][LOSS_L2] = training_loss
        metrics[TEST][LOSS_L2] = test_loss
        Dump.save_json(metrics, exp_dir/f"metrics_{epoch:04d}.json")
        save_checkpoint(model, exp_dir, optimizer, config=config, epoch=epoch)
        torch.cuda.empty_cache()


def main(argv):
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser_def = shared_parser(help="Launch training \nCheck results at: https://wandb.ai/teammd/audio-separation"
                               + ("\n<<<Cuda available>>>" if default_device == "cuda" else ""))
    parser_def.add_argument("-nowb", "--no-wandb", action="store_true")
    parser_def.add_argument("-o", "--output-dir", type=str, default=EXPERIMENT_STORAGE_ROOT)
    parser_def.add_argument("-f", "--force", action="store_true", help="Override existing experiment")

    parser_def.add_argument("-d", "--device", type=str, default=default_device,
                            help="Training device", choices=["cpu", "cuda"])
    args = parser_def.parse_args(argv)
    for exp in args.experiments:
        launch_training(
            exp, wandb_flag=not args.no_wandb, save_dir=Path(args.output_dir),
            override=args.force,
            device=args.device
        )


if __name__ == "__main__":
    main(sys.argv[1:])

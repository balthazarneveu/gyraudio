from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.parser import shared_parser
from gyraudio.audio_separation.properties import (
    TRAIN, TEST, EPOCHS, OPTIMIZER, NAME, MAX_STEPS_PER_EPOCH, LOSS_L2,
    SIGNAL, NOISE, TOTAL,
)
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder, save_checkpoint
from gyraudio.audio_separation.metrics import Metrics
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
    training_loss = Metrics(TRAIN)
    test_loss = Metrics(TEST)
    for epoch in range(config[EPOCHS]):
        training_loss.reset_epoch()
        test_loss.reset_epoch()
        model.to(device)
        # Training loop
        # -----------------------------------------------------------

        metrics = {TRAIN: {}, TEST: {}}
        for step_index, (batch_mix, batch_signal, batch_noise) in tqdm(
                enumerate(dl[TRAIN]), desc=f"Epoch {epoch}", total=len(dl[TRAIN])):
            if max_steps is not None and step_index >= max_steps:
                break
            batch_mix, batch_signal, batch_noise = batch_mix.to(device), batch_signal.to(device), batch_noise.to(device)
            model.zero_grad()
            batch_output_signal, batch_output_noise = model(batch_mix)
            training_loss.update(batch_output_signal, batch_signal, SIGNAL)
            training_loss.update(batch_output_noise, batch_noise, NOISE)
            loss = training_loss.finish_step()
            loss.backward()
            optimizer.step()
        training_loss.finish_epoch()

        # Validation loop
        # -----------------------------------------------------------
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for step_index, (batch_mix, batch_signal, batch_noise) in tqdm(
                    enumerate(dl[TEST]), desc=f"Epoch {epoch}", total=len(dl[TEST])):
                if max_steps is not None and step_index >= max_steps:
                    break
                batch_mix, batch_signal, batch_noise = batch_mix.to(
                    device), batch_signal.to(device), batch_noise.to(device)
                batch_output_signal, batch_output_noise = model(batch_mix)
                test_loss.update(batch_output_signal, batch_signal, SIGNAL)
                test_loss.update(batch_output_noise, batch_noise, NOISE)
                test_loss.finish_step()
        test_loss.finish_epoch()

        print(f"epoch {epoch}:\n{training_loss}\n{test_loss}")
        if wandb_flag:
            wandb.log({
                "loss/training loss signal": training_loss.total_metric[SIGNAL],
                "loss/test loss signal": test_loss.total_metric[SIGNAL],
                "debug loss/training loss total": training_loss.total_metric[TOTAL],
                "debug loss/test loss total": test_loss.total_metric[TOTAL],
                "debug loss/training loss noise": training_loss.total_metric[NOISE],
                "debug loss/test loss noise": test_loss.total_metric[NOISE],

            })
        metrics[TRAIN] = training_loss.total_metric
        metrics[TEST] = test_loss.total_metric
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

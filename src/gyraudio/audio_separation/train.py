from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.parser import shared_parser
from gyraudio.audio_separation.properties import (
    TRAIN, TEST, EPOCHS, OPTIMIZER, NAME, MAX_STEPS_PER_EPOCH,
    SIGNAL, NOISE, TOTAL, SNR, SCHEDULER, SCHEDULER_CONFIGURATION,
    LOSS, LOSS_L2, LOSS_L1, LOSS_TYPE, COEFFICIENT
)
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder, save_checkpoint
from gyraudio.audio_separation.metrics import Costs, snr
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


def update_metrics(metrics, phase, pred, gt, pred_noise, gt_noise):
    metrics[phase].update(pred, gt, SIGNAL)
    metrics[phase].update(pred_noise, gt_noise, NOISE)
    metrics[phase].update(pred, gt, SNR)
    loss = metrics[phase].finish_step()
    return loss


def training_loop(model: torch.nn.Module, config: dict, dl, device: str = "cuda", wandb_flag: bool = False,
                  exp_dir: Path = None):
    optim_params = deepcopy(config[OPTIMIZER])
    optim_name = optim_params[NAME]
    optim_params.pop(NAME)
    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optim_params)

    scheduler = None
    scheduler_config = config.get(SCHEDULER_CONFIGURATION, {})
    scheduler_name = config.get(SCHEDULER, False)
    if scheduler_name:
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, **scheduler_config)
            logging.info(f"Using scheduler {scheduler_name} with config {scheduler_config}")
        else:
            raise NotImplementedError(f"Scheduler {scheduler_name} not implemented")
    max_steps = config.get(MAX_STEPS_PER_EPOCH, None)
    chosen_loss = config.get(LOSS, LOSS_L2)
    if chosen_loss == LOSS_L2:
        costs = {TRAIN:  Costs(TRAIN), TEST: Costs(TEST)}
    elif chosen_loss == LOSS_L1:
        cost_init = {
            SIGNAL: {
                COEFFICIENT: 0.5,
                LOSS_TYPE: torch.nn.functional.l1_loss
            },
            NOISE: {
                COEFFICIENT: 0.5,
                LOSS_TYPE: torch.nn.functional.l1_loss
            },
            SNR: {
                LOSS_TYPE: snr
            }
        }
        costs = {
            TRAIN:  Costs(TRAIN, costs=cost_init),
            TEST: Costs(TEST)
        }
    for epoch in range(config[EPOCHS]):
        costs[TRAIN].reset_epoch()
        costs[TEST].reset_epoch()
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
            loss = update_metrics(
                costs, TRAIN,
                batch_output_signal, batch_signal,
                batch_output_noise, batch_noise
            )
            # costs[TRAIN].update(batch_output_signal, batch_signal, SIGNAL)
            # costs[TRAIN].update(batch_output_noise, batch_noise, NOISE)
            # loss = costs[TRAIN].finish_step()
            loss.backward()
            optimizer.step()
        costs[TRAIN].finish_epoch()

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
                loss = update_metrics(
                    costs, TEST,
                    batch_output_signal, batch_signal,
                    batch_output_noise, batch_noise
                )
        costs[TEST].finish_epoch()
        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(costs[TEST].total_metric[SNR])
        print(f"epoch {epoch}:\n{costs[TRAIN]}\n{costs[TEST]}")
        wandblogs = {}
        if wandb_flag:
            for phase in [TRAIN, TEST]:
                wandblogs[f"{phase} loss signal"] = costs[phase].total_metric[SIGNAL]
                wandblogs[f"debug loss/{phase} loss signal"] = costs[phase].total_metric[SIGNAL]
                wandblogs[f"debug loss/{phase} loss total"] = costs[phase].total_metric[TOTAL]
                wandblogs[f"debug loss/{phase} loss noise"] = costs[phase].total_metric[NOISE]
                wandblogs[f"{phase} snr"] = costs[phase].total_metric[SNR]
                wandblogs["learning rate"] = optimizer.param_groups[0]['lr']
            wandb.log(wandblogs)
        metrics[TRAIN] = costs[TRAIN].total_metric
        metrics[TEST] = costs[TEST].total_metric
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

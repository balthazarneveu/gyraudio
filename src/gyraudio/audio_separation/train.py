from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.parser import shared_parser
from gyraudio.audio_separation.properties import (
    TRAIN, TEST, EPOCHS, OPTIMIZER, NAME
)
import sys
import torch
from tqdm import tqdm
from copy import deepcopy
import wandb
import logging


def prepare_training(exp: int, wandb_flag: bool = True):
    short_name, model, config, dl = get_experience(exp)
    print(short_name)
    print(config)
    logging.info(f"Starting training for {short_name}")
    logging.info(f"Config: {config}")
    if wandb_flag:
        wandb.init(
            project="audio-sep",
            name=short_name,
            tags=["debug"],
            config=config
        )
    training_loop(model, config, dl, wandb_flag=wandb_flag)
    if wandb_flag:
        wandb.finish()


def training_loop(model: torch.nn.Module, config: dict, dl, device: str = "cuda", wandb_flag: bool = False):
    optim_params = deepcopy(config[OPTIMIZER])
    optim_name = optim_params[NAME]
    optim_params.pop(NAME)
    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optim_params)
    for epoch in range(config[EPOCHS]):
        model.to(device)
        training_loss = 0.
        for batch_mix, batch_signal, batch_noise in tqdm(dl[TRAIN], desc=f"Epoch {epoch}", total=len(dl[TRAIN])):
            batch_mix, batch_signal, batch_noise = batch_mix.to(device), batch_signal.to(device), batch_noise.to(device)
            model.zero_grad()
            batch_output_signal, _batch_output_noise = model(batch_mix)
            loss = torch.nn.functional.mse_loss(batch_output_signal, batch_signal)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        print(training_loss/len(dl[TRAIN]))
        model.eval()
        with torch.no_grad():
            val_loss = 0.
            for batch_mix, batch_signal, batch_noise in tqdm(dl[TEST], desc=f"Epoch {epoch}", total=len(dl[TEST])):
                batch_mix, batch_signal, batch_noise = batch_mix.to(
                    device), batch_signal.to(device), batch_noise.to(device)
                batch_output_signal, _batch_output_noise = model(batch_mix)
                loss = torch.nn.functional.mse_loss(batch_output_signal, batch_signal)
                val_loss += loss.item()
        val_loss = val_loss/len(dl[TEST])
        wandb.log({"train/loss": training_loss, "validation/loss": val_loss})


def main(argv):
    parser_def = shared_parser()
    parser_def.add_argument("-nowb", "--no-wandb", action="store_true")
    args = parser_def.parse_args(argv)
    for exp in args.experiments:
        prepare_training(exp, wandb_flag=not args.no_wandb)


if __name__ == "__main__":
    main(sys.argv[1:])

from batch_processing import Batch
import argparse
import sys
from pathlib import Path
from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.properties import SHORT_NAME
import torch
from gyraudio.audio_separation.experiment_tracking.storage import load_checkpoint
from gyraudio.io.audio import load_audio_tensor, save_audio_tensor
from typing import List
import logging


def parse_command_line(batch: Batch) -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description='Batch audio processing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e",  "--experiments", type=int, nargs="+", required=True,
                        help="Experiment ids to be inferred sequentially")
    parser.add_argument("-m", "--model-root", type=str, default=EXPERIMENT_STORAGE_ROOT)
    parser.add_argument("-d", "--device", type=str, default=default_device)
    return batch.parse_args(parser)


def outp(path: Path, suffix: str, extension=".wav"):
    return (path.parent / (path.stem + suffix)).with_suffix(extension)


def audio_separation_processing(
    input: Path, output: Path, args: argparse.Namespace,
    model_list: List[torch.nn.Module],
    config_list: List[dict]
):
    device = args.device
    input_audio_path = list(input.glob("mix*.wav"))[0]
    mixed_signal, sampling_rate = load_audio_tensor(input_audio_path, device=device)
    with torch.no_grad():
        for config, model in zip(config_list, model_list):
            short_name = config.get(SHORT_NAME, "unknown")
            predicted_signal, predicted_noise = model(mixed_signal.unsqueeze(0))
            save_audio_tensor(outp(output, f"_prediction_{short_name}"), predicted_signal.squeeze(0), sampling_rate)
    save_audio_tensor(outp(output, "_mix"), mixed_signal, sampling_rate)


def main(argv):
    batch = Batch(argv)
    batch.set_io_description(
        input_help='input audio files',
        output_help='output directory'
    )
    batch.set_multiprocessing_enabled(False)
    args = parse_command_line(batch)
    exp = args.experiments[0]
    device = args.device
    models_list = []
    config_list = []
    logging.info(f"Loading experiments models {args.experiments}")
    for exp in args.experiments:
        model_dir = Path(args.model_root)
        short_name, model, config, _dl = get_experience(exp)
        _, exp_dir = get_output_folder(config, root_dir=model_dir, override=False)
        assert exp_dir.exists(), f"Experiment {short_name} does not exist in {model_dir}"
        model.eval()
        model.to(device)
        model, __optimizer, epoch, config = load_checkpoint(model, exp_dir, epoch=None, device=args.device)
        config[SHORT_NAME] = short_name
        models_list.append(model)
        config_list.append(config)
        # batch.run(audio_separation_processing, [model], [config])
    logging.info(f"Starting inference:")
    batch.run(audio_separation_processing, models_list, config_list)


if __name__ == "__main__":
    main(sys.argv[1:])

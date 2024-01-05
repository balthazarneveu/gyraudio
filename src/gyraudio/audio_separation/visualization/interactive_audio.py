from batch_processing import Batch
import argparse
import sys
from pathlib import Path
from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.properties import SHORT_NAME, CLEAN, NOISY, MIXED
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
    parser.add_argument("-preload", "--preload", action="store_true", help="Preload audio files")
    parser.add_argument("-p", "--interactive", action="store_true", help="Play = Interactive mode")
    parser.add_argument("-m", "--model-root", type=str, default=EXPERIMENT_STORAGE_ROOT)
    parser.add_argument("-d", "--device", type=str, default=default_device)
    return batch.parse_args(parser)


def outp(path: Path, suffix: str, extension=".wav"):
    return (path.parent / (path.stem + suffix)).with_suffix(extension)


def load_buffers(signal: dict, device="cpu"):
    clean_signal, sampling_rate = load_audio_tensor(signal["paths"][CLEAN], device=device)
    noisy_signal, sampling_rate = load_audio_tensor(signal["paths"][NOISY], device=device)
    mixed_signal, sampling_rate = load_audio_tensor(signal["paths"][MIXED], device=device)
    signal["buffers"] = {
        CLEAN: clean_signal,
        NOISY: noisy_signal,
        MIXED: mixed_signal
    }
    signal["sampling_rate"] = sampling_rate


def audio_loading(
    input: Path, output: Path, args: argparse.Namespace,
):
    name = input.name
    clean_audio_path = input/"voice.wav"
    noisy_audio_path = input/"noise.wav"
    mixed_audio_path = list(input.glob("mix*.wav"))[0]
    signal = {
        "name": name,
        "paths": {
            CLEAN: clean_audio_path,
            NOISY: noisy_audio_path,
            MIXED: mixed_audio_path
        }
    }
    if args.preload:
        load_buffers(signal)
    return signal


def visualization(
    all_signals: List[dict],
    model_list: List[torch.nn.Module],
    config_list: List[dict],
    device="cuda"
):
    from interactive_pipe.data_objects.curves import Curve, SingleCurve
    for signal in all_signals:
        print(signal["name"])
        if "buffers" not in signal:
            load_buffers(signal, device=device)
            # signal["sampling_rate"] = sampling_rate
        clean = SingleCurve(y=signal["buffers"][CLEAN][0, :], label="clean")
        noisy = SingleCurve(y=signal["buffers"][NOISY][0, :], label="noisy")
        Curve([clean, noisy]).show()


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
        model, __optimizer, epoch, config = load_checkpoint(model, exp_dir, epoch=None)
        config[SHORT_NAME] = short_name
        models_list.append(model)
        config_list.append(config)
        # batch.run(audio_separation_processing, [model], [config])
    logging.info("Load audio buffers:")
    all_signals = batch.run(audio_loading)
    visualization(all_signals, models_list, config_list, device=device)

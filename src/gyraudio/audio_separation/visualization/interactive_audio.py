from batch_processing import Batch
import argparse
import sys
from pathlib import Path
from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.properties import SHORT_NAME, CLEAN, NOISY, MIXED, ANNOTATIONS
import torch
from gyraudio.audio_separation.experiment_tracking.storage import load_checkpoint
from gyraudio.io.audio import load_audio_tensor, save_audio_tensor
from typing import List
import numpy as np
import logging
from interactive_pipe.data_objects.curves import Curve, SingleCurve
from interactive_pipe import interactive_pipeline, interactive, KeyboardControl


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


@interactive(
    idx=KeyboardControl(value_default=0, value_range=[0, 1000], modulo=True, keyup="right", keydown="left")
)
def signal_selector(signals, idx=0):
    signal = signals[idx % len(signals)]
    if "buffers" not in signal:
        load_buffers(signal)
    return signal


@interactive(
    snr=(6., [-3., 6.], "extra SNR amplification [dB]")
)
def remix(signals, snr=0.):
    signal = signals["buffers"][CLEAN]
    noisy = signals["buffers"][NOISY]
    # mixed_signal = signal + 10.**(-snr/20.)*noisy
    mixed_signal = 10.**(snr/20.)*signal + noisy
    return mixed_signal


@interactive(
    model=KeyboardControl(value_default=0, value_range=[0, 99], keyup="pagedown", keydown="pageup")
)
def audio_sep_inference(mixed, models, configs, model: int = 0):
    selected_model = models[model % len(models)]
    config = configs[model % len(models)]
    device = "cuda"
    short_name = config.get(SHORT_NAME, "")
    annotations = config.get(ANNOTATIONS, "")
    predicted_signal, predicted_noise = selected_model(mixed.to(device).unsqueeze(0))
    predicted_signal = predicted_signal.squeeze(0)
    pred = SingleCurve(y=5*predicted_signal[0, :].detach().cpu().numpy(),
                       style="b-", label=f"predicted_{short_name} {annotations}")
    return pred


def visualize_audio(signal: dict, mixed_signal, pred):
    dec = 200
    clean = SingleCurve(y=signal["buffers"][CLEAN][0, ::dec], alpha=1., style="r--", linewidth=1, label="clean")
    noisy = SingleCurve(y=signal["buffers"][NOISY][0, ::dec], alpha=0.3, style="y-", linewidth=1, label="noisy")
    mixed = SingleCurve(y=mixed_signal[0, ::dec], style="g-", alpha=0.5, linewidth=2, label="mixed")
    pred.y = pred.y[::dec]
    curves = [noisy, mixed, pred, clean]
    return Curve(curves, ylim=[-0.04, 0.04], xlabel="Time index", ylabel="Amplitude")


def interactive_audio_separation_processing(signals, model_list, config_list):
    sig = signal_selector(signals)
    mixed = remix(sig)
    pred = audio_sep_inference(mixed, model_list, config_list)
    curve = visualize_audio(sig, mixed, pred)
    return curve


def interactive_audio_separation_visualization(
        all_signals: List[dict],
        model_list: List[torch.nn.Module],
        config_list: List[dict],
        device="cuda"
):
    interactive_pipeline(gui="auto")(interactive_audio_separation_processing)(
        all_signals, model_list, config_list
    )


def visualization(
    all_signals: List[dict],
    model_list: List[torch.nn.Module],
    config_list: List[dict],
    device="cuda"
):
    for signal in all_signals:
        print(signal["name"])
        if "buffers" not in signal:
            load_buffers(signal, device="cpu")
        clean = SingleCurve(y=signal["buffers"][CLEAN][0, :], label="clean")
        noisy = SingleCurve(y=signal["buffers"][NOISY][0, :], label="noise", alpha=0.3)
        curves = [clean, noisy]
        for config, model in zip(config_list, model_list):
            short_name = config.get(SHORT_NAME, "unknown")
            predicted_signal, predicted_noise = model(signal["buffers"][MIXED].to(device).unsqueeze(0))
            predicted = SingleCurve(y=predicted_signal.squeeze(0)[0, :].detach().cpu().numpy(),
                                    label=f"predicted_{short_name}")
            curves.append(predicted)
        Curve(curves).show()


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
    if not args.interactive:
        visualization(all_signals, models_list, config_list, device=device)
    else:
        interactive_audio_separation_visualization(all_signals, models_list, config_list, device=device)

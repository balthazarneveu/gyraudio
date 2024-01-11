from batch_processing import Batch
import argparse
from pathlib import Path
from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.properties import SHORT_NAME, CLEAN, NOISY, MIXED, PREDICTED, ANNOTATIONS
import torch
from gyraudio.audio_separation.experiment_tracking.storage import load_checkpoint
from gyraudio.audio_separation.visualization.pre_load_audio import (
    parse_command_line_audio_load, load_buffers, audio_loading_batch)

from typing import List
import numpy as np
import logging
from interactive_pipe.data_objects.curves import Curve, SingleCurve
from interactive_pipe import interactive, KeyboardControl, Control
from interactive_pipe.headless.pipeline import HeadlessPipeline
from interactive_pipe.graphical.qt_gui import InteractivePipeQT
from interactive_pipe.graphical.mpl_gui import InteractivePipeMatplotlib
from gyraudio.audio_separation.visualization.audio_player import audio_selector, audio_trim, audio_player
default_device = "cuda" if torch.cuda.is_available() else "cpu"


@interactive(
    idx=KeyboardControl(value_default=0, value_range=[0, 1000], modulo=True, keyup="8", keydown="2")
)
def signal_selector(signals, idx=0, global_params={}):
    # signals are loaded in CPU
    signal = signals[idx % len(signals)]
    if "buffers" not in signal:
        load_buffers(signal)
    global_params["sampling_rate"] = signal["sampling_rate"]
    global_params["mixed_snr"] = signal.get("mixed_snr", np.NaN)
    return signal


@interactive(
    dataset_mix=(True,),
    snr=(0., [-4., 4.], "extra SNR amplification [dB]")
)
def remix(signals, dataset_mix=True, snr=0.):
    # power_target_sqrt = 15.9054
    power_target_sqrt = 16.
    if dataset_mix:
        mixed_signal = signals["buffers"][MIXED]
        # Can be retrieved by :
        # signal = signals["buffers"][CLEAN]
        # noisy = signals["buffers"][NOISY]
        # mixed_snr = 10 ** (signals.get("mixed_snr", np.NaN) / 10)
        # mixed_signal = mixed_snr ** 0.5 * torch.norm(noisy) / torch.norm(signal) * signal + noisy
        # mixed_signal = mixed_signal * torch.max(signals["buffers"][MIXED]) / torch.max(mixed_signal)
        # or mixed_signal = mixed_signal * power_target_sqrt / torch.norm(mixed_signal)
    else:
        signal = signals["buffers"][CLEAN]
        noisy = signals["buffers"][NOISY]
        alpha = 10 ** (-snr / 20) * torch.norm(signal) / torch.norm(noisy)
        mixed_signal = signal + alpha * noisy
        mixed_signal = mixed_signal * power_target_sqrt / torch.norm(mixed_signal)
    return mixed_signal


@interactive(std_dev=Control(0., value_range=[0., 0.1], name="extra noise std", step=0.0001),
             amplify=(1., [0., 10.], "amplification of everything"))
def augment(signals, mixed, std_dev=0., amplify=1.):
    signals["buffers"][MIXED] *= amplify
    signals["buffers"][NOISY] *= amplify
    signals["buffers"][CLEAN] *= amplify
    mixed = mixed*amplify+torch.randn_like(mixed)*std_dev
    return signals, mixed


# @interactive(
#     device=("cuda", ["cpu", "cuda"]) if default_device == "cuda" else ("cpu", ["cpu"])
# )
def select_device(device=default_device, global_params={}):
    global_params["device"] = device


@interactive(
    model=KeyboardControl(value_default=0, value_range=[0, 99], keyup="pagedown", keydown="pageup")
)
def audio_sep_inference(mixed, models, configs, model: int = 0, global_params={}):
    selected_model = models[model % len(models)]
    config = configs[model % len(models)]
    short_name = config.get(SHORT_NAME, "")
    annotations = config.get(ANNOTATIONS, "")
    device = global_params.get("device", "cpu")
    with torch.no_grad():
        selected_model.eval()
        selected_model.to(device)
        predicted_signal, predicted_noise = selected_model(mixed.to(device).unsqueeze(0))
        predicted_signal = predicted_signal.squeeze(0)
    pred_curve = SingleCurve(y=predicted_signal[0, :].detach().cpu().numpy(),
                             style="g-", label=f"predicted_{short_name} {annotations}")
    return predicted_signal, pred_curve


def get_trim(sig, zoom, center, num_samples=300):
    N = len(sig)
    native_ds = N/num_samples
    center_idx = int(center*N)
    window = int(num_samples/zoom*native_ds)
    start_idx = max(0, center_idx - window//2)
    end_idx = min(N, center_idx + window//2)
    skip_factor = max(1, int(native_ds/zoom))
    return start_idx, end_idx, skip_factor


def zin(sig, zoom, center, num_samples=300):
    start_idx, end_idx, skip_factor = get_trim(sig, zoom, center, num_samples=num_samples)
    out = np.zeros(num_samples)
    trimmed = sig[start_idx:end_idx:skip_factor]
    out[:len(trimmed)] = trimmed[:num_samples]
    return out


@interactive(
    center=KeyboardControl(value_default=0.5, value_range=[0., 1.], step=0.01, keyup="6", keydown="4"),
    zoom=KeyboardControl(value_default=0., value_range=[0., 15.], step=1, keyup="+", keydown="-"),
    zoomy=KeyboardControl(value_default=0., value_range=[-15., 15.], step=1, keyup="up", keydown="down")
)
def visualize_audio(signal: dict, mixed_signal, pred, zoom=1, zoomy=0., center=0.5, global_params={}):
    """Create curves
    """
    zval = 1.5**zoom
    start_idx, end_idx, _skip_factor = get_trim(signal["buffers"][CLEAN][0, :], zval, center)
    global_params["trim"] = dict(start=start_idx, end=end_idx)
    selected = global_params.get("selected_audio", MIXED)
    clean = SingleCurve(y=zin(signal["buffers"][CLEAN][0, :], zval, center),
                        alpha=1.,
                        style="k-",
                        linewidth=0.9,
                        label=("*" if selected == CLEAN else " ")+"clean")
    noisy = SingleCurve(y=zin(signal["buffers"][NOISY][0, :], zval, center),
                        alpha=0.3,
                        style="y--",
                        linewidth=1,
                        label=("*" if selected == NOISY else " ") + "noisy"
                        )
    mixed = SingleCurve(y=zin(mixed_signal[0, :], zval, center), style="r-",
                        alpha=0.1,
                        linewidth=2,
                        label=("*" if selected == MIXED else " ") + "mixed")
    true_mixed = SingleCurve(y=zin(signal["buffers"][MIXED][0, :], zval, center),
                        alpha=0.3, style="b-", linewidth=1, label="true mixed")
    pred.y = zin(pred.y, zval, center)
    pred.label = ("*" if selected ==  PREDICTED else " ") + pred.label
    curves = [noisy, mixed, pred, clean]
    title = f"Premixed SNR : {global_params['mixed_snr']:.1f} dB"
    return Curve(curves, ylim=[-0.04 * 1.5 ** zoomy, 0.04 * 1.5 ** zoomy], xlabel="Time index", ylabel="Amplitude", title=title)


def interactive_audio_separation_processing(signals, model_list, config_list):
    sig = signal_selector(signals)
    mixed = remix(sig)
    # sig, mixed = augment(sig, mixed)
    select_device()
    pred, pred_curve = audio_sep_inference(mixed, model_list, config_list)
    sound = audio_selector(sig, mixed, pred)
    curve = visualize_audio(sig, mixed, pred_curve)
    trimmed_sound = audio_trim(sound)
    audio_player(trimmed_sound)
    return curve


def interactive_audio_separation_visualization(
        all_signals: List[dict],
        model_list: List[torch.nn.Module],
        config_list: List[dict],
        gui="qt"
):
    pip = HeadlessPipeline.from_function(interactive_audio_separation_processing, cache=False)
    if gui == "qt":
        app = InteractivePipeQT(pipeline=pip, name="audio separation", size=(1000, 1000), audio=True)
    else:
        logging.warning("No support for audio player with Matplotlib")
        app = InteractivePipeMatplotlib(pipeline=pip, name="audio separation", size=None, audio=False)
    app(all_signals, model_list, config_list)


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


def parse_command_line(parser: Batch = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = parse_command_line_audio_load()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    iparse = parser.add_argument_group("Audio separation visualization")
    iparse.add_argument("-e",  "--experiments", type=int, nargs="+", required=True,
                        help="Experiment ids to be inferred sequentially")
    iparse.add_argument("-p", "--interactive", action="store_true", help="Play = Interactive mode")
    iparse.add_argument("-m", "--model-root", type=str, default=EXPERIMENT_STORAGE_ROOT)
    iparse.add_argument("-d", "--device", type=str, default=default_device,
                        choices=["cpu", "cuda"] if default_device == "cuda" else ["cpu"])
    iparse.add_argument("-gui", "--gui", type=str, default="qt", choices=["qt", "mpl"])
    return parser


def main(argv):
    batch = Batch(argv)
    batch.set_io_description(
        input_help='input audio files',
        output_help='output directory'
    )
    batch.set_multiprocessing_enabled(False)
    parser = parse_command_line()
    args = batch.parse_args(parser)
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
    logging.info("Load audio buffers:")
    all_signals = batch.run(audio_loading_batch)
    if not args.interactive:
        visualization(all_signals, models_list, config_list, device=device)
    else:
        interactive_audio_separation_visualization(all_signals, models_list, config_list, gui=args.gui)

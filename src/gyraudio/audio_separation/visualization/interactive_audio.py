from batch_processing import Batch
import argparse
from pathlib import Path
from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.properties import (
    SHORT_NAME, CLEAN, NOISY, MIXED, PREDICTED, ANNOTATIONS, PATHS, BUFFERS, SAMPLING_RATE, NAME
)
import torch
from gyraudio.audio_separation.experiment_tracking.storage import load_checkpoint
from gyraudio.audio_separation.visualization.pre_load_audio import (
    parse_command_line_audio_load, load_buffers, audio_loading_batch)
from gyraudio.audio_separation.visualization.pre_load_custom_audio import (
    parse_command_line_generic_audio_load, generic_audio_loading_batch,
    load_buffers_custom
)
from torchaudio.functional import resample
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
LEARNT_SAMPLING_RATE = 8000


@interactive(
    idx=KeyboardControl(value_default=0, value_range=[0, 1000], modulo=True, keyup="8", keydown="2"),
    idn=KeyboardControl(value_default=0, value_range=[0, 1000], modulo=True, keyup="9", keydown="3")
)
def signal_selector(signals, idx=0, idn=0, global_params={}):
    if isinstance(signals, dict):
        clean_sigs = signals[CLEAN]
        clean = clean_sigs[idx % len(clean_sigs)]
        if BUFFERS not in clean:
            load_buffers_custom(clean)
        noise_sigs = signals[NOISY]
        noise = noise_sigs[idn % len(noise_sigs)]
        if BUFFERS not in noise:
            load_buffers_custom(noise)
        cbuf, nbuf = clean[BUFFERS], noise[BUFFERS]
        if clean[SAMPLING_RATE] != LEARNT_SAMPLING_RATE:
            cbuf = resample(cbuf, clean[SAMPLING_RATE], LEARNT_SAMPLING_RATE)
            clean[SAMPLING_RATE] = LEARNT_SAMPLING_RATE
        if noise[SAMPLING_RATE] != LEARNT_SAMPLING_RATE:
            nbuf = resample(nbuf, noise[SAMPLING_RATE], LEARNT_SAMPLING_RATE)
            noise[SAMPLING_RATE] = LEARNT_SAMPLING_RATE
        min_length = min(cbuf.shape[-1], nbuf.shape[-1])
        min_length = min_length - min_length % 1024
        signal = {
            PATHS: {
                CLEAN: clean[PATHS],
                NOISY: noise[PATHS]

            },
            BUFFERS: {
                CLEAN: cbuf[..., :1, :min_length],
                NOISY: nbuf[..., :1, :min_length],
            },
            NAME: f"Clean={clean[NAME]} | Noise={noise[NAME]}",
            SAMPLING_RATE: LEARNT_SAMPLING_RATE
        }
    else:
        # signals are loaded in CPU
        signal = signals[idx % len(signals)]
        if BUFFERS not in signal:
            load_buffers(signal)
        global_params["premixed_snr"] = signal.get("premixed_snr", None)
        signal[NAME] = f"File={signal[NAME]}"
    global_params["selected_info"] = signal[NAME]
    global_params[SAMPLING_RATE] = signal[SAMPLING_RATE]
    return signal


@interactive(
    snr=(0., [-10., 10.], "SNR [dB]")
)
def remix(signals, snr=0., global_params={}):
    signal = signals[BUFFERS][CLEAN]
    noisy = signals[BUFFERS][NOISY]
    alpha = 10 ** (-snr / 20) * torch.norm(signal) / torch.norm(noisy)
    mixed_signal = signal + alpha * noisy
    global_params["snr"] = snr
    return mixed_signal


@interactive(std_dev=Control(0., value_range=[0., 0.1], name="extra noise std", step=0.0001),
             amplify=(1., [0., 10.], "amplification of everything"))
def augment(signals, mixed, std_dev=0., amplify=1.):
    signals[BUFFERS][MIXED] *= amplify
    signals[BUFFERS][NOISY] *= amplify
    signals[BUFFERS][CLEAN] *= amplify
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


def compute_metrics(pred, sig, global_params={}):
    METRICS = "metrics"
    target = sig[BUFFERS][CLEAN]
    global_params[METRICS] = {}
    global_params[METRICS]["MSE"] = torch.mean((target-pred.cpu())**2)
    global_params[METRICS]["SNR"] = 10.*torch.log10(torch.sum(target**2)/torch.sum((target-pred.cpu())**2))


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
    start_idx, end_idx, _skip_factor = get_trim(signal[BUFFERS][CLEAN][0, :], zval, center)
    global_params["trim"] = dict(start=start_idx, end=end_idx)
    selected = global_params.get("selected_audio", MIXED)
    clean = SingleCurve(y=zin(signal[BUFFERS][CLEAN][0, :], zval, center),
                        alpha=1.,
                        style="k-",
                        linewidth=0.9,
                        label=("*" if selected == CLEAN else " ")+"clean")
    noisy = SingleCurve(y=zin(signal[BUFFERS][NOISY][0, :], zval, center),
                        alpha=0.3,
                        style="y--",
                        linewidth=1,
                        label=("*" if selected == NOISY else " ") + "noisy"
                        )
    mixed = SingleCurve(y=zin(mixed_signal[0, :], zval, center), style="r-",
                        alpha=0.1,
                        linewidth=2,
                        label=("*" if selected == MIXED else " ") + "mixed")
    # true_mixed = SingleCurve(y=zin(signal[BUFFERS][MIXED][0, :], zval, center),
    #                          alpha=0.3, style="b-", linewidth=1, label="true mixed")
    pred.y = zin(pred.y, zval, center)
    pred.label = ("*" if selected == PREDICTED else " ") + pred.label
    curves = [noisy, mixed, pred, clean]
    title = f"SNR  in {global_params['snr']:.1f} dB"
    if "selected_info" in global_params:
        title += f" | {global_params['selected_info']}"
    title += "\n"
    for metric_name, metric_value in global_params.get("metrics", {}).items():
        title += f" | {metric_name} "
        title += f"{metric_value:.2e}" if (abs(metric_value) < 1e-2 or abs(metric_value)
                                           > 1000) else f"{metric_value:.2f}"
    # if global_params.get("premixed_snr", None) is not None:
    #     title += f"| Premixed SNR : {global_params['premixed_snr']:.1f} dB"
    return Curve(curves, ylim=[-0.04 * 1.5 ** zoomy, 0.04 * 1.5 ** zoomy], xlabel="Time index", ylabel="Amplitude", title=title)


def interactive_audio_separation_processing(signals, model_list, config_list):
    sig = signal_selector(signals)
    mixed = remix(sig)
    # sig, mixed = augment(sig, mixed)
    select_device()
    pred, pred_curve = audio_sep_inference(mixed, model_list, config_list)
    compute_metrics(pred, sig)
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
        if BUFFERS not in signal:
            load_buffers(signal, device="cpu")
        clean = SingleCurve(y=signal[BUFFERS][CLEAN][0, :], label="clean")
        noisy = SingleCurve(y=signal[BUFFERS][NOISY][0, :], label="noise", alpha=0.3)
        curves = [clean, noisy]
        for config, model in zip(config_list, model_list):
            short_name = config.get(SHORT_NAME, "unknown")
            predicted_signal, predicted_noise = model(signal[BUFFERS][MIXED].to(device).unsqueeze(0))
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


def main(argv: List[str]):
    """Paired signals and noise in folders"""
    batch = Batch(argv)
    batch.set_io_description(
        input_help='input audio files',
        output_help=argparse.SUPPRESS
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


def main_custom(argv: List[str]):
    """Handle custom noise and custom signals
    """
    parser = parse_command_line()
    parser.add_argument("-s", "--signal", type=str, required=True, nargs="+", help="Signal to be preloaded")
    parser.add_argument("-n", "--noise", type=str, required=True, nargs="+", help="Noise to be preloaded")
    args = parser.parse_args(argv)
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
    all_signals = {}
    for args_paths, key in zip([args.signal, args.noise], [CLEAN, NOISY]):
        new_argv = ["-i"] + args_paths
        if args.preload:
            new_argv += ["--preload"]
        batch = Batch(new_argv)
        new_parser = parse_command_line_generic_audio_load()
        batch.set_io_description(
            input_help=argparse.SUPPRESS,  # 'input audio files',
            output_help=argparse.SUPPRESS
        )
        batch.set_multiprocessing_enabled(False)
        _ = batch.parse_args(new_parser)
        all_signals[key] = batch.run(generic_audio_loading_batch)
    interactive_audio_separation_visualization(all_signals, models_list, config_list, gui=args.gui)

from batch_processing import Batch
import argparse
import sys
from pathlib import Path
from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
import torch
from gyraudio.audio_separation.experiment_tracking.storage import load_checkpoint
from gyraudio.io.audio import load_audio_tensor, save_audio_tensor


def parse_command_line(batch: Batch) -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description='Batch audio processing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e",  "--experiments", type=int, nargs="+", required=True,
                        help="Experiment ids to be inferred sequentially")
    parser.add_argument("-p", "--interactive", action="store_true", help="Play = Interactive mode")
    parser.add_argument("-m", "--model-root", type=str, default=EXPERIMENT_STORAGE_ROOT)
    parser.add_argument("-d", "--device", type=str, default=default_device)
    return batch.parse_args(parser)


def outp(path: Path, suffix: str, extension=".wav"):
    return (path.parent / (path.stem + suffix)).with_suffix(extension)


def audio_separation_processing(input: Path, output: Path, args: argparse.Namespace):
    input_audio_path = list(input.glob("mix*.wav"))[0]
    device = args.device
    for exp in args.experiments:
        model_dir = Path(args.model_root)
        short_name, model, config, _dl = get_experience(exp)
        _, exp_dir = get_output_folder(config, root_dir=model_dir, override=False)
        assert exp_dir.exists(), f"Experiment {short_name} does not exist in {model_dir}"
        model.eval()
        model.to(device)
        model, __optimizer, epoch, config = load_checkpoint(model, exp_dir, epoch=None)
        mixed_signal, sampling_rate = load_audio_tensor(input_audio_path, device=device)
        predicted_signal, predicted_noise = model(mixed_signal)
        save_audio_tensor(outp(output, f"_prediction_{short_name}"), predicted_signal, sampling_rate)
        save_audio_tensor(outp(output, "_mix"), mixed_signal, sampling_rate)


def main(argv):
    batch = Batch(argv)
    batch.set_io_description(
        input_help='input audio files',
        output_help='output directory'
    )
    __args = parse_command_line(batch)
    batch.set_multiprocessing_enabled(False)
    batch.run(audio_separation_processing)


if __name__ == "__main__":
    main(sys.argv[1:])

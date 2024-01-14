from batch_processing import Batch
import argparse
import sys
from pathlib import Path
from gyraudio.audio_separation.properties import CLEAN, NOISY, MIXED, PATHS, BUFFERS, NAME, SAMPLING_RATE
from gyraudio.io.audio import load_audio_tensor


def parse_command_line_audio_load() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Batch audio processing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-preload", "--preload", action="store_true", help="Preload audio files")
    return parser


def outp(path: Path, suffix: str, extension=".wav"):
    return (path.parent / (path.stem + suffix)).with_suffix(extension)


def load_buffers(signal: dict, device="cpu") -> None:
    clean_signal, sampling_rate = load_audio_tensor(signal[PATHS][CLEAN], device=device)
    noisy_signal, sampling_rate = load_audio_tensor(signal[PATHS][NOISY], device=device)
    mixed_signal, sampling_rate = load_audio_tensor(signal[PATHS][MIXED], device=device)
    signal[BUFFERS] = {
        CLEAN: clean_signal,
        NOISY: noisy_signal,
        MIXED: mixed_signal
    }
    signal[SAMPLING_RATE] = sampling_rate


def audio_loading(input: Path, preload: bool) -> dict:
    name = input.name
    clean_audio_path = input/"voice.wav"
    noisy_audio_path = input/"noise.wav"
    mixed_audio_path = list(input.glob("mix*.wav"))[0]
    signal = {
        NAME: name,
        PATHS: {
            CLEAN: clean_audio_path,
            NOISY: noisy_audio_path,
            MIXED: mixed_audio_path
        }
    }
    signal["premixed_snr"] = float(mixed_audio_path.stem.split("_")[-1])
    if preload:
        load_buffers(signal)
    return signal


def audio_loading_batch(input: Path, args: argparse.Namespace) -> dict:
    """Wrapper to load audio files from a directory using batch_processing
    """
    return audio_loading(input, preload=args.preload)


def main(argv):
    batch = Batch(argv)
    batch.set_io_description(
        input_help='input audio files',
        output_help=argparse.SUPPRESS
    )
    parser = parse_command_line_audio_load()
    batch.parse_args(parser)
    all_signals = batch.run(audio_loading_batch)
    return all_signals


if __name__ == "__main__":
    main(sys.argv[1:])

from batch_processing import Batch
import argparse
import sys
from pathlib import Path
from gyraudio.audio_separation.properties import PATHS, BUFFERS, NAME, GENERIC
from gyraudio.io.audio import load_audio_tensor


def parse_command_line_generic_audio_load() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Batch audio loading',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-preload", "--preload", action="store_true", help="Preload audio files")
    return parser


def load_buffers_custom(signal: dict, device="cpu") -> None:
    generic_signal, sampling_rate = load_audio_tensor(signal[PATHS], device=device)
    signal[BUFFERS] = generic_signal
    signal["sampling_rate"] = sampling_rate


def audio_loading(input: Path, preload: bool) -> dict:
    name = input.name
    signal = {
        NAME: name,
        PATHS: input,
    }
    if preload:
        load_buffers_custom(signal)
    return signal


def generic_audio_loading_batch(input: Path, args: argparse.Namespace) -> dict:
    """Wrapper to load audio files from a directory using batch_processing
    """
    return audio_loading(input, preload=args.preload)


def main(argv):
    batch = Batch(argv)
    batch.set_io_description(
        input_help='input audio files',
        output_help=argparse.SUPPRESS
    )
    parser = parse_command_line_generic_audio_load()
    batch.parse_args(parser)
    all_signals = batch.run(generic_audio_loading_batch)
    return all_signals


if __name__ == "__main__":
    main(sys.argv[1:])

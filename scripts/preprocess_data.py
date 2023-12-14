import argparse
from batch_processing import Batch
import sys
# from gyraudio import root_dir
from pathlib import Path


def data_processing(input_file: Path, output_dir: Path, args):
    print(input_file)
    pass


def parse_command_line(batch: Batch) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch video processing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-mp", "--multi-processing", action="store_true",
                        help="Enable multiprocessing - Warning with GPU - use -j2")
    parser.add_argument("--override", action="store_true",
                        help="overwrite processed results")
    return batch.parse_args(parser)


def main(argv):
    batch = Batch(argv)
    batch.set_io_description(
        input_help='input video files', output_help='output directory')
    args = parse_command_line(batch)
    # Disable mp - Highly recommended!
    if not args.multi_processing:
        batch.set_multiprocessing_enabled(False)
    batch.run(data_processing)


if __name__ == "__main__":
    main(sys.argv[1:])

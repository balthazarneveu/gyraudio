import argparse
from batch_processing import Batch
import sys
# from gyraudio import root_dir
from gyraudio.io.audio import load_raw_audio
from gyraudio.io.imu import get_imu_data
from pathlib import Path
import logging
import numpy as np


def sanity_check_plot(audio_signal, imu_data, rate_audio=48000, rate_imu=200):
    import matplotlib.pyplot as plt
    import numpy as np
    timeline = np.arange(len(audio_signal))/rate_audio
    timeline_imu = np.arange(len(imu_data))/rate_imu
    for idx in range(audio_signal.shape[1]):
        plt.plot(timeline, audio_signal[:, idx], label=f"audio mic={idx}")
    for idx in range(imu_data.shape[1]):
        plt.plot(timeline_imu, imu_data[:, idx], label=f"gyro {'xyz'[idx]}")
    plt.legend()
    plt.show()


def data_processing(input_file: Path, output_dir: Path, args):
    rate, sig = load_raw_audio(input_file.with_suffix(".WAV"))
    sig = sig.astype(np.float32)
    sig /= np.fabs(sig[rate:-rate]).max()
    sig *= 3.
    # offset = int((sig[:,0]==0).sum())
    # sig = sig[offset:]
    channels = sig.shape[1]
    logging.info(f"Sampling rate {rate/1e3}kHz, length {sig.shape[0]/rate:.1f}, {channels} audio channels")
    gyro = get_imu_data(input_file)
    sanity_check_plot(sig, gyro, rate_audio=rate, rate_imu=200.)
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

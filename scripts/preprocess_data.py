import argparse
from batch_processing import Batch
import sys
# from gyraudio import root_dir
from gyraudio.io.audio import load_raw_audio
from gyraudio.io.imu import get_imu_data
from gyraudio.io.dump import Dump
from gyraudio.properties import GYRO, AUDIO, AUDIO_RATE
from pathlib import Path
import logging
import numpy as np


def sanity_check_plot(audio_signal, imu_data, rate_audio=48000, rate_imu=200):
    # Do a quick uniformization of the audio signal
    audio_signal = np.array(audio_signal).astype(np.float32)
    audio_signal /= np.fabs(audio_signal[rate_audio:-rate_audio]).max()
    audio_signal *= 3.

    import matplotlib.pyplot as plt
    timeline = np.arange(len(audio_signal))/rate_audio
    timeline_imu = np.arange(len(imu_data))/rate_imu
    for idx in range(audio_signal.shape[1]):
        plt.plot(timeline, audio_signal[:, idx], label=f"audio mic={idx}")
    for idx in range(imu_data.shape[1]):
        plt.plot(timeline_imu, imu_data[:, idx], label=f"gyro {'xyz'[idx]}")
    plt.legend()
    plt.show()


def data_processing(input_file: Path, output_dir: Path, args):
    if input_file.suffix == ".pkl":
        preprocessed_file = input_file
    else:
        preprocessed_file = output_dir.with_suffix(".pkl")
    if preprocessed_file.exists() and not args.override:
        logging.info(f"Skipping {input_file.name}")
        data_samples = Dump.load_pickle(preprocessed_file)
    else:
        rate, sig = load_raw_audio(input_file.with_suffix(".WAV"))
        # offset = int((sig[:,0]==0).sum())
        # sig = sig[offset:]
        channels = sig.shape[1]
        logging.info(f"Sampling rate {rate/1e3}kHz, length {sig.shape[0]/rate:.1f}, {channels} audio channels")
        gyro = get_imu_data(input_file.with_suffix(".MP4"))
        data_samples = {
            GYRO: gyro,
            AUDIO_RATE: rate,
            AUDIO: sig,
        }
        Dump.save_pickle(data_samples, preprocessed_file)

    sanity_check_plot(data_samples[AUDIO], data_samples[GYRO], rate_audio=data_samples[AUDIO_RATE], rate_imu=200.)
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

from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.parser import shared_parser
from gyraudio.audio_separation.properties import TEST, NAME, SHORT_NAME, CURRENT_EPOCH, SNR_FILTER
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.experiment_tracking.storage import load_checkpoint
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder
from gyraudio.io.dump import Dump
from pathlib import Path
import sys
import torch
import pandas as pd
import wandb
from gyraudio.io.audio import load_raw_audio
from typing import List
# Files paths
DEFAULT_RECORD_FILE = "infer_record.csv"  # Store the characteristics of the inference record file
DEFAULT_EVALUATION_FILE = "eval_df.csv"  # Store the characteristics of the inference record file
# Record keys
NBATCH = "nb_batch"
BEST_SNR = "best_snr"
BEST_SAVE_SNR = "best_save_snr"
WORST_SNR = "worst_snr"
WORST_SAVE_SNR = "worst_save_snr"
RECORD_KEYS = [NAME, SHORT_NAME, CURRENT_EPOCH, NBATCH, SNR_FILTER, BEST_SAVE_SNR, BEST_SNR, WORST_SAVE_SNR, WORST_SNR]
# Exaluation keys
SAVE_IDX = "save_idx"
SNR_IN = "snr_in"
SNR_OUT = "snr_out"
EVAL_KEYS = [SAVE_IDX, SNR_IN, SNR_OUT]


def load_file(path: Path, keys: List[str]) -> pd.DataFrame:
    if not (path.exists()):
        df = pd.DataFrame(columns=keys)
        df.to_csv(path)
    return pd.read_csv(path)


def launch_infer(exp: int, snr_filter: list = None, device: str = "cuda", model_dir: Path = None,
                 output_dir: Path = EXPERIMENT_STORAGE_ROOT, force_reload=False, max_batches=None,
                 ext=".wav"):
    # Load experience
    run = wandb.init(project="samples-audio-sep",
                     name=f"{exp:04d}",
                     config={
                         "exp": exp,
                     })
    if snr_filter is not None:
        snr_filter = sorted(snr_filter)
    short_name, model, config, dl = get_experience(exp, snr_filter_test=snr_filter)
    exists, exp_dir = get_output_folder(config, root_dir=model_dir, override=False)
    assert exp_dir.exists(), f"Experiment {short_name} does not exist in {model_dir}"
    model.eval()
    model.to(device)
    model, optimizer, epoch, config_checkpt = load_checkpoint(model, exp_dir, epoch=None, device=device)

    # Folder creation
    if output_dir is not None:
        for sample in [0, 1, 2, 10]:
            save_dir = output_dir/(exp_dir.name+"_infer" + (f"_epoch_{epoch:04d}_nbatch_{max_batches if max_batches is not None else len(dl[TEST])}")
                                   + ("" if snr_filter is None else f"_snrs_{'_'.join(map(str, snr_filter))}"))
            print(save_dir)
            audio_conf = list(save_dir.glob(f"{sample:04d}*.json"))[0]
            results_dict = Dump.load_json(audio_conf)
            print(results_dict)
            for sig_mode in ["out", "mixed", "original"]:
                audio_file = audio_conf.parent/(audio_conf.stem + "_" + sig_mode + ext)
                rate, signal = load_raw_audio(audio_file)
                # print(audio)
                snrin, snrout = results_dict["snr_in"], results_dict["snr_out"]
                ampli = 1.
                if sig_mode == "out":
                    caption = f"SNR in = {snrin:.1f}db | out = {snrout:.1f}db"
                elif sig_mode == "mixed":
                    caption = f"MIXED SNR in = {snrin:.1f}db"
                else:
                    caption = f"CLEAN"
                run.log({f"Sample {sample}": wandb.Audio(
                    # (ampli*signal).clip(-1, 1),
                    3*signal,
                    caption=caption,
                    sample_rate=rate)})

    run.finish()


def main(argv):
    ##### THIS IS JUST
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser_def = shared_parser(help="Launch inference on a specific model"
                               + ("\n<<<Cuda available>>>" if default_device == "cuda" else ""))
    parser_def.add_argument("-i", "--input-dir", type=str, default=EXPERIMENT_STORAGE_ROOT)
    parser_def.add_argument("-o", "--output-dir", type=str, default=EXPERIMENT_STORAGE_ROOT)
    parser_def.add_argument("-d", "--device", type=str, default=default_device,
                            help="Training device", choices=["cpu", "cuda"])
    parser_def.add_argument("-r", "--reload", action="store_true",
                            help="Force reload files")
    parser_def.add_argument("-b", "--nb-batch", type=int, default=None,
                            help="Number of batches to process")
    parser_def.add_argument("-s",  "--snr-filter", type=float, nargs="+", default=None,
                            help="SNR filters on the inference dataloader")
    parser_def.add_argument("-ext", "--extension", type=str, default=".wav", help="Extension of the audio files to save",
                            choices=[".wav", ".mp4"])
    args = parser_def.parse_args(argv)
    for exp in args.experiments:
        launch_infer(
            exp,
            model_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            device=args.device,
            force_reload=args.reload,
            max_batches=args.nb_batch,
            snr_filter=args.snr_filter,
            ext=args.extension
        )


if __name__ == "__main__":
    main(sys.argv[1:])

# Example : python src\gyraudio\audio_separation\infer.py -i ./__output_audiosep -e 1002 -d cpu -b 2 -s 4 5 6

from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.parser import shared_parser
from gyraudio.audio_separation.properties import TEST, NAME, SHORT_NAME, CURRENT_EPOCH, SNR_FILTER
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.experiment_tracking.storage import load_checkpoint
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder
from gyraudio.audio_separation.metrics import snr
from pathlib import Path
import sys
import torch
from tqdm import tqdm
import torchaudio
import pandas as pd
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


def launch_infer(exp: int, snr_filter: list = None, device: str = "cuda", model_dir: Path = None, output_dir: Path = EXPERIMENT_STORAGE_ROOT, force_reload=False, max_batches=None):
    # Load experience
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
        record_path = output_dir/DEFAULT_RECORD_FILE
        record_df = load_file(record_path, RECORD_KEYS)

        # Define conditions for filtering
        exist_conditions = {
            NAME: config[NAME],
            SHORT_NAME: config[SHORT_NAME],
            CURRENT_EPOCH: epoch,
            NBATCH: max_batches,
        }
        # Create boolean masks and combine them
        masks = [(record_df[key] == value) for key, value in exist_conditions.items()]
        if snr_filter is None:
            masks.append((record_df[SNR_FILTER]).isnull())
        else:
            masks.append(record_df[SNR_FILTER] == str(snr_filter))
        combined_mask = pd.Series(True, index=record_df.index)
        for mask in masks:
            combined_mask = combined_mask & mask
        filtered_df = record_df[combined_mask]

        save_dir = output_dir/(exp_dir.name+"_infer" + (f"_epoch_{epoch:04d}_nbatch_{max_batches if max_batches is not None else len(dl[TEST])}")
                               + ("" if snr_filter is None else f"_snrs_{'_'.join(map(str, snr_filter))}"))
        evaluation_path = save_dir/DEFAULT_EVALUATION_FILE
        if not (filtered_df.empty) and not (force_reload):
            assert evaluation_path.exists()
            print(f"Inference already exists, see folder {save_dir}")
            record_row_df = filtered_df
        else:
            record_row_df = pd.DataFrame({
                NAME: config[NAME],
                SHORT_NAME: config[SHORT_NAME],
                CURRENT_EPOCH: epoch,
                NBATCH: max_batches,
                SNR_FILTER: [None],
            }, index=[0], columns=RECORD_KEYS)
            record_row_df.at[0, SNR_FILTER] = snr_filter

            save_dir.mkdir(parents=True, exist_ok=True)
            evaluation_df = load_file(evaluation_path, EVAL_KEYS)
            with torch.no_grad():
                test_loss = 0.
                save_idx = 0
                best_snr = 0
                worst_snr = 0
                processed_batches = 0
                for step_index, (batch_mix, batch_signal, batch_noise) in tqdm(
                        enumerate(dl[TEST]), desc=f"Inference epoch {epoch}", total=max_batches if max_batches is not None else len(dl[TEST])):
                    batch_mix, batch_signal, batch_noise = batch_mix.to(
                        device), batch_signal.to(device), batch_noise.to(device)
                    batch_output_signal, _batch_output_noise = model(batch_mix)
                    loss = torch.nn.functional.mse_loss(batch_output_signal, batch_signal)
                    test_loss += loss.item()

                    # SNR stats
                    snr_in = snr(batch_mix, batch_signal, reduce=None)
                    snr_out = snr(batch_output_signal, batch_signal, reduce=None)
                    best_current, best_idx_current = torch.max(snr_out-snr_in, axis=0)
                    worst_current, worst_idx_current = torch.min(snr_out-snr_in, axis=0)
                    if best_current > best_snr:
                        best_snr = best_current
                        best_save_idx = save_idx + best_idx_current
                    if worst_current > worst_snr:
                        worst_snr = worst_current
                        worst_save_idx = save_idx + worst_idx_current

                    # Save by signal
                    batch_output_signal = batch_output_signal.detach().cpu()
                    batch_signal = batch_signal.detach().cpu()
                    batch_mix = batch_mix.detach().cpu()
                    for audio_idx in range(batch_output_signal.shape[0]):
                        new_eval_row = pd.DataFrame({SAVE_IDX: save_idx, SNR_IN: float(
                            snr_in[audio_idx]), SNR_OUT: float(snr_out[audio_idx])}, index=[0])
                        evaluation_df = pd.concat([new_eval_row, evaluation_df.loc[:]], ignore_index=True)

                        # Save .wav
                        torchaudio.save(
                            str(save_dir/f"{save_idx:04d}_mixed.wav"),
                            batch_mix[audio_idx, :, :],
                            sample_rate=dl[TEST].dataset.sampling_rate,
                            channels_first=True
                        )
                        torchaudio.save(
                            str(save_dir/f"{save_idx:04d}_out.wav"),
                            batch_output_signal[audio_idx, :, :],
                            sample_rate=dl[TEST].dataset.sampling_rate,
                            channels_first=True
                        )

                        save_idx += 1
                    processed_batches += 1
                    if max_batches is not None and processed_batches >= max_batches:
                        break
            test_loss = test_loss/len(dl[TEST])
            evaluation_df.to_csv(evaluation_path)

            record_row_df[BEST_SAVE_SNR] = int(best_save_idx)
            record_row_df[BEST_SNR] = float(best_snr)
            record_row_df[WORST_SAVE_SNR] = int(worst_save_idx)
            record_row_df[WORST_SNR] = float(worst_snr)
            record_df = pd.concat([record_row_df, record_df.loc[:]], ignore_index=True)
            record_df.to_csv(record_path, index=0)

            print(f"Test loss: {test_loss:.3e}, \nbest snr performance: {best_save_idx} with {best_snr:.1f}dB, \nworst snr performance: {worst_save_idx} with {worst_snr:.1f}dB")

    return record_row_df, evaluation_path


def main(argv):
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
    args = parser_def.parse_args(argv)
    for exp in args.experiments:
        launch_infer(
            exp,
            model_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            device=args.device,
            force_reload=args.reload,
            max_batches=args.nb_batch,
            snr_filter=args.snr_filter
        )


if __name__ == "__main__":
    main(sys.argv[1:])

# Example : python src\gyraudio\audio_separation\infer.py -i ./__output_audiosep -e 1002 -d cpu -b 2 -s 4 5 6

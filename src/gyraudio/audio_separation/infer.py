from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.parser import shared_parser
from gyraudio.audio_separation.properties import TEST
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.experiment_tracking.storage import load_checkpoint
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder
from gyraudio.audio_separation.metrics import snr
from pathlib import Path
import sys
import json
import torch
from tqdm import tqdm
import torchaudio
import pandas as pd
SAVE_IDX = "save_idx"
SNR_IN = "snr_in"
SNR_OUT = "snr_out"

def launch_infer(exp: int, snr_filter : list = None, device: str = "cuda", model_dir: Path = None, output_dir: Path = EXPERIMENT_STORAGE_ROOT, force_reload = False, max_batches = 1):
    short_name, model, config, dl = get_experience(exp, snr_filter_test=snr_filter)
    exists, exp_dir = get_output_folder(config, root_dir=model_dir, override=False)
    assert exp_dir.exists(), f"Experiment {short_name} does not exist in {model_dir}"
    model.eval()
    model.to(device)
    model, optimizer, epoch, config = load_checkpoint(model, exp_dir, epoch=None, device=device)
    if output_dir is not None:
        save_dir = output_dir/(exp_dir.name+"_infer"+f"_epoch_{epoch:04d}")
        save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        test_loss = 0.
        save_idx = 0
        best_snr_perf = 0
        worst_snr_perf = 0
        processed_batches = 0
        total_df = pd.DataFrame({SAVE_IDX : [], SNR_IN : [], SNR_OUT : []})
        for step_index, (batch_mix, batch_signal, batch_noise) in tqdm(
                enumerate(dl[TEST]), desc=f"Inference epoch {epoch}", total=len(dl[TEST])):
            batch_mix, batch_signal, batch_noise = batch_mix.to(
                device), batch_signal.to(device), batch_noise.to(device)
            batch_output_signal, _batch_output_noise = model(batch_mix)

            loss = torch.nn.functional.mse_loss(batch_output_signal, batch_signal)
            test_loss += loss.item()
            snr_in = snr(batch_mix, batch_signal, reduce=None)
            snr_out = snr(batch_output_signal, batch_signal, reduce=None)
            best_current, best_idx_current = torch.max(snr_out-snr_in, 0)
            worst_current, worst_idx_current = torch.min(snr_out-snr_in, 0)
            worst_current = torch.min(snr_out-snr_in)
            if best_current > best_snr_perf :
                best_snr_perf = best_current
                best_save_idx = save_idx + best_idx_current
            if worst_current > worst_snr_perf :
                worst_snr_perf = worst_current
                worst_save_idx = save_idx + worst_idx_current

            batch_output_signal = batch_output_signal.detach().cpu()
            batch_signal = batch_signal.detach().cpu()
            batch_mix = batch_mix.detach().cpu()
            for audio_idx in range(batch_output_signal.shape[0]):
                new_row = pd.DataFrame({SAVE_IDX : save_idx, SNR_IN : float(snr_in[audio_idx]), SNR_OUT : float(snr_out[audio_idx])}, index = [0])
                total_df = pd.concat([new_row, total_df.loc[:]], ignore_index=True)
                audio_dict = {} 
                audio_dict["SNR_IN"] = float(snr_in[audio_idx])
                audio_dict["SNR_OUT"] = float(snr_out[audio_idx])
                if force_reload or not((save_dir/f"{save_idx:04d}_metrics.json").exists()) :
                    with open(save_dir/f"{save_idx:04d}_metrics.json", "w") as metrics_file :
                        metrics_file.write(json.dumps(audio_dict))
                if force_reload or not((save_dir/f"{save_idx:04d}_mixed.wav").exists()) :
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
                # TODO: save SNR metadata in the name?
                save_idx += 1
            processed_batches += 1
            if processed_batches >= max_batches :
                break
    test_loss = test_loss/len(dl[TEST])
    total_df.to_csv(save_dir/f"total_df.csv")

    total_dict = {"N_SAMPLES" : len(total_df), "BEST_SNR_SAVE" : int(best_save_idx), "BEST_SNR" : float(best_snr_perf), "WORST_SNR_SAVE" : int(worst_save_idx), "WORST_SNR" : float(worst_snr_perf), "LOSS" : float(test_loss)}
    with open(save_dir/f"total_metrics.json", "w") as metrics_file :
        metrics_file.write(json.dumps(total_dict))
    print(f"Test loss: {test_loss:.3e}, \nbest snr performance: {best_save_idx} with {best_snr_perf:.1f}dB, \nworst snr performance: {worst_save_idx} with {worst_snr_perf:.1f}dB")

    return save_dir/f"total_df.csv"


def main(argv):
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser_def = shared_parser(help="Launch training \nCheck results at: https://wandb.ai/balthazarneveu/audio-sep"
                               + ("\n<<<Cuda available>>>" if default_device == "cuda" else ""))
    parser_def.add_argument("-i", "--input-dir", type=str, default=EXPERIMENT_STORAGE_ROOT)
    parser_def.add_argument("-o", "--output-dir", type=str, default=EXPERIMENT_STORAGE_ROOT)
    parser_def.add_argument("-d", "--device", type=str, default=default_device,
                            help="Training device", choices=["cpu", "cuda"])
    parser_def.add_argument("-r", "--reload", action="store_true",
                        help="Force reload files")
    parser_def.add_argument("-b", "--nb-batch", type=int, default=1,
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

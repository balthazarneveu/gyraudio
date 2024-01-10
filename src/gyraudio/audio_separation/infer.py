from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
from gyraudio.audio_separation.parser import shared_parser
from gyraudio.audio_separation.properties import TEST
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.experiment_tracking.storage import load_checkpoint
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder
from pathlib import Path
import sys
import torch
from tqdm import tqdm
import torchaudio


def launch_infer(exp: int, device: str = "cuda", model_dir: Path = None, output_dir: Path = None):
    short_name, model, config, dl = get_experience(exp)
    exists, exp_dir = get_output_folder(config, root_dir=model_dir, override=False)
    assert exp_dir.exists(), f"Experiment {short_name} does not exist in {model_dir}"
    # assert exists, f"Experiment {short_name} does not exist in {model_dir}"
    model.eval()
    model.to(device)
    model, optimizer, epoch, config = load_checkpoint(model, exp_dir, epoch=None, device=device)
    if output_dir is not None:
        save_dir = output_dir/(exp_dir.name+"_infer"+f"_epoch_{epoch:04d}")
        save_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        test_loss = 0.
        save_idx = 0
        for step_index, (batch_mix, batch_signal, batch_noise) in tqdm(
                enumerate(dl[TEST]), desc=f"Inference epoch {epoch}", total=len(dl[TEST])):
            batch_mix, batch_signal, batch_noise = batch_mix.to(
                device), batch_signal.to(device), batch_noise.to(device)
            batch_output_signal, _batch_output_noise = model(batch_mix)
            loss = torch.nn.functional.mse_loss(batch_output_signal, batch_signal)
            test_loss += loss.item()
            batch_output_signal = batch_output_signal.detach().cpu()
            batch_mix = batch_mix.detach().cpu()
            for audio_idx in range(batch_output_signal.shape[0]):
                torchaudio.save(
                    str(save_dir/f"{save_idx:04d}_dirty.wav"),
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
            break  # TODO: limit amount of samples
    test_loss = test_loss/len(dl[TEST])
    print(f"Test loss: {test_loss:.3e}")


def main(argv):
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser_def = shared_parser(help="Launch training \nCheck results at: https://wandb.ai/balthazarneveu/audio-sep"
                               + ("\n<<<Cuda available>>>" if default_device == "cuda" else ""))
    parser_def.add_argument("-i", "--input-dir", type=str, default=EXPERIMENT_STORAGE_ROOT)
    parser_def.add_argument("-o", "--output-dir", type=str, default=EXPERIMENT_STORAGE_ROOT)
    parser_def.add_argument("-d", "--device", type=str, default=default_device,
                            help="Training device", choices=["cpu", "cuda"])
    args = parser_def.parse_args(argv)
    for exp in args.experiments:
        launch_infer(
            exp,
            model_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            device=args.device
        )


if __name__ == "__main__":
    main(sys.argv[1:])

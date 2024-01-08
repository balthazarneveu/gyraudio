import torch
import matplotlib.pyplot as plt
import numpy as np
from gyraudio.default_locations import SAMPLE_ROOT
from gyraudio.audio_separation.visualization.pre_load_audio import audio_loading


def get_silence_mask(
        sig: torch.Tensor, morph_kernel_size: int = 499, k_smooth=21, thresh=0.0001,
        debug: bool = False) -> torch.Tensor:
    with torch.no_grad():
        smooth = torch.nn.Conv1d(1, 1, k_smooth, padding=k_smooth//2, bias=False).to(sig.device)
        smooth.weight.data.fill_(1./k_smooth)
        smoothed = smooth(torch.abs(sig))
        st = 1.*(torch.abs(smoothed) < thresh*torch.ones_like(smoothed, device=sig.device))
        sig_dil = torch.nn.MaxPool1d(morph_kernel_size, stride=1, padding=morph_kernel_size//2)(st)
        sig_ero = -torch.nn.MaxPool1d(morph_kernel_size, stride=1, padding=morph_kernel_size//2)(-sig_dil)
    if debug:
        return sig_ero.squeeze(0), smoothed.squeeze(0), st.squeeze(0)
    else:
        return sig_ero


def visualize_silence_mask(sig: torch.Tensor, silence_thresh: float = 0.0001):
    silence_thresh = 0.0001
    silence_mask, smoothed_amplitude, _ = get_silence_mask(
        sig, k_smooth=21, morph_kernel_size=499, thresh=silence_thresh, debug=True
    )
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(sig.squeeze(0).cpu().numpy(), "k-", label="voice", alpha=0.5)
    plt.plot(0.01*silence_mask.cpu().numpy(), "r-", alpha=1., label="silence mask")
    plt.grid()
    plt.legend()
    plt.title("Voice and silence mask")
    plt.ylim(-0.04, 0.04)

    plt.subplot(122)
    plt.plot(smoothed_amplitude.cpu().numpy(), "g--", alpha=0.5, label="smoothed amplitude")
    plt.plot(np.ones(silence_mask.shape[-1])*silence_thresh, "c--", alpha=1., label="threshold")
    plt.plot(-silence_thresh+silence_thresh*silence_mask.cpu().numpy(), "r-", alpha=1, label="silence mask")
    plt.grid()
    plt.legend()
    plt.title("Thresholding mechanism")
    plt.ylim(-silence_thresh, silence_thresh*10)
    plt.show()


if __name__ == "__main__":
    from gyraudio.audio_separation.properties import CLEAN, BUFFERS
    sample_folder = SAMPLE_ROOT/"0009"
    signals = audio_loading(sample_folder, preload=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sig_in = signals[BUFFERS][CLEAN].to(device)
    visualize_silence_mask(sig_in)

from gyraudio.audio_separation.properties import CLEAN, NOISY, MIXED, PREDICTED
from pathlib import Path
from gyraudio.io.audio import save_audio_tensor
from gyraudio import root_dir
from interactive_pipe import Control
from interactive_pipe import interactive
import logging
import torch

HERE = Path(__file__).parent
MUTE = "mute"
LOGOS = {
    PREDICTED: HERE/"play_logo_pred.png",
    MIXED: HERE/"play_logo_mixed.png",
    CLEAN: HERE/"play_logo_clean.png",
    NOISY: HERE/"play_logo_noise.png",
    MUTE: HERE/"mute_logo.png",
}
ICONS = [it for key, it in LOGOS.items()]
KEYS = [key for key, it in LOGOS.items()]

ping_pong_index = 0


@interactive(
    volume=(100, [0, 1000], "volume"),
    player=Control(MUTE, KEYS, icons=ICONS))
def audio_player(sig, mixed, pred, global_params={}, volume=100, player=MUTE):
    sampling_rate = global_params.get("sampling_rate", 8000)
    try:
        if player == MUTE:
            global_params["__stop"]()
        else:
            if player == CLEAN:
                audio_track = sig["buffers"][CLEAN]
            elif player == NOISY:
                audio_track = sig["buffers"][NOISY]
            elif player == MIXED:
                audio_track = mixed
            elif player == PREDICTED:
                audio_track = pred
            ping_pong_path = root_dir/"__ping_pong"
            ping_pong_path.mkdir(exist_ok=True)
            global ping_pong_index
            audio_track_path = ping_pong_path/f"_tmp_{ping_pong_index}.wav"
            ping_pong_index = (ping_pong_index + 1) % 10
            if global_params.get("trim", False):
                start, end = global_params["trim"]["start"], global_params["trim"]["end"]
                remainder = (end-start) % 8
                audio_trim = audio_track[..., start:end-remainder]
                repeat_factor = int(sampling_rate*4./(end-start))
                logging.debug(f"{repeat_factor}")
                repeat_factor = max(1, repeat_factor)
                audio_trim = audio_trim.repeat(1, repeat_factor)
                logging.debug(f"{audio_trim.shape}")
            else:
                audio_trim = audio_track
            save_audio_tensor(audio_track_path, volume/100.*audio_trim,
                              sampling_rate=global_params.get("sampling_rate", sampling_rate))
            global_params["__set_audio"](audio_track_path)
            global_params["__play"]()
    except Exception as exc:
        logging.debug(f"Exception in audio_player {exc}")
        pass

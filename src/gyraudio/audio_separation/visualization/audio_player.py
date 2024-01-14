from gyraudio.audio_separation.properties import CLEAN, NOISY, MIXED, PREDICTED, SAMPLING_RATE
from pathlib import Path
from gyraudio.io.audio import save_audio_tensor
from gyraudio import root_dir
from interactive_pipe import Control, KeyboardControl
from interactive_pipe import interactive
import logging

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
    player=Control(MUTE, KEYS, icons=ICONS))
def audio_selector(sig, mixed, pred, global_params={}, player=MUTE):

    global_params["selected_audio"] = player if player != MUTE else global_params.get("selected_audio", MIXED)
    global_params[MUTE] = player == MUTE
    if player == CLEAN:
        audio_track = sig["buffers"][CLEAN]
    elif player == NOISY:
        audio_track = sig["buffers"][NOISY]
    elif player == MIXED:
        audio_track = mixed
    elif player == PREDICTED:
        audio_track = pred
    else:
        audio_track = mixed
    return audio_track


@interactive(
    loop=KeyboardControl(True, keydown="l"))
def audio_trim(audio_track, global_params={}, loop=True):
    sampling_rate = global_params.get(SAMPLING_RATE, 8000)
    if global_params.get("trim", False):
        start, end = global_params["trim"]["start"], global_params["trim"]["end"]
        remainder = (end-start) % 8
        audio_trim = audio_track[..., start:end-remainder]
        repeat_factor = int(sampling_rate*4./(end-start))
        logging.debug(f"{repeat_factor}")
        repeat_factor = max(1, repeat_factor)
        if loop:
            repeat_factor = 1
        audio_trim = audio_trim.repeat(1, repeat_factor)
        logging.debug(f"{audio_trim.shape}")
    else:
        audio_trim = audio_track
    return audio_trim


@interactive(
    volume=(100, [0, 1000], "volume"),
)
def audio_player(audio_trim, global_params={}, volume=100):
    sampling_rate = global_params.get(SAMPLING_RATE, 8000)
    try:
        if global_params.get(MUTE, True):
            global_params["__stop"]()
            print("mute!")
        else:
            ping_pong_path = root_dir/"__ping_pong"
            ping_pong_path.mkdir(exist_ok=True)
            global ping_pong_index
            audio_track_path = ping_pong_path/f"_tmp_{ping_pong_index}.wav"
            ping_pong_index = (ping_pong_index + 1) % 10
            save_audio_tensor(audio_track_path, volume/100.*audio_trim,
                              sampling_rate=global_params.get(SAMPLING_RATE, sampling_rate))
            global_params["__set_audio"](audio_track_path)
            global_params["__play"]()
    except Exception as exc:
        logging.warning(f"Exception in audio_player {exc}")
        pass

from gyraudio.audio_separation.properties import CLEAN, NOISY, MIXED, PREDICTED
from pathlib import Path
from gyraudio.io.audio import save_audio_tensor
from interactive_pipe import Control
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


@interactive(
    volume=(100, [0, 1000], "volume"),
    player=Control(MUTE, KEYS, icons=ICONS))
def audio_player(sig, mixed, pred, global_params={}, volume=100, player=MUTE):
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
            audio_track_path = "_tmp.wav"
            save_audio_tensor(audio_track_path, volume/100.*audio_track,
                              sampling_rate=global_params.get("sampling_rate", 8000))
            global_params["__set_audio"](audio_track_path)
            global_params["__play"]()
    except Exception as exc:
        logging.debug(f"Exception in audio_player {exc}")
        pass

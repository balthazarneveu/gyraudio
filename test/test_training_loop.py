import pytest
from gyraudio.audio_separation.train import launch_training

@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_training_loop(device):
    launch_training(0, wandb_flag=False, device=device)

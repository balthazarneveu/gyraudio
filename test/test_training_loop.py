import pytest
from gyraudio.audio_separation.train import launch_training
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_training_loop(device):
    launch_training(0, wandb_flag=False, device=device, save_dir=EXPERIMENT_STORAGE_ROOT)
    

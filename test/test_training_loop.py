from gyraudio.audio_separation.train import launch_training


def test_training_loop():
    launch_training(0, wandb_flag=False)

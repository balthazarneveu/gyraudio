from gyraudio.audio_separation.push_wandb_samples import main as infer_main
import sys

if __name__ == "__main__":
    infer_main(sys.argv[1:])

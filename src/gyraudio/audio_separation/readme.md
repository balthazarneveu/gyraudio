# Audio separation
## :bug: Pytest

```bash
pytest-3 test
```


## :chart_with_downwards_trend: Training

:id: Keep track of experiments by an integer id. 

Each experiment is defined by:
- Dataloader configuration (data, augmentations)
- Model (architecture, sizes)
- Optimizer configuration (hyperparameters)

### Defining experiments
[Code to define new experiments](/src/gyraudio/audio_separation/experiment_tracking/experiments_definition.py)
### Remote training

- :unlock: Create a [scripts/__kaggle_login.py](/scripts/__kaggle_login.py) file locally.
```python
kaggle_users = {
    "user1": {
        "username": "user1_kaggle_name",
        "key": "user1_kaggle_key"
    },
    "user2": {
        "username": "user2_kaggle_name",
        "key": "user2_kaggle_key"
    },
}
```
- :key: Allow Kaggle secrets to access wandb:
  - `wandb_api_key`: weights and biases API key 
  - :phone: a verified kaggle account is required

Run `python scripts/audio_separation_train_remote.py -u user1 -e X`
This will create a dedicated folder for training a specific experiment with a dedicated notebook.

- use **`-p`** (`--push`) will upload/push the notebook and run it.
- use **`-d`** (`--download`) to download the training results and save it to disk)

#### :green_circle: First time setup
> - use **`--cpu`** to setup at the begining (avoid using GPU when you set up :warning: )
> - please note that the first time, you'll need to manually edit the notebook under kaggle web page to allow secrets.
### Local training
`python scripts/audio_separation_train.py -e 1`

### :bar_chart: [Experiment tracking](https://wandb.ai/balthazarneveu/audio-sep)

## Evaluation
Run inference/evaluation from a checkpoint
`python scripts/audio_separation_infer.py -e 1`

### :gear: Batch processing
Process specific audio files
`python scripts/audio_separation_batch.py -i __data_source_separation/source_separation/test/0000 -o __output_audio -e 1`

### :play_or_pause_button: Interactive
Check audio separation visually.
```
python scripts/audio_separation_interactive.py -i __data_source_separation/source_separation/test/000* -o __output/batch_processing -e 1 --preload -p
```

- :arrow_backward: :arrow_forward: = next audio `left` / `right`
- :arrow_double_down: :arrow_double_up: = next model `page up` / `page down` (*compare models*)

### Dataloader
[Mixed dataloader](/src/gyraudio/audio_separation/data/mixed.py)
- Loads mixed, signal, noise.
- Audio tensors of sizes `[N, 1, T]`, `T=8000` by default.
- Possibility to filter by SNR.
Provided data:
```python
SNR: 0 - TRAIN SIZE 1001 TEST SIZE 395
SNR: 1 - TRAIN SIZE 506 TEST SIZE 209
SNR: 2 - TRAIN SIZE 485 TEST SIZE 171
SNR: 3 - TRAIN SIZE 509 TEST SIZE 206
SNR: 4 - TRAIN SIZE 520 TEST SIZE 200
SNR: -1 - TRAIN SIZE 506 TEST SIZE 201
SNR: -2 - TRAIN SIZE 507 TEST SIZE 205
SNR: -4 - TRAIN SIZE 516 TEST SIZE 206
SNR: -3 - TRAIN SIZE 450 TEST SIZE 207
```
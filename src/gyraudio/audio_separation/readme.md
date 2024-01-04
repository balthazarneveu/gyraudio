# Pytest
```bash
pytest-3 test
```
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

### :chart_with_downwards_trend: Training
:id: Keep track of experiments by an integer id. 

Each experiment is defined by:
- Dataloader configuration (data, augmentations)
- Model (architecture, sizes)
- Optimizer configuration (hyperparameters)

### [:bar_chart: Experiment tracking](https://wandb.ai/balthazarneveu/audio-sep)

### Evaluation
Run inference/evaluation from a checkpoint
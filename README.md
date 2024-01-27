# Audio source separation
Authors:
- :star: [Mathilde Dupouy](https://github.com/MathildeDupouy)
- :star: [Balthazar Neveu](https://github.com/balthazarneveu)

Context: Project [MVA - deep and signal](https://www.master-mva.com/cours/apprentissage-profond-et-traitement-du-signal-introduction-et-applications-industrielles/), class given by Thomas Courtat.

# [:scroll: REPORT](https://wandb.ai/teammd/audio-separation/reports/Audio-source-separation--Vmlldzo2NjA2OTg2)



#### Demo samples
<!-- |Clean | Mixed| Audio separation WaveU-Net bias free |
|:---: | :---: | :----: |
| . | -2.47db | 11.5db  |
|https://github.com/balthazarneveu/gyraudio/assets/41070742/d0f01bc7-4868-4cad-a961-eb73748ccdfa   |  https://github.com/balthazarneveu/gyraudio/assets/41070742/76803e54-c614-4a3c-9cae-5d967ecf87ad | https://github.com/balthazarneveu/gyraudio/assets/41070742/a6b02e90-979d-4b06-a5b9-d8590c8c7692 |  -->

:green_circle: Clean

https://github.com/balthazarneveu/gyraudio/assets/41070742/d0f01bc7-4868-4cad-a961-eb73748ccdfa


:red_circle: Mixed (noisy) -2.47db

https://github.com/balthazarneveu/gyraudio/assets/41070742/76803e54-c614-4a3c-9cae-5d967ecf87ad


:-1: Audio separation : Thin WaveUNet  - 7 scales `h_ext=16` (1M parameters): (*11.5db*)

*"Garçon" word disappear*

https://github.com/balthazarneveu/gyraudio/assets/41070742/c8940622-d0fc-4e2c-9409-2d0e4b8285c6


:trophy: Audio separation : Bigger WaveUNet bias free, 7 scales, `h_ext=28` 3.1M parameters (*11.5db*)

https://github.com/balthazarneveu/gyraudio/assets/41070742/a6b02e90-979d-4b06-a5b9-d8590c8c7692



-------
### :rocket: Getting started

```bash
git clone https://github.com/balthazarneveu/gyraudio
cd gyraudio
pip install -e .
```
To get everything working correctly, you'll need `pip install -r requirements.txt` and to have `Pytorch` (including torchaudio) + pyQT or pyside to run the GUI properly.

A few audio samples and pretrained models are provided.

-----

# :speaker: Audio separation

:question: [Audio source separation framework and all details](/src/gyraudio/audio_separation/readme.md)




## Evaluation
:gift: Please note that a few pretrained models are provided and a few test samples are provided aswell.

*Run inference/evaluation from a checkpoint*
`python scripts/audio_separation_infer.py -e 3000`

### :gear: Batch processing
Process specific audio files
`python scripts/audio_separation_batch.py -i '__data_source_separation/source_separation/test/00*' -o __output_audio -e 3000`

### :play_or_pause_button: Interactive audio separation
Listen while checking waveform audio separation :ear: + :eye: .
```
python scripts/audio_separation_interactive.py -i __data_source_separation/source_separation/test/000* -o __output/batch_processing -e 1 --preload -p
```

- Browse across samples:
  - :arrow_left: :arrow_right: = next audio   (:two:/:eight:)
  - :arrow_double_down: :arrow_double_up: = next model `page up` / `page down` (*compare models*)
<!-- - :arrow_backward: :arrow_forward:  -->
- Investigate a specific signal
  - :mag: +/- Zoom in/out,
  - :four: / :six: Navigate audio
  - `L` trigger audio loop
 
-------

# :gear: Development


### :bug: Pytest

```bash
pytest-3 test
```
### :arrow_down: Dataset
[Download dataset on kaggle](https://www.kaggle.com/datasets/balthazarneveu/audio-separation-dataset)
unzip data here. `__data_source_separation` 

### Training scripts
- Training the best model locally: `python scripts/audio_separation_train.py -e 3001 -nowb`
- It is possible to fully train on Kaggle, please refer to the [Remote training](/src/gyraudio/audio_separation/readme.md) documentation for all details.


![Remote training](/report/figures/overview.png)

## :chart_with_downwards_trend: Training

:id: Keep track of experiments by an integer id. 

Each experiment is defined by:
- Dataloader configuration (data, augmentations)
- Model (architecture, sizes)
- Optimizer configuration (hyperparameters)


### Defining experiments
[Code to define new experiments](/src/gyraudio/audio_separation/experiment_tracking/experiments_definition.py)





----------

# :test_tube: Gyraudio *[tentative]*

A kind of novel idea for audio source separation that was not explored:
Exploiting IMU (**i**nertial **m**easurement **u**nits = gyroscope and accelerometer) data to help source separation. 
Unfortunately the track was not pursued.


![multimodal_sanity_check](/report/figures/audio_and_gyro_walk.png)


```bash
python scripts/preprocess_data.py -i __data/GH013090.MP4 -o __out
```


### Requirements
- A few requirements so far: `batch_processing`, `matplotlib`, `PyYAML`, `scipy`
*Optional `interactive_pipe`*

- If you need to process new gopro files, you'll need the [GPMF](https://github.com/alexis-mignon/pygpmf) pyton library
```
pip install gpmf
pip uninstall python-ffmpeg
conda install -c conda-forge ffmpeg-python
```


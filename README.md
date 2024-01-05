# Audio source separation
Authors:
- :star: [Mathilde Dupouy](https://github.com/MathildeDupouy)
- :star: [Balthazar Neveu](https://github.com/balthazarneveu)

Context: Project [MVA - deep and signal](https://www.master-mva.com/cours/apprentissage-profond-et-traitement-du-signal-introduction-et-applications-industrielles/), class given by Thomas Courtat.

# Code

## Setup
```bash
git clone https://github.com/balthazarneveu/gyraudio
cd gyraudio
pip install -e .
```


# :speaker: Audio separation
[Audio source separation framework](/src/gyraudio/audio_separation/readme.md)

# :test_tube: Gyraudio *[tentative]*
 a novel idea for audio source separation:
Exploiting IMU (**i**nertial **m**easurement **u**nits = gyroscope and accelerometer) data to help source separation. 


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
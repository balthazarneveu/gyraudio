# Gyraudio
A novel idea for audio source separation:
Exploiting IMU (**i**nertial **m**easurement **u**nits = gyroscope and accelerometer) data to help source separation. 

Authors:
- :star: Mathilde Dupouy
- :star: Balthazar Neveu

![multimodal_sanity_check](/report/figures/audio_and_gyro_walk.png)


```bash
python scripts/preprocess_data.py -i __data/GH013090.MP4 -o __out
```

## Setup
```bash
git clone https://github.com/balthazarneveu/gyraudio
cd gyraudio
pip install -e .
```


### Requirements
- A few requirements so far: `batch_processing`, `matplotlib`, `PyYAML`, `scipy`

- If you need to process new gopro files, you'll need the [GPMF](https://github.com/alexis-mignon/pygpmf) pyton library
```
pip install gpmf
pip uninstall python-ffmpeg
conda install -c conda-forge ffmpeg-python
```


- *Optional `interactive_pipe`*
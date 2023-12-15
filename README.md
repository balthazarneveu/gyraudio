# Gyraudio
A novel method for audio source separation.
Authors:
- :star: Mathilde Dupouy
- :star: Balthazar Neveu


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
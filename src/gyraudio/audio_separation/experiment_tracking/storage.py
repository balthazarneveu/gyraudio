from gyraudio.audio_separation.properties import SHORT_NAME
from pathlib import Path
from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
import logging


def get_output_folder(config: dict, root_dir: Path = EXPERIMENT_STORAGE_ROOT, override: bool = False) -> Path:
    output_folder = root_dir/config["short_name"]
    exists = False
    if output_folder.exists():
        logging.info(f"Experiment {config[SHORT_NAME]} already exists. Override is set to False. Skipping.")
        if not override:
            logging.warning(f"Experiment {config[SHORT_NAME]} will be OVERRIDDEN")
            exists = True
    else:
        output_folder.mkdir(parents=True, exist_ok=True)
        exists = True
    return exists, output_folder

from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
from torch import Tensor


class AudioDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        augmentation_config: dict = {},
        snr_filter: Optional[float] = None
    ):
        self.data_path = data_path
        self.augmentation_config = augmentation_config
        self.snr_filter = snr_filter
        self.load_data()
        self.length = len(self.file_list)

    def load_data(self):
        raise NotImplementedError("load_data method must be implemented")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tensor:
        raise NotImplementedError("__getitem__ method must be implemented")


import torch
from gyraudio.audio_separation.architecture.model import SeparationModel


class NeutralModel(SeparationModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fake = torch.nn.Conv1d(1, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity function
        """
        n = self.fake(x)
        return x, n

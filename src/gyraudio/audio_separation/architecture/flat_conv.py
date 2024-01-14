import torch
from gyraudio.audio_separation.architecture.model import SeparationModel
from typing import Tuple


class FlatConvolutional(SeparationModel):
    """Convolutional neural network for audio separation,
    No decimation, no bottleneck, just basic signal processing
    """

    def __init__(self,
                 ch_in: int = 1,
                 ch_out: int = 2,
                 h_dim=16,
                 k_size=5,
                 dilation=1
                 ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            ch_in, h_dim, k_size, 
            dilation=dilation, padding=dilation*(k_size//2))
        self.conv2 = torch.nn.Conv1d(
            h_dim, h_dim, k_size,
            dilation=dilation, padding=dilation*(k_size//2))
        self.conv3 = torch.nn.Conv1d(
            h_dim, h_dim, k_size,
            dilation=dilation, padding=dilation*(k_size//2))
        self.conv4 = torch.nn.Conv1d(
            h_dim, h_dim, k_size,
            dilation=dilation, padding=dilation*(k_size//2))
        self.pool = torch.nn.MaxPool1d(kernel_size=2)
        self.relu = torch.nn.ReLU()
        self.encoder = torch.nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            self.conv4,
            self.relu
        )
        self.demux = torch.nn.Sequential(*(
            torch.nn.Conv1d(h_dim, h_dim//2, 1),  # conv1x1
            torch.nn.ReLU(),
            torch.nn.Conv1d(h_dim//2, ch_out, 1),  # conv1x1
        ))

    def forward(self, mixed_sig_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform feature extraction followed by classifier head

        Args:
            sig_in (torch.Tensor): [N, C, T]

        Returns:
            torch.Tensor: logits (not probabilities) [N, n_classes]
        """
        # Convolution backbone
        # [N, C, T] -> [N, h, T]
        features = self.encoder(mixed_sig_in)
        # [N, h, T] -> [N, 2, T]
        demuxed = self.demux(features)
        return torch.chunk(demuxed, 2, dim=1)  # [N, 1, T], [N, 1, T]

import torch
from typing import List


class FilterBank(torch.nn.Module):
    """Convolution filter bank (linear)
    Serves as an embedding for the audio signal
    """

    def __init__(self, ch_in: int, out_dim=16, k_size=5, dilation_list: List[int] = [1, 2, 4, 8]):
        super().__init__()
        self.out_dim = out_dim
        self.source_modality_conv = torch.nn.ModuleList()
        for dilation in dilation_list:
            self.source_modality_conv.append(
                torch.nn.Conv1d(ch_in, out_dim//len(dilation_list), k_size, dilation=dilation, padding=(dilation*(k_size//2)))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([conv(x) for conv in self.source_modality_conv], axis=1)
        assert out.shape[1] == self.out_dim
        return out


class ResConvolution(torch.nn.Module):
    """ResNet building block
    https://paperswithcode.com/method/residual-connection
    """

    def __init__(self, ch, hdim=None, k_size=5):
        super().__init__()
        hdim = hdim or ch
        self.conv1 = torch.nn.Conv1d(ch, hdim, k_size, padding=k_size//2)
        self.conv2 = torch.nn.Conv1d(hdim, ch, k_size, padding=k_size//2)
        self.non_linearity = torch.nn.ReLU()

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.non_linearity(x)
        x = self.conv2(x)
        x += x_in
        x = self.non_linearity(x)
        return x


if __name__ == "__main__":
    model = FilterBank(1, 16)
    inp = torch.rand(2, 1, 2048)
    out = model(inp)
    print(model)
    print(out[0].shape)

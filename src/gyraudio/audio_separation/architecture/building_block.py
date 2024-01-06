import torch


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

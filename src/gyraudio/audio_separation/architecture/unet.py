import torch
from gyraudio.audio_separation.architecture.model import SeparationModel
from gyraudio.audio_separation.architecture.building_block import ResConvolution
from typing import Optional
# import logging


class EncoderSingleStage(torch.nn.Module):
    """
    Extend channels
    Resnet
    Downsample by 2
    """

    def __init__(self, ch: int, ch_out: int, hdim: Optional[int] = None, k_size=5):
        # ch_out ~ ch_in*extension_factor
        super().__init__()
        hdim = hdim or ch
        self.extension_conv = torch.nn.Conv1d(ch, ch_out, k_size, padding=k_size//2)
        self.res_conv = ResConvolution(ch_out, hdim=hdim, k_size=k_size)
        self.max_pool = torch.nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.extension_conv(x)
        x = self.res_conv(x)
        x_ds = self.max_pool(x)
        return x, x_ds


class DecoderSingleStage(torch.nn.Module):
    """
    Upsample by 2
    Resnet
    Extend channels
    """

    def __init__(self, ch: int, ch_out: int, hdim: Optional[int] = None, k_size=5):
        """Decoder stage

        Args:
            ch (int): channel size (downsampled & skip connection have same channel size)
            ch_out (int): number of output channels (shall match the number of input channels of the next stage)
            hdim (Optional[int], optional): Hidden dimension used in the residual block. Defaults to None.
            k_size (int, optional): Convolution size. Defaults to 5.
        Notes:
        ======
        ch_out = 2*ch/extension_factor

        self.scale_mixers_conv 
        - tells how lower decoded (x_ds) scale is merged with current encoded scale (x_skip)
        - could be a pointwise aka conv1x1
        """

        super().__init__()
        hdim = hdim or ch
        self.scale_mixers_conv = torch.nn.Conv1d(2*ch, ch_out, k_size, padding=k_size//2)

        self.res_conv = ResConvolution(ch_out, hdim=hdim, k_size=k_size)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.non_linearity = torch.nn.ReLU()

    def forward(self, x_ds: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """"""
        x_us = self.upsample(x_ds)  # [N, ch, T/2] -> [N, ch, T]

        x = torch.cat([x_us, x_skip], dim=1)  # [N, 2.ch, T]
        x = self.scale_mixers_conv(x)  # [N, ch_out, T]
        x = self.non_linearity(x)
        x = self.res_conv(x)  # [N, ch_out, T]
        return x


class ResUNet(SeparationModel):
    """Convolutional neural network for audio separation,

    Decimation, bottleneck
    """

    def __init__(self,
                 ch_in: int = 1,
                 ch_out: int = 2,
                 channels_extension: float = 1.5,
                 h_dim=16,
                 k_size=5,
                 ) -> None:
        super().__init__()
        self.need_split = ch_out != ch_in
        self.ch_out = ch_out
        self.source_modality_conv = torch.nn.Conv1d(ch_in, h_dim, k_size, padding=k_size//2)
        self.encoder_list = torch.nn.ModuleList()
        self.decoder_list = torch.nn.ModuleList()
        self.non_linearity = torch.nn.ReLU()

        h_dim_current = h_dim
        for _level in range(4):
            h_dim_ds = int(h_dim_current*channels_extension)
            self.encoder_list.append(EncoderSingleStage(h_dim_current, h_dim_ds, k_size=k_size))
            self.decoder_list.append(DecoderSingleStage(h_dim_ds, h_dim_current, k_size=k_size))
            h_dim_current = h_dim_ds
        self.bottleneck = ResConvolution(h_dim_current, k_size=k_size)
        self.target_modality_conv = torch.nn.Conv1d(h_dim, ch_out, 1)  # conv1x1 channel mixer

    def forward(self, x_in):
        # x_in (1, 2048)
        x0 = self.source_modality_conv(x_in)
        x0 = self.non_linearity(x0)
        # x0 -> (16, 2048)

        x1_skip, x1_ds = self.encoder_list[0](x0)
        # x1_skip -> (24, 2048)
        # x1_ds   -> (24, 1024)
        # print(x1_skip.shape, x1_ds.shape)

        x2_skip, x2_ds = self.encoder_list[1](x1_ds)
        # x2_skip -> (36, 1024)
        # x2_ds   -> (36, 512)
        # print(x2_skip.shape, x2_ds.shape)

        x3_skip, x3_ds = self.encoder_list[2](x2_ds)
        # x3_skip -> (54, 512)
        # x3_ds   -> (54, 256)
        # print(x3_skip.shape, x3_ds.shape)

        x4_skip, x4_ds = self.encoder_list[3](x3_ds)
        # x4_skip -> (81, 256)
        # x4_ds   -> (81, 128)
        # print(x4_skip.shape, x4_ds.shape)

        x4_dec = self.bottleneck(x4_ds)
        x3_dec = self.decoder_list[3](x4_dec, x4_skip)
        x2_dec = self.decoder_list[2](x3_dec, x3_skip)
        x1_dec = self.decoder_list[1](x2_dec, x2_skip)
        x0_dec = self.decoder_list[0](x1_dec, x1_skip)
        demuxed = self.target_modality_conv(x0_dec)
        # no relu
        if self.need_split:
            return torch.chunk(demuxed, self.ch_out, dim=1)
        return demuxed, None


if __name__ == "__main__":
    model = ResUNet()
    inp = torch.rand(2, 1, 2048)
    out = model(inp)
    print(model)
    print(model.count_parameters())
    print(out[0].shape)

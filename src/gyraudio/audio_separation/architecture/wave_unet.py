import torch
from gyraudio.audio_separation.architecture.model import SeparationModel
from gyraudio.audio_separation.architecture.building_block import ResConvolution
from typing import Optional


def get_non_linearity(activation: str):
    if activation == "LeakyReLU":
        non_linearity = torch.nn.LeakyReLU()
    else:
        non_linearity = torch.nn.ReLU()
    return non_linearity


class BaseConvolutionBlock(torch.nn.Module):
    def __init__(self, ch_in, ch_out: int, k_size: int, activation="LeakyReLU") -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(ch_in, ch_out, k_size, padding=k_size//2)
        self.non_linearity = get_non_linearity(activation)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.conv(x_in)  # [N, ch_in, T] -> [N, ch_in+channels_extension, T]
        x = self.non_linearity(x)
        return x


class EncoderStage(torch.nn.Module):
    """Conv (and extend channels), downsample 2 by skipping samples
    """

    def __init__(self, ch_in: int, ch_out: int, k_size: int = 15) -> None:

        super().__init__()

        self.conv = BaseConvolutionBlock(ch_in, ch_out, k_size=k_size)

    def forward(self, x):
        x = self.conv(x)
        x_ds = x[..., ::2]
        # ch_out = ch_in+channels_extension
        return x, x_ds


class DecoderStage(torch.nn.Module):
    """Upsample by 2, Concatenate with skip connection, Conv (and shrink channels)
    """

    def __init__(self, ch_in: int, ch_out: int, k_size: int = 5) -> None:
        """Decoder stage
        """

        super().__init__()
        self.conv = BaseConvolutionBlock(ch_in, ch_out, k_size=k_size)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.non_linearity = torch.nn.ReLU()

    def forward(self, x_ds: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """"""
        x_us = self.upsample(x_ds)  # [N, ch, T/2] -> [N, ch, T]
        x = torch.cat([x_us, x_skip], dim=1)  # [N, 2.ch, T]
        x = self.conv(x)  # [N, ch_out, T]
        x = self.non_linearity(x)
        return x


class WaveUNet(SeparationModel):
    """UNET in temporal domain (waveform) 
    = Multiscale convolutional neural network for audio separation
    https://arxiv.org/abs/1806.03185
    """

    def __init__(self,
                 ch_in: int = 1,
                 ch_out: int = 2,
                 channels_extension: int = 24,
                 k_conv_ds: int = 15,
                 k_conv_us: int = 5,
                 num_layers: int = 4,
                 ) -> None:
        assert num_layers == 4, "Only 4 layers supported for now"
        super().__init__()
        self.need_split = ch_out != ch_in
        self.ch_out = ch_out
        self.encoder_list = torch.nn.ModuleList()
        self.decoder_list = torch.nn.ModuleList()
        self.non_linearity = torch.nn.ReLU()
        self.encoder_list.append(EncoderStage(ch_in, channels_extension, k_size=k_conv_ds))
        for level in range(1, num_layers+1):
            ch_i = level*channels_extension
            ch_o = (level+1)*channels_extension
            self.encoder_list.append(EncoderStage(ch_i, ch_o, k_size=k_conv_ds))
            self.decoder_list.append(DecoderStage(ch_o+ch_i, ch_i, k_size=k_conv_us))
        self.bottleneck = BaseConvolutionBlock(
            num_layers*channels_extension,
            (num_layers+1)*channels_extension,
            k_size=k_conv_ds)
        self.target_modality_conv = torch.nn.Conv1d(channels_extension, ch_out, 1)  # conv1x1 channel mixer

    def forward(self, x_in):
        # x_in (1, 2048)
        x1_skip, x1_ds = self.encoder_list[0](x_in)
        # x1_skip -> (24, 2048)
        # x1_ds   -> (24, 1024)
        print(x1_skip.shape, x1_ds.shape)

        x2_skip, x2_ds = self.encoder_list[1](x1_ds)
        # x2_skip -> (48, 1024)
        # x2_ds   -> (48, 512)
        print(x2_skip.shape, x2_ds.shape)

        x3_skip, x3_ds = self.encoder_list[2](x2_ds)
        # x3_skip -> (72, 512)
        # x3_ds   -> (72, 256)
        print(x3_skip.shape, x3_ds.shape)

        x4_skip, x4_ds = self.encoder_list[3](x3_ds)
        # x4_skip -> (96, 256)
        # x4_ds   -> (96, 128)
        print(x4_skip.shape, x4_ds.shape)

        x4_dec = self.bottleneck(x4_ds)
        print(x4_dec.shape)
        x3_dec = self.decoder_list[3](x4_dec, x4_skip)
        print(x3_dec.shape, x3_skip.shape)
        x2_dec = self.decoder_list[2](x3_dec, x3_skip)
        x1_dec = self.decoder_list[1](x2_dec, x2_skip)
        x0_dec = self.decoder_list[0](x1_dec, x1_skip)
        # no relu
        demuxed = self.target_modality_conv(x0_dec)
        if self.need_split:
            return torch.chunk(demuxed, self.ch_out, dim=1)
        return demuxed, None


if __name__ == "__main__":
    model = WaveUNet(ch_out=1)
    inp = torch.rand(2, 1, 2048)
    out = model(inp)
    print(model)
    print(model.count_parameters())
    print(out[0].shape)

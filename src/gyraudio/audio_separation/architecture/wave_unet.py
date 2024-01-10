import torch
from gyraudio.audio_separation.architecture.model import SeparationModel
from typing import Optional, Tuple


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

    def forward(self, x_ds: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """"""
        x_us = self.upsample(x_ds)  # [N, ch, T/2] -> [N, ch, T]
        x = torch.cat([x_us, x_skip], dim=1)  # [N, 2.ch, T]
        x = self.conv_block(x)  # [N, ch_out, T]
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
                 num_layers: int = 6,
                 ) -> None:
        super().__init__()
        self.need_split = ch_out != ch_in
        self.ch_out = ch_out
        self.encoder_list = torch.nn.ModuleList()
        self.decoder_list = torch.nn.ModuleList()
        # Defining first encoder
        self.encoder_list.append(EncoderStage(ch_in, channels_extension, k_size=k_conv_ds))
        for level in range(1, num_layers+1):
            ch_i = level*channels_extension
            ch_o = (level+1)*channels_extension
            if level < num_layers:
                # Skipping last encoder since we defined the first one outside the loop
                self.encoder_list.append(EncoderStage(ch_i, ch_o, k_size=k_conv_ds))
            self.decoder_list.append(DecoderStage(ch_o+ch_i, ch_i, k_size=k_conv_us))
        self.bottleneck = BaseConvolutionBlock(
            num_layers*channels_extension,
            (num_layers+1)*channels_extension,
            k_size=k_conv_ds)
        self.target_modality_conv = torch.nn.Conv1d(channels_extension+ch_in, ch_out, 1)  # conv1x1 channel mixer

    def forward(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward UNET pass

        ```
        (1  , 2048)----------------->(24 , 2048) > (1  , 2048)
            v                            ^
        (24 , 1024)----------------->(48 , 1024)
            v                            ^
        (48 , 512 )----------------->(72 , 512 )
            v                            ^
        (72 , 256 )----------------->(96 , 256 )
            v                            ^
        (96 , 128 )----BOTTLENECK--->(120, 128 )
        ```

        """
        skipped_list = []
        ds_list = [x_in]
        for level, enc in enumerate(self.encoder_list):
            x_skip, x_ds = enc(ds_list[-1])
            skipped_list.append(x_skip)
            ds_list.append(x_ds.clone())
            # print(x_skip.shape, x_ds.shape)
        x_dec = self.bottleneck(ds_list[-1])
        for level, dec in enumerate(self.decoder_list[::-1]):
            x_dec = dec(x_dec, skipped_list[-1-level])
        # print(x_dec.shape)
        x_dec = torch.cat([x_dec, x_in], dim=1)
        # print(x_dec.shape)
        demuxed = self.target_modality_conv(x_dec)
        # print(demuxed.shape)
        if self.need_split:
            return torch.chunk(demuxed, self.ch_out, dim=1)
        return demuxed, None

        # x_skip, x_ds
        # (24, 2048), (24, 1024)
        # (48, 1024), (48, 512 )
        # (72, 512 ), (72, 256 )
        # (96, 256 ), (96, 128 )

        # (120, 128 )
        # (96 , 256 )
        # (72 , 512 )
        # (48 , 1024)
        # (24 , 2048)
        # (25 , 2048) demuxed - after concat
        # (1  , 2048)


if __name__ == "__main__":
    model = WaveUNet(ch_out=1, num_layers=9)
    inp = torch.rand(2, 1, 2048)
    out = model(inp)
    print(model)
    print(model.count_parameters())
    print(out[0].shape)

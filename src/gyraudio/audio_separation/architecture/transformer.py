import torch
from gyraudio.audio_separation.architecture.model import SeparationModel
from gyraudio.audio_separation.architecture.building_block import FilterBank
from typing import Optional


class TransformerModel(SeparationModel):
    """Transformer base model
    =========================
    - Embed signal with a filter bank
    - No positional encoding (Potential  =add/concatenate positional encoding)
    - `nlayers` * transformer blocks
    """

    def __init__(self,
                 nhead: int = 8,  # H
                 nlayers: int = 4,  # L
                 k_size=5,
                 embedding_dim: int = 64,  # D
                 ch_in: int = 1,
                 ch_out: int = 1,
                 dropout: float = 0.,  # dr
                 positional_encoding: str = None
                 ) -> None:
        """Transformer base model

        Args:
            nhead (int): number of heads in each of the MHA models
            embedding_dim (int): D number of channels in the audio embeddings 
                = output of the filter bank
                assume `embedding_dim` = `h_dim`
                h_dim is the hidden dimension of the model.
            nlayers (int): number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dropout (float, optional): dropout value. Defaults to 0.
        """
        super().__init__()
        self.model_type = "Transformer"
        h_dim = embedding_dim  # use the same embedding & hidden dimensions

        self.encoder = FilterBank(ch_in, embedding_dim, k_size=k_size)
        if positional_encoding is None:
            self.pos_encoder = torch.nn.Identity()
        else:
            raise NotImplementedError(
                f"Unknown positional encoding {positional_encoding} - should be add/concat in future")
        # self.pos_encoder = PositionalEncoding(h_dim, dropout=dropout)

        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model=h_dim,  # input dimension to the transformer encoder layer
            nhead=nhead,  # number of heads for MHA (Multi-head attention)
            dim_feedforward=h_dim,  # output dimension of the MLP on top of the transformer.
            dropout=dropout,
            batch_first=True
        )  # we assume h_dim = d_model = dim_feedforward

        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layers,
            num_layers=nlayers
        )
        self.h_dim = h_dim
        self.target_modality_conv = torch.nn.Conv1d(h_dim, ch_out, 1)  # conv1x1 channel mixer
        # Note: we could finish with a few residual conv blocks... this is pure signal processing

    def forward(
        self, src: torch.LongTensor,
        src_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """Embdeddings, positional encoders, go trough `nlayers` of residual {multi (`nhead`) attention heads + MLP}.

        Args:
            src (torch.LongTensor): [N, 1, T] audio signal

        Returns:
            torch.FloatTensor: separated signal [N, 1, T]
        """
        src = self.encoder(src)  # [N, 1, T] -> [N, D, T]
        src = src.transpose(-1, -2)  # [N, D, T] -> [N, T, D] # Transformer  expects (batch N, seq "T", features "D")
        src = self.pos_encoder(src)  # -> [N, T, D]  - add positional encoding

        output = self.transformer_encoder(src, mask=src_mask)  # -> [N, T, D]
        output = output.transpose(-1, -2)  # -> [N, D, T]
        output = self.target_modality_conv(output)  # -> [N, 1, T]
        return output, None


if __name__ == "__main__":
    model = TransformerModel()
    inp = torch.rand(2, 1, 2048)
    out = model(inp)
    print(model)
    print(out[0].shape)

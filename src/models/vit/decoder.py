import torch.nn as nn
from einops import rearrange
from src.models.vit.utils import init_weights


class DecoderLinear(nn.Module):
    def __init__(
        self,
        n_cls,
        d_encoder,
        scale_factor,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.head = nn.Linear(d_encoder, n_cls)
        self.upsampling = nn.Upsample(scale_factor=scale_factor**2, mode="linear")
        self.norm = nn.LayerNorm((n_cls, 24 * scale_factor, 24 * scale_factor))
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()
        self.apply(init_weights)

    def forward(self, x, img_size):
        H, _ = img_size
        x = self.head(x)  ####### (2, 577, 64)
        x = x.transpose(2, 1)  ## (2, 64, 576)
        x = self.upsampling(x)  # (2, 64, 576*scale_factor*scale_factor)
        x = x.transpose(2, 1)  ## (2, 576*scale_factor*scale_factor, 64)
        x = rearrange(x, "b (h w) c -> b c h w", h=H // (16 // self.scale_factor))  # (2, 64, 24*scale_factor, 24*scale_factor)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.gelu(x)

        return x  # (2, 64, a, a)

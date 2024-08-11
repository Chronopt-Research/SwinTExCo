import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import _load_weights
from timm.models.layers import trunc_normal_
from typing import List

from src.models.vit.utils import init_weights, resize_pos_embed
from src.models.vit.blocks import Block
from src.models.vit.decoder import DecoderLinear


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.distilled = distilled
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 2, d_model))
            self.head_dist = nn.Linear(d_model, n_cls)
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, d_model))

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)])

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, im, head_out_idx: List[int], n_dim_output=3, return_features=False):
        B, _, H, W = im.shape
        PS = self.patch_size
        assert n_dim_output == 3 or n_dim_output == 4, "n_dim_output must be 3 or 4"
        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed
        x = self.dropout(x)
        device = x.device

        if n_dim_output == 3:
            heads_out = torch.zeros(size=(len(head_out_idx), B, (H // PS) ** 2 + 1, self.d_model)).to(device)
        else:
            heads_out = torch.zeros(size=(len(head_out_idx), B, self.d_model, H // PS, H // PS)).to(device)
        self.register_buffer("heads_out", heads_out)

        head_idx = 0
        for idx_layer, blk in enumerate(self.blocks):
            x = blk(x)
            if idx_layer in head_out_idx:
                if n_dim_output == 3:
                    heads_out[head_idx] = x
                else:
                    heads_out[head_idx] = x[:, 1:, :].reshape((-1, 24, 24, self.d_model)).permute(0, 3, 1, 2)
                head_idx += 1

        x = self.norm(x)

        if return_features:
            return heads_out

        if self.distilled:
            x, x_dist = x[:, 0], x[:, 1]
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            x = (x + x_dist) / 2
        else:
            x = x[:, 0]
            x = self.head(x)
        return x

    def get_attention_map(self, im, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}.")
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


class FeatureTransform(nn.Module):
    def __init__(self, img_size, d_encoder, nls_list=[128, 256, 512, 512], scale_factor_list=[8, 4, 2, 1]):
        super(FeatureTransform, self).__init__()
        self.img_size = img_size

        self.decoder_0 = DecoderLinear(n_cls=nls_list[0], d_encoder=d_encoder, scale_factor=scale_factor_list[0])
        self.decoder_1 = DecoderLinear(n_cls=nls_list[1], d_encoder=d_encoder, scale_factor=scale_factor_list[1])
        self.decoder_2 = DecoderLinear(n_cls=nls_list[2], d_encoder=d_encoder, scale_factor=scale_factor_list[2])
        self.decoder_3 = DecoderLinear(n_cls=nls_list[3], d_encoder=d_encoder, scale_factor=scale_factor_list[3])

    def forward(self, x_list):
        feat_3 = self.decoder_3(x_list[3][:, 1:, :], self.img_size)  # (2, 512, 24, 24)
        feat_2 = self.decoder_2(x_list[2][:, 1:, :], self.img_size)  # (2, 512, 48, 48)
        feat_1 = self.decoder_1(x_list[1][:, 1:, :], self.img_size)  # (2, 256, 96, 96)
        feat_0 = self.decoder_0(x_list[0][:, 1:, :], self.img_size)  # (2, 128, 192, 192)
        return feat_0, feat_1, feat_2, feat_3
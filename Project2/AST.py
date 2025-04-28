import torch
import torch.nn as nn
from timm.layers.helpers import to_2tuple
from timm.layers.weight_init import trunc_normal_

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(201, 41), patch_size=16, stride=None, in_chans=1, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = stride if stride is not None else patch_size
        stride = to_2tuple(stride)

        self.num_patches = ((img_size[0] - patch_size[0]) // stride[0] + 1) * \
                           ((img_size[1] - patch_size[1]) // stride[1] + 1)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    def __init__(
        self,
        label_dim=23,
        fstride=16,
        tstride=16,
        input_fdim=201,
        input_tdim=41,
        model_size='base384'
    ):
        super(ASTModel, self).__init__()

        if model_size == 'base384':
            self.embed_dim = 768
            depth = 6
            num_heads = 12
        elif model_size == 'small224':
            self.embed_dim = 384
            depth = 12
            num_heads = 6
        elif model_size == 'tiny224':
            self.embed_dim = 192
            depth = 12
            num_heads = 3
        else:
            raise ValueError("Unsupported model size.")

        patch_size = (16, 16)
        stride = (fstride, tstride)

        self.patch_embed = PatchEmbed(
            img_size=(input_fdim, input_tdim),
            patch_size=patch_size,
            stride=stride,
            in_chans=1,
            embed_dim=self.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, label_dim)

    def forward(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.patch_embed(x)
        B, N, D = x.size()

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

"""CNN stem + Transformer visual encoder (Hybrid CNN-Transformer, Week 1 first pass)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.init import trunc_normal_


@dataclass
class EncoderOutput:
    pooled: torch.Tensor        # [B, d_model] -- CLS token, post-transformer
    patch_tokens: torch.Tensor  # [B, N, d_model] -- patch tokens, post-transformer


class CNNStem(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], dropout: float) -> None:
        super().__init__()
        blocks = []
        prev = in_channels
        for out_channels in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(dropout),
                )
            )
            prev = out_channels
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class HybridEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        cnn_channels: list[int],
        d_model: int,
        num_transformer_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cnn_stem = CNNStem(in_channels=3, channels=cnn_channels, dropout=dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat_map = self.cnn_stem(dummy)
        _, feat_channels, h, w = feat_map.shape
        num_patches = h * w

        self.patch_proj = nn.Linear(feat_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

    def forward(self, image: torch.Tensor) -> EncoderOutput:
        feat_map = self.cnn_stem(image)                            # [B,C,H',W']
        batch_size = feat_map.shape[0]
        patches = feat_map.flatten(2).transpose(1, 2)               # [B,N,C]
        patches = self.patch_proj(patches)                          # [B,N,d_model]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)            # [B,N+1,d_model]
        tokens = tokens + self.pos_embed

        encoded = self.transformer(tokens)
        return EncoderOutput(pooled=encoded[:, 0], patch_tokens=encoded[:, 1:])

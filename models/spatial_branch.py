"""
FrequSpatial V2 — Spatial Branch
SwinIR-style deep spatial feature extraction using RSTB blocks.
"""

import torch
import torch.nn as nn
import numpy as np

from .common import conv3x3, LayerNorm2d
from .swin import RSTB


class SpatialBranch(nn.Module):
    """
    Deep spatial feature extractor using Residual Swin Transformer Blocks.

    Architecture:
        Conv3x3
        N × RSTB  (each: M SwinBlocks + Conv + residual)
        LayerNorm2d
        Conv3x3
        + global residual
    """

    def __init__(
        self,
        dim: int,
        num_rstb: int,
        num_swin_per_rstb: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        # Stochastic depth: linearly increase drop_path across all blocks
        total_blocks = num_rstb * num_swin_per_rstb
        dpr = np.linspace(0, drop_path_rate, total_blocks).tolist()

        self.rstb_blocks = nn.ModuleList([
            RSTB(
                dim=dim,
                num_blocks=num_swin_per_rstb,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rates=dpr[i * num_swin_per_rstb:(i + 1) * num_swin_per_rstb],
            )
            for i in range(num_rstb)
        ])

        self.norm = LayerNorm2d(dim)
        self.conv_tail = conv3x3(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]"""
        residual = x
        for blk in self.rstb_blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.conv_tail(x)
        return x + residual

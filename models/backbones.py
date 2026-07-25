"""Alternative local stage for the generality ablation.

To show SR-MoSO is a *reusable* global-context module rather than something tied to Swin attention,
the local stage can be swapped for a plain residual conv group. If SR-MoSO improves this
non-transformer backbone too (`edsr_moso` > `edsr_base`), that is the drop-in-operator evidence.
Mirrors `RSTB`'s structure (body + tail conv + residual) so the two are directly comparable.
"""
import torch
import torch.nn as nn
from .common import ResBlock, conv3x3


class ResidualGroup(nn.Module):
    """EDSR-style residual group: N ResBlocks + conv3x3 + local residual. [B,C,H,W] → [B,C,H,W]."""

    def __init__(self, dim: int, num_blocks: int = 8, res_scale: float = 0.2):
        super().__init__()
        self.body = nn.Sequential(*[ResBlock(dim, res_scale=res_scale) for _ in range(num_blocks)])
        self.conv = conv3x3(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.body(x)) + x

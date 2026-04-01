"""Gated Cross-Domain Fusion - Learned sigmoid gating: the network decides how much of each domain to keep at each spatial location and channel."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import conv1x1, conv3x3, ChannelAttention, LayerNorm2d


class GatedFusion(nn.Module):
    """Gated fusion of spatial and frequency features.
    out = gate_s ⊙ spatial + gate_f ⊙ frequency
    gates = sigmoid(conv(cat([spatial, frequency])))
    This is strictly better than simple concatenation because:
      - The gate is input-dependent (content-aware)
      - Each gate is between [0,1] so the output is a convex combination
      - Gradients flow cleanly through both branches"""
    def __init__(self, channels: int, use_channel_refinement: bool = True):
        super().__init__()
        self.use_channel_refinement = use_channel_refinement

        # Gate computation: input is concat of both → 2 sigmoid gates
        self.gate_net = nn.Sequential(
            conv3x3(channels * 2, channels * 2),
            nn.GELU(),
            conv1x1(channels * 2, channels * 2),  # 2 gates: gate_s, gate_f
            nn.Sigmoid(),)

        # Post-fusion refinement
        self.refine = nn.Sequential(
            conv3x3(channels, channels),
            nn.GELU(),
            conv3x3(channels, channels),)

        # Optional channel attention to further select relevant channels
        if use_channel_refinement:
            self.ca = ChannelAttention(channels, reduction=16)

        self.norm = LayerNorm2d(channels)

    def forward(self,
        spatial: torch.Tensor,    # [B, C, H, W]
        frequency: torch.Tensor,  # [B, C, H, W]
) -> torch.Tensor:
        combined = torch.cat([spatial, frequency], dim=1)  # [B, 2C, H, W]
        gates = self.gate_net(combined)                    # [B, 2C, H, W]

        gate_s, gate_f = gates.chunk(2, dim=1)             # each [B, C, H, W]
        fused = gate_s * spatial + gate_f * frequency      # [B, C, H, W]

        fused = self.refine(fused)
        if self.use_channel_refinement:
            fused = self.ca(fused)

        return self.norm(fused)

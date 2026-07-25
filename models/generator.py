"""SR-MoSO generator.

    LR [B,3,H,W]
      ↓ conv3x3                                        shallow features [B,C,H,W]
      ↓ Trunk: [ local stage (Swin RSTB) → global stage (SR-MoSO) ] x N  + shallow residual
      ↓ PixelShuffle(scale)
      ↓ conv3x3                                        [B,3,sH,sW]
      + bicubic(LR)                                    global residual

The local/global interleaving lives in `trunk.Trunk`; the spectral operator in `spectral.py`.
Ablation flags (config.ModelConfig): use_freq_branch, spectral_stage, freq_use_ccso, ccso_mode,
spectral_grid, spatial_backbone — each toggles cleanly without shape errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import conv3x3, UpscaleBlock
from .trunk import Trunk
from config import ModelConfig


class FrequSpatialGenerator(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        C = cfg.base_channels
        self.scale = cfg.scale

        self.conv_first = conv3x3(cfg.in_channels, C)
        self.trunk = Trunk(C, cfg)
        self.upsample = UpscaleBlock(C, cfg.scale)
        self.conv_last = conv3x3(C, cfg.out_channels)

        self._init_weights()

    def _init_weights(self):
        """Only the 'glue' layers. Branch modules self-initialize (trunc_normal_ for Swin,
        near-identity for the spectral diagonals, zero-init for the residual projections).

        `conv_last` is **zero-initialized** so the model starts exactly at `bicubic(LR)` (~31 dB) and
        improves monotonically. A He-initialized final conv would start ~1.4 std away from bicubic,
        wasting the first thousands of steps — under gradient clipping — just shrinking it back."""
        for m in (self.conv_first, self.upsample):
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    # fan_in He (correct for the channel-expanding PixelShuffle convs), GELU gain
                    nn.init.kaiming_normal_(mod.weight, mode="fan_in", nonlinearity="relu")
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
        nn.init.zeros_(self.conv_last.weight)
        nn.init.zeros_(self.conv_last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: LR image [B, 3, H, W] in [0, 1] → SR [B, 3, scale*H, scale*W]."""
        bicubic = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.conv_first(x)
        deep = self.trunk(feat)
        return self.conv_last(self.upsample(deep)) + bicubic

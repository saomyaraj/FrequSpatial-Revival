"""Deep feature trunk: local stages interleaved with global SR-MoSO stages.

    conv_first(x) ─► [ LocalStage → SpectralStage ] x N ─► LayerNorm2d ─► conv3x3 ─► + shallow

Each stage pairs a **local** operator (Swin RSTB, or a residual conv group for the generality
ablation) with a **global** SR-MoSO operator, so the spectral response is composed with attention and
nonlinearity at multiple depths. This is the placement used by SwinFIR / FFTformer / GFNet; a single
shallow spectral branch running parallel to a deep trunk cannot contribute comparably.

`spectral_stage` selects what fills the global slot:
    "moso" → SpectralBlock (proposed)   "conv" → ConvBlock (param-matched local control)   None → no global stage
"""

import numpy as np
import torch
import torch.nn as nn

from .common import conv3x3, LayerNorm2d
from .swin import RSTB
from .backbones import ResidualGroup
from .spectral import SpectralBlock, ConvBlock


class Trunk(nn.Module):
    def __init__(self, dim: int, cfg):
        super().__init__()
        n = cfg.num_rstb
        self.local_stages = nn.ModuleList()
        self.spectral_stages = nn.ModuleList()

        # stochastic depth spread over all Swin blocks
        dpr = np.linspace(0, cfg.swin_drop_path_rate, n * cfg.num_swin_per_rstb).tolist()

        for i in range(n):
            if cfg.spatial_backbone == "edsr":
                self.local_stages.append(ResidualGroup(dim, num_blocks=cfg.edsr_num_blocks))
            else:
                self.local_stages.append(RSTB(
                    dim=dim,
                    num_blocks=cfg.num_swin_per_rstb,
                    num_heads=cfg.swin_num_heads,
                    window_size=cfg.swin_window_size,
                    mlp_ratio=cfg.swin_mlp_ratio,
                    qkv_bias=cfg.swin_qkv_bias,
                    drop=cfg.swin_drop_rate,
                    attn_drop=cfg.swin_attn_drop_rate,
                    drop_path_rates=dpr[i * cfg.num_swin_per_rstb:(i + 1) * cfg.num_swin_per_rstb],
                ))

            if not cfg.use_freq_branch:
                continue
            if cfg.spectral_stage == "conv":
                self.spectral_stages.append(ConvBlock(dim, target_params=_moso_params(dim, cfg)))
            else:
                self.spectral_stages.append(SpectralBlock(
                    dim, cfg.fno_modes_h, cfg.fno_modes_w,
                    num_experts=cfg.ccso_experts,
                    mode=cfg.ccso_mode if cfg.freq_use_ccso else "static",
                    per_expert_mix=cfg.ccso_per_expert_mix,
                    spectral_grid=cfg.spectral_grid,
                    ref_size=cfg.spectral_ref_size,
                ))

        self.norm = LayerNorm2d(dim)
        self.conv_after_body = conv3x3(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: shallow features [B, C, H, W] → deep features [B, C, H, W] (+ shallow residual)"""
        shallow = x
        for i, local in enumerate(self.local_stages):
            x = local(x)
            if i < len(self.spectral_stages):
                x = self.spectral_stages[i](x)
        return self.conv_after_body(self.norm(x)) + shallow


def _moso_params(dim: int, cfg) -> int:
    """parameter count of one SpectralBlock's operator — the budget the `conv` control must match."""
    K = cfg.ccso_experts if cfg.ccso_mode in ("moso", "mixture") and cfg.freq_use_ccso else 1
    diag = 2 * K * dim * cfg.fno_modes_h * cfg.fno_modes_w
    mix = (2 * K if cfg.ccso_per_expert_mix else 2) * dim * dim
    route = (9 * dim * dim + dim) + (dim * K + K) if cfg.ccso_mode == "moso" and cfg.freq_use_ccso else 0
    return diag + mix + route

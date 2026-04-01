"""Generator (Full SR Model)
Architecture overview:
    LR [B, 3, H, W]
      ↓ Conv3×3  → shallow features [B, C, H, W]
      │
      ├──→ Spatial Branch (N × RSTB with SwinTransformerBlocks)
      │
      ├──→ Frequency Branch (FNO-mixing + band decomp + mag/phase)
      │
      ↓ Cross-Domain Attention Module (CDAM) ← KEY NOVEL CONTRIBUTION
      │    bidirectional: spatial ↔ frequency queries
      │
      ↓ Gated Fusion
      │    gate_s * spatial_enhanced + gate_f * freq_enhanced
      │
      ↓ Deep feature refinement (ResBlocks)
      ↓ PixelShuffle(scale)
      ↓ Conv3×3 → [B, 3, sH, sW]
      + bicubic(LR) global residual"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import conv3x3, ResBlock, UpscaleBlock, LayerNorm2d
from .spatial_branch import SpatialBranch
from .frequency_branch import FrequencyBranch
from .cross_domain import CrossDomainAttentionModule
from .fusion import GatedFusion
from config import ModelConfig


class FrequSpatialGenerator(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        C = cfg.base_channels
        self.scale = cfg.scale

        # ── Shallow feature extraction ──
        self.conv_first = conv3x3(cfg.in_channels, C)

        # ── Spatial branch ──
        self.spatial_branch = SpatialBranch(
            dim=C,
            num_rstb=cfg.num_rstb,
            num_swin_per_rstb=cfg.num_swin_per_rstb,
            num_heads=cfg.swin_num_heads,
            window_size=cfg.swin_window_size,
            mlp_ratio=cfg.swin_mlp_ratio,
            qkv_bias=cfg.swin_qkv_bias,
            drop_rate=cfg.swin_drop_rate,
            attn_drop_rate=cfg.swin_attn_drop_rate,
            drop_path_rate=cfg.swin_drop_path_rate,
        )

        # ── Frequency branch ──
        self.freq_branch = FrequencyBranch(
            channels=C,
            fno_modes_h=cfg.fno_modes_h,
            fno_modes_w=cfg.fno_modes_w,
            num_freq_blocks=cfg.num_freq_blocks,
            num_bands=cfg.freq_bands,
        )

        # ── Cross-Domain Attention Module (CDAM) ──
        self.cdam = CrossDomainAttentionModule(
            dim=C,
            num_layers=cfg.num_cdam_layers,
            num_heads=cfg.cdam_num_heads,
            window_size=cfg.cdam_window_size,
            mlp_ratio=cfg.cdam_mlp_ratio,
            drop_rate=cfg.cdam_drop_rate,
        )

        # ── Gated fusion ──
        self.fusion = GatedFusion(C, use_channel_refinement=cfg.use_channel_refinement)

        # ── Deep feature refinement ──
        self.deep_refine = nn.Sequential(
            *[ResBlock(C, res_scale=0.2) for _ in range(4)],
            LayerNorm2d(C),
            conv3x3(C, C),
        )

        # ── Upscale + output conv ──
        self.upsample = UpscaleBlock(C, cfg.scale)
        self.conv_last = conv3x3(C, cfg.out_channels)

        # Weight init
        self._init_weights()

    def _init_weights(self):
        """
        Initialize only the 'glue' layers (conv_first, deep_refine, upsample, conv_last).
        Branch modules (spatial, frequency, CDAM, fusion) have their own careful
        initialization (trunc_normal_ for Swin, 1/sqrt(C) for FNO, etc.)
        and must NOT be overwritten here.
        """
        for module in [self.conv_first, self.deep_refine, self.upsample, self.conv_last]:
            for m in module.modules() if isinstance(module, nn.Module) else [module]:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LR image [B, 3, H, W] in [0, 1]
        Returns:
            SR image [B, 3, scale*H, scale*W]
        """
        # Global residual: bicubic upscale of input
        bicubic = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )

        # Shallow features
        feat = self.conv_first(x)

        # Dual-domain processing
        spatial_feat = self.spatial_branch(feat)
        freq_feat    = self.freq_branch(feat)

        # Cross-domain attention: mutually enrich both domains
        spatial_feat, freq_feat = self.cdam(spatial_feat, freq_feat)

        # Gated fusion
        fused = self.fusion(spatial_feat, freq_feat)

        # Deep refinement + global residual in feature space
        deep = self.deep_refine(fused) + feat

        # Upsample & output
        out = self.conv_last(self.upsample(deep))

        # Add bicubic global residual (helps with low-frequency content)
        return out + bicubic

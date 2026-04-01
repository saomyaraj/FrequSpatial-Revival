"""U-Net discriminator with spectral normalization
U-Net discriminator provides per-pixel feedback (local) and a global discriminator signal, which is stronger for SR than
standard PatchGAN discriminators. Spectral normalization stabilizes training (no weight clipping needed)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from config import ModelConfig

def sn_conv(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> nn.Conv2d:
    """spectrally-normalized convolution."""
    return spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, bias=False))

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            sn_conv(in_ch, out_ch, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(out_ch, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            sn_conv(in_ch + skip_ch, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(out_ch, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class UNetDiscriminator(nn.Module):
    """U-Net discriminator.
    Encoder produces multi-scale feature maps (skip connections).
    Decoder reconstructs per-pixel discrimination map. Final output: pixel-wise realness score"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        C = cfg.disc_base_channels
        in_ch = cfg.in_channels

        # encoder
        self.enc0 = nn.Sequential(
            sn_conv(in_ch, C),
            nn.LeakyReLU(0.2, inplace=True),
        )  # no downsampling — full resolution features
        self.enc1 = DownBlock(C, C * 2)
        self.enc2 = DownBlock(C * 2, C * 4)
        self.enc3 = DownBlock(C * 4, C * 8)
        self.enc4 = DownBlock(C * 8, C * 8)

        # global discriminator head(scalar per sample)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(C * 8, 1)),
        )

        # decoder(produces per-pixel map)
        self.dec3 = UpBlock(C * 8, C * 8, C * 4)
        self.dec2 = UpBlock(C * 4, C * 4, C * 2)
        self.dec1 = UpBlock(C * 2, C * 2, C)
        self.dec0 = UpBlock(C, C, C)

        # Per-pixel output
        self.pixel_head = spectral_norm(nn.Conv2d(C, 1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        """args: x: image [B, 3, H, W]
        returns:global_score: [B, 1] — real/fake per image
            local_map: [B, 1, H, W] — per-pixel score"""
        # encode
        e0 = self.enc0(x)   # [B, C, H, W]
        e1 = self.enc1(e0)  # [B, 2C, H/2, W/2]
        e2 = self.enc2(e1)  # [B, 4C, H/4, W/4]
        e3 = self.enc3(e2)  # [B, 8C, H/8, W/8]
        e4 = self.enc4(e3)  # [B, 8C, H/16, W/16]

        # global score
        global_score = self.global_head(e4)  # [B, 1]

        # decode
        d3 = self.dec3(e4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        local_map = self.pixel_head(d0) # [B, 1, H, W]
        return global_score, local_map

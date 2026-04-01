"""Frequency Branch
  1. FNO-style global frequency mixing: learnable complex weights applied to top-k frequency modes → captures global long-range dependencies that spatial convolutions miss.
  2. Adaptive Band Decomposition: split spectrum into low/mid/high bands using radial distance from DC, process each with dedicated sub-networks, recombine with learned weights.
  3. Phase path uses proper (sin, cos) encoding to avoid wrapping artifacts and preserve full angular information.
  4. Zero dead-code: all computed features are used."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import ResBlock, conv1x1, conv3x3, ChannelAttention, LayerNorm2d

# FNO-style Spectral Mixing Layer
class SpectralMixingLayer(nn.Module):
    """Fourier Neural Operator-inspired layer.
    Keeps top-k frequency modes and applies a learnable complex linear transformation to mix them globally. This is fundamentally different from
    convolution: it operates on the ENTIRE spatial field simultaneously, capturing patterns at any range.
    Both positive and negative frequency modes along the height axis are processed (rfft2 retains the full height axis)."""

    def __init__(self, channels: int, modes_h: int, modes_w: int):
        super().__init__()
        self.channels = channels
        self.modes_h = modes_h   # Number of frequency modes (height, per side)
        self.modes_w = modes_w   # Number of frequency modes (width, rfft half)

        # Complex weights for BOTH positive and negative height modes. Stored as real+imag for stability
        scale = 1.0 / math.sqrt(channels)
        # Positive height modes
        self.w1_real = nn.Parameter(scale * torch.randn(channels, channels, modes_h, modes_w))
        self.w1_imag = nn.Parameter(scale * torch.randn(channels, channels, modes_h, modes_w))
        # Negative height modes
        self.w2_real = nn.Parameter(scale * torch.randn(channels, channels, modes_h, modes_w))
        self.w2_imag = nn.Parameter(scale * torch.randn(channels, channels, modes_h, modes_w))

    def _complex_mult2d(self, x, w_real, w_imag):
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        real = torch.einsum("bcxy,icxy->bixy", x.real, w_real) - \
               torch.einsum("bcxy,icxy->bixy", x.imag, w_imag)
        imag = torch.einsum("bcxy,icxy->bixy", x.real, w_imag) + \
               torch.einsum("bcxy,icxy->bixy", x.imag, w_real)
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] float32 — works in spatial domain, does mixing in freq."""
        B, C, H, W = x.shape

        # FFT (real input → half spectrum for efficiency)
        x_ft = torch.fft.rfft2(x, norm="ortho")  # [B, C, H, W//2+1]

        # Truncate to top-k modes
        modes_h = min(self.modes_h, H // 2)
        modes_w = min(self.modes_w, W // 2 + 1)

        out_ft = torch.zeros_like(x_ft)

        # Apply learnable complex mixing on top-k POSITIVE height modes
        out_ft[:, :, :modes_h, :modes_w] = self._complex_mult2d(x_ft[:, :, :modes_h, :modes_w],
            self.w1_real[:, :, :modes_h, :modes_w],
            self.w1_imag[:, :, :modes_h, :modes_w],)
        # Apply to NEGATIVE height modes (wrap-around in rfft2: indices -modes_h:)
        out_ft[:, :, -modes_h:, :modes_w] = self._complex_mult2d(x_ft[:, :, -modes_h:, :modes_w],
            self.w2_real[:, :, :modes_h, :modes_w],
            self.w2_imag[:, :, :modes_h, :modes_w],)

        # IFFT back to spatial domain
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


# Frequency Band Processor
class FreqBandProcessor(nn.Module):
    """Processes a single frequency band (low / mid / high). Works on the 2D spectrum slice, treats real+imag as separate channels."""
    def __init__(self, channels: int, num_blocks: int = 2):
        super().__init__()
        # Real + imag concatenated → 2*C input
        self.proj_in = conv1x1(channels * 2, channels)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.proj_out = conv1x1(channels, channels)

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        """x_complex: [B, C, Hf, Wf] complex (a frequency slice)
        Returns: [B, C, Hf, Wf] real (processed features for this band)"""
        # Stack real + imag as channels
        x = torch.cat([x_complex.real, x_complex.imag], dim=1)  # [B, 2C, Hf, Wf]
        x = self.proj_in(x)
        x = self.body(x)
        return self.proj_out(x)

# Magnitude + Phase Paths
class MagnitudePath(nn.Module):
    """Process log-magnitude spectrum. Log-scale handles dynamic range."""
    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.norm = LayerNorm2d(channels)

    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        """mag: [B, C, H, W] — already log-scaled spatial features."""
        return self.norm(self.body(mag))


class PhasePath(nn.Module):
    """Process phase spectrum. Phase lives in [-π, π] — use (sin, cos) encoding to:
      1. Avoid the discontinuity at ±π
      2. Preserve full angular information (sin alone can't distinguish π/3 from 2π/3)"""
    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        # Sin+cos doubles channels → project back
        self.proj_in = conv1x1(channels * 2, channels)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.norm = LayerNorm2d(channels)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """phase: [B, C, H, W] — values in [-π, π]."""
        # Encode phase as (sin, cos) for continuous, wrapping-aware representation
        phase_enc = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)  # [B, 2C, H, W]
        phase_enc = self.proj_in(phase_enc)  # [B, C, H, W]
        return self.norm(self.body(phase_enc))


# Radial Band Mask Utilities
def _build_radial_masks(H: int, W_rfft: int, num_bands: int, device: torch.device):
    """Build radial frequency band masks for rfft2 output. rfft2 output has shape [H, W//2+1]. DC is at [0, 0].
    Height frequencies wrap: index 0 = DC, 1..H//2 = positive, H//2+1..H-1 = negative.
    Width frequencies: 0..W//2 = positive only (rfft symmetry). We compute radial distance from DC and split into `num_bands` equal bands"""
    # Frequency coordinates (normalized to [0, 1] range)
    freq_h = torch.fft.fftfreq(H, device=device)       # [-0.5, 0.5), length H
    freq_w = torch.fft.rfftfreq(H, device=device)[:W_rfft]  # [0, 0.5], length W_rfft

    # Radial distance from DC
    grid_h, grid_w = torch.meshgrid(freq_h, freq_w, indexing="ij")
    radius = torch.sqrt(grid_h ** 2 + grid_w ** 2)  # [H, W_rfft]
    max_radius = radius.max() + 1e-8

    # Split into equal radial bands
    masks = []
    for i in range(num_bands):
        lo = max_radius * i / num_bands
        hi = max_radius * (i + 1) / num_bands
        mask = (radius >= lo) & (radius < hi)
        if i == num_bands - 1:
            mask = mask | (radius >= hi)  # Include boundary in last band
        masks.append(mask)
    return masks

# Full Frequency Branch
class FrequencyBranch(nn.Module):
    """Hybrid frequency processing branch. Pipeline:
        1. FNO-style global mixing (long-range dependencies)
        2. Adaptive radial band decomposition (low/mid/high)
        3. Magnitude + phase dedicated paths (with proper sin+cos encoding)
        4. All paths fused, global residual"""
    def __init__(self, channels: int, fno_modes_h: int, fno_modes_w: int, num_freq_blocks: int, num_bands: int = 3,):
        super().__init__()
        self.channels = channels
        self.num_bands = num_bands

        # 1. FNO global spectral mixing
        self.spectral_mix = SpectralMixingLayer(channels, fno_modes_h, fno_modes_w)
        self.spectral_norm = LayerNorm2d(channels)

        # 2. Adaptive band decomposition
        blocks_per_band = max(1, num_freq_blocks // num_bands)
        self.band_processors = nn.ModuleList([FreqBandProcessor(channels, blocks_per_band) for _ in range(num_bands)])

        # Learnable band importance weights
        self.band_weights = nn.Parameter(torch.ones(num_bands) / num_bands)

        # 3. Magnitude path
        self.mag_path = MagnitudePath(channels, num_freq_blocks)

        # 4. Phase path (with sin+cos encoding)
        self.phase_path = PhasePath(channels, num_freq_blocks)

        # 5. Recombination
        # Fuse: spectral_mix + band_agg + mag_path + phase_path → channels
        self.fuse = nn.Sequential(
            conv1x1(channels * 4, channels * 2),
            nn.GELU(),
            conv1x1(channels * 2, channels),)

        # Channel attention to select what matters post-fusion
        self.ca = ChannelAttention(channels, reduction=16)

        self.conv_tail = conv3x3(channels, channels)

        # Cache for radial masks
        self._mask_cache = {}

    def _get_radial_masks(self, H, W_rfft, device):
        key = (H, W_rfft, device)
        if key not in self._mask_cache:
            self._mask_cache[key] = _build_radial_masks(H, W_rfft, self.num_bands, device)
        return self._mask_cache[key]

    def _extract_band(self, x_ft, mask):
        """Extract a frequency band using a radial mask and return as compact tensor."""
        # Mask and collect non-zero region (use bounding box for efficiency)
        rows = mask.any(dim=1).nonzero(as_tuple=True)[0]
        cols = mask.any(dim=0).nonzero(as_tuple=True)[0]
        if len(rows) == 0 or len(cols) == 0:
            return x_ft[:, :, :1, :1]  # fallback: tiny tensor
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1
        band = x_ft[:, :, r0:r1, c0:c1] * mask[r0:r1, c0:c1].unsqueeze(0).unsqueeze(0)
        return band

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] spatial features (LR space)
        Returns: [B, C, H, W] enriched features in spatial domain"""
        B, C, H, W = x.shape

        # cast to float32 for FFT (avoids cuFFT half-precision crash)
        x_f32 = x.float()

        # 1. FNO global spectral mixing
        fno_out = self.spectral_norm(self.spectral_mix(x_f32))

        # 2. Compute FFT for band decomposition & mag/phase paths
        x_ft = torch.fft.rfft2(x_f32, norm="ortho")  # [B, C, H, W//2+1]
        W_rfft = x_ft.shape[-1]

        magnitude = torch.abs(x_ft)
        phase = torch.angle(x_ft)
        log_mag = torch.log(magnitude + 1.0)  # log1p for stability

        # 3. Adaptive radial band decomposition
        masks = self._get_radial_masks(H, W_rfft, x.device)
        band_feats = []
        band_weights = torch.softmax(self.band_weights, dim=0)

        for i, (mask, proc) in enumerate(zip(masks, self.band_processors)):
            band = self._extract_band(x_ft, mask)
            feat = proc(band)  # [B, C, Hb, Wb]
            # Upsample to full spatial size for aggregation
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
            band_feats.append(feat * band_weights[i])

        band_agg = sum(band_feats)  # [B, C, H, W]

        # 4. Magnitude + phase paths. Convert half-spectrum features to spatial domain for conv processing. irfft2 properly maps [B, C, H, W//2+1] → [B, C, H, W]
        log_mag_spatial = torch.fft.irfft2(torch.complex(log_mag, torch.zeros_like(log_mag)), s=(H, W), norm="ortho")
        phase_spatial = torch.fft.irfft2(torch.complex(phase, torch.zeros_like(phase)), s=(H, W), norm="ortho")

        mag_feat = self.mag_path(log_mag_spatial)    # [B, C, H, W]
        phase_feat = self.phase_path(phase_spatial)    # [B, C, H, W]

        # 5. Fuse all paths
        combined = torch.cat([fno_out, band_agg, mag_feat, phase_feat], dim=1)
        fused = self.fuse(combined) # [B, C, H, W]
        fused = self.ca(fused) # channel attention
        out = self.conv_tail(fused) + x_f32  # global residual

        # Cast back to input dtype (may be fp16 under AMP)
        return out.to(x.dtype)

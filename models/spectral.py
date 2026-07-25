"""SR-MoSO — Spatially-Routed Mixture of Spectral Operators.

A window-attention backbone has a bounded (local) receptive field. SR-MoSO supplies the
complementary adaptive-**global** half: K *global* spectral experts (each a full-FFT operator, so
each has the whole image in its receptive field) combined **per-pixel** by routing weights predicted
from the local spatial features at that depth. Spatial content therefore decides *which* global
spectral operator applies *where* — a spatially-varying spectral response with no windowing and no
block artifacts.

`SpectralBlock` interleaves the operator into the trunk (one per stage, residual, zero-init) so the
spectral response is composed with attention and nonlinearity at multiple depths — the placement used
by SwinFIR / FFTformer / GFNet. `ConvBlock` is the param-matched *local* control used to prove the
gain comes from the global spectral operator rather than from added capacity.

Efficiency: each expert is a per-(channel,mode) complex *diagonal* filter followed by one shared
complex channel-mix (1x1), i.e. ~15x fewer params than a full CxC-per-mode FNO operator.

Note: Hermitian symmetry of the learned filter is not enforced, so `irfft2` applies the Hermitian
projection of the parameterized operator (standard in FNO/GFNet implementations).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import conv1x1, conv3x3, LayerNorm2d


class MoSpectralOperator(nn.Module):
    """K global spectral experts, routed per-pixel by spatial content.

    Each expert is a complex diagonal filter parameterized on a canonical Gh x Gw grid and sampled to
    the live rfft grid [H, W//2+1] every forward. `spectral_grid` selects how the canonical grid is
    indexed — this is the resolution-invariance ablation axis:

      "normalized" (ours) — index by NORMALIZED frequency (fftfreq/rfftfreq). The same normalized
          frequency gets the same response at any resolution → resolution-invariant, full-spectrum.
      "absolute"         — index by ABSOLUTE DFT bin, scaled by a fixed reference resolution. Still
          full-spectrum, but the response is tied to bin index → resolution-DEPENDENT. Identical to
          "normalized" at the reference resolution, so it isolates resolution-invariance alone.
      "fixed"            — vanilla FNO/GFNet: keep the lowest Gh x Gw modes, zero the rest
          (resolution-dependent AND truncated).

    mode:
      "moso"    — K experts + per-pixel softmax routing (headline; default)
      "mixture" — K experts, uniformly averaged (isolates routing)
      "global"  — K=1 expert + one content-predicted complex spectral gain (spatially-invariant)
      "static"  — K=1 expert, no conditioning (an efficient static FNO)
    """

    def __init__(self, channels: int, modes_h: int, modes_w: int, num_experts: int = 4,
                 mode: str = "moso", per_expert_mix: bool = False,
                 spectral_grid: str = "normalized", ref_size: int = 64):
        super().__init__()
        assert mode in ("moso", "mixture", "global", "static")
        assert spectral_grid in ("normalized", "absolute", "fixed")
        self.channels = channels
        self.Gh, self.Gw = modes_h, modes_w        # canonical grid dims
        self.mode = mode
        self.K = num_experts if mode in ("moso", "mixture") else 1
        self.per_expert_mix = per_expert_mix
        self.spectral_grid = spectral_grid
        self.ref_size = ref_size                   # reference resolution for "absolute" indexing

        # per-expert complex diagonal on the canonical grid.
        # near-identity init (re~1, im~0) so the operator starts as pass-through.
        init_std = 0.02
        shape = (self.K, channels, self.Gh, self.Gw)
        self.grid_re = nn.Parameter(1.0 + init_std * torch.randn(*shape))
        self.grid_im = nn.Parameter(init_std * torch.randn(*shape))

        # complex channel-mix (1x1), identity at init. Shared across experts by default.
        eye = torch.eye(channels)
        if per_expert_mix:
            self.mix_re = nn.Parameter(eye.expand(self.K, channels, channels).clone()
                                       + init_std * torch.randn(self.K, channels, channels))
            self.mix_im = nn.Parameter(init_std * torch.randn(self.K, channels, channels))
        else:
            self.mix_re = nn.Parameter(eye + init_std * torch.randn(channels, channels))
            self.mix_im = nn.Parameter(init_std * torch.randn(channels, channels))

        if mode == "moso":
            # per-pixel routing over experts, conditioned on the spatial features at this depth
            self.route = nn.Sequential(conv3x3(channels, channels), nn.GELU(), conv1x1(channels, self.K))
            nn.init.zeros_(self.route[-1].weight)                  # uniform routing at init
            nn.init.zeros_(self.route[-1].bias)
        elif mode == "global":
            hidden = max(64, channels)
            self.gain_net = nn.Sequential(nn.Linear(channels, hidden), nn.GELU())
            self.gain_head = nn.Linear(hidden, 2 * channels)
            nn.init.zeros_(self.gain_head.weight); nn.init.zeros_(self.gain_head.bias)

        # routing capture (visualization only; off by default → zero training overhead)
        self.capture_routing = False
        self._last_routing = None
        self._coord_cache = {}

    # ── canonical-grid sampling ────────────────────────────────────────────────
    def _coords(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """grid_sample coords [1,H,Wr,2] mapping each live rfft bin onto the canonical grid.
        Both indexings share the same final mapping (f_h -> y=2f_h, f_w -> x=4f_w-1), so
        "absolute" coincides exactly with "normalized" at the reference resolution."""
        key = (H, W, device, self.spectral_grid)
        if key not in self._coord_cache:
            if self.spectral_grid == "absolute":
                # index by absolute DFT bin, scaled by the fixed reference resolution
                h = torch.arange(H, device=device)
                h_signed = torch.where(h < (H + 1) // 2, h, h - H)     # wrapped, like fftfreq
                fh = h_signed.float() / self.ref_size
                fw = torch.arange(W // 2 + 1, device=device).float() / self.ref_size
            else:
                fh = torch.fft.fftfreq(H, device=device)               # [H]  in [-1/2, 1/2)
                fw = torch.fft.rfftfreq(W, device=device)              # [Wr] in [0, 1/2]
            x = fw * 4.0 - 1.0
            y = fh * 2.0
            gy, gx = torch.meshgrid(y, x, indexing="ij")
            self._coord_cache[key] = torch.stack([gx, gy], dim=-1).unsqueeze(0)   # [1,H,Wr,2]
        return self._coord_cache[key]

    def _resample_filters(self, H: int, W: int, device: torch.device):
        """sample the canonical complex diagonals onto the live rfft grid → D_re, D_im [K,C,H,Wr]."""
        C = self.channels
        coords = self._coords(H, W, device)
        inp = torch.cat([self.grid_re.reshape(self.K * C, self.Gh, self.Gw),
                         self.grid_im.reshape(self.K * C, self.Gh, self.Gw)], dim=0).unsqueeze(0)
        sampled = F.grid_sample(inp, coords, mode="bilinear", align_corners=True,
                                padding_mode="border")[0]              # [2KC,H,Wr]
        Wr = sampled.shape[-1]
        return (sampled[:self.K * C].reshape(self.K, C, H, Wr),
                sampled[self.K * C:].reshape(self.K, C, H, Wr))

    def _fixed_filters(self, H: int, W: int):
        """vanilla FNO/GFNet: place the canonical params on the lowest Gh x Gw DFT modes and zero the
        rest → the operator's normalized cutoff shrinks as resolution grows."""
        C, Wr = self.channels, W // 2 + 1
        gh = min(self.Gh // 2, H // 2)
        gw = min(self.Gw, Wr)
        D_re = self.grid_re.new_zeros(self.K, C, H, Wr)
        D_im = self.grid_im.new_zeros(self.K, C, H, Wr)
        D_re[:, :, :gh, :gw] = self.grid_re[:, :, :gh, :gw]            # low positive-height modes
        D_im[:, :, :gh, :gw] = self.grid_im[:, :, :gh, :gw]
        if gh > 0:
            D_re[:, :, H - gh:, :gw] = self.grid_re[:, :, gh:2 * gh, :gw]   # low negative-height modes
            D_im[:, :, H - gh:, :gw] = self.grid_im[:, :, gh:2 * gh, :gw]
        return D_re, D_im

    def _filters(self, H: int, W: int, device: torch.device):
        if self.spectral_grid == "fixed":
            return self._fixed_filters(H, W)
        return self._resample_filters(H, W, device)

    # ── operator ──────────────────────────────────────────────────────────────
    def _channel_mix(self, s_re, s_im, k):
        m_re = self.mix_re[k] if self.per_expert_mix else self.mix_re
        m_im = self.mix_im[k] if self.per_expert_mix else self.mix_im
        o_re = torch.einsum("bchw,co->bohw", s_re, m_re) - torch.einsum("bchw,co->bohw", s_im, m_im)
        o_im = torch.einsum("bchw,co->bohw", s_re, m_im) + torch.einsum("bchw,co->bohw", s_im, m_re)
        return o_re, o_im

    def _expert_spectrum(self, x_ft, k, D_re, D_im):
        """expert k: full-spectrum complex diagonal + complex channel mix → spectrum [B,C,H,Wr]"""
        fr = x_ft.real * D_re[k] - x_ft.imag * D_im[k]
        fi = x_ft.real * D_im[k] + x_ft.imag * D_re[k]
        mr, mi = self._channel_mix(fr, fi, k)
        return torch.complex(mr, mi)

    @torch.no_grad()
    def expert_response(self, k: int, H: int, W: int, device: torch.device = None) -> torch.Tensor:
        """channel-averaged |D_k| on the rfft grid at (H, W) — for the expert-spectrum figure."""
        device = device or self.grid_re.device
        D_re, D_im = self._filters(H, W, device)
        return torch.sqrt(D_re[k] ** 2 + D_im[k] ** 2).mean(0)        # [H, Wr]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,C,H,W] spatial features (also the routing conditioner). Returns [B,C,H,W]."""
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")                        # [B,C,H,Wr]
        D_re, D_im = self._filters(H, W, x.device)                     # [K,C,H,Wr]

        if self.mode in ("moso", "mixture"):
            if self.mode == "mixture":
                # uniform average (no routing) — accumulate, no [B,K,C,H,W] stack
                out = None
                for k in range(self.K):
                    y = torch.fft.irfft2(self._expert_spectrum(x_ft, k, D_re, D_im), s=(H, W), norm="ortho")
                    out = y if out is None else out + y
                return out / self.K
            r = torch.softmax(self.route(x), dim=1)                     # [B,K,H,W]
            if self.capture_routing:
                self._last_routing = r.detach()
            out = None
            for k in range(self.K):                                     # routed sum, accumulated
                y = torch.fft.irfft2(self._expert_spectrum(x_ft, k, D_re, D_im), s=(H, W), norm="ortho")
                y = y * r[:, k:k + 1]
                out = y if out is None else out + y
            return out

        # static / global: single expert (K == 1)
        spec = self._expert_spectrum(x_ft, 0, D_re, D_im)
        if self.mode == "global":
            d = x.mean(dim=(-2, -1))                                   # [B,C]
            g = self.gain_head(self.gain_net(d)).view(B, C, 2)
            gain = torch.complex(1.0 + g[..., 0], g[..., 1])[:, :, None, None]   # identity at init
            spec = spec * gain
        return torch.fft.irfft2(spec, s=(H, W), norm="ortho")


class SpectralBlock(nn.Module):
    """One interleaved SR-MoSO stage: norm → operator → 1x1, added residually.

    The output 1x1 is **zero-initialized**, so the block is an exact identity at init: a `proposed`
    model and its `no_freq` counterpart produce identical outputs at step 0, and training starts from
    a clean attention-only baseline that the spectral path grows into.

    FFT/complex math is forced to fp32: `autocast` casts at the *op* level, so `einsum` would
    otherwise run in fp16 and `torch.complex` would produce ComplexHalf, which `irfft2` rejects.
    """

    def __init__(self, channels: int, modes_h: int, modes_w: int, num_experts: int = 4,
                 mode: str = "moso", per_expert_mix: bool = False,
                 spectral_grid: str = "normalized", ref_size: int = 64):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.op = MoSpectralOperator(channels, modes_h, modes_w, num_experts=num_experts, mode=mode,
                                     per_expert_mix=per_expert_mix, spectral_grid=spectral_grid,
                                     ref_size=ref_size)
        self.proj = conv1x1(channels, channels)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)   # identity at init
        self.collect_stats = False      # off by default → zero training overhead
        self.stats = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.collect_stats and self.op.mode == "moso":
                self.op.capture_routing = True
            h = self.op(self.norm(x.float()))
            self.op.capture_routing = False
        delta = self.proj(h.to(x.dtype))
        if self.collect_stats:
            self._record(x, delta)
        return x + delta

    @torch.no_grad()
    def _record(self, x: torch.Tensor, delta: torch.Tensor):
        """cheap scalars for monitoring the two things that decide the paper (see utils docs):

        contrib      ||delta|| / ||x||  — is the spectral path contributing at all? The projection is
                     zero-init, so contrib==0 means it never grew and the mechanism is inert.
        route_entropy normalized entropy of r(p) in [0,1] — 1.0 means routing stayed UNIFORM
                     (SR-MoSO degenerates to the `no_routing` mixture); lower means it specializes.
        expert_usage mean routing weight per expert — spots dead/collapsed experts.
        """
        s = {"contrib": (delta.norm() / x.norm().clamp_min(1e-12)).item()}
        r = self.op._last_routing
        if r is not None:
            K = r.shape[1]
            p = r.clamp_min(1e-12)
            ent = -(p * p.log()).sum(1).mean() / math.log(K)      # [0,1], 1 == uniform
            s["route_entropy"] = ent.item()
            s["expert_usage"] = r.mean(dim=(0, 2, 3)).tolist()
            self.op._last_routing = None
        self.stats = s


def enable_stats(model: nn.Module, on: bool = True):
    """toggle diagnostic collection on every SpectralBlock (cheap; intended for one batch per epoch)."""
    for m in model.modules():
        if isinstance(m, SpectralBlock):
            m.collect_stats = on


def collect_stats(model: nn.Module) -> dict:
    """flatten the last recorded diagnostics into {"moso/s0/contrib": ..., ...} plus means across
    stages, which are the two numbers to actually watch during training."""
    out, contribs, entropies = {}, [], []
    i = 0
    for m in model.modules():
        if isinstance(m, SpectralBlock) and m.stats:
            for k, v in m.stats.items():
                out[f"moso/s{i}/{k}"] = v
            if "contrib" in m.stats:
                contribs.append(m.stats["contrib"])
            if "route_entropy" in m.stats:
                entropies.append(m.stats["route_entropy"])
            i += 1
    if contribs:
        out["moso/contrib_mean"] = sum(contribs) / len(contribs)
    if entropies:
        out["moso/route_entropy_mean"] = sum(entropies) / len(entropies)
    return out


class ConvBlock(nn.Module):
    """Param-matched **local** control for SpectralBlock (the `conv_block` ablation).

    Same residual slot, same zero-init identity, same parameter budget — but a stack of 3x3 convs
    (local receptive field) instead of a global spectral operator. This isolates "global spectral
    operator" from "extra capacity", which a plain no-frequency baseline cannot do.
    """

    def __init__(self, channels: int, target_params: int):
        super().__init__()
        per_conv = 9 * channels * channels + channels
        n = max(1, round(target_params / per_conv))
        layers, self.norm = [], LayerNorm2d(channels)
        for i in range(n):
            layers.append(conv3x3(channels, channels))
            if i < n - 1:
                layers.append(nn.GELU())
        self.body = nn.Sequential(*layers)
        self.proj = conv1x1(channels, channels)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)   # identity at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(self.body(self.norm(x)))

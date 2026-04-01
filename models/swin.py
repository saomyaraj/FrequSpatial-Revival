"""
FrequSpatial V2 — Swin Transformer Blocks
Proper SwinIR-style implementation:
  - Relative position bias: computed and USED (V1 bug: unused)
  - Dynamic padding for arbitrary resolutions
  - Alternating regular / shifted windows
  - Residual Swin Transformer Block (RSTB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from .common import MLP, window_partition, window_reverse, LayerNorm2d, conv3x3


# ──────────────────────────────────────────────
# Window Multi-Head Self-Attention
# ──────────────────────────────────────────────
class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention (W-MSA / SW-MSA).
    Relative position bias is computed once and used in every forward pass.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww (square)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table: (2*Wh-1) * (2*Ww-1), num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    [nW*B, N, C]   (N = window_size^2 tokens)
            mask: [nW, N, N] or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: [B_, num_heads, N, head_dim]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, num_heads, N, N]

        # Add relative position bias — THIS WAS UNUSED IN V1, NOW PROPERLY APPLIED
        rel_pos_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()  # num_heads, N, N
        attn = attn + rel_pos_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ──────────────────────────────────────────────
# Swin Transformer Block
# ──────────────────────────────────────────────
class SwinTransformerBlock(nn.Module):
    """
    One Swin Transformer block with optional cyclic shift.
    Alternates between regular (shift=0) and shifted (shift=ws//2) windows.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

        # Cyclic shift mask (computed lazily, cached by input resolution)
        self._attn_mask_cache = {}

    def _get_attn_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, device)
        if key not in self._attn_mask_cache:
            mask = self._compute_attn_mask(H, W, device)
            self._attn_mask_cache[key] = mask
        return self._attn_mask_cache[key]

    def _compute_attn_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, ws, ws, 1
        mask_windows = mask_windows.view(-1, self.window_size ** 2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, C]"""
        B, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)

        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Window partition
        x_windows = window_partition(x, self.window_size)  # nW*B, ws, ws, C
        x_windows = x_windows.view(-1, self.window_size ** 2, C)

        # Window attention
        attn_mask = self._get_attn_mask(H, W, x.device)
        attn_out = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_out, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ──────────────────────────────────────────────
# Residual Swin Transformer Block (RSTB)
# ──────────────────────────────────────────────
class RSTB(nn.Module):
    """
    Residual Swin Transformer Block (SwinIR-style).
    M alternating SwinTransformerBlocks + Conv3x3 + global residual.
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int,
        num_heads: int,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rates: list = None,
    ):
        super().__init__()
        self.window_size = window_size

        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_blocks

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path_rates[i],
            )
            for i in range(num_blocks)
        ])

        self.conv = conv3x3(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]"""
        from .common import pad_to_multiple, unpad

        # Dynamic padding to nearest window-size multiple
        x, (ph, pw) = pad_to_multiple(x, self.window_size)
        _, _, H, W = x.shape

        residual = x
        # BCHW → BHWC for transformer blocks
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        x = self.conv(x) + residual

        # Remove padding
        return unpad(x, ph, pw)

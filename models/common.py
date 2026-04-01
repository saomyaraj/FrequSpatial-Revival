"""common building blocks, shared primitives used across the model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

# norms
class LayerNorm2d(nn.Module):
    """channel-first LayerNorm (BCHW input) — preferred over BN for SR"""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# basic conv wrappers
def conv3x3(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias)


def conv1x1(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)


# residual block
class ResBlock(nn.Module):
    """basic residual block: Conv -> GELU -> Conv + skip. uses GELU instead of ReLU (better for transformer-adjacent architectures). no BN — avoids artifacts in SR due to statistics mismatch"""
    def __init__(self, channels: int, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            conv3x3(channels, channels),
            nn.GELU(),
            conv3x3(channels, channels),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.body(x)


# channel attention
class ChannelAttention(nn.Module):
    """squeeze-and-excitation channel attention"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            conv1x1(channels, reduced, bias=False),
            nn.GELU(),
            conv1x1(reduced, channels, bias=False),)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx) * x


# upscale block(pixelshuffle)
class UpscaleBlock(nn.Module):
    """sub-pixel convolution upsampling. process at LR resolution all the way, shuffle only at the end -> more efficient than bicubic-first approaches"""
    def __init__(self, in_channels: int, scale: int):
        super().__init__()
        if scale == 3:
            self.up = nn.Sequential(
                conv3x3(in_channels, in_channels * 9),
                nn.PixelShuffle(3),
                nn.GELU(),)
        elif scale == 4:
            # Two 2× stages for better quality
            self.up = nn.Sequential(
                conv3x3(in_channels, in_channels * 4),
                nn.PixelShuffle(2),
                nn.GELU(),
                conv3x3(in_channels, in_channels * 4),
                nn.PixelShuffle(2),
                nn.GELU(),)
        else:  # scale == 2
            self.up = nn.Sequential(
                conv3x3(in_channels, in_channels * 4),
                nn.PixelShuffle(2),
                nn.GELU(),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# MLP
class MLP(nn.Module):
    """feed-forward MLP used in Swin and CDAM blocks"""
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.0,):
        super().__init__()
        hidden = hidden_features or in_features
        out = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# window utilities(shared by Swin and CDAM)
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """partition feature map into non-overlapping windows
    args: x: [B, H, W, C], window_size: window size (square)
    returns: windows: [num_windows * B, window_size, window_size, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """reverse window partition
    args:windows: [num_windows * B, window_size, window_size, C]
        window_size: window size
        H, W: original spatial dimensions
    returns: x: [B, H, W, C]"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def pad_to_multiple(x: torch.Tensor, window_size: int):
    """pad spatial dims of BCHW tensor to nearest multiple of window_size. returns (padded_tensor, (pad_h, pad_w)) for unpadding later"""
    _, _, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def unpad(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """remove padding added by pad_to_multiple"""
    H, W = x.shape[-2], x.shape[-1]
    return x[:, :, : H - pad_h if pad_h else H, : W - pad_w if pad_w else W]
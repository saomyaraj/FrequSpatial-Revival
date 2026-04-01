"""Cross-Domain Attention Module - Windowed bidirectional cross-attention between spatial and frequency feature maps"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from .common import MLP, window_partition, window_reverse, pad_to_multiple, unpad, LayerNorm2d, conv3x3


# windowed cross-attention
class WindowedCrossAttention(nn.Module):
    """cross-attention between two feature maps (query domain <- key/value domain) using the same window partitioning as Swin for efficiency"""
    def __init__(self, dim: int, num_heads: int, window_size: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0,):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # query projection(from query domain)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # key + value projection(from source domain)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # Relative position bias (shared, same structure as Swin)
        self.rel_pos_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        trunc_normal_(self.rel_pos_bias_table, std=0.02)

        coords = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(coords, coords, indexing="ij")).flatten(1)  # 2, N
        rel = grid[:, :, None] - grid[:, None, :]  # 2, N, N
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_pos_index", rel.sum(-1))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query_tokens: torch.Tensor, source_tokens: torch.Tensor,) -> torch.Tensor:   # [nW*B, N, C] — query domain, # [nW*B, N, C] — key/value domain
        B_, N, C = query_tokens.shape

        q = self.q_proj(query_tokens).reshape(B_, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3) * self.scale  # [B_, heads, N, head_dim]

        kv = self.kv_proj(source_tokens).reshape(B_, N, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # each: [B_, heads, N, head_dim]

        attn = q @ k.transpose(-2, -1)  # [B_, heads, N, N]

        # relative position bias
        rel_bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)]
        rel_bias = rel_bias.view(self.window_size ** 2, self.window_size ** 2, -1)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()  # heads, N, N
        attn = attn + rel_bias.unsqueeze(0)

        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(out))


# CDAM layer
class CDAMLayer(nn.Module):
    """one CDAM layer. performs:
      1. spatial <- frequency (spatial queries attend to freq keys/values)
      2. frequency <- spatial (freq queries attend to spatial keys/values)
      3. FFN on each updated feature map
    both directions share the same window partitioning"""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8, mlp_ratio: float = 2.0, qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,):
        super().__init__()
        self.window_size = window_size

        # spatial queries attend to frequency(spatial <- freq)
        self.norm_s1 = nn.LayerNorm(dim)
        self.norm_f_for_s = nn.LayerNorm(dim)
        self.cross_attn_s = WindowedCrossAttention(dim, num_heads, window_size, qkv_bias, attn_drop, drop)
        self.norm_s2 = nn.LayerNorm(dim)
        self.ffn_s = MLP(dim, int(dim * mlp_ratio), drop=drop)

        # frequency queries attend to spatial(freq <- spatial)
        self.norm_f1 = nn.LayerNorm(dim)
        self.norm_s_for_f = nn.LayerNorm(dim)
        self.cross_attn_f = WindowedCrossAttention(dim, num_heads, window_size, qkv_bias, attn_drop, drop)
        self.norm_f2 = nn.LayerNorm(dim)
        self.ffn_f = MLP(dim, int(dim * mlp_ratio), drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _partition(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """BCHW -> [nW*B, N, C]"""
        x = x.permute(0, 2, 3, 1)  # BHWC
        windows = window_partition(x, self.window_size)  # nW*B, ws, ws, C
        return windows.view(-1, self.window_size ** 2, x.shape[-1])

    def _unpartition(self, tokens: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
        """[nW*B, N, C] -> BCHW"""
        C = tokens.shape[-1]
        tokens = tokens.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(tokens, self.window_size, H, W)  # BHWC
        return x.permute(0, 3, 1, 2)  # BCHW

    def forward(self, spatial: torch.Tensor, frequency: torch.Tensor,):
        B, C, H, W = spatial.shape
        # spatial <- frequency
        s_win  = self._partition(spatial,   H, W)
        f_win  = self._partition(frequency, H, W)

        s_norm = self.norm_s1(s_win)
        f_norm = self.norm_f_for_s(f_win)
        delta_s = self.cross_attn_s(s_norm, f_norm)
        s_win = s_win + self.drop_path(delta_s)
        s_win = s_win + self.drop_path(self.ffn_s(self.norm_s2(s_win)))
        spatial_out = self._unpartition(s_win, B, H, W)

        # frequency <- spatial. re-partition with updated spatial
        s_win_upd = self._partition(spatial_out, H, W)
        f_norm2 = self.norm_f1(f_win)
        s_norm2 = self.norm_s_for_f(s_win_upd)
        delta_f = self.cross_attn_f(f_norm2, s_norm2)
        f_win = f_win + self.drop_path(delta_f)
        f_win = f_win + self.drop_path(self.ffn_f(self.norm_f2(f_win)))
        freq_out = self._unpartition(f_win, B, H, W)

        return spatial_out, freq_out


# full CDAM stack
class CrossDomainAttentionModule(nn.Module):
    """stack of CDAM layers with dynamic padding for arbitrary resolutions. takes spatial and frequency features, returns mutually-enriched versions"""
    def __init__(self, dim: int, num_layers: int, num_heads: int, window_size: int = 8, mlp_ratio: float = 2.0, drop_rate: float = 0.0, attn_drop_rate: float = 0.0, drop_path_rate: float = 0.1,):
        super().__init__()
        self.window_size = window_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            CDAMLayer(dim=dim, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],)
            for i in range(num_layers)])

        # output projections to calibrate outputs
        self.proj_s = conv3x3(dim, dim)
        self.proj_f = conv3x3(dim, dim)

    def forward(self, spatial: torch.Tensor, frequency: torch.Tensor):
        """args: spatial: [B, C, H, W]
            frequency: [B, C, H, W]
        returns: enhanced_spatial, enhanced_frequency(same shapes)"""
        # pad to window-size multiple
        spatial,   (ph_s, pw_s) = pad_to_multiple(spatial,   self.window_size)
        frequency, (ph_f, pw_f) = pad_to_multiple(frequency, self.window_size)

        s, f = spatial, frequency
        for layer in self.layers:
            s, f = layer(s, f)

        # project + residual
        s = self.proj_s(s) + spatial
        f = self.proj_f(f) + frequency

        # remove padding
        s = unpad(s, ph_s, pw_s)
        f = unpad(f, ph_f, pw_f)
        return s, f

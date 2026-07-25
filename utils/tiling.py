"""tiled + self-ensemble inference for SR.

Full benchmark/validation images (e.g. DIV2K 2K) are too large to run through the model
in one shot. `tiled_forward` splits the LR image into overlapping tiles, super-resolves each,
and blends them with a cosine taper to remove seams. `self_ensemble_forward` averages the
8 dihedral (flip/rotate) transforms — the standard SR test-time augmentation.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def tiled_forward(model, lr: torch.Tensor, scale: int, tile: int = 96, overlap: int = 16) -> torch.Tensor:
    """Super-resolve a (possibly large) LR image with overlapping tiles + cosine blending.
    Args:
        model: callable LR[B,C,H,W] -> SR[B,C,scale*H,scale*W]
        lr:    [B, C, H, W]
        tile:  LR tile size; overlap: LR overlap between tiles.
    Returns: SR [B, C, scale*H, scale*W]."""
    B, C, H, W = lr.shape
    if H <= tile and W <= tile:
        return model(lr)

    stride = max(1, tile - overlap)
    out_h, out_w = H * scale, W * scale
    sr = lr.new_zeros(B, C, out_h, out_w)
    weight = lr.new_zeros(B, 1, out_h, out_w)

    # cosine taper window (LR-tile sized, upscaled) to feather tile borders
    def taper(n):
        w = torch.hann_window(n, periodic=False, device=lr.device).clamp_min(1e-3)
        return w

    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if ys[-1] != H - tile and H > tile:
        ys.append(H - tile)
    if xs[-1] != W - tile and W > tile:
        xs.append(W - tile)

    for y in ys:
        for x in xs:
            th = min(tile, H - y)
            tw = min(tile, W - x)
            patch = lr[:, :, y:y + th, x:x + tw]
            sr_patch = model(patch)                                  # [B, C, th*scale, tw*scale]
            ph, pw = th * scale, tw * scale
            win = (taper(ph)[:, None] * taper(pw)[None, :])[None, None]  # [1,1,ph,pw]
            oy, ox = y * scale, x * scale
            sr[:, :, oy:oy + ph, ox:ox + pw] += sr_patch * win
            weight[:, :, oy:oy + ph, ox:ox + pw] += win

    return sr / weight.clamp_min(1e-6)


# 8 dihedral transforms (identity, flips, rotations) and their inverses
def _augment(x, mode):
    if mode == 0: return x
    if mode == 1: return x.flip(-1)
    if mode == 2: return x.flip(-2)
    if mode == 3: return x.flip(-1).flip(-2)
    if mode == 4: return x.transpose(-1, -2)
    if mode == 5: return x.transpose(-1, -2).flip(-1)
    if mode == 6: return x.transpose(-1, -2).flip(-2)
    if mode == 7: return x.transpose(-1, -2).flip(-1).flip(-2)


def _deaugment(x, mode):
    # inverse of _augment (transpose is its own inverse; flips applied in reverse)
    if mode == 0: return x
    if mode == 1: return x.flip(-1)
    if mode == 2: return x.flip(-2)
    if mode == 3: return x.flip(-2).flip(-1)
    if mode == 4: return x.transpose(-1, -2)
    if mode == 5: return x.flip(-1).transpose(-1, -2)
    if mode == 6: return x.flip(-2).transpose(-1, -2)
    if mode == 7: return x.flip(-2).flip(-1).transpose(-1, -2)


@torch.no_grad()
def self_ensemble_forward(forward_fn, lr: torch.Tensor) -> torch.Tensor:
    """average the model output over the 8 dihedral transforms.
    forward_fn: callable LR -> SR (may itself be a tiled_forward closure)."""
    out = 0.0
    for m in range(8):
        sr = forward_fn(_augment(lr, m))
        out = out + _deaugment(sr, m)
    return out / 8.0

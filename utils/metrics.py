"""metrics - PSNR and SSIM on the Y (luminance) channel, with border-shave.

This matches the standard SR benchmark protocol used by EDSR / SwinIR / HAT:
  - convert to Y channel (BT.601 luma),
  - crop `scale` pixels from each border (boundary pixels are unreliable),
  - then compute PSNR / SSIM.
Use this protocol when comparing to published numbers — do NOT use RGB PSNR or omit the shave.
"""
import math
import numpy as np
import torch
from skimage.metrics import structural_similarity


def match_size(pred: torch.Tensor, target: torch.Tensor):
    """Crop both tensors to their common spatial size.

    `scale * LR` need not equal the HR size (integer rounding when HR dims aren't divisible by
    `scale`, or an LR pack generated from uncropped HR). Without this guard the numpy subtraction in
    `calc_psnr` either raises a broadcast error or — worse — silently compares misaligned pixels."""
    h = min(pred.shape[-2], target.shape[-2])
    w = min(pred.shape[-1], target.shape[-1])
    return pred[..., :h, :w], target[..., :h, :w]


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] tensor [0,1] → [H, W, C] numpy float32"""
    return t.cpu().float().clamp(0, 1).numpy().transpose(1, 2, 0)


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """RGB [H,W,3] float [0,1] → Y channel (BT.601 luma), standard for SR PSNR/SSIM."""
    return (65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2]) / 255.0 + 16.0 / 255.0


def _shave(y: np.ndarray, border: int) -> np.ndarray:
    if border > 0 and y.shape[0] > 2 * border and y.shape[1] > 2 * border:
        return y[border:-border, border:-border]
    return y


def calc_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0, crop_border: int = 0) -> float:
    """PSNR on the Y channel with `crop_border`-pixel border shave."""
    pred_y = _shave(rgb_to_y(pred), crop_border)
    target_y = _shave(rgb_to_y(target), crop_border)
    mse = np.mean((pred_y - target_y) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 20 * math.log10(data_range / math.sqrt(mse))


def calc_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0, crop_border: int = 0) -> float:
    """SSIM on the Y channel with `crop_border`-pixel border shave.
    Uses an 11×11 Gaussian window (σ=1.5) and population covariance to match the MATLAB SSIM
    reported by EDSR / SwinIR / HAT (skimage defaults differ: 7×7 uniform, sample covariance)."""
    pred_y = _shave(rgb_to_y(pred), crop_border)
    target_y = _shave(rgb_to_y(target), crop_border)
    return structural_similarity(pred_y, target_y, data_range=data_range,
                                 gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False, win_size=11)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, crop_border: int = 0):
    """compute average PSNR and SSIM for a batch.
    Args:
        pred, target: [B, C, H, W] tensors in [0, 1]
        crop_border:  pixels to shave from each border (set to `scale` for benchmark protocol)
    Returns: (avg_psnr, avg_ssim)"""
    B = pred.shape[0]
    psnrs, ssims = [], []
    for i in range(B):
        p = tensor_to_np(pred[i])
        t = tensor_to_np(target[i])
        psnrs.append(calc_psnr(p, t, crop_border=crop_border))
        ssims.append(calc_ssim(p, t, crop_border=crop_border))
    return sum(psnrs) / B, sum(ssims) / B

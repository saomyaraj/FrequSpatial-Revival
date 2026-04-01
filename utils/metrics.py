"""metrics - PSNR and SSIM with Y-channel (luminance) evaluation, which is the standard for SR benchmarking"""
import math
import numpy as np
import torch
from skimage.metrics import structural_similarity

def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] tensor [0,1] → [H, W, C] numpy float32"""
    return t.cpu().float().clamp(0, 1).numpy().transpose(1, 2, 0)


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """convert RGB [H,W,3] float [0,1] to Y channel (luminance) [H,W]. standard for PSNR/SSIM in SR papers (benchmark on luminance only)"""
    return (65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2]) / 255.0 + 16.0 / 255.0


def calc_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """PSNR on Y channel (luminance), standard SR benchmark protocol"""
    pred_y = rgb_to_y(pred)
    target_y = rgb_to_y(target)
    mse = np.mean((pred_y - target_y) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 20 * math.log10(data_range / math.sqrt(mse))


def calc_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """SSIM on Y channel (luminance)"""
    pred_y = rgb_to_y(pred)
    target_y = rgb_to_y(target)
    return structural_similarity(pred_y, target_y, data_range=data_range)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    """compute PSNR and SSIM for a batch.
    Args: pred, target: [B, C, H, W] tensors in [0, 1]
    Returns: avg_psnr, avg_ssim (floats)"""
    B = pred.shape[0]
    psnrs, ssims = [], []
    for i in range(B):
        p = tensor_to_np(pred[i])
        t = tensor_to_np(target[i])
        psnrs.append(calc_psnr(p, t))
        ssims.append(calc_ssim(p, t))
    return sum(psnrs) / B, sum(ssims) / B

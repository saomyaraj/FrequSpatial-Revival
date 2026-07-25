"""Tile-size / cross-resolution robustness evaluation (contribution 2).

The resolution-invariant spectral experts (`spectral_grid="normalized"`) apply the same normalized
frequency response at any resolution, so inference quality is ~independent of the tile size used to
process a large image. A fixed-grid FNO baseline (`spectral_grid="fixed"`) keeps a fixed number of DFT
modes per tile, so its effective normalized cutoff shifts with tile size → quality varies. This module
measures PSNR/SSIM vs inference tile size to produce that figure.
"""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .tiling import tiled_forward
from .metrics import compute_metrics


@torch.no_grad()
def eval_tile_robustness(model, lr: torch.Tensor, hr: torch.Tensor, scale: int,
                         tiles=(64, 128, 256, None), overlap_frac: int = 6):
    """PSNR/SSIM of one image super-resolved with different LR tile sizes.
    tiles: list of LR tile sizes; `None` = whole-image (no tiling). Returns {label: {psnr, ssim}}."""
    model.eval()
    results = {}
    for t in tiles:
        if t is None:
            sr = model(lr)
            label = "full"
        else:
            sr = tiled_forward(model, lr, scale=scale, tile=t, overlap=max(1, t // overlap_frac))
            label = str(t)
        sr = sr.clamp(0, 1)
        psnr, ssim = compute_metrics(sr, hr, crop_border=scale)
        results[label] = {"psnr": psnr, "ssim": ssim}
    return results


def save_tile_robustness_plot(results_by_model: dict, save_path: str):
    """results_by_model: {model_name: {tile_label: {psnr, ssim}}}. Plots PSNR vs tile size —
    ours (normalized) should be ~flat; abs_grid / fixed_grid_fno should drift."""
    if not results_by_model:
        raise ValueError("results_by_model is empty — nothing to plot")
    # x-axis is the union of tile labels in first-seen order, so curves measured over different
    # tile lists stay correctly aligned (plotting per-curve indices would mislabel one of them).
    labels = list(dict.fromkeys(k for res in results_by_model.values() for k in res))
    pos = {k: i for i, k in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, res in results_by_model.items():
        xs = [pos[k] for k in res]
        ys = [res[k]["psnr"] for k in res]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("inference tile size (LR px)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Tile-size robustness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path

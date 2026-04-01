"""visualization Utils"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from .metrics import tensor_to_np

# result grid
def save_result_grid(lr: torch.Tensor, # [B, 3, H_lr, W_lr]
    sr: torch.Tensor, # [B, 3, H_hr, W_hr]
    hr: torch.Tensor, # [B, 3, H_hr, W_hr]
    psnrs: list, ssims: list, save_path: str, max_images: int = 4,):
    """save a grid of LR | SR | HR comparisons"""
    n = min(lr.shape[0], max_images)
    fig, axes = plt.subplots(n, 3, figsize=(14, 5 * n))
    if n == 1:
        axes = axes[None, :]  # make 2D always

    titles = ["LR (input)", "SR (ours)", "HR (target)"]
    for i in range(n):
        lr_np = tensor_to_np(lr[i])
        sr_np = tensor_to_np(sr[i])
        hr_np = tensor_to_np(hr[i])

        # bilinear upsample LR for display (same size as SR/HR)
        import torch.nn.functional as F
        lr_up = F.interpolate(lr[i:i+1], size=hr.shape[-2:], mode="bilinear", align_corners=False)
        lr_np = tensor_to_np(lr_up[0])

        for j, (img, title) in enumerate(zip([lr_np, sr_np, hr_np], titles)):
            axes[i, j].imshow(img.clip(0, 1))
            axes[i, j].axis("off")
            if j == 1:
                axes[i, j].set_title(f"{title}\nPSNR {psnrs[i]:.2f} dB | SSIM {ssims[i]:.4f}", fontsize=9,)
            else:
                axes[i, j].set_title(title, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path


# training Curves
def save_training_curves(train_losses: list, val_psnrs: list, loss_components: dict, save_path: str,): # {"l1": [...], "perc": [...], "freq": [...], "adv": [...]}
    epochs = list(range(1, len(train_losses) + 1))

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # total loss
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(epochs, train_losses, color="#e74c3c", linewidth=1.5)
    ax0.set_title("total generator loss")
    ax0.set_xlabel("Epoch"); ax0.set_ylabel("Loss")
    ax0.grid(True, alpha=0.3)

    # validation PSNR
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(epochs[:len(val_psnrs)], val_psnrs, color="#2ecc71", linewidth=1.5)
    best_idx = int(np.argmax(val_psnrs)) if val_psnrs else 0
    if val_psnrs:
        ax1.axvline(best_idx + 1, color="gray", linestyle="--", alpha=0.6, label=f"Best: {max(val_psnrs):.2f} dB @ epoch {best_idx+1}")
    ax1.set_title("validation PSNR (Y-channel)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("PSNR (dB)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # individual loss components
    ax2 = fig.add_subplot(gs[1, :])
    colors = {"l1": "#3498db", "perc": "#9b59b6", "freq": "#e67e22", "adv": "#e74c3c"}
    for name, vals in loss_components.items():
        if vals:
            ax2.plot(epochs[:len(vals)], vals, label=name.upper(), color=colors.get(name, "black"), linewidth=1.2, alpha=0.85)
    ax2.set_title("loss components")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss (unweighted)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path


# frequency visualization
def save_freq_comparison(sr: torch.Tensor, hr: torch.Tensor, save_path: str,): # [1, 3, H, W]
    """visualize magnitude spectrum of SR vs HR. useful for diagnosing whether the frequency branch is helping"""
    def spectrum(img):
        gray = img.mean(dim=0, keepdim=True)  # [1, H, W]
        ft = torch.fft.fftshift(torch.fft.fft2(gray.float(), norm="ortho"))
        return torch.log(torch.abs(ft).squeeze() + 1e-8).cpu().numpy()

    sr_spec = spectrum(sr[0])
    hr_spec = spectrum(hr[0])
    diff = np.abs(sr_spec - hr_spec)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, img, title in zip(axes, [sr_spec, hr_spec, diff], ["SR Spectrum (log-mag)", "HR Spectrum (log-mag)", "Spectrum Error |SR-HR|"],):
        im = ax.imshow(img, cmap="inferno")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path

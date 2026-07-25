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
def save_training_curves(train_losses: list, val_psnrs: list, save_path: str):
    """L1 loss + validation PSNR. (Single-loss training → no per-component panel, and nothing to
    misalign after a resume.)"""
    epochs = list(range(1, len(train_losses) + 1))

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # training loss
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(epochs, train_losses, color="#e74c3c", linewidth=1.5)
    ax0.set_title("training L1 loss")
    ax0.set_xlabel("Epoch"); ax0.set_ylabel("L1")
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

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── SR-MoSO mechanism figures (routing + expert spectra) ───────────────────────
def _get_spectral_ops(model):
    """all interleaved MoSpectralOperator stages in the trunk, in depth order (may be empty)."""
    from models.spectral import MoSpectralOperator
    return [m for m in model.modules() if isinstance(m, MoSpectralOperator)]


def _get_spectral_op(model, stage: int = -1):
    """one operator stage (default: the deepest). None if the model has no spectral stages."""
    ops = _get_spectral_ops(model)
    return ops[stage] if ops else None


@torch.no_grad()
def save_routing_maps(model, lr: torch.Tensor, save_path: str, max_experts: int = 8, stage: int = -1):
    """Per-pixel routing weights r_k(p) of one SR-MoSO stage for one LR image: LR input + K expert
    heatmaps + the argmax (dominant-expert) map. The payoff figure — shows experts specialize
    spatially. Only meaningful in 'moso' mode. lr: [1,3,H,W]."""
    op = _get_spectral_op(model, stage)
    if op is None or op.mode != "moso":
        return None
    device = next(model.parameters()).device
    was, model_training = op.capture_routing, model.training
    op.capture_routing = True
    model.eval()
    _ = model(lr[:1].to(device))
    r = op._last_routing                                           # [1,K,h,w]
    op.capture_routing = was
    op._last_routing = None
    if model_training:
        model.train()
    if r is None:
        return None

    r = r[0].cpu()                                                 # [K,h,w]
    K = min(r.shape[0], max_experts)
    lr_np = tensor_to_np(lr[0])
    argmax = r[:K].argmax(0).numpy()                               # [h,w] dominant expert

    fig, axes = plt.subplots(1, K + 2, figsize=(3 * (K + 2), 3.2))
    axes[0].imshow(lr_np.clip(0, 1)); axes[0].set_title("LR input", fontsize=9); axes[0].axis("off")
    for k in range(K):
        im = axes[k + 1].imshow(r[k].numpy(), cmap="viridis")
        axes[k + 1].set_title(f"expert {k}", fontsize=9); axes[k + 1].axis("off")
        plt.colorbar(im, ax=axes[k + 1], fraction=0.046, pad=0.04)
    im = axes[K + 1].imshow(argmax, cmap="tab10", vmin=0, vmax=max(9, K - 1))
    axes[K + 1].set_title("dominant expert", fontsize=9); axes[K + 1].axis("off")
    plt.colorbar(im, ax=axes[K + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_erf(model, img: torch.Tensor, save_path: str):
    """Effective Receptive Field: |∂ SR(center pixel) / ∂ input| over the input image. A global
    operator (SR-MoSO) spreads across the whole input; windowed-local attention / local conv stays
    compact. Compare `proposed` vs `no_freq` to show the frequency branch makes the ERF global.
    (Not @no_grad — it needs a backward pass.)"""
    was_training = model.training
    model.eval()
    x = img[:1].clone().detach().requires_grad_(True)
    out = model(x)                                                 # [1,3,sH,sW]
    sh, sw = out.shape[-2:]
    resp = out[0, :, sh // 2, sw // 2].mean()                      # center-pixel response
    model.zero_grad(set_to_none=True)
    resp.backward()
    grad = x.grad[0].abs().mean(0).cpu().numpy()                   # [H,W]
    grad = np.log1p(grad / (grad.max() + 1e-12) * 1e3)            # log-scale for visibility

    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(grad, cmap="viridis")
    ax.set_title("Effective Receptive Field", fontsize=9); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    if was_training:
        model.train()
    return save_path


@torch.no_grad()
def save_expert_spectra(model, save_path: str, H: int = 64, W: int = 64, max_experts: int = 8,
                        stage: int = -1):
    """Each expert's learned magnitude response |D_k| over normalized frequency (DC centered
    vertically; width is the rfft half-spectrum 0→Nyquist). Shows experts cover complementary /
    anisotropic frequency bands — distinguishes SR-MoSO from a single static filter."""
    op = _get_spectral_op(model, stage)
    if op is None:
        return None
    K = min(op.K, max_experts)
    fig, axes = plt.subplots(1, K, figsize=(3 * K, 3.2))
    if K == 1:
        axes = [axes]
    for k in range(K):
        resp = torch.fft.fftshift(op.expert_response(k, H, W), dim=0).cpu().numpy()  # [H,Wr], DC-centered
        im = axes[k].imshow(resp, cmap="inferno", aspect="auto")
        axes[k].set_title(f"expert {k}  |D|", fontsize=9)
        axes[k].set_xlabel("f_w →"); axes[k].set_yticks([])
        plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
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

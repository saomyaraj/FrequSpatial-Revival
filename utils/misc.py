"""misc utilities: reproducibility seeding + model complexity reporting"""
import os
import json
import math
import random
import subprocess
import time
import dataclasses
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42, deterministic: bool = False):
    """seed all RNGs for reproducibility.
    deterministic=True forces cuDNN determinism (slower, exact repro)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # FFT/attention kernels may lack deterministic impls — warn_only avoids hard crash
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def count_params(model: nn.Module, trainable_only: bool = True) -> float:
    """return parameter count in millions."""
    if trainable_only:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        n = sum(p.numel() for p in model.parameters())
    return n / 1e6


def _fft_flops(model: nn.Module, H: int, W: int) -> float:
    """Analytic FLOPs for what fvcore cannot see.

    fvcore has NO handler for `fft_rfft2` / `fft_irfft2` / `grid_sampler` and ignores elementwise
    ops, so it silently *undercounts SR-MoSO* — the headline module — which would flatter the paper.
    Per SpectralBlock, per expert: 1 rfft2 + 1 irfft2 (~5·N·log2 N each) plus the complex diagonal
    (6 flops/bin). The complex channel-mix is an einsum, which fvcore DOES count, so it is excluded
    here to avoid double counting."""
    from models.spectral import SpectralBlock
    total, N = 0.0, H * W
    for m in model.modules():
        if isinstance(m, SpectralBlock):
            C, K = m.op.channels, m.op.K
            total += (2 * K) * C * 5.0 * N * math.log2(max(N, 2))     # rfft2 + irfft2 per expert
            total += K * 6.0 * C * H * (W // 2 + 1)                   # complex diagonal multiply
    return total


def count_flops(model: nn.Module, input_shape=(1, 3, 64, 64), device="cpu", verbose: bool = False) -> float:
    """GFLOPs for one forward pass at `input_shape`, including an analytic FFT term.
    Returns -1.0 if fvcore is unavailable (callers skip gracefully)."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return -1.0
    was_training = model.training
    model.eval()
    x = torch.randn(*input_shape, device=device)
    try:
        with torch.no_grad():
            fca = FlopCountAnalysis(model, x)
            fca.uncalled_modules_warnings(False)
            if not verbose:
                fca.unsupported_ops_warnings(False)
            g = fca.total() / 1e9
            if verbose:
                unsupported = {k: v for k, v in fca.unsupported_ops().items() if v}
                if unsupported:
                    print(f"  [flops] not counted by fvcore (added analytically): {dict(unsupported)}")
    except Exception:
        g = -1.0
    if g >= 0:
        g += _fft_flops(model, input_shape[-2], input_shape[-1]) / 1e9
    if was_training:
        model.train()
    return g


@torch.no_grad()
def measure_latency(model: nn.Module, input_shape, device="cpu", warmup: int = 3, iters: int = 10) -> float:
    """median forward latency in ms (CUDA-synchronized). Lightweight-SR papers report this alongside
    params/FLOPs; without it a 'lightweight' claim isn't verifiable."""
    was_training = model.training
    model.eval()
    dev = torch.device(device)
    x = torch.randn(*input_shape, device=dev)
    for _ in range(warmup):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    if was_training:
        model.train()
    times.sort()
    return times[len(times) // 2]


class MetricLogger:
    """append one JSON object per epoch to `metrics.jsonl`.

    Plain-text, append-only, and readable while training runs:
        py -3 -c "import json;[print(json.loads(l)['epoch'],json.loads(l)['psnr']) for l in open('results/metrics.jsonl')]"
    Survives crashes and resumes (unlike an in-memory list), and needs no wandb account."""

    def __init__(self, save_dir: str, filename: str = "metrics.jsonl"):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, filename)

    def log(self, **kw):
        with open(self.path, "a") as f:
            f.write(json.dumps(kw, default=float) + "\n")
        return kw


def format_eta(seconds: float) -> str:
    """'2h 14m' / '9m 30s' — so a 500-epoch run tells you when it will finish."""
    s = int(max(seconds, 0))
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """True if the worktree has uncommitted/untracked changes — i.e. `git_commit` alone does NOT
    reproduce this run. Recorded so a manifest can never silently overstate reproducibility."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except Exception:
        return True


def write_run_manifest(save_dir: str, cfg, seed: int, deterministic: bool) -> str:
    """write a self-describing run manifest (resolved config + env + git commit) for reproducibility."""
    manifest = {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": seed,
        "deterministic": deterministic,
        "model": dataclasses.asdict(cfg.model),
        "train": dataclasses.asdict(cfg.train),
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def report_complexity(model: nn.Module, name: str, scale: int = 4, device="cpu",
                      out_hw=(720, 1280), latency: bool = False):
    """print params + FLOPs (+ latency) at the conventional 1280x720 OUTPUT size.
    Returns (params_M, gflops, latency_ms); gflops/latency are -1.0 when unavailable."""
    p = count_params(model)
    shape = (1, 3, out_hw[0] // scale, out_hw[1] // scale)
    g = count_flops(model, input_shape=shape, device=device, verbose=True)
    ms = measure_latency(model, shape, device=device) if latency else -1.0
    parts = [f"{p:.3f}M params",
             f"{g:.2f} GFLOPs @ {out_hw[1]}x{out_hw[0]} out" if g >= 0 else "(install fvcore for FLOPs)"]
    if ms >= 0:
        parts.append(f"{ms:.1f} ms")
    print(f"{name}: " + " | ".join(parts))
    return p, g, ms

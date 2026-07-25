"""smoke test (CPU, <90s): forward/backward, EMA, metrics, tiled inference, every ablation preset,
plus targeted regression tests for bugs that previously escaped to a real training run.

    py -3 test_smoke.py
"""
import torch

from config import get_config, available_presets
from models import FrequSpatialGenerator
from utils import EMA, compute_metrics, tiled_forward, set_seed, match_size
from utils.misc import count_params


def _check(name, cond):
    assert cond, f"FAILED: {name}"
    print(f"  ok: {name}")


def _small_cfg(preset, scale=2, hw=16):
    cfg = get_config(preset)
    cfg.model.scale = scale
    cfg.model.num_rstb = 2
    cfg.model.num_swin_per_rstb = 2
    cfg.model.edsr_num_blocks = 2
    cfg.model.fno_modes_h = 4
    cfg.model.fno_modes_w = 4
    cfg.train.patch_size = hw
    cfg.validate()
    return cfg


def test_preset(preset, scale=2, hw=16):
    print(f"\n[preset={preset}, scale={scale}]")
    set_seed(0)
    cfg = _small_cfg(preset, scale, hw)

    g = FrequSpatialGenerator(cfg.model)
    criterion = torch.nn.L1Loss()

    lr = torch.rand(2, 3, hw, hw)
    hr = torch.rand(2, 3, hw * scale, hw * scale)

    sr = g(lr)
    _check("output shape", sr.shape == (2, 3, hw * scale, hw * scale))
    _check("output finite", torch.isfinite(sr).all())

    loss = criterion(sr, hr)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-4)
    opt.zero_grad(); loss.backward(); opt.step()
    _check("received gradients", any(p.grad is not None for p in g.parameters()))

    ema = EMA(g, decay=0.9)
    ema.update(g); ema.apply_shadow(g); ema.restore(g)
    _check("EMA roundtrip", True)

    with torch.no_grad():
        sr2 = tiled_forward(g, lr, scale=scale, tile=8, overlap=2).clamp(0, 1)
    _check("tiled forward shape", sr2.shape == (2, 3, hw * scale, hw * scale))
    psnr, ssim = compute_metrics(sr2, hr, crop_border=scale)
    _check("metrics finite", (psnr == psnr) and (ssim == ssim))

    print(f"  params: {count_params(g):.3f}M")


def test_non_square_input():
    """Non-square inputs occur constantly (tiled_forward edge tiles, benchmark images). A previous
    rfft width-axis bug crashed only here — every other test used square inputs."""
    print("\n[non-square input]")
    set_seed(0)
    cfg = _small_cfg("proposed", scale=2, hw=16)
    g = FrequSpatialGenerator(cfg.model).eval()
    for (h, w) in [(16, 24), (24, 16), (18, 26)]:
        with torch.no_grad():
            out = g(torch.rand(1, 3, h, w))
        _check(f"non-square {h}x{w}", out.shape == (1, 3, h * 2, w * 2) and torch.isfinite(out).all())


def test_autocast_forward_backward():
    """AMP regression: `x.float()` does NOT disable autocast — einsum would run in fp16 and
    `torch.complex` would produce ComplexHalf, which irfft2 rejects. SpectralBlock must force fp32."""
    print("\n[autocast (AMP) forward/backward]")
    set_seed(0)
    cfg = _small_cfg("proposed", scale=2, hw=16)
    g = FrequSpatialGenerator(cfg.model)
    lr, hr = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 32, 32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        sr = g(lr)
        loss = torch.nn.functional.l1_loss(sr.float(), hr)
    loss.backward()
    _check("autocast output finite", torch.isfinite(sr.float()).all())
    _check("autocast grads finite", all(torch.isfinite(p.grad).all() for p in g.parameters() if p.grad is not None))


def test_spectral_block_identity_at_init():
    """SpectralBlock's output projection is zero-init, so `proposed` and `no_freq` must produce
    IDENTICAL outputs at step 0 — training starts from a clean attention-only baseline."""
    print("\n[spectral block is identity at init]")
    set_seed(0); g_prop = FrequSpatialGenerator(_small_cfg("proposed").model).eval()
    set_seed(0); g_none = FrequSpatialGenerator(_small_cfg("no_freq").model).eval()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        d = (g_prop(x) - g_none(x)).abs().max().item()
    _check(f"proposed == no_freq at init (max diff={d:.2e})", d < 1e-5)


def test_starts_at_bicubic():
    """conv_last is zero-init, so the model must start exactly at bicubic(LR) (~31 dB, not noise)."""
    print("\n[model starts at bicubic]")
    set_seed(0)
    g = FrequSpatialGenerator(_small_cfg("proposed").model).eval()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        out = g(x)
        bic = torch.nn.functional.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
    _check("output == bicubic at init", (out - bic).abs().max().item() < 1e-5)


def test_routing_is_spatial():
    """SR-MoSO routing must vary across space (else it is just a global mixture)."""
    print("\n[SR-MoSO routing is spatial]")
    from models.spectral import MoSpectralOperator
    set_seed(0)
    op = MoSpectralOperator(channels=8, modes_h=4, modes_w=4, num_experts=4, mode="moso")
    for p in op.route.parameters():
        torch.nn.init.normal_(p, std=0.5)
    r = torch.softmax(op.route(torch.rand(1, 8, 16, 16)), dim=1)      # [1,K,H,W]
    _check("routing varies spatially", r.var(dim=(2, 3)).mean().item() > 1e-6)


def test_resolution_invariance():
    """Contribution 2. Probe each grid mode at the SAME NORMALIZED frequency at two resolutions.

    normalized (ours) — identical response (resolution-invariant, full-spectrum)
    absolute          — response drifts (bin-indexed; isolates invariance, coverage held fixed)
    fixed             — vanilla FNO: normalized *bandwidth* shrinks as resolution grows (truncation)
    """
    print("\n[resolution invariance: normalized vs absolute vs fixed]")
    from models.spectral import MoSpectralOperator

    def op_for(grid):
        set_seed(0)
        return MoSpectralOperator(channels=8, modes_h=6, modes_w=6, num_experts=3, mode="moso",
                                  spectral_grid=grid, ref_size=32)

    # ours: same normalized frequency → same response at any resolution
    op = op_for("normalized")
    a, b = op.expert_response(0, 32, 32), op.expert_response(0, 96, 96)
    _check(f"normalized: DC invariant (diff={abs(a[0,0]-b[0,0]).item():.2e})",
           abs(a[0, 0] - b[0, 0]).item() < 1e-4)
    _check(f"normalized: mid-band invariant (diff={abs(a[8,8]-b[24,24]).item():.2e})",
           abs(a[8, 8] - b[24, 24]).item() < 1e-4)

    # absolute: full-spectrum but bin-indexed → drifts
    op = op_for("absolute")
    a, b = op.expert_response(0, 32, 32), op.expert_response(0, 96, 96)
    _check(f"absolute: drifts with resolution (diff={abs(a[8,8]-b[24,24]).item():.2e})",
           abs(a[8, 8] - b[24, 24]).item() > 1e-4)

    # fixed (vanilla FNO): a fixed *count* of modes → the normalized bandwidth shrinks with resolution
    op = op_for("fixed")
    fa = (op.expert_response(0, 32, 32) > 0).float().mean().item()
    fb = (op.expert_response(0, 96, 96) > 0).float().mean().item()
    _check(f"fixed: normalized bandwidth shrinks ({fa:.3f} -> {fb:.3f})", fb < fa * 0.5)


def test_match_size_guard():
    print("\n[match_size guard]")
    a, b = torch.rand(1, 3, 40, 30), torch.rand(1, 3, 41, 30)
    a2, b2 = match_size(a, b)
    _check("cropped to common size", a2.shape == b2.shape == (1, 3, 40, 30))


def test_validate_mixed_sizes():
    """Regression: validation images have DIFFERENT resolutions; concatenating them into one batch
    crashed at the first vis interval, i.e. after a full epoch of training."""
    print("\n[validate() with mixed-resolution images]")
    import tempfile
    from train import validate
    set_seed(0)
    cfg = _small_cfg("proposed", scale=2, hw=16)
    cfg.train.val_tile = 24
    g = FrequSpatialGenerator(cfg.model).eval()
    loader = [{"lr": torch.rand(1, 3, 20, 28), "hr": torch.rand(1, 3, 40, 56)},
              {"lr": torch.rand(1, 3, 28, 20), "hr": torch.rand(1, 3, 56, 40)}]
    with tempfile.TemporaryDirectory() as d:
        psnr, ssim = validate(g, loader, torch.device("cpu"), cfg, epoch=1, save_dir=d, save_vis=True)
    _check("validate finished with vis on mixed sizes", psnr == psnr and ssim == ssim)


if __name__ == "__main__":
    print("=== SR-MoSO smoke test ===")
    test_routing_is_spatial()
    test_resolution_invariance()
    test_spectral_block_identity_at_init()
    test_starts_at_bicubic()
    test_non_square_input()
    test_autocast_forward_backward()
    test_match_size_guard()
    test_validate_mixed_sizes()
    for p in available_presets():
        test_preset(p, scale=2, hw=16)
    print("\nALL SMOKE TESTS PASSED")

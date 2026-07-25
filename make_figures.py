"""Generate the paper's mechanism figures from a trained checkpoint.

    py -3 make_figures.py --ckpt results/checkpoints/best.pth --image benchmarks/Urban100/HR/img_001.png
    py -3 make_figures.py --ckpt ... --baseline_ckpt results/ablations/x4/abs_grid/checkpoints/best.pth

Produces (into --out):
    routing.png        per-pixel routing weights r_k(p) + dominant-expert map  → experts specialize
    expert_spectra.png each expert's learned |D_k| over normalized frequency   → complementary bands
    erf.png            effective receptive field                              → genuinely global
    robustness.png     PSNR vs inference tile size (ours vs baseline)          → contribution 2
"""
import os
import argparse
import torch
from PIL import Image
from torchvision import transforms

from config import get_config, available_presets
from evaluate import load_model
from utils import (save_routing_maps, save_expert_spectra, save_erf,
                   eval_tile_robustness, save_tile_robustness_plot, set_seed)

_to_tensor = transforms.ToTensor()


def _load_lr_hr(path, scale):
    """HR image → (LR, HR) pair, HR cropped to be divisible by scale."""
    hr = Image.open(path).convert("RGB")
    w, h = hr.size
    hr = hr.crop((0, 0, w - w % scale, h - h % scale))
    w, h = hr.size
    lr = hr.resize((w // scale, h // scale), Image.BICUBIC)
    return _to_tensor(lr).unsqueeze(0), _to_tensor(hr).unsqueeze(0)


def main():
    p = argparse.ArgumentParser(description="Generate SR-MoSO mechanism figures")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--preset", type=str, default="proposed", choices=available_presets())
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--image", type=str, required=True, help="an HR image (LR is derived by bicubic)")
    p.add_argument("--out", type=str, default="figures")
    p.add_argument("--stage", type=int, default=-1, help="which interleaved SR-MoSO stage to visualize")
    p.add_argument("--baseline_ckpt", type=str, default=None,
                   help="checkpoint of a resolution-DEPENDENT arm (abs_grid / fixed_grid_fno) for the "
                        "tile-robustness comparison")
    p.add_argument("--baseline_preset", type=str, default="abs_grid", choices=available_presets())
    p.add_argument("--tiles", type=int, nargs="*", default=[64, 128, 256])
    args = p.parse_args()

    set_seed(0)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = get_config(args.preset); cfg.model.scale = args.scale; cfg.validate()
    model = load_model(args.ckpt, cfg, device)
    lr, hr = _load_lr_hr(args.image, args.scale)
    lr, hr = lr.to(device), hr.to(device)

    made = []
    for fn, path in ((lambda: save_routing_maps(model, lr, os.path.join(args.out, "routing.png"),
                                                stage=args.stage), "routing.png"),
                     (lambda: save_expert_spectra(model, os.path.join(args.out, "expert_spectra.png"),
                                                  stage=args.stage), "expert_spectra.png"),
                     (lambda: save_erf(model, lr, os.path.join(args.out, "erf.png")), "erf.png")):
        made.append((path, fn() is not None))

    # contribution-2 figure: quality vs inference tile size
    tiles = tuple(args.tiles) + (None,)     # None = whole image, no tiling
    results = {"ours (normalized)": eval_tile_robustness(model, lr, hr, args.scale, tiles=tiles)}
    if args.baseline_ckpt:
        bcfg = get_config(args.baseline_preset); bcfg.model.scale = args.scale; bcfg.validate()
        base = load_model(args.baseline_ckpt, bcfg, device)
        results[args.baseline_preset] = eval_tile_robustness(base, lr, hr, args.scale, tiles=tiles)
    save_tile_robustness_plot(results, os.path.join(args.out, "robustness.png"))
    made.append(("robustness.png", True))

    for name, ok in made:
        print(f"  {'wrote' if ok else 'skipped (n/a for this preset)':<8} {name}")
    print("\ntile-robustness PSNR (dB):")
    for name, res in results.items():
        print(f"  {name:22s} " + "  ".join(f"{k}:{v['psnr']:.3f}" for k, v in res.items()))


if __name__ == "__main__":
    main()

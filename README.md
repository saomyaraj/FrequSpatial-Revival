# SR-MoSO: Spatially-Routed Mixture of Spectral Operators for Lightweight Super-Resolution

A lightweight (~0.93M-param, PSNR-oriented) single-image SR network. Window attention has a bounded
**local** receptive field; SR-MoSO supplies the complementary adaptive-**global** half — a per-pixel,
content-routed mixture of *global* spectral operators, interleaved into the trunk so it acts on deep
features at multiple depths.

## Contributions

```text
out(p) = Σ_k r_k(p) · iFFT( M · D_k ⊙ FFT(x) )
```

1. **SR-MoSO — routed mixture of global spectral operators.** K global experts (`D_k`, complex
   diagonals; `M`, a shared complex channel-mix) combined by dense **per-pixel routing**
   `r(p) = softmax(conv(x))` predicted from the local features at that depth: content decides *which*
   global operator applies *where*. One SR-MoSO stage follows every local stage, so the spectral
   response is composed with attention and nonlinearity rather than bolted on as a side branch.
2. **Resolution-invariant parameterization → measured cross-resolution robustness.** Each expert lives
   on a canonical **normalized-frequency** grid, sampled to the live rfft grid every forward, so the
   same normalized frequency gets the same response at any resolution (a plain FNO/GFNet indexes the
   discrete grid, so its cutoff drifts). Two ablation arms isolate this: `abs_grid` (same
   full-spectrum coverage, but **absolute-bin** indexed → resolution-dependent) and `fixed_grid_fno`
   (vanilla FNO mode truncation).
3. **Generality — SR-MoSO as a drop-in module.** Swapping the Swin trunk for a plain residual CNN at
   matched budget (`edsr_base` 0.63M → `edsr_moso` 0.94M, the same +0.31M SR-MoSO adds to
   `no_freq` 0.62M → `proposed` 0.93M) tests whether the operator helps a non-transformer backbone too.

Supporting: efficiency-frontier PSNR/SSIM at ≤1M params with **FLOPs and latency** reported, plus
interpretability (routing maps, expert spectra, effective receptive field).

**Positioning (honest).** SR-MoSO is a recombination made specific to SR: conditional/dynamic
convolution (CondConv/DynamicConv) whose experts are resolution-invariant *global spectral* filters
(FNO/GFNet/AFNO family), routed densely per pixel. It must be read against **SwinFIR** (Swin + a
*static* Fourier block — the `static_fno` row is that baseline, and routing must beat it), **FDConv**
(spatially-variant per-band *weight* modulation, but local and for detection), and **SRNO** (a single
neural operator for SR). The resolution-invariant parameterization is methodology (cf.
continuous/SIREN-FNO operators), not a separate claim. The case is empirical — see the gate below.
Note also that Hermitian symmetry of the learned filter is not enforced, so `irfft2` applies the
Hermitian projection of the parameterized operator (standard in FNO/GFNet).

## Architecture

```text
LR ─ conv3x3 ─┬─────────────────────────────────────────────────┐  (shallow residual)
              └─ [ RSTB (local) → SpectralBlock (global) ] × N ──┴─ LayerNorm2d ─ conv3x3
                 ─ PixelShuffle(scale) ─ conv3x3 ─ + bicubic(LR) ─ SR
```

`SpectralBlock` = `norm → MoSpectralOperator → 1x1`, added residually with the projection
**zero-initialized**, so every block is an exact identity at step 0: `proposed` and `no_freq` produce
identical outputs at init, and `conv_last` is zero-init too so the model starts exactly at
`bicubic(LR)`. FFT/complex math is forced to fp32 inside the block (`autocast` casts per-op, so
`einsum` would otherwise yield ComplexHalf, which `irfft2` rejects).

## Repository layout (flat)

```text
config.py         # all hyperparameters + ablation presets
train.py          # training (L1, iteration-based schedule, fast validation, live diagnostics)
evaluate.py       # benchmark eval → PSNR/SSIM + params/FLOPs/latency → CSV/JSON
run_ablations.py  # ablation sweep (train + eval each preset) AND results aggregation (--aggregate_only)
make_figures.py   # routing maps, expert spectra, ERF, tile-robustness
test_smoke.py     # CPU smoke test: all presets + regression tests (non-square, AMP, mixed-size val, identity-at-init)
models/           # generator, trunk (interleaving), spectral (SR-MoSO + diagnostics), swin, backbones, common
utils/            # metrics (Y+shave, MATLAB SSIM), tiling, EMA/checkpoint, misc (FLOPs/latency/manifest/logging), robustness, visualization
data/             # dataset + dataloaders (virtual epochs, sub-image aware) + prepare.py (download & sub-images)
```

## Model size (x4)

| preset | params |
|---|---|
| `proposed` | 0.93M |
| `no_freq` (SwinIR-like) | 0.62M |
| `conv_block` (param-matched local control) | 0.94M |

`proposed` is 0.84M / 0.95M / 0.93M at x2 / x3 / x4 — comparable to OmniSR (0.79M),
SRFormer-light (0.87M), ATD-light (0.77M). Sizing knobs: `base_channels` (÷ `swin_num_heads`),
`num_rstb`, `num_swin_per_rstb`, `ccso_experts`, `fno_modes_h/w` (canonical grid).

## Getting the data

```bash
# DIV2K (train + valid, HR + LR bicubic) — downloads and unpacks, then cuts sub-images.
# ~4.5 GB for one scale. Skips anything already present, so it is safe to re-run.
python -m data.prepare --download --scale 4
```

Direct links if you prefer to fetch manually (from
[the DIV2K page](https://data.vision.ee.ethz.ch/cvl/DIV2K/)): `DIV2K_train_HR.zip`,
`DIV2K_valid_HR.zip`, `DIV2K_train_LR_bicubic_X4.zip`, `DIV2K_valid_LR_bicubic_X4.zip`.

**DIV2K alone is enough to train and to run the falsification gate.** The benchmark sets
(Set5/Set14/BSD100/Urban100/Manga109) are only needed for the final comparison table; grab them from
the BasicSR / KAIR (SwinIR) release links or a HuggingFace mirror and drop them under `benchmarks/`.
`evaluate.py` needs only the **HR** images — it derives LR by PIL bicubic when no LR set is present.
For numbers directly comparable to published tables, supply the official LR sets (they use MATLAB
bicubic, which differs marginally).

## Quick start

```bash
pip install -r requirements.txt

# runs with NO data — every preset + the regression tests
python test_smoke.py

# get data (see above)
python -m data.prepare --download --scale 4

# train the proposed model (pure L1), x4, DIV2K under ./DIV2K
python train.py
python train.py --scale 2 --bs 32
python train.py --wandb             # + remote monitoring
python train.py --resume            # resume is explicit; a bare re-run trains fresh

# benchmark eval (EMA + tiled inference; add --self_ensemble for x8)
python evaluate.py --ckpt results/checkpoints/best.pth --scale 4 --out_json results/eval.json
```

## Monitoring a run

Each epoch prints, and appends one JSON line to `<save_dir>/metrics.jsonl`:

```text
epoch [007/500] L1 0.0231 | PSNR 29.412 dB | SSIM 0.8331 | contrib 0.065 | H(route) 0.909 | LR 2.00e-04 | 214s | ETA 29h 18m
```

Two of those are **SR-MoSO health checks** — they monitor the paper's kill-criteria live, so you learn
within an hour whether the mechanism is working instead of after a week:

| field | meaning | what to worry about |
|---|---|---|
| `contrib` | ‖spectral block output‖ / ‖input‖ | starts at exactly **0** (zero-init). If it stays ~0, the spectral path never grew — the mechanism is inert. |
| `H(route)` | routing entropy, normalized to [0,1] | starts at exactly **1.0** (uniform). If it stays 1.0, routing never specialized and SR-MoSO has degenerated into the `no_routing` mixture. |

Per-stage values and per-expert usage (spots dead experts) are in `metrics.jsonl` under
`moso/s{i}/...`. Also written: `checkpoints/{latest,best,epoch_NNN}.pth` (generator + optimizer +
scheduler + EMA, so resume is exact), `training_curves.png`, periodic `vis_epoch_*.png`, and
`run_manifest.json`.

`--wandb` adds the same scalars to Weights & Biases. Worth it here: single-GPU runs are long, and
comparing 13 ablation presets on shared axes (especially `contrib` / `H(route)` across arms) is much
easier than reading 13 JSONL files. `run_ablations.py --wandb` enables it for every run in the sweep.
It is entirely optional — `metrics.jsonl` holds the same data with no account needed.

## Ablations

One preset == one ablation-table row.

| preset | isolates |
|---|---|
| `proposed` | the full model (SR-MoSO, K=4, routed, interleaved) |
| `no_freq` | local-only (SwinIR-like) baseline |
| `conv_block` | **param-matched LOCAL control** — global spectral operator vs merely extra capacity |
| `static_fno` | single **static** spectral op (SwinFIR-analog; routing must beat this) |
| `no_routing` | K experts, uniform average (isolates **routing**) |
| `ccso_global` | content gain, spatially-invariant (isolates **per-pixel**) |
| `per_expert_mix` | per-expert channel-mix (is the shared mix limiting? — "weak MoE" check) |
| `abs_grid` | full-spectrum but absolute-bin indexed → **isolates resolution-invariance** |
| `fixed_grid_fno` | vanilla FNO/GFNet mode truncation |
| `k2`, `k8` | expert-count sweep (proposed uses K=4) |
| `edsr_base`, `edsr_moso` | generality: SR-MoSO on a non-Swin CNN backbone at matched budget |

```bash
# full sweep (train + eval each preset), then auto-aggregate into a table
python run_ablations.py --scale 4 --evaluate

# the decisive rows first — the cheapest falsification of the thesis
python run_ablations.py --presets proposed conv_block static_fno no_freq --evaluate --epochs 60

# aggregate an existing sweep
python run_ablations.py --aggregate_only --scale 4
```

**Falsify first.** Before committing to a long run, confirm on a short schedule that `proposed` beats
**`conv_block`** (not merely `no_freq` — that would only prove extra capacity helps) and `static_fno`,
and that routing maps specialize (they start uniform: the routing head is zero-init). If
`proposed ≈ conv_block`, the spectral operator adds nothing beyond capacity — stop.

## Metric protocol

PSNR/SSIM on the **Y (luminance) channel** with a **`scale`-pixel border shave** (EDSR / SwinIR / HAT).
SSIM uses an 11×11 Gaussian window (σ=1.5) with population covariance (MATLAB convention). FLOPs and
latency are reported at the conventional **1280×720 output**; the FLOPs count includes an **analytic
FFT term**, because fvcore has no handler for `fft_rfft2`/`fft_irfft2` and would otherwise undercount
the headline module. Report single-forward and self-ensemble (`+`) numbers **separately**. Keep the
inference tile size identical between training-time validation and evaluation — SR-MoSO is a *global*
operator, so tile size changes its effective receptive field.

## Mechanism figures

```bash
python make_figures.py --ckpt results/checkpoints/best.pth \
    --image benchmarks/Urban100/HR/img_001.png \
    --baseline_ckpt results/ablations/x4/abs_grid/checkpoints/best.pth
```

Produces `routing.png` (experts specialize spatially), `expert_spectra.png` (complementary /
anisotropic bands), `erf.png` (genuinely global receptive field vs local methods), and
`robustness.png` (PSNR vs inference tile size — ours flat, `abs_grid` drifts: contribution 2).

## Data layout

```text
DIV2K/
  DIV2K_train_HR/*.png
  DIV2K_train_LR_bicubic/X{2,3,4}/*.png
  DIV2K_valid_HR/*.png
  DIV2K_valid_LR_bicubic/X{2,3,4}/*.png
  # created by `python -m data.prepare`:
  DIV2K_train_HR_sub/*.png
  DIV2K_train_LR_bicubic/X{2,3,4}_sub/*.png

benchmarks/                       # for evaluate.py (HR required; LR optional)
  Set5/HR/*.png  [+ Set5/LR_bicubic/X4/*.png]
  Set14/ ...  BSD100/ ...  Urban100/ ...  Manga109/ ...
```

## Reproducibility

`train.py` seeds all RNGs and writes `run_manifest.json` (resolved config, git commit, **`git_dirty`
flag**, torch/CUDA versions) per run — commit before a real run so the manifest actually reproduces it.
Training is iteration-based (`iters_per_epoch × num_epochs` ≈ 500K, the lightweight-SR convention).
EMA weights are used for validation/eval, and the EMA shadow is re-seeded when EMA starts so the
random init never contaminates reported PSNR. Checkpoints store generator/optimizer/scheduler/EMA
state for exact resume.

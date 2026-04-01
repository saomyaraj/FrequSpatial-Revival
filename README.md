# Hybrid Frequency-Spatial Super-Resolution with Cross-Domain Attention

## Contributions

### 1. Cross-Domain Attention Module (CDAM)

Bidirectional windowed cross-attention between spatial and frequency feature maps.

- Spatial features query frequency space → pull in long-range harmonic context
- Frequency features query spatial space → anchor frequency info to local structure
- Windowed implementation (shared with Swin) keeps complexity O(N·ws²)
- Both directions share relative position bias → spatial coherence is preserved

Prior work (e.g., SwinIR, FRAN) uses late-stage concatenation or element-wise fusion. CDAM enables explicit information exchange before fusion.

### 2. Multi-Scale Phase-Coherence Loss (MSPCL)

Phase loss computed at 3 spatial scales (full, ½, ¼ resolution). Phase encodes WHERE structures are; magnitude encodes how strong they are. Multi-scale enforcement recovers details at coarse AND fine granularities.

### 3. FNO-Style Spectral Mixing

Learnable complex weight mixing of top-k frequency modes. Captures global periodicities unreachable by any local convolution. Parameterizes only top-k modes -> efficient, acts as implicit low-pass prior.

### 4. Adaptive Band Decomposition

Spectrum split into low/mid/high bands with dedicated sub-networks. Learned band importance weights (softmax) → model decides how much each band contributes.

## Quick Start

```bash
# install dependencies
pip install torch torchvision timm scikit-image matplotlib tqdm wandb

# smoke test (cpu, <60s)
cd version_2
python test_smoke.py

# train (default: scale x4, DIV2K)
python train.py

# train with overrides
python train.py --scale 2 --bs 16 --epochs 300

# resume
python train.py --resume
```

## Metrics Protocol

PSNR and SSIM are computed on the Y channel (luminance) only, following the standard SR benchmark protocol used in EDSR, SwinIR, HAT. This matches reported numbers in the literature — do not use RGB PSNR when comparing to published results.

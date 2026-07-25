"""centralized config - all hyperparameters live here. Change once → propagates everywhere"""

from dataclasses import dataclass, field
from typing import List

# model config
@dataclass
class ModelConfig:
    # global
    scale: int = 4               # SR upscale factor: 2, 3, or 4
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 48      # must be divisible by swin_num_heads (lightweight tier: ~0.98M total)

    # spatial branch
    spatial_backbone: str = "swin"  # "swin" (SwinIR-style RSTB) | "edsr" (plain residual CNN, generality ablation)
    num_rstb: int = 3            # no of residual swin transformer blocks (swin backbone)
    num_swin_per_rstb: int = 6   # SwinTransformerBlocks per RSTB (swin backbone)
    edsr_num_blocks: int = 3     # ResBlocks per local stage (edsr backbone; sized to match the
                                 # Swin trunk's budget so the generality rows stay comparable)
    swin_window_size: int = 8    # window size for window attention
    swin_num_heads: int = 6      # attention heads (base_channels // swin_num_heads = head_dim)
    swin_mlp_ratio: float = 2.0
    swin_qkv_bias: bool = True
    swin_drop_rate: float = 0.0
    swin_attn_drop_rate: float = 0.0
    swin_drop_path_rate: float = 0.1  # stochastic depth rate

    # ── SR-MoSO global stage (headline contribution) ──
    # Each expert is a learned complex diagonal on an fno_modes_h × fno_modes_w canonical grid,
    # sampled to the live rfft grid at runtime (resolution-invariant, full-spectrum, anisotropic).
    # One SR-MoSO stage is interleaved after each local stage.
    fno_modes_h: int = 14        # canonical grid height (normalized freq f_h ∈ [-½,½])
    fno_modes_w: int = 14        # canonical grid width  (normalized freq f_w ∈ [0,½])
    freq_use_ccso: bool = True       # if False → static (single, unconditioned) spectral operator
    ccso_mode: str = "moso"          # "moso" (routed mixture) | "mixture" | "global" | "static"
    ccso_experts: int = 4            # number of spectral experts K (mode="moso"/"mixture")
    ccso_per_expert_mix: bool = False  # each expert gets its own complex channel-mix (ablation)
    # canonical-grid indexing — the resolution-invariance ablation axis:
    #   "normalized" (ours) resolution-invariant | "absolute" full-spectrum but bin-indexed
    #   (resolution-dependent; isolates invariance) | "fixed" vanilla FNO/GFNet mode truncation
    spectral_grid: str = "normalized"
    spectral_ref_size: int = 64       # reference resolution for "absolute" indexing (= train patch size)

    # ── ablation switches (all honored by the trunk) ──
    use_freq_branch: bool = True     # if False → local-only (SwinIR-like baseline)
    spectral_stage: str = "moso"     # "moso" (proposed) | "conv" (param-matched LOCAL control)


# training config
@dataclass
class TrainConfig:
    # data
    data_root: str = "DIV2K"
    patch_size: int = 64         # LR patch size for training
    num_workers: int = 4

    # reproducibility
    seed: int = 42
    deterministic: bool = False  # set True for fully deterministic (slower) runs

    # training schedule
    # Iteration-based convention (lightweight-SR standard ≈ 500K iters). One "epoch" is a fixed
    # `iters_per_epoch` random-patch steps, so total iters = num_epochs * iters_per_epoch.
    batch_size: int = 32         # lightweight-SR convention (32–64); lower via --bs if VRAM-limited
    iters_per_epoch: int = 1000  # steps per epoch (virtual epoch; None → one pass over the images)
    num_epochs: int = 500        # 500 * 1000 = 500K iters
    warmup_epochs: int = 5       # linear LR warmup (= 5K iters)

    # optimizers
    lr_g: float = 2e-4           # generator learning rate
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    gradient_clip_norm: float = 0.5

    # EMA
    ema_decay: float = 0.999
    ema_start_epoch: int = 5     # start EMA after this epoch
    ema_update_every: int = 1    # update EMA every N generator steps (amortize)
    ema_cpu_offload: bool = False  # keep EMA shadow on CPU (low-VRAM only; slower)

    # AMP
    use_amp: bool = True

    # validation (fast subset each epoch; full valid set periodically & at the end)
    val_subset: int = 10         # #valid images for the per-epoch fast validation (0 → full set)
    val_tile: int = 256          # LR tile size for validation tiled inference
    full_val_interval: int = 25  # run the FULL valid set every N epochs (and always at the end)

    # checkpointing & logging
    save_dir: str = "results"
    checkpoint_interval: int = 10
    vis_interval: int = 5
    use_wandb: bool = False
    wandb_project: str = "sr-moso"
    wandb_entity: str = ""       # set your wandb entity if needed


# top-level config (single object passed everywhere)
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        self.validate()

    def validate(self):
        m = self.model
        assert m.base_channels % m.swin_num_heads == 0, (
            f"base_channels ({m.base_channels}) must be divisible by swin_num_heads ({m.swin_num_heads})")
        assert m.scale in [2, 3, 4], (f"scale must be 2, 3, or 4, got {m.scale}")
        assert m.ccso_mode in ("moso", "mixture", "global", "static"), (
            f"ccso_mode must be 'moso', 'mixture', 'global', or 'static', got {m.ccso_mode}")
        assert m.ccso_experts >= 1, f"ccso_experts must be >= 1, got {m.ccso_experts}"
        assert m.spatial_backbone in ("swin", "edsr"), (
            f"spatial_backbone must be 'swin' or 'edsr', got {m.spatial_backbone}")
        assert m.spectral_grid in ("normalized", "absolute", "fixed"), (
            f"spectral_grid must be 'normalized', 'absolute', or 'fixed', got {m.spectral_grid}")
        assert m.spectral_stage in ("moso", "conv"), (
            f"spectral_stage must be 'moso' or 'conv', got {m.spectral_stage}")
        assert self.train.num_epochs > self.train.warmup_epochs, (
            f"num_epochs ({self.train.num_epochs}) must exceed warmup_epochs ({self.train.warmup_epochs})")


# ── ablation presets ──────────────────────────────────────────────────────────
# One preset == one ablation-table row.
_PRESETS = {
    # the proposed model: SR-MoSO routed mixture, interleaved after every local stage
    "proposed":        {},
    # ── does the GLOBAL SPECTRAL operator earn its place? ──
    "no_freq":         {"use_freq_branch": False},          # local-only (SwinIR-like) baseline
    "conv_block":      {"spectral_stage": "conv"},          # param-matched LOCAL control: isolates
                                                            # "global spectral" from "extra capacity"
    # ── which part of SR-MoSO matters? ──
    "static_fno":      {"freq_use_ccso": False},            # single static spectral op (SwinFIR-analog)
    "no_routing":      {"ccso_mode": "mixture"},            # K experts, uniform avg (isolates routing)
    "ccso_global":     {"ccso_mode": "global"},             # content gain, spatially-invariant
    "per_expert_mix":  {"ccso_per_expert_mix": True},       # per-expert channel mix (weak-MoE check)
    # ── resolution-invariance (contribution 2) ──
    "abs_grid":        {"spectral_grid": "absolute"},       # full-spectrum but bin-indexed → isolates
                                                            # resolution-invariance alone
    "fixed_grid_fno":  {"spectral_grid": "fixed"},          # vanilla FNO/GFNet mode truncation
    # ── expert-count sweep (proposed uses K=4) ──
    "k2":              {"ccso_experts": 2},
    "k8":              {"ccso_experts": 8},
    # ── generality (contribution 3): SR-MoSO dropped into a non-Swin CNN backbone ──
    "edsr_base":       {"spatial_backbone": "edsr", "use_freq_branch": False},
    "edsr_moso":       {"spatial_backbone": "edsr"},
}


def get_config(preset: str = "proposed") -> Config:
    """return a config for the given ablation preset. Override fields as needed afterwards."""
    if preset not in _PRESETS:
        raise ValueError(f"unknown preset '{preset}'. choose from {list(_PRESETS)}")
    cfg = Config()
    for k, v in _PRESETS[preset].items():
        setattr(cfg.model, k, v)
    cfg.validate()
    return cfg


def available_presets() -> List[str]:
    return list(_PRESETS)

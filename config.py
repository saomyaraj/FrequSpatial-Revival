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
    base_channels: int = 60      # must be divisible by swin_num_heads

    # spatial branch(SwinIR-style RSTB)
    num_rstb: int = 6            # no of residual swin transformer blocks
    num_swin_per_rstb: int = 6   # SwinTransformerBlocks per RSTB
    swin_window_size: int = 8    # window size for window attention
    swin_num_heads: int = 6      # attention heads (base_channels // swin_num_heads = head_dim)
    swin_mlp_ratio: float = 2.0
    swin_qkv_bias: bool = True
    swin_drop_rate: float = 0.0
    swin_attn_drop_rate: float = 0.0
    swin_drop_path_rate: float = 0.1  # stochastic depth rate

    # frequency branch
    # FNO-style global mixing: learn to mix top-k frequency modes
    fno_modes_h: int = 16        # top-k modes to keep along height
    fno_modes_w: int = 16        # top-k modes to keep along width
    num_freq_blocks: int = 4     # frequency residual processing blocks per band

    # adaptive band decomposition: split spectrum into N bands
    freq_bands: int = 3          # low / mid / high

    # cross-domain attention module (CDAM)
    num_cdam_layers: int = 2     # no of CDAM layers
    cdam_num_heads: int = 6      # cross-attention heads
    cdam_window_size: int = 8    # window size for windowed cross-attention
    cdam_mlp_ratio: float = 2.0
    cdam_drop_rate: float = 0.0

    # fusion
    use_channel_refinement: bool = True   # channel attention after fusion

    # discriminator (U-Net with spectral norm)
    disc_base_channels: int = 64
    disc_num_layers: int = 4


# training config
@dataclass
class TrainConfig:
    # data
    data_root: str = "DIV2K"
    patch_size: int = 64         # LR patch size for training
    aug_probability: float = 0.5
    num_workers: int = 4

    # training schedule
    batch_size: int = 8
    num_epochs: int = 200
    warmup_epochs: int = 5       # linear LR warmup

    # optimizers
    lr_g: float = 2e-4           # generator learning rate
    lr_d: float = 1e-4           # discriminator learning rate
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    gradient_clip_norm: float = 0.5

    # EMA
    ema_decay: float = 0.999
    ema_start_epoch: int = 5     # start EMA after this epoch

    # AMP
    use_amp: bool = True

    # loss weights
    l1_weight: float = 1.0
    perceptual_weight: float = 0.1
    freq_weight: float = 0.05
    adv_weight: float = 0.005

    # perceptual loss VGG layers (conv1_2, conv2_2, conv3_4, conv4_4, conv5_4)
    vgg_layers: List[int] = field(default_factory=lambda: [2, 7, 16, 25, 34])

    # checkpointing & logging
    save_dir: str = "results"
    checkpoint_interval: int = 10
    vis_interval: int = 5
    log_interval: int = 100      # batches between logs
    use_wandb: bool = False
    wandb_project: str = "FrequSpatial-V2"
    wandb_entity: str = ""       # set your wandb entity if needed


# top-level config (single object passed everywhere)
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        # validation
        assert self.model.base_channels % self.model.swin_num_heads == 0, (f"base_channels ({self.model.base_channels}) must be divisible by " f"swin_num_heads ({self.model.swin_num_heads})")
        assert self.model.base_channels % self.model.cdam_num_heads == 0, (f"base_channels ({self.model.base_channels}) must be divisible by " f"cdam_num_heads ({self.model.cdam_num_heads})")
        assert self.model.scale in [2, 3, 4], (f"scale must be 2, 3, or 4, got {self.model.scale}")
        assert self.model.fno_modes_h <= self.train.patch_size // 2, (f"fno_modes_h ({self.model.fno_modes_h}) must be <= patch_size//2 " f"({self.train.patch_size // 2})")


def get_config() -> Config:
    """return a default config. Override fields as needed"""
    return Config()

"""DIV2K Dataset - augmentation pipeline: flips + 90 deg rotations (all structure-preserving)"""

import os
import glob
import random
from typing import Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import TrainConfig


class DIV2KDataset(Dataset):
    def __init__(self, root: str, split: str = "train", scale: int = 4, patch_size: int = 64,
                 augment: bool = True, epoch_size: Optional[int] = None): # LR patch size (None = full image for validation)
        self.split = split
        self.scale = scale
        self.patch_size = patch_size  # LR patch size
        self.augment = augment and (split == "train")

        hr_dir = os.path.join(root, f"DIV2K_{split}_HR")
        lr_dir = os.path.join(root, f"DIV2K_{split}_LR_bicubic", f"X{scale}")

        # prefer pre-cut sub-images for training (see data/prepare.py): decoding a 480px tile instead
        # of a full 2K PNG for every 64px crop is the difference between a fast and a slow epoch.
        self.using_sub = False
        if split == "train":
            hr_sub = os.path.join(root, "DIV2K_train_HR_sub")
            lr_sub = os.path.join(root, "DIV2K_train_LR_bicubic", f"X{scale}_sub")
            if os.path.isdir(hr_sub) and os.path.isdir(lr_sub):
                hr_dir, lr_dir, self.using_sub = hr_sub, lr_sub, True

        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        if not self.hr_paths:
            raise RuntimeError(f"no images found in {hr_dir}. check data_root "
                               f"(get the data with: python -m data.prepare --download)")

        self.lr_dir = lr_dir
        self.to_tensor = transforms.ToTensor()
        # virtual epoch length: with random-crop training we sample `epoch_size` random patches per
        # epoch (decoupling epoch length from #images) so one epoch == a fixed number of iterations.
        self.epoch_size = epoch_size if (epoch_size and split == "train") else None
        ps_str = f"patch_size(LR)={patch_size}" if patch_size else "full images"
        es_str = f" | epoch_size={self.epoch_size}" if self.epoch_size else ""
        sub_str = " | sub-images" if self.using_sub else ""
        print(f"[DIV2K] {split}: {len(self.hr_paths)} images | scale={scale} | {ps_str}{es_str}{sub_str}")

    def __len__(self):
        return self.epoch_size if self.epoch_size else len(self.hr_paths)

    def __getitem__(self, idx: int):
        hr_path = self.hr_paths[idx % len(self.hr_paths)]
        img_id  = os.path.splitext(os.path.basename(hr_path))[0]
        # sub-images share the HR filename; full DIV2K LR files carry an `x{scale}` suffix
        lr_name = f"{img_id}.png" if self.using_sub else f"{img_id}x{self.scale}.png"
        lr_path = os.path.join(self.lr_dir, lr_name)

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        if self.split == "train" and self.patch_size is not None:
            lr, hr = self._random_crop(lr, hr)
        if self.augment:
            lr, hr = self._augment(lr, hr)

        return {"lr": self.to_tensor(lr), "hr": self.to_tensor(hr)}

    def _random_crop(self, lr: Image.Image, hr: Image.Image):
        lw, lh = lr.size
        ps_lr  = self.patch_size
        ps_hr  = ps_lr * self.scale

        if lw < ps_lr or lh < ps_lr:
            # resize small images rather than crash
            lr = lr.resize((ps_lr, ps_lr), Image.BICUBIC)
            hr = hr.resize((ps_hr, ps_hr), Image.BICUBIC)
            return lr, hr

        x = random.randint(0, lw - ps_lr)
        y = random.randint(0, lh - ps_lr)
        lr = lr.crop((x, y, x + ps_lr, y + ps_lr))
        hr = hr.crop((x * self.scale, y * self.scale, (x + ps_lr) * self.scale, (y + ps_lr) * self.scale))
        return lr, hr

    def _augment(self, lr: Image.Image, hr: Image.Image):
        """flips + 90-degree rotations, all exact (pixel permutations).

        Uses `transpose(ROTATE_*)` rather than `rotate()`: `rotate` is an affine resample that is only
        exact on the square fast path, and on a non-square image it would resample LR and HR with
        different sub-pixel offsets — silently misaligning the training pair."""
        if random.random() < 0.5:
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
        rot = random.choice([None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
        if rot is not None:
            lr = lr.transpose(rot)
            hr = hr.transpose(rot)
        return lr, hr


def get_dataloaders(cfg: TrainConfig, scale: int):
    from torch.utils.data import DataLoader

    epoch_size = cfg.iters_per_epoch * cfg.batch_size if cfg.iters_per_epoch else None
    train_ds = DIV2KDataset(cfg.data_root, "train", scale, cfg.patch_size, augment=True, epoch_size=epoch_size)
    val_ds = DIV2KDataset(cfg.data_root, "valid", scale, patch_size=None, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True,)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers,)
    return train_loader, val_loader

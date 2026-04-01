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
    def __init__(self, root: str, split: str = "train", scale: int = 4, patch_size: int = 64, augment: bool = True,): # LR patch size (None = full image for validation)
        self.split = split
        self.scale = scale
        self.patch_size = patch_size  # LR patch size
        self.augment = augment and (split == "train")

        hr_dir = os.path.join(root, f"DIV2K_{split}_HR")
        lr_dir = os.path.join(root, f"DIV2K_{split}_LR_bicubic", f"X{scale}")

        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        if not self.hr_paths:
            raise RuntimeError(f"no images found in {hr_dir}. check data_root")

        self.lr_dir = lr_dir
        self.to_tensor = transforms.ToTensor()
        ps_str = f"patch_size(LR)={patch_size}" if patch_size else "full images"
        print(f"[DIV2K] {split}: {len(self.hr_paths)} images | scale={scale} | {ps_str}")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx: int):
        hr_path = self.hr_paths[idx]
        img_id  = os.path.splitext(os.path.basename(hr_path))[0]
        lr_path = os.path.join(self.lr_dir, f"{img_id}x{self.scale}.png")

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
        if random.random() < 0.5:
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.choice([0, 1, 2, 3])
        if k:
            lr = lr.rotate(90 * k)
            hr = hr.rotate(90 * k)
        return lr, hr


def get_dataloaders(cfg: TrainConfig, scale: int):
    from torch.utils.data import DataLoader

    train_ds = DIV2KDataset(cfg.data_root, "train", scale, cfg.patch_size, augment=True)
    val_ds = DIV2KDataset(cfg.data_root, "valid", scale, patch_size=None, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True,)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers,)
    return train_loader, val_loader

# Frequ-Spatial: Hybrid Frequency-Spatial Image Super-Resolution

A U-Net based image super-resolution model that combines frequency and spatial domain processing for enhanced image restoration. This project implements a hybrid network architecture that uses both frequency and spatial domain features to achieve SOTA super-resolution results.

## Features

- Hybrid architecture combining frequency and spatial domain processing
- Dual-branch network:
  - Spatial branch with U-Net for capturing local & global features
  - Frequency branch using differentiable FFT to process magnitude & phase
- Spatial and Channel Attention for better feature fusion across domains
- Adaptive fusion module with residual connections to preserve detail
- Multi-scale feature extraction for better representation at different resolutions
- loss functions:
  - Pixel-wise losses: L1, L2
  - Perceptual loss using pretrained VGG
  - Adversarial loss via GAN setup for realism
- Mixed precision training for improved memory & speed
- Automatic checkpointing and early stopping to prevent overfitting

## Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU training)
- Other dependencies:

  ```bash
  pip install torch torchvision numpy matplotlib tqdm wandb scikit-image opencv-python
  ```

## Dataset Setup

### DIV2K Dataset Structure

The project uses the DIV2K dataset for training and validation. The dataset should be organized as follows:

```
DIV2K/
├── DIV2K_train_HR/           # Training high-resolution images
├── DIV2K_train_LR_bicubic/   # Training low-resolution images
│   └── X2/                   # x2 downscaled images
├── DIV2K_valid_HR/           # Validation high-resolution images
└── DIV2K_valid_LR_bicubic/   # Validation low-resolution images
    └── X2/                   # x2 downscaled images
```

### Download Instructions

1. Visit the [DIV2K dataset website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
2. Download the following files:
   - Training HR images: `DIV2K_train_HR.zip`
   - Training LR images (bicubic x2): `DIV2K_train_LR_bicubic_X2.zip`
   - Validation HR images: `DIV2K_valid_HR.zip`
   - Validation LR images (bicubic x2): `DIV2K_valid_LR_bicubic_X2.zip`

3. Extract all zip files into a single directory named `DIV2K`

## Configuration

The model can be configured through the `Config` class in `train.py`. Key parameters include:

```python
class Config:
    # Dataset
    data_root = 'DIV2K'  # Path to DIV2K dataset
    scale = 2           # Upscaling factor
    patch_size = 128    # HR patch size for training
    
    # Model
    base_channels = 64
    use_channel_attention = True
    use_spatial_attention = True
    num_residual_blocks = 16
    
    # Training
    batch_size = 16
    num_workers = 4
    lr = 2e-4
    num_epochs = 200
    warmup_epochs = 5
    
    # Loss weights
    l1_weight = 1.0
    perceptual_weight = 0.1
    freq_loss_weight = 0.05
```

## Training

To train the model:

```bash
run cells of model.ipynb
```

The training process includes:

- Learning rate warmup
- Mixed precision training
- Automatic checkpointing
- Early stopping
- Regular validation
- Visualization of results

## Results

The model saves:

- Best model checkpoints
- Training curves
- Example results
- Validation metrics

## Model Architecture

The model consists of three main components:

1. **Spatial Branch**: Processes image features in the spatial domain using residual blocks and attention mechanisms
2. **Frequency Branch**: Processes image features in the frequency domain using complex convolutions
3. **Fusion Module**: Combines features from both branches using attention mechanisms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

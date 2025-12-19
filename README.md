# XAl-ChangeNet

## Overview
XAl-ChangeNet is a lightweight yet expressive change-detection pipeline built for the xBD/xView2 large-scale disaster dataset. It fuses pre- and post-disaster satellite imagery using a Siamese ResNet18 encoder and a UNet-style decoder to generate dense change masks that align pixel-wise with the source inputs. The project is optimized for GTX 1650-class GPUs via small batch sizes, mixed precision, and lean data pipelines.

## Architecture
```
Pre Image ─┐           ┌─ ResNet18 Encoder Levels ─┐
          │           │                           │
Post Image┤─ Shared ResNet18 encoders ── feature pyramid (5 scales)
          │           │                           │
          └─>| Concatenate pre, post, |post-pre| |─┐
                                                │
                                         UNet Decoder (upsample + conv)
                                                │
                                         1x1 Conv → Change logits
```

## Dataset

This project uses the **xView2 (xBD) dataset** for training and evaluation. The dataset is **not included in this repository** for the following reasons:

- **Size**: The dataset is very large (several GB), making it impractical to host on GitHub
- **Licensing**: xView2 has specific terms of use that require downloading from the official source
- **Best Practice**: Datasets should be downloaded separately to keep repositories lightweight

### Download Instructions

1. **Visit the official xView2 website**: https://challenge.xview2.org/
2. **Register/Login** to access the dataset download page
3. **Download the training data** (typically includes `train_images.tar.gz` and `train_labels.tar.gz`)
4. **Extract the files** to your local machine

### Expected Folder Structure

After downloading and extracting, organize your data as follows:

```
data/
└── train/
    ├── images/          # Pre- and post-disaster satellite images
    ├── labels/          # JSON annotation files
    └── targets/         # (Optional) Ground truth masks
```

**Note**: The dataset folder (`data/`) is ignored by `.gitignore` to prevent accidental commits of large files.

## Dataset Preparation
1. Place raw xBD files under `data/train/` preserving the `images/`, `labels/`, and optional `targets/` folders.
2. Run the preparation script to generate event-wise folders, masks, and manifest files under `data/xbd/`:
   ```
   python scripts/prepare_xbd.py --src data/train --out data/xbd
   ```
3. Inspect the resulting structure:
   - `images/<event>/pre|post`
   - `masks/<event>`
   - `annotations/<event>`
   - `pairs_<event>.json`

## Usage
- Prepare Dataset:
  ```
  python scripts/prepare_xbd.py --src data/train --out data/xbd
  ```
- Train:
  ```
  python train.py --config configs/train.yaml
  ```
- Debug Train:
  ```
  python train.py --config configs/debug.yaml
  ```
  
 - Run locally (examples) 💡
   - Baseline smoke run (short, no strong augmentations):
     ```powershell
     python train_v2.py --config configs/train_smoke.yaml --logdir runs/trainsmoke
     ```
   - Augmented run:
     - Use `configs/train_all_events_aug.yaml` (included) or add your own config.
     ```powershell
     python train_v2.py --config configs/train_all_events_aug.yaml --logdir runs/train_all_events_aug
     ```
    - Request a specific device (GPU/CPU):
      ```powershell
      # Use the GPU (if available). On Linux you can also set CUDA_VISIBLE_DEVICES.
      python train_v2.py --config configs/train_smoke.yaml --logdir runs/trainsmoke --device cuda

      # Force CPU mode
      python train_v2.py --config configs/train_smoke.yaml --logdir runs/trainsmoke --device cpu
      ```
    - If using Windows & you want to prefer a Linux-like environment, use WSL2 or Docker with NVIDIA container runtime (recommended for robust GPU runs):
      - WSL2: https://learn.microsoft.com/windows/wsl/install
      - Docker + NVIDIA: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   - Long training run (50+ epochs):
     ```powershell
     python train_v2.py --config configs/train_long.yaml --logdir runs/train_long
     ```
   - Evaluate across all events (uses `checkpoints/latest.pth` by default):
     ```powershell
     python run.py eval --ckpt checkpoints/latest.pth
     ```
   - Visualize and generate heatmaps (saves to `outputs/`):
     ```powershell
     python run.py visualize --events --ckpt checkpoints/latest.pth --output outputs
     ```
- Explain (Grad-CAM / LIME / SHAP placeholder):
  ```
  python scripts/explain.py --ckpt checkpoints/latest.pth --pre path/to/pre.png --post path/to/post.png
  ```
- Evaluate:
  ```
  python scripts/evaluate.py --pairs data/xbd/pairs_event.json
  ```
  
 - Notes on validation & checkpoints:
   - Use a separate validation pairs manifest in your config to enable validation and automatic best-checkpoint saving (train_v2 will save `checkpoints/best.pth` when a new best validation IoU is observed).
   - Example (edit one of the included configs, e.g. `configs/train_smoke.yaml`):
     ```yaml
     data:
       pairs_file: data/xbd/pairs_guatemala-volcano.json
       val_pairs: data/xbd/pairs_mexico-earthquake.json
     training:
       scheduler: true
       scheduler_type: "cosine"
       max_grad_norm: 1.0
     ```
   - TensorBoard runs: validation images for a sample are logged to the `runs/` directory (`writer.add_image`) and saved under `outputs/val_sample_epoch_{epoch:03d}.png` for quick visual checks.

## Additional Notes
- Checkpoints are saved per epoch under `checkpoints/` with `latest.pth` updated each epoch.
- Visual outputs (Grad-CAM, overlays, notebook renders) land in `outputs/` by default.
- Use the provided notebook `notebooks/visualize.ipynb` to interactively inspect predictions and explanations.

## GitHub / what to commit
- This repo is intended to commit **code + configs + small curated reports** only.
- Do **not** commit large/generated folders (they are ignored by `.gitignore`):
  - `data/` (dataset must be downloaded separately from xView2 website - see [Dataset](#dataset) section)
  - `checkpoints/` (model weights - too large for GitHub)
  - `runs/` (TensorBoard logs)
  - `outputs/` (full prediction dumps)
- To export a small, GitHub-friendly set of results (metrics + a few prediction images) into `reports/`:

```powershell
python scripts/export_reports.py
```

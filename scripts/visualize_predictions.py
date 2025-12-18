"""Generate and save CAM heatmaps, predictions, and masks to outputs/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Ensure project root is on sys.path when running as a script
import os
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.siamese_unet import SiameseResNet18UNet


TRANSFORM = A.Compose([A.Resize(512, 512), A.Normalize()])


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize predictions and CAMs")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/latest.pth"))
    parser.add_argument("--pairs", type=Path, default=None, help="Single pairs_event.json path (optional if --events provided)")
    parser.add_argument("--events", action="store_true", help="Run on all event pairs in data/xbd/")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="output directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--single-file", action="store_true", help="Combine all sample grids into a single image per event")
    return parser.parse_args()


def load_model(ckpt: Path, device: torch.device) -> SiameseResNet18UNet:
    model = SiameseResNet18UNet()
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def load_samples(pairs_file: Path) -> List[Dict]:
    with pairs_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        m = img.convert("L")
        m = m.resize((512, 512), resample=Image.NEAREST)
        return np.array(m) / 255.0


def preprocess(np_img: np.ndarray) -> torch.Tensor:
    data = TRANSFORM(image=np_img)
    tensor = torch.from_numpy(data["image"].transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0)


def gradcam_map(model, pre_tensor, post_tensor, device):
    target_module = model.decoder0.block[-2]
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    handle_f = target_module.register_forward_hook(forward_hook)

    logits = model(pre_tensor, post_tensor)
    score = logits.mean()

    acts = activations[0]

    def act_grad_hook(grad):
        gradients.append(grad.detach())

    acts.register_hook(act_grad_hook)

    model.zero_grad()
    score.backward()

    grads = gradients[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=pre_tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach()

    handle_f.remove()
    return cam


def normalize_and_colorize(heatmap: np.ndarray) -> np.ndarray:
    """Normalize heatmap to [0, 1] and apply jet colormap."""
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-6:
        h_norm = (heatmap - h_min) / (h_max - h_min)
    else:
        h_norm = np.zeros_like(heatmap)
    
    h_uint8 = (h_norm * 255).astype(np.uint8)
    h_color = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
    return h_color


def make_comparison_grid(post_img: np.ndarray, pred_binary: np.ndarray, mask_gt: np.ndarray, cam: np.ndarray, size: int = 256) -> np.ndarray:
    """Return a 4-panel grid image (post, pred, gt, cam) as uint8 BGR array."""
    h, w = size, size
    post_rs = cv2.resize(post_img, (w, h))
    pred_rs = cv2.resize(pred_binary, (w, h))
    mask_rs = cv2.resize(mask_gt, (w, h))
    cam_rs = cv2.resize(cam, (w, h))
    pred_color = np.stack([pred_rs, np.zeros_like(pred_rs), np.zeros_like(pred_rs)], axis=2) * 255
    mask_color = np.stack([mask_rs, np.zeros_like(mask_rs), np.zeros_like(mask_rs)], axis=2) * 255
    cam_color = normalize_and_colorize(cam_rs)
    post_rs_3ch = cv2.cvtColor(post_rs, cv2.COLOR_RGB2BGR) if post_rs.ndim == 3 else cv2.cvtColor(
        cv2.cvtColor(post_rs, cv2.COLOR_GRAY2BGR), cv2.COLOR_RGB2BGR)
    top = np.hstack([post_rs_3ch, pred_color.astype(np.uint8)])
    bottom = np.hstack([mask_color.astype(np.uint8), cam_color])
    grid = np.vstack([top, bottom])
    return grid


def save_comparison(output_dir: Path, idx: int, post_img: np.ndarray, pred_binary: np.ndarray, 
                     mask_gt: np.ndarray, cam: np.ndarray):
    """Save a 4-panel figure: post image, prediction, ground truth, CAM heatmap."""
    # Resize all to same size for display
    h, w = 256, 256
    
    post_rs = cv2.resize(post_img, (w, h))
    pred_rs = cv2.resize(pred_binary, (w, h))
    mask_rs = cv2.resize(mask_gt, (w, h))
    cam_rs = cv2.resize(cam, (w, h))
    
    # Create colorized masks (red for change)
    pred_color = np.stack([pred_rs, np.zeros_like(pred_rs), np.zeros_like(pred_rs)], axis=2) * 255
    mask_color = np.stack([mask_rs, np.zeros_like(mask_rs), np.zeros_like(mask_rs)], axis=2) * 255
    
    # Colorize CAM
    cam_color = normalize_and_colorize(cam_rs)
    
    # Convert post to RGB if needed (assume BGR from cv2)
    post_rs_3ch = cv2.cvtColor(post_rs, cv2.COLOR_RGB2BGR) if post_rs.ndim == 3 else cv2.cvtColor(
        cv2.cvtColor(post_rs, cv2.COLOR_GRAY2BGR), cv2.COLOR_RGB2BGR)
    
    # Stack into 2x2 grid
    top = np.hstack([post_rs_3ch, pred_color.astype(np.uint8)])
    bottom = np.hstack([mask_color.astype(np.uint8), cam_color])
    grid = np.vstack([top, bottom])
    
    output_path = output_dir / f"sample_{idx:03d}.png"
    cv2.imwrite(str(output_path), grid)
    return grid


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Determine which pairs files to process
    if args.events:
        pairs_files = sorted(Path("data/xbd").glob("pairs_*.json"))
        # Skip any global manifest like pairs_all.json so we get per-event folders only
        pairs_files = [p for p in pairs_files if p.stem != "pairs_all"]
        if not pairs_files:
            print("No per-event pairs_*.json found in data/xbd/")
            return
    elif args.pairs:
        pairs_files = [args.pairs]
    else:
        print("Error: provide either --pairs or --events")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model(args.ckpt, device)
    
    for pairs_file in pairs_files:
        # Extract event name from filename (e.g., pairs_guatemala-volcano.json -> guatemala-volcano)
        event_name = pairs_file.stem.replace("pairs_", "")
        event_output_dir = args.output / event_name
        event_output_dir.mkdir(parents=True, exist_ok=True)
        
        samples = load_samples(pairs_file)
        root = pairs_file.parent
        
        print(f"\nGenerating visualizations for event: {event_name} ({len(samples)} samples)...")
        
        grids = []
        for idx, sample in enumerate(tqdm(samples, desc=f"Visualizing {event_name}")):
            pre_np = load_image(root / sample["pre_image"])
            post_np = load_image(root / sample["post_image"])
            mask_np = load_mask(root / sample["mask"])
            
            pre_tensor = preprocess(pre_np).to(device)
            post_tensor = preprocess(post_np).to(device)
            
            with torch.no_grad():
                logits = model(pre_tensor, post_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            pred_binary = (probs > args.threshold).astype(np.uint8)
            
            cam = gradcam_map(model, pre_tensor, post_tensor, device).cpu().numpy()
            
            # Save comparison figure and collect grid
            grid = save_comparison(event_output_dir, idx, post_np, pred_binary, mask_np, cam)
            grids.append(grid)

        # If requested, combine all grids into a single tiled image per event
        if args.single_file and grids:
            cols = 4
            rows = int(np.ceil(len(grids) / cols))
            h, w, c = grids[0].shape
            total_h = h * rows
            total_w = w * cols
            combined = np.zeros((total_h, total_w, c), dtype=np.uint8)
            for i, g in enumerate(grids):
                r = i // cols
                cidx = i % cols
                combined[r*h:(r+1)*h, cidx*w:(cidx+1)*w] = g
            combined_path = args.output / f"{event_name}_all.png"
            cv2.imwrite(str(combined_path), combined)
    
    print(f"\n✓ All visualizations saved to {args.output}/")
    print(f"  Organized by event: {args.output}/[event-name]/sample_NNN.png")
    print(f"  Each sample_NNN.png shows: [post image, prediction] / [ground truth, CAM heatmap]")


if __name__ == "__main__":
    main()

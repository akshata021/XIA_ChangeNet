"""Test model on random samples from each event - compare predictions vs ground truth."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.siamese_unet import SiameseResNet18UNet


TRANSFORM = A.Compose([A.Resize(512, 512), A.Normalize()])


def parse_args():
    parser = argparse.ArgumentParser(description="Test model on random samples from each event")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/best.pth"))
    parser.add_argument("--samples-per-event", type=int, default=2, help="Number of random samples per event")
    parser.add_argument("--output", type=Path, default=Path("outputs/test_samples"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bin = (pred > 0.5).astype(np.float32)
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return float(inter / (union + 1e-6))


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bin = (pred > 0.5).astype(np.float32)
    inter = (pred_bin * target).sum()
    total = pred_bin.sum() + target.sum()
    return float((2 * inter) / (total + 1e-6))


def create_comparison_grid(pre_img: np.ndarray, post_img: np.ndarray, 
                          pred_mask: np.ndarray, gt_mask: np.ndarray,
                          iou: float, dice: float, size: int = 256) -> np.ndarray:
    """Create a 2x2 grid: [pre, post] / [prediction, ground truth]"""
    h, w = size, size
    
    # Resize all to same size
    pre_rs = cv2.resize(pre_img, (w, h))
    post_rs = cv2.resize(post_img, (w, h))
    pred_rs = cv2.resize(pred_mask, (w, h))
    gt_rs = cv2.resize(gt_mask, (w, h))
    
    # Convert masks to RGB (red for change)
    pred_color = np.stack([pred_rs, np.zeros_like(pred_rs), np.zeros_like(pred_rs)], axis=2) * 255
    gt_color = np.stack([gt_rs, np.zeros_like(gt_rs), np.zeros_like(gt_rs)], axis=2) * 255
    
    # Convert images to BGR for cv2
    pre_bgr = cv2.cvtColor(pre_rs, cv2.COLOR_RGB2BGR)
    post_bgr = cv2.cvtColor(post_rs, cv2.COLOR_RGB2BGR)
    
    # Add text labels with metrics
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(pre_bgr, "Pre-disaster", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(post_bgr, "Post-disaster", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(pred_color.astype(np.uint8), f"Prediction (IoU:{iou:.3f})", (10, 30), font, 0.6, (255, 255, 255), 2)
    cv2.putText(gt_color.astype(np.uint8), f"Ground Truth (Dice:{dice:.3f})", (10, 30), font, 0.6, (255, 255, 255), 2)
    
    # Stack into grid
    top = np.hstack([pre_bgr, post_bgr])
    bottom = np.hstack([pred_color.astype(np.uint8), gt_color.astype(np.uint8)])
    grid = np.vstack([top, bottom])
    
    return grid


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model(args.ckpt, device)
    
    # Get all event pairs files (exclude pairs_all.json)
    root = Path("data/xbd")
    pairs_files = sorted([p for p in root.glob("pairs_*.json") if p.name != "pairs_all.json"])
    
    print(f"\nFound {len(pairs_files)} events")
    print(f"Testing {args.samples_per_event} random samples per event...\n")
    
    all_results = []
    
    for pairs_file in pairs_files:
        event_name = pairs_file.stem.replace("pairs_", "")
        samples = load_samples(pairs_file)
        
        if len(samples) < args.samples_per_event:
            print(f"  {event_name}: Only {len(samples)} samples available, using all")
            selected_indices = list(range(len(samples)))
        else:
            selected_indices = random.sample(range(len(samples)), args.samples_per_event)
        
        print(f"\n{event_name} ({len(samples)} total samples):")
        
        for idx, sample_idx in enumerate(selected_indices):
            sample = samples[sample_idx]
            
            # Load images and mask
            pre_np = load_image(root / sample["pre_image"])
            post_np = load_image(root / sample["post_image"])
            mask_np = load_mask(root / sample["mask"])
            
            # Run prediction
            pre_tensor = preprocess(pre_np).to(device)
            post_tensor = preprocess(post_np).to(device)
            
            with torch.no_grad():
                logits = model(pre_tensor, post_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            pred_binary = (probs > args.threshold).astype(np.float32)
            
            # Compute metrics
            iou = compute_iou(pred_binary, mask_np)
            dice = compute_dice(pred_binary, mask_np)
            
            # Create comparison grid
            grid = create_comparison_grid(pre_np, post_np, pred_binary, mask_np, iou, dice)
            
            # Save
            output_path = args.output / f"{event_name}_sample_{idx+1:02d}.png"
            cv2.imwrite(str(output_path), grid)
            
            print(f"  Sample {idx+1}: IoU={iou:.3f}, Dice={dice:.3f} -> {output_path.name}")
            
            all_results.append({
                "event": event_name,
                "sample": idx+1,
                "iou": iou,
                "dice": dice,
                "file": str(output_path.name)
            })
    
    # Save summary
    summary_path = args.output / "test_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_samples": len(all_results),
            "samples_per_event": args.samples_per_event,
            "threshold": args.threshold,
            "results": all_results
        }, f, indent=2)
    
    # Print summary stats
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    avg_iou = np.mean([r["iou"] for r in all_results])
    avg_dice = np.mean([r["dice"] for r in all_results])
    print(f"Average IoU: {avg_iou:.3f}")
    print(f"Average Dice: {avg_dice:.3f}")
    print(f"\nAll results saved to: {args.output}/")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()


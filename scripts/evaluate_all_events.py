"""Evaluate model across all events and generate a performance report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import albumentations as A
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


def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum().item()
    union = pred_bin.sum().item() + target.sum().item() - inter
    return inter / (union + 1e-6)


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum().item()
    total = pred_bin.sum().item() + target.sum().item()
    return (2 * inter) / (total + 1e-6)


def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_bin = (pred > 0.5).float()
    eq = (pred_bin == target).float()
    return float(eq.mean().item())


def compute_precision_recall(pred: torch.Tensor, target: torch.Tensor) -> tuple:
    """Compute pixel-level precision and recall."""
    pred_bin = (pred > 0.5).float()
    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    model = load_model(Path("checkpoints/latest.pth"), device)
    
    pairs_files = sorted(Path("data/xbd").glob("pairs_*.json"))
    
    all_results = {}
    global_metrics = {"iou": [], "dice": [], "accuracy": [], "precision": [], "recall": []}
    
    for pairs_file in pairs_files:
        event_name = pairs_file.stem.replace("pairs_", "")
        samples = load_samples(pairs_file)
        root = pairs_file.parent
        
        metrics = {"iou": [], "dice": [], "accuracy": [], "precision": [], "recall": []}
        
        for sample in tqdm(samples, desc=f"{event_name:25s}", leave=False):
            pre_np = load_image(root / sample["pre_image"])
            post_np = load_image(root / sample["post_image"])
            mask_np = load_mask(root / sample["mask"])
            
            pre_tensor = preprocess(pre_np).to(device)
            post_tensor = preprocess(post_np).to(device)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(pre_tensor, post_tensor)
                probs = torch.sigmoid(logits)
            
            pred_bin = (probs > 0.5).float()
            
            iou = compute_iou(pred_bin, mask_tensor)
            dice = compute_dice(pred_bin, mask_tensor)
            acc = compute_accuracy(pred_bin, mask_tensor)
            prec, rec = compute_precision_recall(pred_bin, mask_tensor)
            
            metrics["iou"].append(iou)
            metrics["dice"].append(dice)
            metrics["accuracy"].append(acc)
            metrics["precision"].append(prec)
            metrics["recall"].append(rec)
            
            # Add to global
            global_metrics["iou"].append(iou)
            global_metrics["dice"].append(dice)
            global_metrics["accuracy"].append(acc)
            global_metrics["precision"].append(prec)
            global_metrics["recall"].append(rec)
        
        # Summarize event
        event_summary = {k: float(np.mean(v)) for k, v in metrics.items()}
        event_summary["samples"] = len(samples)
        event_summary["std_iou"] = float(np.std(metrics["iou"]))
        event_summary["std_dice"] = float(np.std(metrics["dice"]))
        all_results[event_name] = event_summary
        # Save per-event metrics JSON (overwrites any existing)
        out_dir = Path("outputs") / event_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metrics.json", "w") as ef:
            json.dump({"event": event_name, **event_summary}, ef, indent=2)
    
    # Print report
    print("\n" + "=" * 120)
    print("PER-EVENT PERFORMANCE METRICS")
    print("=" * 120)
    print(f"{'Event':<30} {'Samples':>8} {'IoU':>10} {'Dice':>10} {'Acc':>10} {'Prec':>10} {'Rec':>10}")
    print("-" * 120)
    
    for event_name in sorted(all_results.keys()):
        res = all_results[event_name]
        print(f"{event_name:<30} {res['samples']:>8d} "
              f"{res['iou']:>10.4f} {res['dice']:>10.4f} {res['accuracy']:>10.4f} "
              f"{res['precision']:>10.4f} {res['recall']:>10.4f}")
    
    print("-" * 120)
    
    # Global summary
    global_summary = {k: float(np.mean(v)) for k, v in global_metrics.items()}
    print(f"{'GLOBAL AVERAGE':<30} {sum(r['samples'] for r in all_results.values()):>8d} "
          f"{global_summary['iou']:>10.4f} {global_summary['dice']:>10.4f} "
          f"{global_summary['accuracy']:>10.4f} {global_summary['precision']:>10.4f} "
          f"{global_summary['recall']:>10.4f}")
    print("=" * 120)
    
    # Analysis and interpretation
    print("\n" + "=" * 120)
    print("PERFORMANCE ANALYSIS & INTERPRETATION")
    print("=" * 120)
    
    iou = global_summary['iou']
    dice = global_summary['dice']
    prec = global_summary['precision']
    rec = global_summary['recall']
    f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
    
    print(f"\nOverall Metrics:")
    print(f"  • IoU (Intersection over Union):  {iou:.4f} (50% good, 60%+ very good)")
    print(f"  • Dice (F1-Score equivalent):     {dice:.4f} (50% good, 70%+ very good)")
    print(f"  • Precision:                      {prec:.4f} (% of predicted changes are correct)")
    print(f"  • Recall:                         {rec:.4f} (% of actual changes are detected)")
    print(f"  • F1-Score (harmonic mean):       {f1:.4f} (balanced precision-recall)")
    
    print(f"\n📊 Performance Assessment:")
    if iou < 0.3:
        iou_level = "⚠️  POOR - needs improvement"
    elif iou < 0.5:
        iou_level = "⚠️  FAIR - room for improvement"
    elif iou < 0.65:
        iou_level = "✓ GOOD - acceptable performance"
    else:
        iou_level = "✓✓ EXCELLENT - strong performance"
    
    print(f"  IoU {iou:.4f}: {iou_level}")
    
    if rec < prec:
        print(f"  • Model is CONSERVATIVE: high precision ({prec:.4f}), low recall ({rec:.4f})")
        print(f"    → Misses some actual changes but rarely false alarms")
    elif rec > prec:
        print(f"  • Model is AGGRESSIVE: high recall ({rec:.4f}), low precision ({prec:.4f})")
        print(f"    → Detects most changes but with some false alarms")
    else:
        print(f"  • Model is BALANCED: precision ≈ recall ({prec:.4f})")
    
    print(f"\n📈 Per-Event Variation:")
    ious = [all_results[e]['iou'] for e in all_results]
    dices = [all_results[e]['dice'] for e in all_results]
    print(f"  IoU range: {min(ious):.4f} – {max(ious):.4f} (std: {np.std(ious):.4f})")
    print(f"  Dice range: {min(dices):.4f} – {max(dices):.4f} (std: {np.std(dices):.4f})")
    
    best_event = max(all_results.items(), key=lambda x: x[1]['iou'])
    worst_event = min(all_results.items(), key=lambda x: x[1]['iou'])
    print(f"  Best event: {best_event[0]} (IoU {best_event[1]['iou']:.4f})")
    print(f"  Hardest event: {worst_event[0]} (IoU {worst_event[1]['iou']:.4f})")
    
    print(f"\n🎯 Recommendations:")
    if iou < 0.5:
        print(f"  1. Train longer: current {iou:.4f} IoU can improve with more epochs")
        print(f"  2. Tune learning rate: try lr 5e-5 or 2e-4")
        print(f"  3. Increase batch size: from 2 to 4 for better gradient estimates")
    else:
        print(f"  Model is performing well! Consider:")
        print(f"  1. Fine-tuning on hard events: {worst_event[0]}")
        print(f"  2. Data augmentation to improve robustness")
        print(f"  3. Ensemble with other models for production use")
    
    print("\n" + "=" * 120)
    
    # Save report to JSON
    report = {
        "global_metrics": global_summary,
        "per_event": all_results,
        "f1_score": float(f1),
    }
    
    with open("outputs/performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to outputs/performance_report.json")


if __name__ == "__main__":
    main()

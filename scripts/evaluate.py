"""Evaluation script for XAl-ChangeNet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import albumentations as A
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.siamese_unet import SiameseResNet18UNet


TRANSFORM = A.Compose([A.Resize(512, 512), A.Normalize()])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XAl-ChangeNet")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/latest.pth"))
    parser.add_argument("--pairs", type=Path, required=True, help="pairs_event.json path")
    parser.add_argument("--threshold", type=float, default=0.5)
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
    # Ensure mask is resized to the model input size (512x512) and kept as a single channel
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
    """Pixel-wise accuracy between binary prediction and target.

    Both tensors should be in {0,1} float format and same shape.
    """
    pred_bin = (pred > 0.5).float()
    eq = (pred_bin == target).float()
    return float(eq.mean().item())


def gradcam_map(model, pre_tensor, post_tensor, device):
    target_module = model.decoder0.block[-2]
    activations = []
    gradients = []

    # Capture the forward activation (not detached) and register a hook on the
    # activation tensor to collect gradients. Using activation.register_hook
    # avoids issues with module-level backward hooks and inplace ops.
    def forward_hook(_, __, output):
        activations.append(output)

    handle_f = target_module.register_forward_hook(forward_hook)

    logits = model(pre_tensor, post_tensor)
    score = logits.mean()

    # The forward hook has populated activations[0]
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


def localization_iou(cam: torch.Tensor, mask: torch.Tensor) -> float:
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam_bin = (cam > cam.mean()).float()
    inter = (cam_bin * mask).sum().item()
    union = cam_bin.sum().item() + mask.sum().item() - inter
    return inter / (union + 1e-6)


def insertion_deletion(model, pre_tensor, post_tensor, cam, device):
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    flat = cam_norm.view(-1)
    order = torch.argsort(flat, descending=True)
    total = order.numel()
    fractions = torch.linspace(0.0, 1.0, steps=11)

    def score(t_post):
        with torch.no_grad():
            logits = model(pre_tensor, t_post)
            return torch.sigmoid(logits).mean().item()

    baseline = torch.zeros_like(post_tensor)
    current = baseline.clone()
    insertion_scores = []
    deletion_scores = []

    for frac in fractions:
        k = int(frac.item() * total)
        idx = order[:k]
        # Use reshape instead of view to handle non-contiguous tensors
        current.reshape(-1)[idx] = post_tensor.reshape(-1)[idx]
        insertion_scores.append(score(current))

        mask = post_tensor.clone()
        mask.reshape(-1)[idx] = 0
        deletion_scores.append(score(mask))

    insertion_auc = float(torch.trapz(torch.tensor(insertion_scores), fractions).item())
    deletion_auc = float(torch.trapz(torch.tensor(deletion_scores), fractions).item())
    return insertion_auc, deletion_auc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    samples = load_samples(args.pairs)
    root = args.pairs.parent

    metrics = {"iou": [], "dice": [], "accuracy": [], "xai_iou": [], "insertion": [], "deletion": []}

    for sample in tqdm(samples, desc="Evaluating"):
        pre_np = load_image(root / sample["pre_image"])
        post_np = load_image(root / sample["post_image"])
        mask_np = load_mask(root / sample["mask"])

        pre_tensor = preprocess(pre_np).to(device)
        post_tensor = preprocess(post_np).to(device)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(pre_tensor, post_tensor)
            probs = torch.sigmoid(logits)

        metrics["iou"].append(compute_iou((probs > args.threshold).float(), mask_tensor))
        metrics["dice"].append(compute_dice((probs > args.threshold).float(), mask_tensor))
        metrics["accuracy"].append(compute_accuracy((probs > args.threshold).float(), mask_tensor))

        cam = gradcam_map(model, pre_tensor, post_tensor, device)
        metrics["xai_iou"].append(localization_iou(cam, mask_tensor))

        ins, dele = insertion_deletion(model, pre_tensor, post_tensor, cam, device)
        metrics["insertion"].append(ins)
        metrics["deletion"].append(dele)

    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

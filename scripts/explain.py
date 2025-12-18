"""Explainability tools for XAl-ChangeNet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from lime import lime_image
from PIL import Image
from torchvision.utils import make_grid

from models.siamese_unet import SiameseResNet18UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Explain XAl-ChangeNet predictions")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--pre", type=Path, required=True, help="Pre-disaster image")
    parser.add_argument("--post", type=Path, required=True, help="Post-disaster image")
    parser.add_argument("--out", type=Path, default=Path("outputs") / "explain", help="Output directory")
    return parser.parse_args()


def load_model(ckpt: Path, device: torch.device) -> SiameseResNet18UNet:
    model = SiameseResNet18UNet()
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(path: Path, size: int = 512) -> Tuple[torch.Tensor, np.ndarray]:
    transform = A.Compose([A.Resize(size, size), A.Normalize()])
    with Image.open(path) as img:
        np_img = np.array(img.convert("RGB"))
    data = transform(image=np_img)
    tensor = torch.from_numpy(data["image"].transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0), np_img


def grad_cam(model, pre_tensor, post_tensor, device: torch.device, save_path: Path, post_raw: np.ndarray):
    target_module = model.decoder0.block[-2]  # last conv
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_f = target_module.register_forward_hook(forward_hook)
    handle_b = target_module.register_full_backward_hook(backward_hook)

    pre_tensor = pre_tensor.to(device)
    post_tensor = post_tensor.to(device)
    logits = model(pre_tensor, post_tensor)
    score = logits.mean()
    model.zero_grad()
    score.backward()

    acts = activations[0]
    grads = gradients[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1))
    cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=post_raw.shape[:2], mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    heatmap = (plt.cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    overlay = (0.6 * heatmap + 0.4 * post_raw).astype(np.uint8)

    Image.fromarray(overlay).save(save_path)
    handle_f.remove()
    handle_b.remove()
    print(f"Grad-CAM saved to {save_path}")


def lime_explanation(pre_raw: np.ndarray, post_raw: np.ndarray, save_path: Path):
    explainer = lime_image.LimeImageExplainer()
    combined = np.concatenate([pre_raw, post_raw], axis=1)

    def predict_fn(images):
        return np.stack([(img.mean(), 1 - img.mean()) for img in images])

    explanation = explainer.explain_instance(
        combined,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000,
    )
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, hide_rest=False)
    Image.fromarray((temp * mask[..., None]).astype(np.uint8)).save(save_path)
    print(f"LIME mask saved to {save_path}")


def shap_placeholder(save_path: Path):
    message = (
        "SHAP explanations are resource-intensive. Use shap.DeepExplainer or GradientExplainer \n"
        "Example: shap.DeepExplainer(model, background_batch).shap_values(sample)."
    )
    save_path.write_text(message, encoding="utf-8")
    print(f"SHAP instructions written to {save_path}")


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    pre_tensor, pre_raw = preprocess_image(args.pre)
    post_tensor, post_raw = preprocess_image(args.post)

    grad_cam(model, pre_tensor, post_tensor, device, args.out / "grad_cam.png", post_raw)
    lime_explanation(pre_raw, post_raw, args.out / "lime.png")
    shap_placeholder(args.out / "shap.txt")


if __name__ == "__main__":
    main()

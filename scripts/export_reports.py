import argparse
import shutil
from pathlib import Path


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _collect_sample_images(outputs_dir: Path, limit: int) -> list[Path]:
    # Prefer small "sample_*.png" images (your repo already creates these).
    # Fall back to "val_sample_epoch_*.png" if present.
    patterns = [
        "**/sample_*.png",
        "val_sample_epoch_*.png",
        "test_samples/*.png",
    ]
    found: list[Path] = []
    for pat in patterns:
        found.extend(sorted(outputs_dir.glob(pat)))
    # Deduplicate while preserving order
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in found:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq[: max(0, limit)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export small, GitHub-friendly artifacts into reports/.")
    parser.add_argument("--outputs", type=str, default="outputs", help="Path to outputs directory (default: outputs)")
    parser.add_argument("--reports", type=str, default="reports", help="Path to reports directory (default: reports)")
    parser.add_argument("--max-images", type=int, default=12, help="Max number of prediction images to copy")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = (repo_root / args.outputs).resolve()
    reports_dir = (repo_root / args.reports).resolve()

    metrics_dir = reports_dir / "metrics"
    preds_dir = reports_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    copied_any = False

    # Metrics
    copied_any |= _copy_if_exists(outputs_dir / "performance_report.json", metrics_dir / "performance_report.json")
    copied_any |= _copy_if_exists(outputs_dir / "all" / "metrics.json", metrics_dir / "all_metrics.json")

    # Predictions (small curated set)
    images = _collect_sample_images(outputs_dir, limit=args.max_images)
    for i, src in enumerate(images):
        # Flatten into a single folder with stable names
        dst = preds_dir / f"sample_{i:03d}_{src.name}"
        copied_any |= _copy_if_exists(src, dst)

    if not copied_any:
        print("No artifacts copied. Did you run evaluation/visualization to generate outputs/ ?")
        return 2

    print(f"Exported reports to: {reports_dir}")
    print(f"- metrics: {metrics_dir}")
    print(f"- predictions: {preds_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



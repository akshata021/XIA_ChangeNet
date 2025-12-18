"""Simple hyperparameter sweep for `train_v2.py`.

This script programmatically generates small temporary configs (based on a base config)
with overridden lr and batch size, runs short training jobs, and collects performance
metrics from outputs/performance_report.json.

Usage:
  python scripts/hp_sweep.py --base-config configs/train_smoke.yaml --val-pairs data/xbd/pairs_guatemala-volcano.json

"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, List
import time
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("configs/train_smoke.yaml"))
    parser.add_argument("--val-pairs", type=Path, default=Path("data/xbd/pairs_guatemala-volcano.json"))
    parser.add_argument("--lrs", nargs="+", default=["5e-5", "1e-4", "2e-4"])  # strings to preserve format
    parser.add_argument("--batch-sizes", nargs="+", default=["2", "4"])  # strings
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--logdir", type=Path, default=Path("runs/hp_sweep"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: Dict[str, Any]):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def run_train(train_script: Path, config: Path, val_pairs: Path, logdir: Path, run_name: str):
    cmd = ["python", str(train_script), "--config", str(config), "--val_pairs", str(val_pairs), "--logdir", str(logdir / run_name)]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=Path("."), capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("Error: ", proc.stderr)
    return proc.returncode


def read_report(output_dir: Path) -> Dict[str, Any]:
    rpt_file = output_dir / "performance_report.json"
    if rpt_file.exists():
        with rpt_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    args = parse_args()
    base_config = load_yaml(args.base_config)

    results: List[Dict[str, Any]] = []

    args.logdir.mkdir(parents=True, exist_ok=True)

    for lr in args.lrs:
        for bs in args.batch_sizes:
            cfg = base_config.copy()
            cfg["training"]["lr"] = float(lr)
            cfg["data"]["batch_size"] = int(bs)
            cfg["training"]["epochs"] = args.epochs
            # optionally enable scheduler
            cfg["training"]["scheduler"] = True

            tmp_dir = tempfile.mkdtemp(prefix="hp_")
            tmp_cfg_path = Path(tmp_dir) / f"config_lr{lr}_bs{bs}.yaml"
            write_yaml(tmp_cfg_path, cfg)

            run_name = f"lr{lr}_bs{bs}_e{args.epochs}"
            if args.dry_run:
                print(f"DRY-RUN: would run with lr={lr}, bs={bs}, config {tmp_cfg_path}")
                shutil.rmtree(tmp_dir)
                continue

            start = time.time()
            rc = run_train(Path("train_v2.py"), tmp_cfg_path, args.val_pairs, args.logdir, run_name)
            end = time.time()
            out_dir = Path("outputs")  # train_v2 writes performance_report.json to outputs/
            rpt = read_report(out_dir)

            results.append({"lr": float(lr), "batch_size": int(bs), "rc": rc, "time": end - start, "report": rpt, "run_name": run_name})

            # cleanup created tmp dir
            shutil.rmtree(tmp_dir)
            # small sleep to make logs clearer
            time.sleep(3)

    # Save sweep results
    with (args.logdir / "sweep_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print("Sweep complete. Results saved to", args.logdir)


if __name__ == "__main__":
    main()

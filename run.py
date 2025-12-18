"""Unified CLI to run training, evaluation, visualization and hyperparameter sweeps.

Run `python run.py <command> --help` for subcommand options.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_train(args):
    # use train_v2.py
    cmd = [sys.executable, "train_v2.py", "--config", str(args.config)]
    if args.val_pairs:
        cmd += ["--val_pairs", str(args.val_pairs)]
    if args.logdir:
        cmd += ["--logdir", str(args.logdir)]
    print("Running train:", " ".join(cmd))
    return subprocess.call(cmd)


def run_eval(args):
    cmd = [sys.executable, "scripts/evaluate_all_events.py"]
    if args.ckpt:
        cmd += ["--ckpt", str(args.ckpt)]
    print("Running eval:", " ".join(cmd))
    return subprocess.call(cmd)


def run_visualize(args):
    cmd = [sys.executable, "scripts/visualize_predictions.py"]
    if args.events:
        cmd += ["--events"]
    if args.pairs:
        cmd += ["--pairs", str(args.pairs)]
    if args.ckpt:
        cmd += ["--ckpt", str(args.ckpt)]
    if args.output:
        cmd += ["--output", str(args.output)]
    print("Running visualize:", " ".join(cmd))
    return subprocess.call(cmd)


def run_sweep(args):
    cmd = [sys.executable, "scripts/hp_sweep.py", "--base-config", str(args.base_config), "--val-pairs", str(args.val_pairs)]
    if args.lrs:
        cmd += ["--lrs"] + args.lrs
    if args.batch_sizes:
        cmd += ["--batch-sizes"] + args.batch_sizes
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.logdir:
        cmd += ["--logdir", str(args.logdir)]
    print("Running sweep:", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run training/eval/visualizations and sweeps")
    sub = parser.add_subparsers(dest="cmd")

    # train
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--config", type=Path, required=True)
    p_train.add_argument("--val_pairs", type=Path, default=None)
    p_train.add_argument("--logdir", type=Path, default=Path("runs"))
    p_train.set_defaults(func=run_train)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate model across events")
    p_eval.add_argument("--ckpt", type=Path, default=Path("checkpoints/latest.pth"))
    p_eval.set_defaults(func=run_eval)

    # visualize
    p_vis = sub.add_parser("visualize", help="Generate heatmaps/visualizations")
    p_vis.add_argument("--events", action="store_true")
    p_vis.add_argument("--pairs", type=Path, default=None)
    p_vis.add_argument("--ckpt", type=Path, default=Path("checkpoints/latest.pth"))
    p_vis.add_argument("--output", type=Path, default=Path("outputs"))
    p_vis.set_defaults(func=run_visualize)

    # sweep
    p_sweep = sub.add_parser("sweep", help="Run hyperparameter sweep")
    p_sweep.add_argument("--base-config", type=Path, default=Path("configs/train_smoke.yaml"))
    p_sweep.add_argument("--val-pairs", type=Path, default=Path("data/xbd/pairs_guatemala-volcano.json"))
    p_sweep.add_argument("--lrs", nargs="*", default=["5e-05", "1e-04", "2e-04"])
    p_sweep.add_argument("--batch-sizes", nargs="*", default=["2", "4"])
    p_sweep.add_argument("--epochs", type=int, default=3)
    p_sweep.add_argument("--logdir", type=Path, default=Path("runs/hp_sweep"))
    p_sweep.set_defaults(func=run_sweep)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

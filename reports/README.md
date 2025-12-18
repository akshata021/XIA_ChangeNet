## What is `reports/`?

This folder is **intended to be committed to GitHub**. It should contain only small, curated artifacts:

- `metrics/`: summary JSONs you want to showcase (IoU/F1/etc.)
- `predictions/`: a small number of representative prediction images

Everything big or reproducible is **not committed** and is ignored by `.gitignore`:

- `data/` (dataset)
- `checkpoints/` (model weights)
- `runs/` (TensorBoard event files)
- `outputs/` (full prediction dumps)

## How to populate it

After you train/evaluate and produce `outputs/`, run:

```powershell
python scripts/export_reports.py
```

This will copy:
- `outputs/performance_report.json` → `reports/metrics/performance_report.json`
- `outputs/all/metrics.json` → `reports/metrics/all_metrics.json`
- a handful of sample `outputs/**/sample_*.png` images → `reports/predictions/`



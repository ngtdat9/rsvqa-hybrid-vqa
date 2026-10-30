"""Why does a run's validation collapse? Evaluates one checkpoint on val three ways:

1. as trained (eval mode, mixed precision as in training),
2. the same in float32 (rules float16 overflow in the forward pass in or out),
3. after recomputing BatchNorm running statistics from training batches.

If (3) recovers while (1) and (2) don't, the weights are fine and the stored
BatchNorm statistics broke eval-mode predictions.

Usage: python scripts/diagnose_run.py <run_dir> <config_path> [checkpoint_name] [num_batches]
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate, load_run, recalibrate_batchnorm, train_count_median  # noqa: E402


def main() -> None:
    run_dir = Path(sys.argv[1])
    config = yaml.safe_load(Path(sys.argv[2]).read_text())
    checkpoint_name = sys.argv[3] if len(sys.argv) > 3 else "last.pt"
    num_batches = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    model, dataloaders, device = load_run(run_dir, config, checkpoint_name)
    baseline_count = train_count_median(dataloaders["train"])

    def report(label: str, mixed_precision: bool) -> None:
        metrics = evaluate(
            model, dataloaders["val"], device, baseline_count=baseline_count,
            question_types=config["question_types"], desc=label, mixed_precision=mixed_precision,
        )
        print(f"{label:<40} val_class_acc={metrics['class_accuracy']:.4f} "
              f"val_count_mae={metrics['count_mae']:.4f}", flush=True)

    print(f"{run_dir.name} / {checkpoint_name}")
    report("1. as trained (mixed precision)", config["mixed_precision"])
    report("2. float32", False)
    recalibrate_batchnorm(model, dataloaders["train"], device, num_batches, config["mixed_precision"])
    report(f"3. BatchNorm recomputed ({num_batches} batches)", config["mixed_precision"])


if __name__ == "__main__":
    main()

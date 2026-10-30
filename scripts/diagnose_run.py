"""Why does a run's validation collapse while its train loss looks fine? The train
loss is measured in train mode on train data; validation in eval mode on val data.
This separates the possible causes, one check each:

1. val, as trained (eval mode, mixed precision as in training): the reference,
2. val in float32: float16 overflow in the forward pass,
3. val with dropout on (BatchNorm still in eval mode): dropout, i.e. train vs eval mode,
4. train data in eval mode (first ``num_batches`` batches): train vs val data,
5. val after recomputing BatchNorm running statistics from training batches: stale
   BatchNorm statistics.

The check that recovers the train-mode quality names the cause.

Usage: python scripts/diagnose_run.py <run_dir> <config_path> [checkpoint_name] [num_batches]
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate, load_run, recalibrate_batchnorm, train_count_median  # noqa: E402


def main() -> None:
    run_dir = Path(sys.argv[1])
    config = yaml.safe_load(Path(sys.argv[2]).read_text())
    checkpoint_name = sys.argv[3] if len(sys.argv) > 3 else "last.pt"
    num_batches = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    model, dataloaders, device = load_run(run_dir, config, checkpoint_name)
    baseline_count = train_count_median(dataloaders["train"])

    def report(label: str, split: str = "val", mixed_precision: bool = config["mixed_precision"],
               keep_mode: bool = False, max_batches: int | None = None) -> None:
        metrics = evaluate(
            model, dataloaders[split], device, baseline_count=baseline_count,
            question_types=config["question_types"], desc=label, mixed_precision=mixed_precision,
            keep_mode=keep_mode, max_batches=max_batches,
        )
        print(f"{label:<44} class_acc={metrics['class_accuracy']:.4f} "
              f"count_mae={metrics['count_mae']:.4f}", flush=True)

    print(f"{run_dir.name} / {checkpoint_name}")
    report("1. val, as trained (mixed precision)")
    report("2. val, float32", mixed_precision=False)
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    report("3. val, dropout on", keep_mode=True)
    report(f"4. train data, eval mode ({num_batches} batches)", split="train", max_batches=num_batches)
    recalibrate_batchnorm(model, dataloaders["train"], device, num_batches, config["mixed_precision"])
    report(f"5. val, BatchNorm recomputed ({num_batches} batches)")


if __name__ == "__main__":
    main()

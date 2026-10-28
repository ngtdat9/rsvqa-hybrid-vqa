"""Evaluation metrics for HybridVQAModel: classification accuracy by question
type, and count-regression MAE/RMSE."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data_pipeline import prepare_batch

QUESTION_TYPES = ("comp", "rural_urban", "presence")


def train_count_median(dataloader: DataLoader) -> float | None:
    """Median count answer in a dataloader's records (no images are loaded)."""
    records = getattr(dataloader.dataset, "qa_records", [])
    counts = sorted(float(r["answer"]) for r in records if r["type"] == "count")
    if not counts:
        return None
    mid = len(counts) // 2
    return counts[mid] if len(counts) % 2 else (counts[mid - 1] + counts[mid]) / 2


def selection_score(metrics: dict[str, Any]) -> float:
    """Higher-is-better score that weighs both heads equally, used to pick best.pt:

        classification accuracy + (1 - count MAE / MAE of always predicting the median)

    The count half is how much better than the median baseline the count head is
    (0 = no better, 1 = perfect). Without count rows or a baseline it is accuracy alone.
    """
    score = metrics["class_accuracy"]
    baseline_mae = metrics.get("count_mae_baseline")
    if baseline_mae is not None and baseline_mae > 0 and not math.isnan(metrics["count_mae"]):
        score += 1 - metrics["count_mae"] / baseline_mae
    return score


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | None = None,
    baseline_count: float | None = None,
    question_types: tuple[str, ...] | list[str] = QUESTION_TYPES,
    desc: str = "evaluate",
    mixed_precision: bool = False,
) -> dict[str, Any]:
    """Metrics for a model. If ``baseline_count`` is given, also report the MAE/RMSE
    of always predicting that constant (e.g. the train-set median count).
    ``mixed_precision`` runs the forward pass in float16 on CUDA (ignored elsewhere)."""
    if device is not None:
        model = model.to(device)
    model.eval()
    use_amp = mixed_precision and device is not None and device.type == "cuda"

    correct_by_type = {t: 0 for t in question_types}
    total_by_type = {t: 0 for t in question_types}
    abs_errors: list[torch.Tensor] = []
    sq_errors: list[torch.Tensor] = []
    base_abs: list[torch.Tensor] = []
    base_sq: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False, unit="batch"):
            unexpected_types = set(batch["question_type"]) - set(question_types) - {"count"}
            if unexpected_types:
                raise ValueError(
                    f"batch contains question_type(s) {sorted(unexpected_types)} not in "
                    f"question_types={list(question_types)} (or 'count') -- update the "
                    "question_types config/argument to match the actual data, or these "
                    "rows will silently be excluded from the accuracy metrics"
                )
            batch = prepare_batch(batch, device, dataloader.dataset)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(batch)
            outputs = {k: v.float() for k, v in outputs.items()}

            label = batch["label"]
            class_mask = label != -1
            if class_mask.any():
                preds = outputs["class_logits"].argmax(dim=-1)
                correct = preds == label
                for qtype in question_types:
                    type_mask = class_mask & torch.tensor(
                        [t == qtype for t in batch["question_type"]], device=label.device
                    )
                    total_by_type[qtype] += int(type_mask.sum().item())
                    correct_by_type[qtype] += int((correct & type_mask).sum().item())

            count_target = batch["count_target"]
            count_mask = ~count_target.isnan()
            if count_mask.any():
                diff = outputs["count_pred"][count_mask] - count_target[count_mask]
                abs_errors.append(diff.abs())
                sq_errors.append(diff ** 2)
                if baseline_count is not None:
                    base_diff = count_target[count_mask] - baseline_count
                    base_abs.append(base_diff.abs())
                    base_sq.append(base_diff ** 2)

    class_accuracy_by_type = {
        qtype: (correct_by_type[qtype] / total_by_type[qtype]) if total_by_type[qtype] > 0 else float("nan")
        for qtype in question_types
    }
    total_correct = sum(correct_by_type.values())
    total_count = sum(total_by_type.values())
    class_accuracy = (total_correct / total_count) if total_count > 0 else float("nan")

    if abs_errors:
        all_abs = torch.cat(abs_errors)
        all_sq = torch.cat(sq_errors)
        count_mae = all_abs.mean().item()
        count_rmse = all_sq.mean().sqrt().item()
    else:
        count_mae = float("nan")
        count_rmse = float("nan")

    result = {
        "class_accuracy": class_accuracy,
        "class_accuracy_by_type": class_accuracy_by_type,
        "count_mae": count_mae,
        "count_rmse": count_rmse,
    }
    if baseline_count is not None:
        result["count_mae_baseline"] = torch.cat(base_abs).mean().item() if base_abs else float("nan")
        result["count_rmse_baseline"] = torch.cat(base_sq).mean().sqrt().item() if base_sq else float("nan")
    return result


def render_markdown_table(metrics: dict[str, Any]) -> str:
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Classification accuracy (overall) | {metrics['class_accuracy']:.4f} |",
    ]
    for qtype in metrics["class_accuracy_by_type"]:
        lines.append(f"| Classification accuracy ({qtype}) | {metrics['class_accuracy_by_type'][qtype]:.4f} |")
    lines.append(f"| Count MAE | {metrics['count_mae']:.4f} |")
    lines.append(f"| Count RMSE | {metrics['count_rmse']:.4f} |")
    if "count_mae_baseline" in metrics:
        lines.append(f"| Count MAE (always predict train median) | {metrics['count_mae_baseline']:.4f} |")
        lines.append(f"| Count RMSE (always predict train median) | {metrics['count_rmse_baseline']:.4f} |")
    return "\n".join(lines) + "\n"


def _main() -> None:
    from src.data_pipeline import get_dataloaders
    from src.train import build_model, get_device

    run_dir = Path(sys.argv[1])
    config_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("configs/model_config.yaml")
    config = yaml.safe_load(config_path.read_text())

    device = get_device()
    checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    ckpt_config = dict(checkpoint.get("config", config))
    # checkpoints saved before the fusion_type option existed are the concat baseline
    ckpt_config.setdefault("fusion_type", "concat")
    # Model and data must agree on image size, question length and answer vocab, so
    # take those from the checkpoint; file locations come from the live config.
    shape_keys = ("image_size", "max_question_len", "vocab_size")
    eval_config = {**config, **{k: ckpt_config[k] for k in shape_keys}}
    model_config = {**ckpt_config, "data_dir": config["data_dir"], "raw_dir": config["raw_dir"]}

    dataloaders = get_dataloaders(eval_config)
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    baseline_count = train_count_median(dataloaders["train"])
    # every test-like split (HR has two: "test" and the Philadelphia "test_phili")
    sections = []
    for split in [s for s in config["splits"] if s.startswith("test")]:
        metrics = evaluate(
            model, dataloaders[split], device,
            baseline_count=baseline_count, question_types=config["question_types"],
            desc=f"evaluate {split}", mixed_precision=config["mixed_precision"],
        )
        sections.append(f"## Split: {split}\n\n{render_markdown_table(metrics)}")
    report = f"# Results: {run_dir.name}\n\n" + "\n".join(sections)

    out_path = Path("reports") / f"baseline_results_{run_dir.name}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote {out_path}")
    print(report)


if __name__ == "__main__":
    _main()

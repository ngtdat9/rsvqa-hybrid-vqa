"""Evaluation metrics for HybridVQAModel: classification accuracy by question
type, and count-regression MAE/RMSE."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

QUESTION_TYPES = ("comp", "rural_urban", "presence")


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if device is not None:
        model = model.to(device)
    model.eval()

    correct_by_type = {t: 0 for t in QUESTION_TYPES}
    total_by_type = {t: 0 for t in QUESTION_TYPES}
    abs_errors: list[torch.Tensor] = []
    sq_errors: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            if device is not None:
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
            outputs = model(batch)

            label = batch["label"]
            class_mask = label != -1
            if class_mask.any():
                preds = outputs["class_logits"].argmax(dim=-1)
                correct = preds == label
                for qtype in QUESTION_TYPES:
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

    class_accuracy_by_type = {
        qtype: (correct_by_type[qtype] / total_by_type[qtype]) if total_by_type[qtype] > 0 else float("nan")
        for qtype in QUESTION_TYPES
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

    return {
        "class_accuracy": class_accuracy,
        "class_accuracy_by_type": class_accuracy_by_type,
        "count_mae": count_mae,
        "count_rmse": count_rmse,
    }


def render_markdown_table(metrics: dict[str, Any]) -> str:
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Classification accuracy (overall) | {metrics['class_accuracy']:.4f} |",
    ]
    for qtype in QUESTION_TYPES:
        lines.append(f"| Classification accuracy ({qtype}) | {metrics['class_accuracy_by_type'][qtype]:.4f} |")
    lines.append(f"| Count MAE | {metrics['count_mae']:.4f} |")
    lines.append(f"| Count RMSE | {metrics['count_rmse']:.4f} |")
    return "\n".join(lines) + "\n"


def _main() -> None:
    from src.data_pipeline import QuestionVocab, get_dataloaders
    from src.models.vqa_head import HybridVQAModel
    from src.train import get_device

    run_dir = Path(sys.argv[1])
    config = yaml.safe_load(Path("configs/model_config.yaml").read_text())

    dataloaders = get_dataloaders(config)
    question_vocab = QuestionVocab.load(Path(config["data_dir"]) / "vocab" / "question_vocab.json")

    device = get_device()
    checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model = HybridVQAModel(
        image_size=config["image_size"],
        cnn_channels=config["cnn_channels"],
        d_model=config["d_model"],
        num_transformer_layers=config["num_transformer_layers"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        dropout=config["dropout"],
        question_vocab_size=len(question_vocab),
        question_embed_dim=config["question_embed_dim"],
        fusion_hidden_dim=config["fusion_hidden_dim"],
        vocab_size=config["vocab_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(model, dataloaders["test"], device)
    table = render_markdown_table(metrics)

    out_path = Path("reports") / f"baseline_results_{run_dir.name}.md"
    out_path.write_text(f"# Baseline results: {run_dir.name}\n\n{table}")
    print(f"Wrote {out_path}")
    print(table)


if __name__ == "__main__":
    _main()

"""Training loop for HybridVQAModel on RSVQA-LR (Week 1 first pass)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data_pipeline import QuestionVocab, get_dataloaders, prepare_batch
from src.evaluate import evaluate, train_count_median
from src.models.vqa_head import HybridVQAModel, compute_loss


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: dict[str, Any]) -> HybridVQAModel:
    question_vocab = QuestionVocab.load(Path(config["data_dir"]) / "vocab" / "question_vocab.json")
    return HybridVQAModel(
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
        fusion_type=config["fusion_type"],
        max_question_len=config["max_question_len"],
    )


def run_epoch(
    model: HybridVQAModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: dict[str, Any],
    desc: str = "",
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc=desc, leave=False, unit="batch")
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in progress:
            batch = prepare_batch(batch, device, dataloader.dataset)
            outputs = model(batch)
            loss = compute_loss(
                outputs, batch,
                count_loss_weight=config["count_loss_weight"],
                count_transform=config["count_loss_transform"],
            )

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(avg_loss=total_loss / num_batches)

    return total_loss / num_batches


def train(config: dict[str, Any]) -> Path:
    dataloaders = get_dataloaders(config)
    device = get_device()
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    # median train count: the "always predict the median" baseline for count MAE/RMSE
    baseline_count = train_count_median(dataloaders["train"])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config["experiments_dir"]) / f"{config['experiment_name']}_{timestamp}"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    best_val_loss = float("inf")
    patience_counter = 0

    epochs = config["epochs"]
    with open(log_path, "a") as log_file:
        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            train_loss = run_epoch(
                model, dataloaders["train"], optimizer, device, config,
                desc=f"Epoch {epoch + 1}/{epochs} [train]",
            )
            val_loss = run_epoch(
                model, dataloaders["val"], None, device, config,
                desc=f"Epoch {epoch + 1}/{epochs} [val]",
            )
            val_metrics = evaluate(
                model, dataloaders["val"], device,
                baseline_count=baseline_count, question_types=config["question_types"],
                desc=f"Epoch {epoch + 1}/{epochs} [val metrics]",
            )
            tqdm.write(
                f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_class_acc={val_metrics['class_accuracy']:.4f} val_count_mae={val_metrics['count_mae']:.4f}"
            )

            log_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_class_acc": val_metrics["class_accuracy"],
                "val_count_mae": val_metrics["count_mae"],
                "val_count_rmse": val_metrics["count_rmse"],
                "val_count_mae_baseline": val_metrics.get("count_mae_baseline"),
                "timestamp": datetime.now().isoformat(),
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            torch.save(
                {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "epoch": epoch, "config": config},
                checkpoints_dir / "last.pt",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                     "epoch": epoch, "config": config},
                    checkpoints_dir / "best.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= config["early_stopping_patience"]:
                    tqdm.write(
                        f"Early stopping after epoch {epoch + 1}: val_loss has not improved for "
                        f"{patience_counter} epochs (best {best_val_loss:.4f}, kept in best.pt)."
                    )
                    break

    return checkpoints_dir / "best.pt"


if __name__ == "__main__":
    import sys

    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/model_config.yaml")
    model_config = yaml.safe_load(config_path.read_text())
    best_checkpoint = train(model_config)
    print(f"Best checkpoint: {best_checkpoint}")

"""Training loop for HybridVQAModel (RSVQA-LR and RSVQA-HR)."""
from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data_pipeline import QuestionVocab, get_dataloaders, prepare_batch
from src.evaluate import evaluate, selection_score, train_count_median
from src.models.vqa_head import HybridVQAModel, compute_loss

LR_SCHEDULES = ("cosine", "constant")
SELECTION_METRICS = ("score", "val_loss")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (all devices): weight init, shuffle order and
    dropout repeat. GPU kernels can still differ in the last bits between runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, schedule: str) -> float:
    """Factor applied to the base learning rate at optimizer step ``step``.

    ``cosine``: linear warmup from ~0 to 1 over ``warmup_steps``, then cosine decay
    to 0 at ``total_steps``. ``constant``: always 1.
    """
    if schedule == "constant":
        return 1.0
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


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
    scheduler: LambdaLR | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """One pass over ``dataloader``; trains when ``optimizer`` is given. Returns mean loss."""
    model.train(optimizer is not None)
    use_amp = config["mixed_precision"] and device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=False)  # plain backward/step
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc=desc, leave=False, unit="batch")
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in progress:
            batch = prepare_batch(batch, device, dataloader.dataset)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(batch)
                loss = compute_loss(
                    outputs, batch,
                    count_loss_weight=config["count_loss_weight"],
                    count_transform=config["count_loss_transform"],
                )

            if optimizer is not None:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(avg_loss=total_loss / num_batches)

    return total_loss / num_batches


def train(config: dict[str, Any]) -> Path:
    if config["lr_schedule"] not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {config['lr_schedule']!r}")
    selection = config["selection_metric"]
    if selection not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}, got {selection!r}")

    dataloaders = get_dataloaders(config)
    # Seed after the data loaders: building the image cache draws worker seeds from the
    # global RNG, and whether it was built or reused must not change the model's init.
    set_seed(config["seed"])
    device = get_device()
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    total_steps = config["epochs"] * len(dataloaders["train"])
    warmup_steps = int(config["warmup_fraction"] * total_steps)
    scheduler = LambdaLR(
        optimizer, lambda step: lr_multiplier(step, total_steps, warmup_steps, config["lr_schedule"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["mixed_precision"] and device.type == "cuda")
    # median train count: the "always predict the median" baseline for count MAE/RMSE
    baseline_count = train_count_median(dataloaders["train"])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config["experiments_dir"]) / f"{config['experiment_name']}_{timestamp}"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    best = -math.inf  # higher is better: the score itself, or -val_loss
    patience_counter = 0

    epochs = config["epochs"]
    with open(log_path, "a") as log_file:
        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            train_loss = run_epoch(
                model, dataloaders["train"], optimizer, device, config,
                desc=f"Epoch {epoch + 1}/{epochs} [train]", scheduler=scheduler, scaler=scaler,
            )
            val_loss = run_epoch(
                model, dataloaders["val"], None, device, config,
                desc=f"Epoch {epoch + 1}/{epochs} [val]",
            )
            val_metrics = evaluate(
                model, dataloaders["val"], device,
                baseline_count=baseline_count, question_types=config["question_types"],
                desc=f"Epoch {epoch + 1}/{epochs} [val metrics]", mixed_precision=config["mixed_precision"],
            )
            val_score = selection_score(val_metrics)
            lr = optimizer.param_groups[0]["lr"]
            tqdm.write(
                f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_class_acc={val_metrics['class_accuracy']:.4f} val_count_mae={val_metrics['count_mae']:.4f} "
                f"val_score={val_score:.4f} lr={lr:.2e}"
            )

            log_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_class_acc": val_metrics["class_accuracy"],
                "val_count_mae": val_metrics["count_mae"],
                "val_count_rmse": val_metrics["count_rmse"],
                "val_count_mae_baseline": val_metrics.get("count_mae_baseline"),
                "val_score": val_score,
                "lr": lr,
                "timestamp": datetime.now().isoformat(),
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "config": config,
            }
            torch.save(checkpoint, checkpoints_dir / "last.pt")

            current = val_score if selection == "score" else -val_loss
            if current > best:
                best = current
                patience_counter = 0
                torch.save(checkpoint, checkpoints_dir / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= config["early_stopping_patience"]:
                    best_shown = best if selection == "score" else -best
                    tqdm.write(
                        f"Early stopping after epoch {epoch + 1}: {selection} has not improved for "
                        f"{patience_counter} epochs (best {best_shown:.4f}, kept in best.pt)."
                    )
                    break

    return checkpoints_dir / "best.pt"


if __name__ == "__main__":
    import sys

    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/model_config.yaml")
    model_config = yaml.safe_load(config_path.read_text())
    best_checkpoint = train(model_config)
    print(f"Best checkpoint: {best_checkpoint}")

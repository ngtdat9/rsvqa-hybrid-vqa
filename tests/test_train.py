import json
from pathlib import Path

import pytest
import torch
import yaml

from src import data_pipeline as dp
from src.data_pipeline import get_dataloaders
from src.models.vqa_head import compute_loss
from src.train import build_model, train


@pytest.fixture(autouse=True)
def small_qa_records(monkeypatch):
    """Truncate each split to its first 150 QA records so tests run in
    seconds instead of iterating the full 57K-row train split. 150 is large
    enough that the train slice covers all 4 answer classes (verified
    empirically: full coverage reached at record 101)."""
    original = dp.build_qa_records

    def truncated(questions, answers):
        return original(questions, answers)[:150]

    monkeypatch.setattr(dp, "build_qa_records", truncated)

TINY_CONFIG_OVERRIDES = dict(
    image_size=32,
    cnn_channels=[8, 16],
    d_model=32,
    num_transformer_layers=1,
    num_heads=2,
    ffn_dim=64,
    question_embed_dim=32,
    fusion_hidden_dim=32,
    batch_size=8,
    epochs=1,
    early_stopping_patience=1,
)


def make_tiny_config(tmp_path: Path) -> dict:
    config = yaml.safe_load(Path("configs/model_config.yaml").read_text())
    config.update(TINY_CONFIG_OVERRIDES)
    config["data_dir"] = str(tmp_path / "processed")
    config["experiments_dir"] = str(tmp_path / "experiments")
    return config


def test_train_runs_and_writes_checkpoint_and_log(tmp_path):
    config = make_tiny_config(tmp_path)
    best_checkpoint = train(config)

    assert best_checkpoint.exists()

    run_dir = best_checkpoint.parent.parent
    log_path = run_dir / "log.jsonl"
    assert log_path.exists()

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) >= 1
    for line in lines:
        entry = json.loads(line)
        for key in ("epoch", "train_loss", "val_loss", "val_class_acc", "val_count_mae", "val_count_rmse", "timestamp"):
            assert key in entry


def test_overfit_single_batch_loss_decreases(tmp_path):
    config = make_tiny_config(tmp_path)
    dataloaders = get_dataloaders(config)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    batch = next(iter(dataloaders["train"]))

    losses = []
    for _ in range(20):
        outputs = model(batch)
        loss = compute_loss(outputs, batch, count_loss_weight=config["count_loss_weight"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]

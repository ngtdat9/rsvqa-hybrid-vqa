import json
from pathlib import Path

import pytest
import torch
import yaml

from src import data_pipeline as dp
from src.data_pipeline import get_dataloaders
from src.models.vqa_head import compute_loss
from src.train import build_model, lr_multiplier, train


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
        for key in ("epoch", "train_loss", "val_loss", "val_class_acc", "val_count_mae", "val_count_rmse",
                    "val_score", "lr", "timestamp"):
            assert key in entry

    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    assert {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "config"} <= set(checkpoint)


def read_log(best_checkpoint: Path) -> list[dict]:
    lines = (best_checkpoint.parent.parent / "log.jsonl").read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def test_same_seed_gives_the_same_run(tmp_path):
    config = make_tiny_config(tmp_path)
    config["epochs"] = 2
    config["early_stopping_patience"] = 5
    run_a = read_log(train({**config, "experiments_dir": str(tmp_path / "a")}))
    run_b = read_log(train({**config, "experiments_dir": str(tmp_path / "b")}))
    for a, b in zip(run_a, run_b):
        for key in ("train_loss", "val_loss", "val_class_acc", "val_count_mae"):
            assert a[key] == pytest.approx(b[key], rel=1e-5), key
    run_c = read_log(train({**config, "experiments_dir": str(tmp_path / "c"), "seed": 1}))
    assert run_c[0]["train_loss"] != pytest.approx(run_a[0]["train_loss"], rel=1e-5)


def test_cosine_schedule_warms_up_then_decays_to_zero():
    total, warmup = 100, 10
    factors = [lr_multiplier(step, total, warmup, "cosine") for step in range(total)]
    assert factors[0] == pytest.approx(0.1)  # 1/warmup at the first step
    assert factors[warmup - 1] == pytest.approx(1.0)  # peak at the end of warmup
    after = factors[warmup:]
    assert all(x >= y for x, y in zip(after, after[1:]))  # never increases after warmup
    assert lr_multiplier(total, total, warmup, "cosine") == pytest.approx(0.0, abs=1e-12)
    assert all(lr_multiplier(s, total, warmup, "constant") == 1.0 for s in range(total))


def test_lr_follows_the_schedule_during_training(tmp_path):
    config = make_tiny_config(tmp_path)
    config["epochs"] = 2
    config["early_stopping_patience"] = 5
    log = read_log(train(config))
    peak = config["learning_rate"]
    assert 0 < log[0]["lr"] < peak  # past warmup, partway through the decay
    assert log[1]["lr"] == pytest.approx(0.0, abs=peak * 1e-6)  # decayed to 0 at the last step


@pytest.mark.parametrize("key, value", [("lr_schedule", "linear"), ("selection_metric", "accuracy")])
def test_unknown_schedule_or_selection_metric_rejected(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        train({**make_tiny_config(tmp_path), key: value})


def test_overfit_single_batch_loss_decreases(tmp_path):
    config = make_tiny_config(tmp_path)
    dataloaders = get_dataloaders(config)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    batch = dp.prepare_batch(next(iter(dataloaders["train"])), torch.device("cpu"), dataloaders["train"].dataset)

    losses = []
    for _ in range(20):
        outputs = model(batch)
        loss = compute_loss(outputs, batch, count_loss_weight=config["count_loss_weight"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]

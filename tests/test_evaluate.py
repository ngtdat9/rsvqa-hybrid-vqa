import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluate import QUESTION_TYPES, evaluate, selection_score


def test_selection_score_adds_count_gain_over_the_median_baseline():
    metrics = {"class_accuracy": 0.8, "count_mae": 1.0, "count_mae_baseline": 2.0}
    assert selection_score(metrics) == pytest.approx(0.8 + 0.5)
    # count no better than always predicting the median contributes nothing
    assert selection_score({**metrics, "count_mae": 2.0}) == pytest.approx(0.8)


def test_selection_score_is_accuracy_alone_without_count_rows():
    assert selection_score({"class_accuracy": 0.7, "count_mae": float("nan")}) == pytest.approx(0.7)
    assert selection_score({"class_accuracy": 0.7, "count_mae": 1.0, "count_mae_baseline": None}) \
        == pytest.approx(0.7)


class FixedOutputModel(nn.Module):
    """Returns the same hand-chosen outputs regardless of input, so expected
    metrics can be computed by hand."""

    def __init__(self, class_logits: torch.Tensor, count_pred: torch.Tensor) -> None:
        super().__init__()
        self.class_logits = class_logits
        self.count_pred = count_pred

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        batch_size = batch["label"].shape[0]
        return {
            "class_logits": self.class_logits[:batch_size],
            "count_pred": self.count_pred[:batch_size],
        }


class ListDataset(Dataset):
    def __init__(self, batches: list[dict]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> dict:
        return self.batches[idx]


def make_dataloader(batch: dict) -> DataLoader:
    # single-batch "dataloader": wrap the pre-made batch dict so iterating
    # yields it unchanged (batch_size=None avoids re-collation).
    return DataLoader(ListDataset([batch]), batch_size=None, collate_fn=lambda x: x)


def test_class_accuracy_by_type_has_exactly_three_keys():
    batch = {
        "label": torch.tensor([0, 1, -1]),
        "question_type": ["comp", "rural_urban", "count"],
        "count_target": torch.tensor([float("nan"), float("nan"), 5.0]),
    }
    model = FixedOutputModel(
        class_logits=torch.tensor([[10.0, 0, 0, 0], [0, 10.0, 0, 0], [0, 0, 0, 0]]),
        count_pred=torch.tensor([0.0, 0.0, 5.0]),
    )
    result = evaluate(model, make_dataloader(batch))
    assert set(result["class_accuracy_by_type"]) == set(QUESTION_TYPES)


def test_count_rows_excluded_from_classification_and_vice_versa():
    # rows: 0=comp(correct), 1=rural_urban(wrong), 2=count(target=10,pred=8), 3=count(target=4,pred=4)
    batch = {
        "label": torch.tensor([0, 1, -1, -1]),
        "question_type": ["comp", "rural_urban", "count", "count"],
        "count_target": torch.tensor([float("nan"), float("nan"), 10.0, 4.0]),
    }
    model = FixedOutputModel(
        class_logits=torch.tensor(
            [[10.0, 0, 0, 0], [10.0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        ),
        count_pred=torch.tensor([999.0, 999.0, 8.0, 4.0]),  # count_pred on class rows must be ignored
    )
    result = evaluate(model, make_dataloader(batch))

    assert result["class_accuracy_by_type"]["comp"] == 1.0
    assert result["class_accuracy_by_type"]["rural_urban"] == 0.0
    assert math.isnan(result["class_accuracy_by_type"]["presence"])
    assert result["class_accuracy"] == 0.5

    expected_mae = (abs(10.0 - 8.0) + abs(4.0 - 4.0)) / 2
    expected_rmse = (((10.0 - 8.0) ** 2 + (4.0 - 4.0) ** 2) / 2) ** 0.5
    assert math.isclose(result["count_mae"], expected_mae, rel_tol=1e-6)
    assert math.isclose(result["count_rmse"], expected_rmse, rel_tol=1e-6)


def test_median_baseline_metrics():
    # count rows: targets 10 and 4; constant baseline 6 -> abs errors 4 and 2
    batch = {
        "label": torch.tensor([0, -1, -1]),
        "question_type": ["comp", "count", "count"],
        "count_target": torch.tensor([float("nan"), 10.0, 4.0]),
    }
    model = FixedOutputModel(
        class_logits=torch.tensor([[10.0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        count_pred=torch.tensor([0.0, 10.0, 4.0]),
    )
    result = evaluate(model, make_dataloader(batch), baseline_count=6.0)
    assert math.isclose(result["count_mae_baseline"], 3.0, rel_tol=1e-6)
    assert math.isclose(result["count_rmse_baseline"], (((4.0 ** 2) + (2.0 ** 2)) / 2) ** 0.5, rel_tol=1e-6)
    assert result["count_mae"] == 0.0

    without = evaluate(model, make_dataloader(batch))
    assert "count_mae_baseline" not in without


def test_train_count_median_uses_records_not_images():
    from types import SimpleNamespace

    records = [
        {"type": "count", "answer": "10"}, {"type": "count", "answer": "2"},
        {"type": "count", "answer": "4"}, {"type": "presence", "answer": "yes"},
    ]
    loader = SimpleNamespace(dataset=SimpleNamespace(qa_records=records))
    from src.evaluate import train_count_median

    assert train_count_median(loader) == 4.0
    assert train_count_median(SimpleNamespace(dataset=SimpleNamespace(qa_records=[]))) is None


def test_custom_question_types_for_hr():
    batch = {
        "label": torch.tensor([0, 1, -1]),
        "question_type": ["presence", "area", "count"],
        "count_target": torch.tensor([float("nan"), float("nan"), 2.0]),
    }
    model = FixedOutputModel(
        class_logits=torch.tensor([[10.0, 0, 0], [10.0, 0, 0], [0, 0, 0]]),
        count_pred=torch.tensor([0.0, 0.0, 2.0]),
    )
    result = evaluate(model, make_dataloader(batch), question_types=["presence", "comp", "area"])
    assert set(result["class_accuracy_by_type"]) == {"presence", "comp", "area"}
    assert result["class_accuracy_by_type"]["presence"] == 1.0
    assert result["class_accuracy_by_type"]["area"] == 0.0
    assert math.isnan(result["class_accuracy_by_type"]["comp"])


def test_unexpected_question_type_raises_instead_of_silently_dropping():
    batch = {
        "label": torch.tensor([0, 1]),
        "question_type": ["presence", "rural_urban"],  # "rural_urban" not in question_types below
        "count_target": torch.tensor([float("nan"), float("nan")]),
    }
    model = FixedOutputModel(
        class_logits=torch.tensor([[10.0, 0], [0, 10.0]]),
        count_pred=torch.tensor([0.0, 0.0]),
    )
    with pytest.raises(ValueError, match="rural_urban"):
        evaluate(model, make_dataloader(batch), question_types=["presence", "comp"])


def test_loss_kwargs_adds_mean_loss_matching_compute_loss():
    from src.models.vqa_head import compute_loss

    batch = {
        "label": torch.tensor([0, 1, -1]),
        "question_type": ["comp", "comp", "count"],
        "count_target": torch.tensor([float("nan"), float("nan"), 3.0]),
    }
    model = FixedOutputModel(
        class_logits=torch.tensor([[2.0, 0, 0, 0], [2.0, 0, 0, 0], [0, 0, 0, 0]]),
        count_pred=torch.tensor([0.0, 0.0, 1.0]),
    )
    result = evaluate(model, make_dataloader(batch), loss_kwargs={"count_transform": "log1p"})
    expected = compute_loss(model(batch), batch, count_transform="log1p").item()
    assert result["loss"] == pytest.approx(expected)
    assert "loss" not in evaluate(model, make_dataloader(batch))


class BatchNormOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(2)
        self.drop = nn.Dropout(0.5)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        self.saw_dropout_training = self.drop.training
        return {"class_logits": self.bn(batch["image"])}


def test_recalibrate_batchnorm_replaces_corrupted_running_stats():
    from src.evaluate import recalibrate_batchnorm

    torch.manual_seed(0)
    data = torch.randn(64, 2) * torch.tensor([3.0, 0.5]) + torch.tensor([10.0, -2.0])
    batches = [{"image": chunk} for chunk in data.split(16)]
    loader = DataLoader(ListDataset(batches), batch_size=None, collate_fn=lambda x: x)

    model = BatchNormOnly()
    model.bn.running_mean.fill_(1e4)  # what a collapsed run's stats might look like
    model.bn.running_var.fill_(1e6)
    recalibrate_batchnorm(model, loader, torch.device("cpu"), num_batches=4)

    assert torch.allclose(model.bn.running_mean, data.mean(0), atol=1e-4)
    # equal-weight average of the 4 per-batch (unbiased) variances
    expected_var = torch.stack([b["image"].var(0) for b in batches]).mean(0)
    assert torch.allclose(model.bn.running_var, expected_var, atol=1e-4)
    assert not model.saw_dropout_training  # only BatchNorm was in train mode
    assert not model.training and not model.bn.training


def test_max_batches_and_keep_mode():
    batch = {
        "label": torch.tensor([0, 1]),
        "question_type": ["comp", "comp"],
        "count_target": torch.tensor([float("nan"), float("nan")]),
    }
    wrong = {**batch, "label": torch.tensor([1, 0])}
    model = FixedOutputModel(class_logits=torch.tensor([[10.0, 0], [0, 10.0]]), count_pred=torch.zeros(2))
    loader = DataLoader(ListDataset([batch, wrong]), batch_size=None, collate_fn=lambda x: x)
    assert evaluate(model, loader, max_batches=1)["class_accuracy"] == 1.0
    assert evaluate(model, loader)["class_accuracy"] == 0.5

    model.train()
    evaluate(model, loader, keep_mode=True)
    assert model.training
    evaluate(model, loader)
    assert not model.training

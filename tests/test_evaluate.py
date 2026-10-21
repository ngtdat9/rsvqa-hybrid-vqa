import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluate import QUESTION_TYPES, evaluate


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

import json
import math
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.evaluate import count_median, train_count_median
from src.models.clip_baseline import (
    TileEmbeddings,
    answer_candidates,
    answer_text,
    build_prompt,
    encode_tiles,
    majority_accuracy,
    run_baseline,
    score_split,
    select_template,
)
from tests.test_data_layout import hr_config, write_fake_hr

TEMPLATE = "{question} {answer}"
TYPES = ["presence", "comp", "area"]


def record(img_id: int, question: str, qtype: str, answer: str) -> dict:
    return {"img_id": img_id, "question": question, "type": qtype, "answer": answer}


class FakeEncoder:
    """Text embeddings come from a hand-chosen table; unknown sentences get a fixed vector
    that matches no tile. Records every call, so the tests can check what got encoded."""

    def __init__(self, text_vectors: dict[str, list[float]] | None = None) -> None:
        self.text_vectors = text_vectors or {}
        self.texts: list[str] = []
        self.images_encoded = 0
        self.preprocess = lambda image: torch.zeros(2)

    def encode_images(self, pixels: torch.Tensor) -> torch.Tensor:
        self.images_encoded += len(pixels)
        return torch.tensor([1.0, 0.0]).repeat(len(pixels), 1)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        self.texts += texts
        return torch.tensor([self.text_vectors.get(t, [0.0, 0.0]) for t in texts])


def tiles_at(*vectors: list[float]) -> TileEmbeddings:
    return TileEmbeddings({i: i for i in range(len(vectors))}, torch.tensor(vectors))


def test_candidates_come_from_the_given_records_by_type_and_counts_sort_numerically():
    train = [record(0, "q", "count", a) for a in ("10", "2", "0")] + [
        record(0, "q", "presence", "yes"), record(0, "q", "presence", "no"), record(0, "q", "presence", "yes"),
        record(0, "q", "area", "0m2"),
    ]
    assert answer_candidates(train) == {"count": ["0", "2", "10"], "presence": ["no", "yes"], "area": ["0m2"]}


@pytest.mark.parametrize("label, text", [
    ("0m2", "0 square meters"),
    ("1-10m2", "1 to 10 square meters"),
    ("11-100m2", "11 to 100 square meters"),
    ("101-1000m2", "101 to 1000 square meters"),
    (">1000m2", "more than 1000 square meters"),
    ("<=10m2", "at most 10 square meters"),
    ("yes", "yes"),
    ("3", "3"),
])
def test_answer_text(label, text):
    assert answer_text(label) == text


def test_prompt_follows_the_template_and_uses_the_readable_answer():
    assert build_prompt("Question: {question} Answer: {answer}", "What is the area?", "0m2") \
        == "Question: What is the area? Answer: 0 square meters"


def test_score_split_matches_hand_computed_metrics_and_keeps_heads_apart():
    # tile 0 points along x, tile 1 along y; the sentence nearest the tile wins
    tiles = tiles_at([1.0, 0.0], [0.0, 1.0])
    candidates = {"presence": ["no", "yes"], "comp": ["no", "yes"], "area": ["0m2"], "count": ["0", "2", "5"]}
    encoder = FakeEncoder({
        "p no": [1.0, 0.0], "p yes": [0.0, 1.0],   # tile 0 -> no, tile 1 -> yes
        "c 0": [0.0, 1.0], "c 2": [1.0, 0.0], "c 5": [0.0, 0.0],  # tile 0 -> 2, tile 1 -> 0
    })
    records = [
        record(0, "p", "presence", "no"),   # right
        record(1, "p", "presence", "no"),   # wrong
        record(1, "p", "presence", "yes"),  # right
        record(0, "c", "count", "3"),       # predicts 2: error 1
        record(1, "c", "count", "4"),       # predicts 0: error 4
    ]
    metrics = score_split(records, tiles, candidates, TEMPLATE, encoder, TYPES, baseline_count=2.0)

    assert metrics["class_accuracy"] == pytest.approx(2 / 3)
    assert metrics["class_accuracy_by_type"]["presence"] == pytest.approx(2 / 3)
    assert math.isnan(metrics["class_accuracy_by_type"]["comp"])
    assert metrics["count_mae"] == pytest.approx((1 + 4) / 2)
    assert metrics["count_rmse"] == pytest.approx(math.sqrt((1 + 16) / 2))
    assert metrics["count_mae_baseline"] == pytest.approx((1 + 2) / 2)
    assert metrics["count_rmse_baseline"] == pytest.approx(math.sqrt((1 + 4) / 2))


def test_each_unique_questions_sentences_are_encoded_once():
    tiles = tiles_at([1.0, 0.0], [0.0, 1.0])
    candidates = {"presence": ["no", "yes"]}
    encoder = FakeEncoder()
    records = [record(i, q, "presence", "no") for i in (0, 1) for q in ("q1", "q2")]
    # a batch size of 1 flushes after every question, so chunking is exercised too
    metrics = score_split(records, tiles, candidates, TEMPLATE, encoder, TYPES, text_batch_size=1)
    assert sorted(encoder.texts) == ["q1 no", "q1 yes", "q2 no", "q2 yes"]
    assert metrics["sentences_encoded"] == 4


def test_chunking_does_not_change_the_result():
    tiles = tiles_at([1.0, 0.0], [0.0, 1.0])
    candidates = {"presence": ["no", "yes"], "count": ["0", "2"]}
    vectors = {"a no": [1.0, 0.0], "a yes": [0.0, 1.0], "b 0": [0.0, 1.0], "b 2": [1.0, 0.0]}
    records = [record(0, "a", "presence", "no"), record(1, "a", "presence", "no"),
               record(0, "b", "count", "2"), record(1, "b", "count", "2")]
    results = [
        score_split(records, tiles, candidates, TEMPLATE, FakeEncoder(vectors), TYPES, text_batch_size=size)
        for size in (1, 3, 512)
    ]
    assert all(r["class_accuracy"] == pytest.approx(0.5) for r in results)
    assert all(r["count_mae"] == pytest.approx(1.0) for r in results)


def test_unexpected_question_type_raises():
    with pytest.raises(ValueError, match="rural_urban"):
        score_split([record(0, "q", "rural_urban", "rural")], tiles_at([1.0, 0.0]),
                    {"rural_urban": ["rural"]}, TEMPLATE, FakeEncoder(), TYPES)


def test_each_tile_is_encoded_once(tmp_path: Path):
    for img_id in (3, 7):
        Image.new("RGB", (4, 4)).save(tmp_path / f"{img_id}.png")
    encoder = FakeEncoder()
    records = [record(i, q, "presence", "no") for i in (3, 7) for q in ("a", "b", "c")]
    tiles = encode_tiles(records, tmp_path, ".png", encoder, batch_size=1)
    assert encoder.images_encoded == 2
    assert tiles.row_of == {3: 0, 7: 1}
    assert tiles.embeddings.shape == (2, 2)


def test_select_template_picks_the_best_val_score():
    tiles = tiles_at([1.0, 0.0])
    candidates = {"presence": ["no", "yes"]}
    # only the second template's sentences point the tile at the right answer
    encoder = FakeEncoder({"B q no": [0.0, 1.0], "B q yes": [1.0, 0.0],
                           "A q no": [1.0, 0.0], "A q yes": [0.0, 1.0]})
    best, metrics = select_template([record(0, "q", "presence", "yes")], tiles, candidates,
                                   ["A {question} {answer}", "B {question} {answer}"], encoder, TYPES)
    assert best == "B {question} {answer}"
    assert metrics["A {question} {answer}"]["class_accuracy"] == 0.0
    assert metrics["B {question} {answer}"]["class_accuracy"] == 1.0


def test_majority_accuracy_uses_train_answers_only():
    train = [record(0, "q", "presence", "yes")] * 3 + [record(0, "q", "presence", "no"),
                                                        record(0, "q", "count", "5")]
    # "no" is the majority in this split, but "yes" is the train answer that gets scored
    split = [record(1, "q", "presence", "no")] * 3 + [record(1, "q", "presence", "yes"),
                                                       record(1, "q", "count", "1")]
    result = majority_accuracy(train, split)
    assert result["class_accuracy"] == pytest.approx(0.25)
    assert result["class_accuracy_by_type"] == {"presence": pytest.approx(0.25)}


def test_count_median_matches_train_count_median():
    records = [record(0, "q", "count", a) for a in ("1", "9", "2", "4")] + [record(0, "q", "presence", "no")]
    loader = DataLoader(type("D", (), {"qa_records": records, "__len__": lambda self: 0})())
    assert count_median(records) == train_count_median(loader) == pytest.approx(3.0)
    assert count_median([]) is None


def clip_config(tmp_path: Path, **overrides) -> dict:
    return hr_config(tmp_path, clip_model_name="fake/clip-test", clip_batch_size=2, clip_text_batch_size=8,
                     clip_prompt_templates=["A {question} {answer}", "B {question} {answer}"], **overrides)


def test_run_baseline_scores_val_with_every_template_and_test_with_the_winner_only(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    write_fake_hr(raw_dir)
    # one "no" in train makes presence a real choice between two candidates
    answers_path = raw_dir / "USGS_split_train_answers.json"
    answers = json.loads(answers_path.read_text())
    answers["answers"][0]["answer"] = "no"
    answers_path.write_text(json.dumps(answers))

    # every tile is [1, 0], so only template B's "yes" sentence matches; A falls to "no", which is wrong
    question = "Is there a presence thing near the road?"
    encoder = FakeEncoder({f"B {question} yes": [1.0, 0.0]})
    report = run_baseline(clip_config(tmp_path), encoder)

    assert "## Split: test\n" in report and "## Split: test_phili\n" in report
    assert "| Template | Val score | Val accuracy | Val count MAE |" in report
    assert "`B {question} {answer}` | 1.0000 (chosen) | 1.0000 |" in report
    assert "Accuracy, always the most common train answer (overall)" in report
    # 5 sentences per pass: A for val only, B for val and the 2 test splits
    assert sum(t.startswith("A ") for t in encoder.texts) == 5
    assert sum(t.startswith("B ") for t in encoder.texts) == 3 * 5
    # 4 tiles per split: val + test + test_phili, each once
    assert encoder.images_encoded == 3 * 4


def test_pilot_limits_scored_splits_but_not_train_candidates(tmp_path: Path):
    write_fake_hr(tmp_path / "raw")
    config = clip_config(tmp_path)
    config["max_images_per_split"] = {"train": 1, "val": 1, "test": 1, "test_phili": 1}
    encoder = FakeEncoder()
    assert "(pilot subsets)" in run_baseline(config, encoder, pilot=True)
    assert encoder.images_encoded == 3

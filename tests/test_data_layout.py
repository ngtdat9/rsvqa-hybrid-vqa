"""Config-driven dataset layout (LR and HR) on a tiny synthetic RSVQA-style dataset,
so the HR code paths are tested without the 14 GB download."""
import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from src import data_pipeline as dp

SPLITS = ["train", "val", "test", "test_phili"]
TYPES_AND_ANSWERS = [("presence", "yes"), ("comp", "no"), ("area", "between 10m2 and 100m2"), ("count", "3")]


def write_fake_hr(raw_dir: Path, tiles_per_split: int = 4) -> None:
    (raw_dir / "Data").mkdir(parents=True)
    img_id = 0
    for split in SPLITS:
        questions, answers = [], []
        for _ in range(tiles_per_split):
            # random pixels: flat-colour tiles would give a zero channel std
            Image.frombytes("RGB", (64, 64), os.urandom(64 * 64 * 3)).save(raw_dir / "Data" / f"{img_id}.tif")
            for qtype, answer in TYPES_AND_ANSWERS:
                qid = len(questions)
                questions.append({"id": qid, "img_id": img_id, "type": qtype, "active": True,
                                  "question": f"Is there a {qtype} thing near the road?"})
                answers.append({"question_id": qid, "answer": answer, "active": True})
            img_id += 1
        (raw_dir / f"USGS_split_{split}_questions.json").write_text(json.dumps({"questions": questions}))
        (raw_dir / f"USGS_split_{split}_answers.json").write_text(json.dumps({"answers": answers}))


def hr_config(tmp_path: Path, **overrides) -> dict:
    config = dict(
        image_size=32, vocab_size=3, data_dir=str(tmp_path / "processed"), raw_dir=str(tmp_path / "raw"),
        split_file_prefix="USGS_split", images_subdir="Data", image_ext=".tif", splits=SPLITS,
        question_types=["presence", "comp", "area"], norm_stats_max_images=2,
        max_images_per_split=None, subset_seed=0, max_question_len=12, batch_size=4, num_workers=0,
    )
    config.update(overrides)
    return config


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    write_fake_hr(tmp_path / "raw")
    return tmp_path / "raw"


def test_hr_layout_builds_every_configured_split(raw_dir, tmp_path):
    loaders = dp.get_dataloaders(hr_config(tmp_path))
    assert set(loaders) == set(SPLITS)
    batch = next(iter(loaders["test_phili"]))
    assert batch["image"].shape[1:] == (3, 32, 32)
    assert batch["question_tokens"].shape[1] == 12


def test_count_regressed_and_area_classified(raw_dir, tmp_path):
    loaders = dp.get_dataloaders(hr_config(tmp_path))
    ds = loaders["train"].dataset
    for i, record in enumerate(ds.qa_records):
        sample = ds[i]
        if record["type"] == "count":
            assert sample.label == -1 and sample.count_target == 3.0
        else:
            assert sample.label >= 0 and torch.isnan(torch.tensor(sample.count_target))


def test_wrong_vocab_size_fails_loudly_with_the_real_answers(raw_dir, tmp_path):
    with pytest.raises(ValueError, match="between 10m2 and 100m2"):
        dp.get_dataloaders(hr_config(tmp_path, vocab_size=9))


def test_pilot_subset_limits_tiles_per_split(raw_dir, tmp_path):
    loaders = dp.get_dataloaders(hr_config(tmp_path, max_images_per_split={"train": 2, "test_phili": 1}))
    assert len({r["img_id"] for r in loaders["train"].dataset.qa_records}) == 2
    assert len({r["img_id"] for r in loaders["test_phili"].dataset.qa_records}) == 1
    assert len({r["img_id"] for r in loaders["val"].dataset.qa_records}) == 4  # not limited


def test_select_image_subset_keeps_whole_tiles_and_is_deterministic():
    records = [{"img_id": i // 3, "q": i} for i in range(30)]  # 10 tiles x 3 questions
    a = dp.select_image_subset(records, 4, seed=0)
    b = dp.select_image_subset(records, 4, seed=0)
    assert a == b
    per_tile = {}
    for r in a:
        per_tile[r["img_id"]] = per_tile.get(r["img_id"], 0) + 1
    assert len(per_tile) == 4 and set(per_tile.values()) == {3}
    assert dp.select_image_subset(records, None, seed=0) == records


def test_vocab_built_from_full_train_even_with_pilot(raw_dir, tmp_path):
    dp.get_dataloaders(hr_config(tmp_path, max_images_per_split={"train": 1}))
    vocab = dp.AnswerVocab.load(tmp_path / "processed" / "vocab" / "answer_vocab.json")
    assert set(vocab.class_to_idx) == {"yes", "no", "between 10m2 and 100m2"}


def test_reusing_data_dir_for_a_different_raw_dataset_raises(raw_dir, tmp_path):
    shared_processed_dir = tmp_path / "shared_processed"
    config_a = hr_config(tmp_path, data_dir=str(shared_processed_dir))
    dp.get_dataloaders(config_a)  # builds the cache from this raw_dir

    other_raw_dir = tmp_path / "raw_other"
    write_fake_hr(other_raw_dir)
    config_b = hr_config(tmp_path, data_dir=str(shared_processed_dir), raw_dir=str(other_raw_dir))
    with pytest.raises(ValueError, match="data_dir"):
        dp.get_dataloaders(config_b)

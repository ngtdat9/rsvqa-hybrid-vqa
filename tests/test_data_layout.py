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
# area answers use the real HR format: an exact value in m2
TYPES_AND_ANSWERS = [("presence", "yes"), ("comp", "no"), ("area", "45m2"), ("count", "3")]
AREA_BUCKETS = [0, 10, 100, 1000]


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
        question_types=["presence", "comp", "area"], area_buckets=AREA_BUCKETS, norm_stats_max_images=2,
        max_images_per_split=None, subset_seed=0, max_question_len=12, batch_size=4, num_workers=0,
        cache_images=False,
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
    with pytest.raises(ValueError, match="11-100m2"):
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
    assert set(vocab.class_to_idx) == {"yes", "no", "11-100m2"}


def test_reusing_data_dir_for_a_different_raw_dataset_raises(raw_dir, tmp_path):
    shared_processed_dir = tmp_path / "shared_processed"
    config_a = hr_config(tmp_path, data_dir=str(shared_processed_dir))
    dp.get_dataloaders(config_a)  # builds the cache from this raw_dir

    other_raw_dir = tmp_path / "raw_other"
    write_fake_hr(other_raw_dir)
    config_b = hr_config(tmp_path, data_dir=str(shared_processed_dir), raw_dir=str(other_raw_dir))
    with pytest.raises(ValueError, match="data_dir"):
        dp.get_dataloaders(config_b)


@pytest.mark.parametrize("answer, expected", [
    ("0m2", "0m2"),
    ("1m2", "1-10m2"),
    ("10m2", "1-10m2"),
    ("11m2", "11-100m2"),
    ("100m2", "11-100m2"),
    ("101m2", "101-1000m2"),
    ("1000m2", "101-1000m2"),
    ("1001m2", ">1000m2"),
    ("20465m2", ">1000m2"),
])
def test_bucket_area_answer_boundaries(answer, expected):
    assert dp.bucket_area_answer(answer, AREA_BUCKETS) == expected


def test_area_buckets_only_touch_area_records():
    records = [
        {"type": "area", "answer": "640m2"},
        {"type": "count", "answer": "640"},
        {"type": "presence", "answer": "yes"},
    ]
    out = dp.apply_area_buckets(records, AREA_BUCKETS)
    assert [r["answer"] for r in out] == ["101-1000m2", "640", "yes"]
    assert dp.apply_area_buckets(records, None) == records


def test_area_answers_use_bucket_labels_in_every_split(raw_dir, tmp_path):
    loaders = dp.get_dataloaders(hr_config(tmp_path))
    for split in SPLITS:
        area = {r["answer"] for r in loaders[split].dataset.qa_records if r["type"] == "area"}
        assert area == {"11-100m2"}


@pytest.fixture
def decode_counter(monkeypatch):
    """Counts tile decodes (the expensive step the image cache exists to avoid)."""
    calls = []
    original = dp._decode_tile

    def counting(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(dp, "_decode_tile", counting)
    return calls


def test_cached_images_match_uncached_exactly(raw_dir, tmp_path):
    # image_size 64 = the fake tiles' native size, as with real HR (512) and LR (256)
    plain = dp.get_dataloaders(hr_config(tmp_path, image_size=64))["train"].dataset
    cached = dp.get_dataloaders(hr_config(tmp_path, image_size=64, cache_images=True))["train"].dataset
    assert len(plain) == len(cached)
    for i in range(len(plain)):
        assert torch.equal(plain[i].image, cached[i].image)


def test_cache_decodes_each_tile_once(raw_dir, tmp_path, decode_counter):
    loaders = dp.get_dataloaders(hr_config(tmp_path, cache_images=True))
    assert len(decode_counter) == 4 * len(SPLITS)  # 4 tiles per split, decoded while building
    decode_counter.clear()
    ds = loaders["train"].dataset
    for i in range(len(ds)):  # 16 questions over 4 tiles
        assert ds[i].image.shape == (3, 32, 32)
    assert decode_counter == []


def test_tile_cache_reused_then_rebuilt_when_size_changes(raw_dir, tmp_path, decode_counter):
    dp.get_dataloaders(hr_config(tmp_path, cache_images=True))
    decode_counter.clear()
    dp.get_dataloaders(hr_config(tmp_path, cache_images=True))
    assert decode_counter == []  # same tiles and size: reused
    loaders = dp.get_dataloaders(hr_config(tmp_path, cache_images=True, image_size=16))
    assert len(decode_counter) == 4 * len(SPLITS)  # different size: rebuilt
    assert loaders["train"].dataset[0].image.shape == (3, 16, 16)


def test_tile_cache_never_pickles_the_mapped_array(raw_dir, tmp_path):
    import pickle

    ds = dp.get_dataloaders(hr_config(tmp_path, cache_images=True))["train"].dataset
    ds[0]  # opens the memory map in this process
    assert len(pickle.dumps(ds.tile_cache)) < 10_000


def test_changing_area_buckets_on_a_cached_data_dir_raises(raw_dir, tmp_path):
    dp.get_dataloaders(hr_config(tmp_path))
    with pytest.raises(ValueError, match="delete"):
        dp.get_dataloaders(hr_config(tmp_path, area_buckets=[0, 50, 500]))

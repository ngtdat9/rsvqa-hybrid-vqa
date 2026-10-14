import math
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from src import data_pipeline as dp

RAW_DIR = Path("data/raw/rsvqa_lr")
IMAGES_DIR = RAW_DIR / "images" / "Images_LR"
CONFIG = yaml.safe_load(Path("configs/model_config.yaml").read_text())


@pytest.fixture(scope="module")
def split_records() -> dict[str, list[dict]]:
    records = {}
    for split in dp.SPLITS:
        questions, answers = dp.load_split_raw(RAW_DIR, split)
        records[split] = dp.build_qa_records(questions, answers)
    return records


@pytest.fixture(scope="module")
def train_question_vocab(split_records) -> dp.QuestionVocab:
    return dp.QuestionVocab.build([r["question"] for r in split_records["train"]])


@pytest.fixture(scope="module")
def train_answer_vocab(split_records) -> dp.AnswerVocab:
    return dp.AnswerVocab.build(split_records["train"])


def test_active_counts_match_known_values(split_records):
    assert len(split_records["train"]) == 57223
    assert len(split_records["val"]) == 10005
    assert len(split_records["test"]) == 10004
    assert len({r["img_id"] for r in split_records["train"]}) == 572
    assert len({r["img_id"] for r in split_records["val"]}) == 100
    assert len({r["img_id"] for r in split_records["test"]}) == 100


def test_answer_vocab_is_exactly_four_non_count_classes(train_answer_vocab):
    assert len(train_answer_vocab) == 4 == CONFIG["vocab_size"]
    assert set(train_answer_vocab.class_to_idx) == {"yes", "no", "rural", "urban"}


def test_question_vocab_built_from_train_only(split_records, train_question_vocab):
    train_words = {w for r in split_records["train"] for w in dp.tokenize(r["question"])}
    other_words = {
        w
        for split in ("val", "test")
        for r in split_records[split]
        for w in dp.tokenize(r["question"])
    }
    unseen = other_words - train_words
    assert unseen, "expected at least one val/test-only word to test <unk> routing"
    word = next(iter(unseen))
    assert train_question_vocab.word_to_idx.get(word, train_question_vocab.unk_idx) == train_question_vocab.unk_idx


def test_count_questions_excluded_from_answer_vocab(train_answer_vocab):
    for answer in train_answer_vocab.class_to_idx:
        assert not answer.isdigit()


def test_train_val_test_image_ids_are_disjoint(split_records):
    train_ids = {r["img_id"] for r in split_records["train"]}
    val_ids = {r["img_id"] for r in split_records["val"]}
    test_ids = {r["img_id"] for r in split_records["test"]}
    assert train_ids & val_ids == set()
    assert train_ids & test_ids == set()
    assert val_ids & test_ids == set()


@pytest.fixture(scope="module")
def norm_stats(split_records) -> tuple[torch.Tensor, torch.Tensor]:
    train_img_ids = sorted({r["img_id"] for r in split_records["train"]})
    return dp.compute_image_norm_stats([IMAGES_DIR / f"{i}.tif" for i in train_img_ids])


@pytest.fixture
def dataset(split_records, train_question_vocab, train_answer_vocab, norm_stats):
    mean, std = norm_stats
    return dp.RSVQADataset(
        qa_records=split_records["train"],
        images_dir=IMAGES_DIR,
        question_vocab=train_question_vocab,
        answer_vocab=train_answer_vocab,
        image_mean=mean,
        image_std=std,
        image_size=CONFIG["image_size"],
        max_question_len=CONFIG["max_question_len"],
    )


def test_getitem_returns_correctly_shaped_tensors(dataset):
    sample = dataset[0]
    assert sample.image.shape == (3, CONFIG["image_size"], CONFIG["image_size"])
    assert sample.image.dtype == torch.float32
    assert sample.question_tokens.shape == (CONFIG["max_question_len"],)
    assert sample.question_tokens.dtype == torch.long


def test_count_question_routes_to_regression_not_classification(dataset, split_records):
    idx = next(i for i, r in enumerate(split_records["train"]) if r["type"] == "count")
    sample = dataset[idx]
    assert sample.label == -1
    assert not math.isnan(sample.count_target)


def test_non_count_question_routes_to_classification_not_regression(dataset, split_records):
    idx = next(i for i, r in enumerate(split_records["train"]) if r["type"] != "count")
    sample = dataset[idx]
    assert 0 <= sample.label < CONFIG["vocab_size"]
    assert math.isnan(sample.count_target)


def test_collate_fn_batches_mixed_question_types(dataset, split_records):
    count_idx = next(i for i, r in enumerate(split_records["train"]) if r["type"] == "count")
    non_count_idx = next(i for i, r in enumerate(split_records["train"]) if r["type"] != "count")
    batch = [dataset[count_idx], dataset[non_count_idx]]
    collated = dp.vqa_collate_fn(batch)
    assert collated["image"].shape == (2, 3, CONFIG["image_size"], CONFIG["image_size"])
    assert collated["label"].tolist().count(-1) == 1
    assert torch.isnan(collated["count_target"]).sum().item() == 1
    assert len(collated["question_type"]) == 2


def test_dataloader_end_to_end_one_batch(tmp_path):
    config = dict(CONFIG)
    config["data_dir"] = str(tmp_path / "processed")
    dataloaders = dp.get_dataloaders(config)
    batch = next(iter(dataloaders["train"]))
    assert batch["image"].shape[1:] == (3, config["image_size"], config["image_size"])
    assert batch["question_tokens"].shape[1] == config["max_question_len"]


def test_image_norm_stats_fit_on_train_split_only(split_records, norm_stats):
    train_img_ids = sorted({r["img_id"] for r in split_records["train"]})
    expected_mean, expected_std = dp.compute_image_norm_stats(
        [IMAGES_DIR / f"{i}.tif" for i in train_img_ids]
    )
    mean, std = norm_stats
    assert torch.allclose(mean, expected_mean)
    assert torch.allclose(std, expected_std)


def test_missing_image_file_raises_clear_error(split_records, train_question_vocab, train_answer_vocab, norm_stats, tmp_path):
    mean, std = norm_stats
    dataset = dp.RSVQADataset(
        qa_records=split_records["train"][:1],
        images_dir=tmp_path,
        question_vocab=train_question_vocab,
        answer_vocab=train_answer_vocab,
        image_mean=mean,
        image_std=std,
        image_size=CONFIG["image_size"],
        max_question_len=CONFIG["max_question_len"],
    )
    with pytest.raises(RuntimeError):
        dataset[0]

"""RSVQA-LR data pipeline: raw JSON/images -> training-ready batches.

Count-type questions are a regression target (scalar), not part of the
classification vocab -- see CLAUDE.md and RSVQA_Project_Plan.md.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------

def load_split_raw(raw_dir: Path, split: str) -> tuple[list[dict], list[dict]]:
    with open(raw_dir / f"LR_split_{split}_questions.json") as f:
        questions = json.load(f)["questions"]
    with open(raw_dir / f"LR_split_{split}_answers.json") as f:
        answers = json.load(f)["answers"]
    return questions, answers


def filter_active(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["active"]]


def build_qa_records(questions: list[dict], answers: list[dict]) -> list[dict]:
    """Join active answers to active questions by question_id."""
    active_questions = {q["id"]: q for q in filter_active(questions)}
    records = []
    for answer in filter_active(answers):
        question = active_questions[answer["question_id"]]
        records.append(
            {
                "img_id": question["img_id"],
                "question": question["question"],
                "type": question["type"],
                "answer": answer["answer"],
            }
        )
    return records


def tokenize(question: str) -> list[str]:
    return question.rstrip("?").lower().split()


# ---------------------------------------------------------------------------
# Vocab (train-split-only)
# ---------------------------------------------------------------------------

class QuestionVocab:
    def __init__(self, word_to_idx: dict[str, int]) -> None:
        self.word_to_idx = word_to_idx
        self.idx_to_word = sorted(word_to_idx, key=word_to_idx.get)
        self.pad_idx = word_to_idx[PAD_TOKEN]
        self.unk_idx = word_to_idx[UNK_TOKEN]

    @classmethod
    def build(cls, questions: list[str], min_freq: int = 1) -> "QuestionVocab":
        counts: dict[str, int] = {}
        for question in questions:
            for word in tokenize(question):
                counts[word] = counts.get(word, 0) + 1
        word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for word in sorted(counts):
            if counts[word] >= min_freq:
                word_to_idx[word] = len(word_to_idx)
        return cls(word_to_idx)

    def encode(self, question: str, max_len: int) -> torch.Tensor:
        ids = [self.word_to_idx.get(word, self.unk_idx) for word in tokenize(question)]
        ids = ids[:max_len]
        ids = ids + [self.pad_idx] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"word_to_idx": self.word_to_idx}))

    @classmethod
    def load(cls, path: Path) -> "QuestionVocab":
        return cls(json.loads(path.read_text())["word_to_idx"])

    def __len__(self) -> int:
        return len(self.word_to_idx)


class AnswerVocab:
    """Classification vocab for non-count answers only."""

    def __init__(self, class_to_idx: dict[str, int]) -> None:
        self.class_to_idx = class_to_idx
        self.idx_to_class = sorted(class_to_idx, key=class_to_idx.get)

    @classmethod
    def build(cls, qa_records: list[dict]) -> "AnswerVocab":
        answers = sorted({r["answer"] for r in qa_records if r["type"] != "count"})
        return cls({answer: idx for idx, answer in enumerate(answers)})

    def encode(self, answer: str) -> int:
        return self.class_to_idx[answer]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"class_to_idx": self.class_to_idx}))

    @classmethod
    def load(cls, path: Path) -> "AnswerVocab":
        return cls(json.loads(path.read_text())["class_to_idx"])

    def __len__(self) -> int:
        return len(self.class_to_idx)


# ---------------------------------------------------------------------------
# Preprocessing stats (train-split-only)
# ---------------------------------------------------------------------------

def compute_image_norm_stats(image_paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0
    for path in image_paths:
        with Image.open(path) as im:
            arr = TF.to_tensor(im.convert("RGB")).double()
        total += arr.sum(dim=(1, 2))
        total_sq += (arr ** 2).sum(dim=(1, 2))
        n_pixels += arr.shape[1] * arr.shape[2]
    mean = total / n_pixels
    var = total_sq / n_pixels - mean ** 2
    return mean.float(), var.clamp(min=0).sqrt().float()


def save_norm_stats(mean: torch.Tensor, std: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}))


def load_norm_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = json.loads(path.read_text())
    return torch.tensor(data["mean"]), torch.tensor(data["std"])


# ---------------------------------------------------------------------------
# Cached-artifact builder
# ---------------------------------------------------------------------------

def build_processed_artifacts(raw_dir: Path, processed_dir: Path, config: dict[str, Any]) -> None:
    vocab_dir = processed_dir / "vocab"
    question_vocab_path = vocab_dir / "question_vocab.json"
    answer_vocab_path = vocab_dir / "answer_vocab.json"
    norm_stats_path = vocab_dir / "image_norm_stats.json"
    if question_vocab_path.exists() and answer_vocab_path.exists() and norm_stats_path.exists():
        return

    train_questions, train_answers = load_split_raw(raw_dir, "train")
    train_records = build_qa_records(train_questions, train_answers)

    question_vocab = QuestionVocab.build([r["question"] for r in train_records])
    question_vocab.save(question_vocab_path)

    answer_vocab = AnswerVocab.build(train_records)
    answer_vocab.save(answer_vocab_path)

    images_dir = raw_dir / "images" / "Images_LR"
    train_img_ids = sorted({r["img_id"] for r in train_records})
    mean, std = compute_image_norm_stats([images_dir / f"{img_id}.tif" for img_id in train_img_ids])
    save_norm_stats(mean, std, norm_stats_path)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class VQASample:
    image: torch.Tensor
    question_tokens: torch.Tensor
    question_len: int
    question_type: str
    label: int
    count_target: float


class RSVQADataset(Dataset):
    def __init__(
        self,
        qa_records: list[dict],
        images_dir: Path,
        question_vocab: QuestionVocab,
        answer_vocab: AnswerVocab,
        image_mean: torch.Tensor,
        image_std: torch.Tensor,
        image_size: int,
        max_question_len: int,
        transform: Callable | None = None,
    ) -> None:
        self.qa_records = qa_records
        self.images_dir = images_dir
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab
        self.image_mean = image_mean
        self.image_std = image_std
        self.image_size = image_size
        self.max_question_len = max_question_len
        self.transform = transform

    def __len__(self) -> int:
        return len(self.qa_records)

    def __getitem__(self, idx: int) -> VQASample:
        record = self.qa_records[idx]
        image_path = self.images_dir / f"{record['img_id']}.tif"
        try:
            with Image.open(image_path) as im:
                image = im.convert("RGB").copy()
        except (OSError, Image.UnidentifiedImageError) as e:
            raise RuntimeError(f"Failed to load image {image_path.name}: {e}") from e

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.resize(image_tensor, [self.image_size, self.image_size])
        image_tensor = TF.normalize(image_tensor, self.image_mean.tolist(), self.image_std.tolist())
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        tokens = tokenize(record["question"])
        question_tokens = self.question_vocab.encode(record["question"], self.max_question_len)
        question_len = min(len(tokens), self.max_question_len)

        if record["type"] == "count":
            label = -1
            count_target = float(record["answer"])
        else:
            label = self.answer_vocab.encode(record["answer"])
            count_target = float("nan")

        return VQASample(
            image=image_tensor,
            question_tokens=question_tokens,
            question_len=question_len,
            question_type=record["type"],
            label=label,
            count_target=count_target,
        )


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def vqa_collate_fn(batch: list[VQASample]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "image": torch.stack([s.image for s in batch]),
        "question_tokens": torch.stack([s.question_tokens for s in batch]),
        "question_len": torch.tensor([s.question_len for s in batch], dtype=torch.long),
        "question_type": [s.question_type for s in batch],
        "label": torch.tensor([s.label for s in batch], dtype=torch.long),
        "count_target": torch.tensor([s.count_target for s in batch], dtype=torch.float),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    raw_dir = Path(config["raw_dir"])
    processed_dir = Path(config["data_dir"])
    images_dir = raw_dir / "images" / "Images_LR"

    build_processed_artifacts(raw_dir, processed_dir, config)
    question_vocab = QuestionVocab.load(processed_dir / "vocab" / "question_vocab.json")
    answer_vocab = AnswerVocab.load(processed_dir / "vocab" / "answer_vocab.json")
    image_mean, image_std = load_norm_stats(processed_dir / "vocab" / "image_norm_stats.json")

    dataloaders = {}
    for split in SPLITS:
        questions, answers = load_split_raw(raw_dir, split)
        qa_records = build_qa_records(questions, answers)
        dataset = RSVQADataset(
            qa_records=qa_records,
            images_dir=images_dir,
            question_vocab=question_vocab,
            answer_vocab=answer_vocab,
            image_mean=image_mean,
            image_std=image_std,
            image_size=config["image_size"],
            max_question_len=config["max_question_len"],
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=(split == "train"),
            collate_fn=vqa_collate_fn,
        )
    return dataloaders

"""RSVQA data pipeline (LR and HR): raw JSON/images -> training-ready batches.

Which dataset is read is entirely config-driven: ``split_file_prefix``
("LR_split" / "USGS_split"), ``images_subdir``, ``image_ext`` and ``splits``.

Count-type questions are a regression target (scalar), not part of the
classification vocab -- see CLAUDE.md and RSVQA_Project_Plan.md.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------

def load_split_raw(
    raw_dir: Path, split: str, prefix: str = "LR_split"
) -> tuple[list[dict], list[dict]]:
    with open(raw_dir / f"{prefix}_{split}_questions.json") as f:
        questions = json.load(f)["questions"]
    with open(raw_dir / f"{prefix}_{split}_answers.json") as f:
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


def bucket_area_answer(answer: str, edges: list[int]) -> str:
    """Map an exact RSVQA-HR area answer (e.g. '1234m2') to a range label.

    ``edges`` are inclusive upper bounds in whole m2: [0, 10, 100, 1000] gives
    '0m2', '1-10m2', '11-100m2', '101-1000m2' and '>1000m2'.
    """
    value = float(answer.removesuffix("m2"))
    lower = None
    for edge in edges:
        if value <= edge:
            if lower is None:
                return f"{edge}m2" if edge == 0 else f"<={edge}m2"
            return f"{lower + 1}-{edge}m2"
        lower = edge
    return f">{edges[-1]}m2"


def apply_area_buckets(records: list[dict], edges: list[int] | None) -> list[dict]:
    """Replace each area answer with its range label (no-op when ``edges`` is None)."""
    if edges is None:
        return records
    return [
        {**r, "answer": bucket_area_answer(r["answer"], edges)} if r["type"] == "area" else r
        for r in records
    ]


def _sample_ids(ids: list[int], n: int, seed: int) -> list[int]:
    """A deterministic, seeded sample of ``n`` ids (or all of them if there
    are fewer than ``n``), used for both the pilot-subset and norm-stats
    sampling below."""
    if len(ids) <= n:
        return ids
    return random.Random(seed).sample(ids, n)


def select_image_subset(records: list[dict], max_images: int | None, seed: int) -> list[dict]:
    """Keep only the QA records of up to ``max_images`` tiles (all of a tile's
    questions stay together). Deterministic for a given seed."""
    if max_images is None:
        return records
    img_ids = sorted({r["img_id"] for r in records})
    if len(img_ids) <= max_images:
        return records
    keep = set(_sample_ids(img_ids, max_images, seed))
    return [r for r in records if r["img_id"] in keep]


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

def _dataset_manifest(raw_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Identifies which raw dataset/layout (and area bucketing) a data_dir's
    cached artifacts came from, so stale vocab/norm-stats are caught instead
    of silently reused."""
    return {
        "raw_dir": str(raw_dir),
        "split_file_prefix": config["split_file_prefix"],
        "images_subdir": config["images_subdir"],
        "image_ext": config["image_ext"],
        "area_buckets": config["area_buckets"],
    }


def build_processed_artifacts(raw_dir: Path, processed_dir: Path, config: dict[str, Any]) -> None:
    vocab_dir = processed_dir / "vocab"
    question_vocab_path = vocab_dir / "question_vocab.json"
    answer_vocab_path = vocab_dir / "answer_vocab.json"
    norm_stats_path = vocab_dir / "image_norm_stats.json"
    manifest_path = vocab_dir / "manifest.json"
    manifest = _dataset_manifest(raw_dir, config)

    if question_vocab_path.exists() and answer_vocab_path.exists() and norm_stats_path.exists() \
            and manifest_path.exists():
        cached_manifest = json.loads(manifest_path.read_text())
        if cached_manifest != manifest:
            raise ValueError(
                f"data_dir={processed_dir} already holds cached vocab/norm-stats built from "
                f"{cached_manifest}, but the current config describes {manifest}. "
                f"Use a separate data_dir per dataset, or delete {vocab_dir} to rebuild it."
            )
        return

    prefix = config["split_file_prefix"]
    train_questions, train_answers = load_split_raw(raw_dir, "train", prefix)
    # vocabularies come from the FULL train split, even when training on a pilot subset
    train_records = apply_area_buckets(build_qa_records(train_questions, train_answers), config["area_buckets"])

    question_vocab = QuestionVocab.build([r["question"] for r in train_records])
    question_vocab.save(question_vocab_path)

    answer_vocab = AnswerVocab.build(train_records)
    answer_vocab.save(answer_vocab_path)

    images_dir = raw_dir / config["images_subdir"]
    train_img_ids = sorted({r["img_id"] for r in train_records})
    # a seeded sample of train tiles is enough for per-channel mean/std
    train_img_ids = sorted(_sample_ids(train_img_ids, config["norm_stats_max_images"], config["subset_seed"]))
    ext = config["image_ext"]
    mean, std = compute_image_norm_stats([images_dir / f"{img_id}{ext}" for img_id in train_img_ids])
    save_norm_stats(mean, std, norm_stats_path)

    manifest_path.write_text(json.dumps(manifest))


# ---------------------------------------------------------------------------
# Decoded-tile cache
# ---------------------------------------------------------------------------

def _decode_tile(image_path: Path) -> Image.Image:
    try:
        with Image.open(image_path) as im:
            return im.convert("RGB").copy()
    except (OSError, Image.UnidentifiedImageError) as e:
        raise RuntimeError(f"Failed to load image {image_path.name}: {e}") from e


class _TileDecodeDataset(Dataset):
    """Decodes one tile per item to uint8 HWC at ``image_size`` (used to fill the cache)."""

    def __init__(self, paths: list[Path], image_size: int) -> None:
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> np.ndarray:
        pixels = TF.pil_to_tensor(_decode_tile(self.paths[idx]))  # uint8 [3,H,W]
        if pixels.shape[1:] != (self.image_size, self.image_size):
            pixels = TF.resize(pixels, [self.image_size, self.image_size], antialias=True)
        return pixels.permute(1, 2, 0).numpy()


class TileCache:
    """Decoded, resized RGB tiles (uint8, [N,H,W,3]) in a .npy file on local disk.

    Every tile has ~100 questions, so caching decodes each tile once instead of
    once per question. The file is memory-mapped, not loaded into RAM.
    """

    def __init__(self, path: Path, img_ids: list[int]) -> None:
        self.path = path
        self.row_of = {img_id: row for row, img_id in enumerate(img_ids)}
        self._array: np.ndarray | None = None

    def get(self, img_id: int) -> np.ndarray:
        if self._array is None:  # opened lazily, so each DataLoader worker maps it itself
            self._array = np.load(self.path, mmap_mode="r")
        return self._array[self.row_of[img_id]]

    def __getstate__(self) -> dict[str, Any]:
        # never pickle the mapped array into worker processes
        return {**self.__dict__, "_array": None}


def build_tile_cache(
    img_ids: list[int],
    images_dir: Path,
    image_ext: str,
    image_size: int,
    cache_dir: Path,
    name: str,
    num_workers: int,
) -> TileCache:
    """Return a cache for ``img_ids``, decoding them only if no matching cache exists.

    A cache is reused when its recorded tiles, image folder, extension and size all
    match; otherwise (e.g. pilot -> full run) it is rebuilt.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f"{name}.npy"
    meta_path = cache_dir / f"{name}.json"
    meta = {"images_dir": str(images_dir), "image_ext": image_ext, "image_size": image_size, "img_ids": img_ids}
    if data_path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == meta:
        return TileCache(data_path, img_ids)

    meta_path.unlink(missing_ok=True)  # an interrupted build must not look complete
    array = np.lib.format.open_memmap(
        data_path, mode="w+", dtype=np.uint8, shape=(len(img_ids), image_size, image_size, 3)
    )
    paths = [images_dir / f"{img_id}{image_ext}" for img_id in img_ids]
    loader = DataLoader(_TileDecodeDataset(paths, image_size), batch_size=None, num_workers=num_workers)
    for row, pixels in enumerate(tqdm(loader, desc=f"caching {name} tiles", unit="tile")):
        array[row] = pixels
    array.flush()
    del array
    meta_path.write_text(json.dumps(meta))
    return TileCache(data_path, img_ids)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class VQASample:
    image: torch.Tensor  # uint8 [3,H,W]; see prepare_batch
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
        image_ext: str = ".tif",
        tile_cache: TileCache | None = None,
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
        self.image_ext = image_ext
        self.tile_cache = tile_cache

    def __len__(self) -> int:
        return len(self.qa_records)

    def _load_image(self, img_id: int) -> torch.Tensor:
        """uint8 [3, image_size, image_size]; normalized later by ``prepare_batch``."""
        if self.tile_cache is not None:
            pixels = torch.from_numpy(np.array(self.tile_cache.get(img_id)))  # copy: the map is read-only
            return pixels.permute(2, 0, 1).contiguous()
        pixels = TF.pil_to_tensor(_decode_tile(self.images_dir / f"{img_id}{self.image_ext}"))
        if pixels.shape[1:] != (self.image_size, self.image_size):
            pixels = TF.resize(pixels, [self.image_size, self.image_size], antialias=True)
        return pixels

    def __getitem__(self, idx: int) -> VQASample:
        record = self.qa_records[idx]
        image_tensor = self._load_image(record["img_id"])
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


def prepare_batch(batch: dict[str, Any], device: torch.device | None, dataset: Any) -> dict[str, Any]:
    """Move a batch to ``device`` and turn its uint8 images into normalized floats there.

    Images leave the DataLoader workers as uint8, a quarter of the bytes of float32,
    and converting/normalizing a whole batch on the GPU is far cheaper than doing it
    per sample on the workers' CPU cores. ``dataset`` supplies the train-split
    ``image_mean``/``image_std``; it is only read when the batch has uint8 images.
    """
    if device is not None:
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
    image = batch.get("image")
    if image is not None and image.dtype == torch.uint8:
        mean = dataset.image_mean.to(image.device).view(1, -1, 1, 1)
        std = dataset.image_std.to(image.device).view(1, -1, 1, 1)
        batch = {**batch, "image": (image.float().div(255) - mean) / std}
    return batch


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    raw_dir = Path(config["raw_dir"])
    processed_dir = Path(config["data_dir"])
    images_dir = raw_dir / config["images_subdir"]
    prefix = config["split_file_prefix"]

    build_processed_artifacts(raw_dir, processed_dir, config)
    question_vocab = QuestionVocab.load(processed_dir / "vocab" / "question_vocab.json")
    answer_vocab = AnswerVocab.load(processed_dir / "vocab" / "answer_vocab.json")
    image_mean, image_std = load_norm_stats(processed_dir / "vocab" / "image_norm_stats.json")
    if len(answer_vocab) != config["vocab_size"]:
        raise ValueError(
            f"config vocab_size={config['vocab_size']} but the train split has "
            f"{len(answer_vocab)} non-count answers: {answer_vocab.idx_to_class}"
        )

    max_images_per_split = config["max_images_per_split"] or {}
    dataloaders = {}
    for split in config["splits"]:
        questions, answers = load_split_raw(raw_dir, split, prefix)
        qa_records = apply_area_buckets(build_qa_records(questions, answers), config["area_buckets"])
        qa_records = select_image_subset(qa_records, max_images_per_split.get(split), config["subset_seed"])
        tile_cache = None
        if config["cache_images"]:
            tile_cache = build_tile_cache(
                sorted({r["img_id"] for r in qa_records}), images_dir, config["image_ext"],
                config["image_size"], processed_dir / "image_cache", split, config["num_workers"],
            )
        dataset = RSVQADataset(
            qa_records=qa_records,
            images_dir=images_dir,
            question_vocab=question_vocab,
            answer_vocab=answer_vocab,
            image_mean=image_mean,
            image_std=image_std,
            image_size=config["image_size"],
            max_question_len=config["max_question_len"],
            image_ext=config["image_ext"],
            tile_cache=tile_cache,
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=(split == "train"),
            collate_fn=vqa_collate_fn,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),  # lets prepare_batch's GPU copy overlap compute
        )
    return dataloaders

"""The Week 7 demo bundle: everything the local app needs, exported once on Colab where the full HR data is.

The app runs on a Mac without the 14 GB dataset. It needs the demo tiles it can be asked about, what
each retrieval-index row is (its question, true answer and tile), small pictures of the training tiles
for the neighbour cards, and precomputed results for the showcase questions so the demo is instant.

Demo tiles are chosen by rule, per test split, walking the seeded question pool (one question per tile
and type) in its random order: first one tile with a warned question per type, then tiles whose answer
is wrong without a warning, then tiles answered right. The showcase questions are the pool questions of
the demo tiles; their facts, occlusion map and explanation are computed here, on the GPU, from the same
index and neighbours the reported warning statistics used.

    python -m src.demo.bundle <run_dir> configs/hr_config.yaml --index-dir DIR [--out DIR] [--no-llm]

Writes ``<demo_bundle_dir>/``: ``checkpoint.pt``, ``vocab/`` (vocabularies and the train-split norm
stats), ``config.json``, ``tiles/<img_id>.png``, ``demo_records.json``, ``train_table.npz`` with
``train_strings.json``, ``thumbs/<img_id>.jpg``, ``cache.json`` and ``manifest.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data_pipeline import AnswerVocab, prepare_batch, vqa_collate_fn
from src.evaluate import load_run
from src.rag.explain_agent import (
    LLMClient,
    OpenAICompatibleClient,
    PoolResult,
    _grid,
    build_facts,
    explain,
    score_pool,
)
from src.rag.retrieval_index import COUNT, RetrievalIndex, _Tiles, build_index


# ---------------------------------------------------------------------------
# What each index row is
# ---------------------------------------------------------------------------

class _Records(Sequence):
    """The table's rows as the record dicts the RAG code reads, built on access: 625K dicts at once
    would take hundreds of MB on an 8 GB laptop."""

    def __init__(self, table: "TrainTable") -> None:
        self.table = table

    def __len__(self) -> int:
        return len(self.table.img_id)

    def __getitem__(self, i: int) -> dict[str, Any]:  # type: ignore[override]
        t = self.table
        return {"img_id": int(t.img_id[i]), "question": t.questions[t.question_code[i]],
                "type": t.types[t.type_code[i]], "answer": t.answers[t.answer_code[i]]}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return (self[i] for i in range(len(self)))


@dataclass
class TrainTable:
    """Per retrieval-index row (same order as the index): tile id, question type, question and true
    answer, stored as codes into small string tables."""
    img_id: np.ndarray         # [N] int32
    type_code: np.ndarray      # [N] int8
    question_code: np.ndarray  # [N] int32
    answer_code: np.ndarray    # [N] int32
    types: list[str]
    questions: list[str]
    answers: list[str]

    @classmethod
    def from_records(cls, records: Sequence[dict[str, Any]]) -> "TrainTable":
        tables: dict[str, dict[str, int]] = {"type": {}, "question": {}, "answer": {}}
        codes = {name: np.empty(len(records), dtype=np.int64) for name in tables}
        img_id = np.empty(len(records), dtype=np.int32)
        for i, record in enumerate(records):
            img_id[i] = record["img_id"]
            for name, table in tables.items():
                codes[name][i] = table.setdefault(record[name], len(table))
        return cls(img_id=img_id, type_code=codes["type"].astype(np.int8),
                   question_code=codes["question"].astype(np.int32), answer_code=codes["answer"].astype(np.int32),
                   types=list(tables["type"]), questions=list(tables["question"]), answers=list(tables["answer"]))

    @property
    def qa_records(self) -> _Records:
        return _Records(self)

    def type_list(self) -> list[str]:
        return [self.types[c] for c in self.type_code]

    def truths(self, answer_vocab: AnswerVocab) -> np.ndarray:
        """The true answer of every row as the RAG code compares them: the count, or the class index."""
        count_value = np.array([float(a) if _is_number(a) else np.nan for a in self.answers])
        class_value = np.array([answer_vocab.class_to_idx.get(a, -1) for a in self.answers], dtype=np.float64)
        is_count = np.array([t == COUNT for t in self.types])[self.type_code]
        return np.where(is_count, count_value[self.answer_code], class_value[self.answer_code])

    def save(self, out_dir: Path) -> None:
        np.savez_compressed(out_dir / "train_table.npz", img_id=self.img_id, type_code=self.type_code,
                            question_code=self.question_code, answer_code=self.answer_code)
        (out_dir / "train_strings.json").write_text(json.dumps(
            {"types": self.types, "questions": self.questions, "answers": self.answers}))

    @classmethod
    def load(cls, bundle_dir: Path) -> "TrainTable":
        arrays = np.load(bundle_dir / "train_table.npz")
        strings = json.loads((bundle_dir / "train_strings.json").read_text())
        return cls(img_id=arrays["img_id"], type_code=arrays["type_code"], question_code=arrays["question_code"],
                   answer_code=arrays["answer_code"], **strings)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Choosing the demo tiles
# ---------------------------------------------------------------------------

def choose_demo_tiles(
    pool: PoolResult, img_of_row: dict[int, int], warned_per_type: int, wrong: int, right: int,
) -> list[int]:
    """Demo tiles for one test split, in the pool's seeded random order: ``warned_per_type`` tiles with a
    warned question per type, then ``wrong`` tiles with a wrong answer and no warning, then ``right``
    tiles answered right. Each tile is taken once; a group with too few tiles stays short."""
    chosen: list[int] = []
    entries = list(zip(pool.rows, pool.types, pool.correct, pool.warned))

    def take(limit: int, keep: Any) -> None:
        taken = 0
        for row, qtype, correct, warned in entries:
            tile = img_of_row[row]
            if taken < limit and tile not in chosen and keep(qtype, correct, warned):
                chosen.append(tile)
                taken += 1

    for qtype in dict.fromkeys(pool.types):
        take(warned_per_type, lambda t, c, w, qtype=qtype: w and t == qtype)
    take(wrong, lambda t, c, w: not c and not w)
    take(right, lambda t, c, w: c)
    return chosen


# ---------------------------------------------------------------------------
# Shared with the app
# ---------------------------------------------------------------------------

@torch.no_grad()
def occlusion_for(model: nn.Module, dataset: Any, row: int, device: torch.device) -> torch.Tensor:
    """The occlusion map [N] of one question: the confirmed explanation map."""
    from src.explain import occlusion_map

    batch = prepare_batch(vqa_collate_fn([dataset[row]]), device, dataset)
    outputs = model(batch)
    grid = _grid(model)
    return occlusion_map(model, batch, outputs["class_logits"], outputs["count_pred"], grid * grid)[0]


def tile_image(dataset: Any, img_id: int) -> Image.Image:
    """The tile exactly as the model reads it (decoded and resized to the model's image size)."""
    return Image.fromarray(dataset._load_image(img_id).permute(1, 2, 0).numpy())


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _save_thumbnails(train: Any, out_dir: Path, size: int, config: dict[str, Any]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_ids = sorted({int(r["img_id"]) for r in train.qa_records})
    loader = DataLoader(_Tiles(train, img_ids), batch_size=None, num_workers=config["num_workers"])
    for img_id, pixels in zip(img_ids, tqdm(loader, desc="thumbnails", unit="tile", leave=False)):
        image = Image.fromarray(pixels.permute(1, 2, 0).numpy())
        image.resize((size, size), Image.Resampling.LANCZOS).save(out_dir / f"{img_id}.jpg", quality=82)
    return len(img_ids)


def export_bundle(
    run_dir: Path, config: dict[str, Any], index_dir: Path, out_dir: Path, client: LLMClient | None,
) -> str:
    """Writes the bundle to ``out_dir`` and returns a short markdown summary."""
    from src.explain import question_pool

    started = time.monotonic()
    config = {**config, "cache_images": False, "max_images_per_split": None}
    model, dataloaders, device = load_run(run_dir, config)
    model.eval()
    train = dataloaders["train"].dataset
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    checkpoint = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    features = build_index(model, train, index_dir, config, device, checkpoint)  # reused when it matches
    index = RetrievalIndex(features.vectors, [r["type"] for r in train.qa_records], config["rag_search_chunk_rows"])
    table = TrainTable.from_records(train.qa_records)
    train_truths = table.truths(train.answer_vocab)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tiles").mkdir(exist_ok=True)
    demo_records: list[dict[str, Any]] = []
    cache: list[dict[str, Any]] = []
    summary_rows = []
    for split in [s for s in config["splits"] if s.startswith("test")]:
        dataset = dataloaders[split].dataset
        records = dataset.qa_records
        pool_rows = question_pool(records, config["rag_seed"])
        pool = score_pool(model, dataset, pool_rows, index, train_truths, config, device, f"pool {split}")
        tiles = choose_demo_tiles(pool, {row: records[row]["img_id"] for row in pool.rows},
                                  config["demo_warned_per_type"], config["demo_wrong_tiles"], config["demo_right_tiles"])
        for img_id in tiles:
            tile_image(dataset, img_id).save(out_dir / "tiles" / f"{img_id}.png")
        on_tiles = set(tiles)
        demo_records += [{"split": split, "source_row": row, **record}
                         for row, record in enumerate(records) if record["img_id"] in on_tiles]
        position = {row: j for j, row in enumerate(pool.rows)}
        showcase = [row for row in pool.rows if records[row]["img_id"] in on_tiles]
        for row in tqdm(showcase, desc=f"showcase {split}", unit="question", leave=False):
            j = position[row]
            scores = occlusion_for(model, dataset, row, device)
            facts = build_facts(model, dataset, split, row, pool.prediction[j], pool.neighbours[j], train,
                                train_truths, config, device, scores=scores)
            cache.append({"split": split, "img_id": int(records[row]["img_id"]), "question": records[row]["question"],
                          "facts": asdict(facts), "explanation": explain(facts, client, config, True),
                          "occlusion": [round(float(s), 6) for s in scores.cpu()]})
        summary_rows.append(f"| {split} | {len(tiles)} | {sum(r['split'] == split for r in demo_records):,} "
                            f"| {len(showcase)} |")

    table.save(out_dir)
    thumbs = _save_thumbnails(train, out_dir / "thumbs", config["demo_thumb_size"], config)
    shutil.copyfile(checkpoint_path, out_dir / "checkpoint.pt")
    shutil.copytree(Path(config["data_dir"]) / "vocab", out_dir / "vocab", dirs_exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=1, default=str))
    (out_dir / "demo_records.json").write_text(json.dumps(demo_records))
    (out_dir / "cache.json").write_text(json.dumps(cache))
    llm = sum(item["explanation"]["source"] == "llm" for item in cache)
    manifest = {"run": run_dir.name, "checkpoint": checkpoint,
                "index": json.loads((index_dir / "manifest.json").read_text()),
                "tiles": len({r["img_id"] for r in demo_records}), "questions": len(demo_records),
                "showcase": len(cache), "showcase_by_llm": llm, "thumbnails": thumbs}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6
    return "\n".join([
        f"# Demo bundle: {run_dir.name}\n",
        "| Test split | Demo tiles | Questions on them | Showcase (precomputed) |", "|---|---|---|---|",
        *summary_rows, "",
        f"Showcase neighbour paragraphs by the LLM: {llm} of {len(cache)} (the rest use the template). "
        f"Training thumbnails: {thumbs:,}. Bundle: {size:.0f} MB in `{out_dir}`. "
        f"Took {(time.monotonic() - started) / 60:.1f} min on {device.type}.",
    ]) + "\n"


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--index-dir", type=Path, help="default: <run_dir>/retrieval")
    parser.add_argument("--out", type=Path, help="default: the config's demo_bundle_dir")
    parser.add_argument("--no-llm", action="store_true", help="template paragraphs only, no API key needed")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    client = None if args.no_llm else OpenAICompatibleClient(config)
    summary = export_bundle(args.run_dir, config, args.index_dir or args.run_dir / "retrieval",
                            args.out or Path(config["demo_bundle_dir"]), client)
    print(summary)


if __name__ == "__main__":
    _main()

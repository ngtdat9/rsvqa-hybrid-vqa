"""The demo's core, kept apart from the UI: load the bundle, list tiles and questions, answer one question.

A showcase question (precomputed on Colab with the GPU index) is answered from ``cache.json`` at once.
Any other question on a demo tile is answered live on the laptop by the same code the reported results
used: the model's answer, the occlusion map, the neighbours from the local index, the warning, and the
explanation, whose neighbour paragraph comes from the LLM when a key and a connection are available and
from the fixed template otherwise. Live neighbours can differ slightly from a GPU run: the index holds
float16 vectors, and near-tied neighbours (similarity 0.99-1.00) can swap order.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from PIL import Image

from src.data_pipeline import AnswerVocab, QuestionVocab, RSVQADataset, load_norm_stats
from src.demo.bundle import TrainTable, occlusion_for
from src.explain import no_data_cells
from src.rag.explain_agent import (
    Facts,
    Neighbour,
    OpenAICompatibleClient,
    build_facts,
    explain,
    first_sentence,
    fixed_sentences,
    score_pool,
)
from src.rag.retrieval_index import RetrievalIndex
from src.train import build_model


@dataclass
class NeighbourCard:
    rank: int
    img_id: int
    question: str
    answer: str
    thumb: Path | None


@dataclass
class DemoAnswer:
    split: str
    img_id: int
    question: str
    qtype: str
    answer: str             # readable model answer
    confidence: float | None
    truth: str
    correct: bool
    warning: bool
    strip: bool
    text: str               # the full explanation
    parts: list[tuple[str, str]]  # (who wrote it: "code" or "llm" or "template", text), in order
    occlusion: np.ndarray   # [N] occlusion scores
    neighbours: list[NeighbourCard]
    precomputed: bool
    seconds: dict[str, float] = field(default_factory=dict)


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def split_parts(text: str, source: str, first: str, fixed: str) -> list[tuple[str, str]]:
    """Who wrote which part of an explanation: code writes the first and the fixed sentences, and the
    neighbour paragraph between them comes from the LLM or the template."""
    middle = text[len(first):len(text) - len(fixed)].strip() if text.startswith(first) and text.endswith(fixed) else ""
    parts = [("code", first)]
    if middle:
        parts.append(("llm" if source == "llm" else "template", middle))
    parts.append(("code", fixed))
    return parts


def heatmap_overlay(tile: Image.Image, scores: np.ndarray, alpha: float) -> Image.Image:
    """The occlusion map drawn over the tile: dimmed tile, cells coloured by their score (max = 1)."""
    grid = round(math.sqrt(scores.size))
    top = float(scores.max())
    scaled = (scores / top if top > 0 else np.zeros_like(scores)).reshape(grid, grid)
    colours = matplotlib.colormaps["inferno"](scaled)[..., :3]
    size = tile.size[0]
    cells = np.kron(colours, np.ones((size // grid, size // grid, 1)))
    cells = np.asarray(Image.fromarray((cells * 255).astype(np.uint8)).resize(tile.size, Image.Resampling.NEAREST))
    base = np.asarray(tile.convert("RGB"), dtype=np.float32) * 0.65
    return Image.fromarray(((1 - alpha) * base + alpha * cells).clip(0, 255).astype(np.uint8))


class Demo:
    """Everything the app needs, loaded once."""

    def __init__(self, bundle_dir: Path, index_dir: Path, config: dict[str, Any], device: torch.device | None = None,
                 use_llm: bool = True) -> None:
        self.bundle_dir = bundle_dir
        self.device = device or pick_device()
        exported = json.loads((bundle_dir / "config.json").read_text())
        self.config = {**exported, **{k: v for k, v in config.items() if k.startswith(("demo_", "rag_llm"))},
                       "data_dir": str(bundle_dir), "num_workers": 0, "mixed_precision": False,
                       "rag_llm_timeout_s": config["demo_llm_timeout_s"],
                       "rag_llm_max_retries": config["demo_llm_max_retries"]}
        self.manifest = json.loads((bundle_dir / "manifest.json").read_text())
        index_manifest = json.loads((index_dir / "manifest.json").read_text())
        if index_manifest != self.manifest["index"]:
            raise ValueError(f"{index_dir} is not the index this bundle was made with")

        checkpoint = torch.load(bundle_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        self.model = build_model({**checkpoint["config"], "data_dir": str(bundle_dir)})
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

        vocab = bundle_dir / "vocab"
        mean, std = load_norm_stats(vocab / "image_norm_stats.json")
        self.records: list[dict[str, Any]] = json.loads((bundle_dir / "demo_records.json").read_text())
        self.dataset = RSVQADataset(
            qa_records=self.records, images_dir=bundle_dir / "tiles", question_vocab=QuestionVocab.load(
                vocab / "question_vocab.json"), answer_vocab=AnswerVocab.load(vocab / "answer_vocab.json"),
            image_mean=mean, image_std=std, image_size=self.config["image_size"],
            max_question_len=self.config["max_question_len"], image_ext=".png")
        self.train = TrainTable.load(bundle_dir)
        self.train_truths = self.train.truths(self.dataset.answer_vocab)
        vectors = np.load(index_dir / "vectors.npy", mmap_mode="r")
        self.index = RetrievalIndex(vectors, self.train.type_list(), self.config["rag_search_chunk_rows"])
        self.cache = {(item["img_id"], item["question"]): item
                      for item in json.loads((bundle_dir / "cache.json").read_text())}
        self.client = None
        if use_llm and os.environ.get(self.config["rag_llm_api_key_env"]):
            self.client = OpenAICompatibleClient(self.config)

    # -- browsing ---------------------------------------------------------

    def tiles(self) -> list[dict[str, Any]]:
        """Demo tiles in bundle order, with their split and whether they have a no-data strip."""
        seen: dict[int, dict[str, Any]] = {}
        for record in self.records:
            seen.setdefault(record["img_id"], {"img_id": record["img_id"], "split": record["split"]})
        for tile in seen.values():
            tile["path"] = self.bundle_dir / "tiles" / f"{tile['img_id']}.png"
            tile["strip"] = self._strip(tile["img_id"])
        return list(seen.values())

    def questions(self, img_id: int) -> list[dict[str, Any]]:
        """The tile's questions, showcase ones first: (row in the demo records, question, type, showcase)."""
        rows = [{"row": row, "question": r["question"], "type": r["type"],
                 "showcase": (img_id, r["question"]) in self.cache}
                for row, r in enumerate(self.records) if r["img_id"] == img_id]
        return sorted(rows, key=lambda q: (not q["showcase"], q["type"], q["question"]))

    def tile(self, img_id: int) -> Image.Image:
        return Image.open(self.bundle_dir / "tiles" / f"{img_id}.png").convert("RGB")

    def thumb(self, img_id: int) -> Path | None:
        path = self.bundle_dir / "thumbs" / f"{img_id}.jpg"
        return path if path.exists() else None

    def _strip(self, img_id: int) -> bool:
        pixels = torch.from_numpy(np.asarray(self.tile(img_id), dtype=np.float32) / 255).permute(2, 0, 1)[None]
        grid = round(math.sqrt(self.model.encoder.pos_embed.size(1) - 1))
        return bool(no_data_cells(pixels, grid, self.config["xai_nodata_pixel_value"],
                                  self.config["xai_nodata_cell_fraction"]).any())

    # -- answering --------------------------------------------------------

    def ask(self, row: int) -> DemoAnswer:
        """Answer the demo record at ``row``: from the cache when it is a showcase question, else live."""
        record = self.records[row]
        item = self.cache.get((record["img_id"], record["question"]))
        if item is not None:
            return self._answer(record, item["facts"], item["explanation"], np.array(item["occlusion"]), True, {})
        seconds: dict[str, float] = {}
        started = time.monotonic()
        pool = score_pool(self.model, self.dataset, [row], self.index, self.train_truths, self.config, self.device,
                          "live")
        seconds["answer and search"] = time.monotonic() - started
        started = time.monotonic()
        scores = occlusion_for(self.model, self.dataset, row, self.device)
        seconds["occlusion map"] = time.monotonic() - started
        facts = build_facts(self.model, self.dataset, record["split"], row, pool.prediction[0], pool.neighbours[0],
                            self.train, self.train_truths, self.config, self.device, scores=scores)
        started = time.monotonic()
        explanation = explain(facts, self.client, self.config, True)
        seconds["explanation"] = time.monotonic() - started
        return self._answer(record, asdict(facts), explanation, scores.cpu().numpy(), False, seconds)

    def _answer(self, record: dict[str, Any], facts: dict[str, Any], explanation: dict[str, Any],
                occlusion: np.ndarray, precomputed: bool, seconds: dict[str, float]) -> DemoAnswer:
        f = Facts(**{**facts, "neighbours": [Neighbour(**n) for n in facts["neighbours"]],
                     "count_range": tuple(facts["count_range"]) if facts["count_range"] else None})
        cards = [NeighbourCard(n.rank, n.img_id, n.question, n.answer, self.thumb(n.img_id)) for n in f.neighbours]
        return DemoAnswer(
            split=record["split"], img_id=record["img_id"], question=record["question"], qtype=record["type"],
            answer=f.answer, confidence=f.confidence, truth=f.truth, correct=f.correct, warning=f.warning,
            strip=f.strip, text=explanation["text"],
            parts=split_parts(explanation["text"], explanation["source"], first_sentence(f), fixed_sentences(f)),
            occlusion=occlusion, neighbours=cards, precomputed=precomputed, seconds=seconds)


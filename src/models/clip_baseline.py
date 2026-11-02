"""Zero-shot CLIP baseline: no training on RSVQA. For each question, one sentence per
candidate answer is scored against the tile, and the best-scoring answer is the prediction.

    python -m src.models.clip_baseline configs/hr_config.yaml [--pilot]

Each tile is encoded once and each unique question's sentences once, so the QA
dataloaders (which load a tile about 100 times) are not used.
"""
from __future__ import annotations

import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data_pipeline import (
    _decode_tile,
    apply_area_buckets,
    build_qa_records,
    load_split_raw,
    select_image_subset,
)
from src.evaluate import count_median, render_markdown_table, selection_score


class Encoder(Protocol):
    preprocess: Callable[[Image.Image], torch.Tensor]

    def encode_images(self, pixels: torch.Tensor) -> torch.Tensor: ...

    def encode_texts(self, texts: list[str]) -> torch.Tensor: ...


class _ClipPreprocess:
    """Picklable on its own, so DataLoader workers don't receive the whole model."""

    def __init__(self, image_processor: Any) -> None:
        self.image_processor = image_processor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]


class ClipEncoder:
    """Pretrained CLIP at its own input size and normalization; embeddings are L2-normalized."""

    def __init__(self, model_name: str, device: torch.device, mixed_precision: bool = False) -> None:
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

        self.device = device
        self.use_amp = mixed_precision and device.type == "cuda"
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.preprocess = _ClipPreprocess(CLIPImageProcessor.from_pretrained(model_name))

    @torch.no_grad()
    def encode_images(self, pixels: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
            features = self.model.get_image_features(pixel_values=pixels.to(self.device))
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
            features = self.model.get_text_features(**tokens)
        return F.normalize(features.float(), dim=-1)


def answer_candidates(train_records: list[dict]) -> dict[str, list[str]]:
    """Distinct train answers per question type; count candidates sorted numerically."""
    by_type: dict[str, set[str]] = defaultdict(set)
    for record in train_records:
        by_type[record["type"]].add(record["answer"])
    return {
        qtype: sorted(answers, key=float) if qtype == "count" else sorted(answers)
        for qtype, answers in by_type.items()
    }


def answer_text(answer: str) -> str:
    """Readable wording for area range labels ('11-100m2'); other answers pass through."""
    if not answer.endswith("m2"):
        return answer
    value = answer.removesuffix("m2")
    if value.startswith(">"):
        return f"more than {value[1:]} square meters"
    if value.startswith("<="):
        return f"at most {value[2:]} square meters"
    if "-" in value:
        low, high = value.split("-")
        return f"{low} to {high} square meters"
    return f"{value} square meters"


def build_prompt(template: str, question: str, answer: str) -> str:
    return template.format(question=question, answer=answer_text(answer))


@dataclass
class TileEmbeddings:
    row_of: dict[int, int]  # img_id -> row in embeddings
    embeddings: torch.Tensor


class _TileDataset(Dataset):
    def __init__(self, paths: list[Path], preprocess: Callable[[Image.Image], torch.Tensor]) -> None:
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.preprocess(_decode_tile(self.paths[idx]))


def encode_tiles(
    records: list[dict], images_dir: Path, image_ext: str, encoder: Encoder,
    batch_size: int, num_workers: int = 0, desc: str = "encode tiles",
) -> TileEmbeddings:
    img_ids = sorted({r["img_id"] for r in records})
    dataset = _TileDataset([images_dir / f"{i}{image_ext}" for i in img_ids], encoder.preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    chunks = [encoder.encode_images(pixels) for pixels in tqdm(loader, desc=desc, leave=False, unit="batch")]
    return TileEmbeddings({img_id: row for row, img_id in enumerate(img_ids)}, torch.cat(chunks))


def _question_chunks(
    questions: list[tuple[str, str]], candidates: dict[str, list[str]], template: str, size: int,
) -> Iterator[tuple[list[tuple[str, str]], list[str]]]:
    """Group unique (type, question) pairs so each chunk's sentences encode in one batch."""
    chunk: list[tuple[str, str]] = []
    prompts: list[str] = []
    for qtype, question in questions:
        chunk.append((qtype, question))
        prompts += [build_prompt(template, question, answer) for answer in candidates[qtype]]
        if len(prompts) >= size:
            yield chunk, prompts
            chunk, prompts = [], []
    if chunk:
        yield chunk, prompts


def score_split(
    records: list[dict], tiles: TileEmbeddings, candidates: dict[str, list[str]], template: str,
    encoder: Encoder, question_types: list[str], baseline_count: float | None = None,
    text_batch_size: int = 512, desc: str = "score",
) -> dict[str, Any]:
    """The same metric keys as ``src.evaluate.evaluate``, plus ``sentences_encoded``. Text
    embeddings are dropped after each chunk: count questions have many candidates, so
    keeping them all could take gigabytes."""
    by_question: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record["type"] != "count" and record["type"] not in question_types:
            raise ValueError(
                f"question_type {record['type']!r} is not in question_types={question_types} "
                "(or 'count') -- update the config, or these rows would be excluded from the metrics"
            )
        by_question[(record["type"], record["question"])].append(record)

    correct_by_type = {t: 0 for t in question_types}
    total_by_type = {t: 0 for t in question_types}
    abs_sum = sq_sum = base_abs_sum = base_sq_sum = 0.0
    num_counts = 0
    sentences = 0

    for chunk, prompts in tqdm(
        _question_chunks(list(by_question), candidates, template, text_batch_size),
        desc=desc, leave=False, unit="chunk",
    ):
        text = encoder.encode_texts(prompts)
        sentences += len(prompts)
        start = 0
        for qtype, question in chunk:
            options = candidates[qtype]
            rows = by_question[(qtype, question)]
            image_rows = torch.tensor([tiles.row_of[r["img_id"]] for r in rows], device=tiles.embeddings.device)
            best = (tiles.embeddings[image_rows] @ text[start:start + len(options)].T).argmax(dim=1).tolist()
            start += len(options)
            for record, choice in zip(rows, best):
                if qtype == "count":
                    target = float(record["answer"])
                    error = float(options[choice]) - target
                    abs_sum += abs(error)
                    sq_sum += error ** 2
                    num_counts += 1
                    if baseline_count is not None:
                        base_abs_sum += abs(target - baseline_count)
                        base_sq_sum += (target - baseline_count) ** 2
                else:
                    total_by_type[qtype] += 1
                    correct_by_type[qtype] += options[choice] == record["answer"]

    total = sum(total_by_type.values())
    result: dict[str, Any] = {
        "class_accuracy": sum(correct_by_type.values()) / total if total else float("nan"),
        "class_accuracy_by_type": {
            t: correct_by_type[t] / total_by_type[t] if total_by_type[t] else float("nan") for t in question_types
        },
        "count_mae": abs_sum / num_counts if num_counts else float("nan"),
        "count_rmse": math.sqrt(sq_sum / num_counts) if num_counts else float("nan"),
        "sentences_encoded": sentences,
    }
    if baseline_count is not None:
        result["count_mae_baseline"] = base_abs_sum / num_counts if num_counts else float("nan")
        result["count_rmse_baseline"] = math.sqrt(base_sq_sum / num_counts) if num_counts else float("nan")
    return result


def select_template(
    val_records: list[dict], tiles: TileEmbeddings, candidates: dict[str, list[str]], templates: list[str],
    encoder: Encoder, question_types: list[str], baseline_count: float | None = None,
    text_batch_size: int = 512,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """The template with the best ``selection_score`` on val (the first one on a tie), and each
    template's val metrics."""
    metrics = {
        template: score_split(
            val_records, tiles, candidates, template, encoder, question_types, baseline_count,
            text_batch_size, desc=f"val: {template}",
        )
        for template in templates
    }
    return max(templates, key=lambda template: selection_score(metrics[template])), metrics


def majority_accuracy(train_records: list[dict], records: list[dict]) -> dict[str, Any]:
    """Accuracy of always giving each classification type's most common train answer."""
    answers: dict[str, Counter[str]] = defaultdict(Counter)
    for record in train_records:
        if record["type"] != "count":
            answers[record["type"]][record["answer"]] += 1
    majority = {qtype: counter.most_common(1)[0][0] for qtype, counter in answers.items()}

    correct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for record in records:
        if record["type"] != "count":
            total[record["type"]] += 1
            correct[record["type"]] += majority.get(record["type"]) == record["answer"]
    return {
        "class_accuracy": sum(correct.values()) / sum(total.values()) if total else float("nan"),
        "class_accuracy_by_type": {qtype: correct[qtype] / total[qtype] for qtype in total},
    }


def _load_records(config: dict[str, Any], split: str, pilot: bool) -> list[dict]:
    questions, answers = load_split_raw(Path(config["raw_dir"]), split, config["split_file_prefix"])
    records = apply_area_buckets(build_qa_records(questions, answers), config["area_buckets"])
    max_images = (config["max_images_per_split"] or {}).get(split) if pilot else None
    return select_image_subset(records, max_images, config["subset_seed"])


def _render_section(split: str, metrics: dict[str, Any], majority: dict[str, Any], seconds: float) -> str:
    rows = [f"| Accuracy, always the most common train answer (overall) | {majority['class_accuracy']:.4f} |"]
    rows += [
        f"| Accuracy, always the most common train answer ({qtype}) | {value:.4f} |"
        for qtype, value in majority["class_accuracy_by_type"].items()
    ]
    return (
        f"## Split: {split}\n\n{render_markdown_table(metrics)}" + "\n".join(rows) + "\n\n"
        f"{metrics['sentences_encoded']:,} sentences encoded, {seconds / 60:.1f} min\n"
    )


def run_baseline(config: dict[str, Any], encoder: Encoder, pilot: bool = False) -> str:
    """Pick the template on val, score the test splits with it once, and return a markdown report.
    Candidates, the median and the majority answers always come from the complete train split."""
    images_dir = Path(config["raw_dir"]) / config["images_subdir"]
    train_records = _load_records(config, "train", pilot=False)
    candidates = answer_candidates(train_records)
    baseline_count = count_median(train_records)
    text_batch_size = config["clip_text_batch_size"]

    def encode(records: list[dict], split: str) -> TileEmbeddings:
        return encode_tiles(records, images_dir, config["image_ext"], encoder, config["clip_batch_size"],
                            config["num_workers"], desc=f"encode {split} tiles")

    val_records = _load_records(config, "val", pilot)
    template, val_metrics = select_template(
        val_records, encode(val_records, "val"), candidates, config["clip_prompt_templates"], encoder,
        config["question_types"], baseline_count, text_batch_size,
    )

    title = f"# Results: zero-shot CLIP ({config['clip_model_name']})" + (" (pilot subsets)" if pilot else "")
    sections = [
        title + "\n",
        "## Sentence template, chosen on val\n\n| Template | Val score | Val accuracy | Val count MAE |\n|---|---|---|---|\n"
        + "".join(
            f"| `{t}` | {selection_score(m):.4f}{' (chosen)' if t == template else ''} "
            f"| {m['class_accuracy']:.4f} | {m['count_mae']:.4f} |\n"
            for t, m in val_metrics.items()
        ),
    ]
    for split in [s for s in config["splits"] if s.startswith("test")]:
        started = time.monotonic()
        records = _load_records(config, split, pilot)
        metrics = score_split(
            records, encode(records, split), candidates, template, encoder, config["question_types"],
            baseline_count, text_batch_size, desc=f"score {split}",
        )
        sections.append(_render_section(split, metrics, majority_accuracy(train_records, records),
                                        time.monotonic() - started))
    return "\n".join(sections)


def _main() -> None:
    from src.train import get_device

    args = [a for a in sys.argv[1:] if a != "--pilot"]
    pilot = "--pilot" in sys.argv[1:]
    config = yaml.safe_load(Path(args[0] if args else "configs/hr_config.yaml").read_text())

    device = get_device()
    encoder = ClipEncoder(config["clip_model_name"], device, config["mixed_precision"])
    report = run_baseline(config, encoder, pilot)

    short_name = config["clip_model_name"].split("/")[-1].removeprefix("clip-")
    out_path = Path("reports") / f"baseline_results_clip_{short_name}{'_pilot' if pilot else ''}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote {out_path}")
    print(report)


if __name__ == "__main__":
    _main()

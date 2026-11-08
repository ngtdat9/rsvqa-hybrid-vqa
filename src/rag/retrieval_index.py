"""Retrieval index for the RAG explanation layer, and a test of whether retrieval carries information.

Every training QA pair is stored as the trained model's fused image-question vector: the
vector both heads read, so the nearest neighbours are the cases this model treats as alike.
Nothing is trained here. Each tile is encoded once and fused with each of its questions.

The test, fixed before any result: for test questions, does low agreement between the
model's answer and its neighbours' true answers flag the model's mistakes better than the
same number of random same-type training questions? Per type, the AUROC difference
(retrieval minus random, test splits pooled) must have a 95% bootstrap lower bound above 0.
Only types that pass may use neighbour disagreement as an uncertainty warning.

    python -m src.rag.retrieval_index <run_dir> configs/hr_config.yaml [--pilot]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data_pipeline import prepare_batch, tokenize
from src.evaluate import load_run
from src.models.hybrid_encoder import EncoderOutput

COUNT = "count"


@dataclass
class Features:
    vectors: np.ndarray     # [N, D] float16, L2-normalized fused vectors
    pred_class: np.ndarray  # [N] int16: predicted class index, -1 on count rows
    pred_count: np.ndarray  # [N] float32: predicted count, nan on classification rows


class _Tiles(Dataset):
    """One uint8 tile per item, through the dataset's own loading (cache or decode)."""

    def __init__(self, dataset: Any, img_ids: list[int]) -> None:
        self.dataset = dataset
        self.img_ids = img_ids

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.dataset._load_image(self.img_ids[idx])


def empty_features(rows: int, dim: int) -> Features:
    return Features(
        vectors=np.zeros((rows, dim), dtype=np.float16),
        pred_class=np.full(rows, -1, dtype=np.int16),
        pred_count=np.full(rows, np.nan, dtype=np.float32),
    )


@torch.no_grad()
def compute_features(
    model: nn.Module, dataset: Any, rows: list[int], out: Features, config: dict[str, Any],
    device: torch.device, desc: str,
) -> None:
    """Fill ``out`` (position j holds record ``rows[j]`` of ``dataset``) with fused vectors and
    predictions. Each tile goes through the encoder once, then is fused with each of its questions."""
    model.eval()
    positions: dict[int, list[int]] = defaultdict(list)
    for j, row in enumerate(rows):
        positions[dataset.qa_records[row]["img_id"]].append(j)
    img_ids = list(positions)
    loader = DataLoader(_Tiles(dataset, img_ids), batch_size=config["rag_tile_batch_size"],
                        shuffle=False, num_workers=config["num_workers"])
    use_amp = config.get("mixed_precision", False) and device.type == "cuda"
    max_len = config["max_question_len"]
    chunk = config["rag_question_batch_size"]
    start = 0
    for images in tqdm(loader, desc=desc, unit="batch", leave=False):
        tile_ids = img_ids[start:start + images.size(0)]
        start += images.size(0)
        image = prepare_batch({"image": images}, device, dataset)["image"]
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            encoded = model.encoder(image)
        jobs = [(b, j) for b, img_id in enumerate(tile_ids) for j in positions[img_id]]
        for s in range(0, len(jobs), chunk):
            part = jobs[s:s + chunk]
            tile_index = torch.tensor([b for b, _ in part], device=device)
            records = [dataset.qa_records[rows[j]] for _, j in part]
            tokens = torch.stack([dataset.question_vocab.encode(r["question"], max_len) for r in records]).to(device)
            lengths = torch.tensor([min(len(tokenize(r["question"])), max_len) for r in records], device=device)
            enc = EncoderOutput(pooled=encoded.pooled[tile_index], patch_tokens=encoded.patch_tokens[tile_index])
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                fused, _ = model.fuse(enc, tokens, lengths)
                outputs = model.heads(fused)
            index = np.array([j for _, j in part])
            is_count = np.array([r["type"] == COUNT for r in records])
            out.vectors[index] = F.normalize(fused.float(), dim=-1).cpu().numpy().astype(np.float16)
            classes = outputs["class_logits"].argmax(-1).cpu().numpy()
            counts = outputs["count_pred"].float().cpu().numpy()
            out.pred_class[index] = np.where(is_count, -1, classes)
            out.pred_count[index] = np.where(is_count, counts, np.nan)


def records_fingerprint(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for r in records:
        digest.update(f"{r['img_id']}\t{r['type']}\t{r['question']}\t{r['answer']}\n".encode())
    return digest.hexdigest()


def build_index(
    model: nn.Module, dataset: Any, index_dir: Path, config: dict[str, Any], device: torch.device,
    checkpoint: str,
) -> Features:
    """Fused vectors for every record of ``dataset``, memory-mapped in ``index_dir``. An index whose
    manifest matches (same records, checkpoint hash and width) is reused; otherwise it is rebuilt."""
    index_dir.mkdir(parents=True, exist_ok=True)
    rows = len(dataset.qa_records)
    dim = model.classifier.in_features
    manifest = {"checkpoint": checkpoint, "rows": rows, "dim": dim, "records": records_fingerprint(dataset.qa_records)}
    manifest_path = index_dir / "manifest.json"
    names = ("vectors", "pred_class", "pred_count")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) == manifest:
        return Features(*(np.load(index_dir / f"{name}.npy", mmap_mode="r") for name in names))

    manifest_path.unlink(missing_ok=True)  # an interrupted build must not look complete
    shapes = {"vectors": ((rows, dim), np.float16), "pred_class": ((rows,), np.int16), "pred_count": ((rows,), np.float32)}
    arrays = {name: np.lib.format.open_memmap(index_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
              for name, (shape, dtype) in shapes.items()}
    arrays["pred_class"][:] = -1
    arrays["pred_count"][:] = np.nan
    features = Features(**arrays)
    compute_features(model, dataset, list(range(rows)), features, config, device, "index train")
    for array in arrays.values():
        array.flush()
    del features, arrays
    manifest_path.write_text(json.dumps(manifest))
    return Features(*(np.load(index_dir / f"{name}.npy", mmap_mode="r") for name in names))


class RetrievalIndex:
    """Exact cosine search within one question type, reading the index in chunks of rows so
    memory stays bounded however large it is."""

    def __init__(self, vectors: np.ndarray, types: list[str], chunk_rows: int) -> None:
        self.vectors = vectors
        self.chunk_rows = chunk_rows
        by_type: dict[str, list[int]] = defaultdict(list)
        for row, qtype in enumerate(types):
            by_type[qtype].append(row)
        self.rows_of = {qtype: np.array(rows, dtype=np.int64) for qtype, rows in by_type.items()}

    def search(
        self, queries: np.ndarray, qtype: str, k: int, device: torch.device | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` (similarities [Q,k], index rows [Q,k]) among ``qtype`` rows, most similar first."""
        device = device or torch.device("cpu")
        candidates = self.rows_of[qtype]
        k = min(k, len(candidates))
        q = torch.as_tensor(np.asarray(queries, dtype=np.float32), device=device)
        best_sims = torch.full((q.size(0), 0), -math.inf, device=device)
        best_pos = torch.zeros((q.size(0), 0), dtype=torch.long, device=device)
        for s in range(0, len(candidates), self.chunk_rows):
            block = torch.as_tensor(np.asarray(self.vectors[candidates[s:s + self.chunk_rows]], dtype=np.float32), device=device)
            sims = torch.cat([best_sims, q @ block.T], dim=1)
            pos = torch.cat([best_pos, torch.arange(s, s + block.size(0), device=device).expand(q.size(0), -1)], dim=1)
            best_sims, order = sims.topk(min(k, sims.size(1)), dim=1)
            best_pos = pos.gather(1, order)
        return best_sims.cpu().numpy(), candidates[best_pos.cpu().numpy()]

    def random_rows(self, qtype: str, queries: int, k: int, rng: np.random.Generator) -> np.ndarray:
        """``k`` distinct random ``qtype`` rows per query: the control."""
        candidates = self.rows_of[qtype]
        k = min(k, len(candidates))
        return np.stack([rng.choice(candidates, size=k, replace=False) for _ in range(queries)])


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def agreement(prediction: float, neighbour_truths: np.ndarray, is_count: bool) -> float:
    """How well the neighbours' true answers agree with the model's answer. Classification:
    the share of neighbours whose true answer is the predicted class. Count: minus the gap
    between the prediction and the neighbours' median true count, in log(1 + count)."""
    if is_count:
        return -abs(math.log1p(prediction) - math.log1p(float(np.median(neighbour_truths))))
    return float(np.mean(neighbour_truths == prediction))


def neighbour_vote(neighbour_truths: np.ndarray, sims: np.ndarray, is_count: bool) -> float:
    """The neighbours' answer: their median count, or the most common class (ties go to the
    class with the larger summed similarity)."""
    if is_count:
        return float(np.median(neighbour_truths))
    totals: dict[float, tuple[int, float]] = {}
    for truth, sim in zip(neighbour_truths.tolist(), sims.tolist()):
        votes, weight = totals.get(truth, (0, 0.0))
        totals[truth] = (votes + 1, weight + sim)
    return max(totals, key=totals.get)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):  # ties share their average rank
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
    return ranks


def auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (ties count half);
    nan without both classes."""
    positives = positives.astype(bool)
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(scores, dtype=np.float64))
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_difference(
    scores_a: np.ndarray, scores_b: np.ndarray, positives: np.ndarray, samples: int, seed: int,
) -> dict[str, float]:
    """AUROC(a) - AUROC(b) on the same questions, with a 95% paired bootstrap interval."""
    rng = np.random.default_rng(seed)
    n = len(positives)
    differences = []
    for _ in range(samples):
        pick = rng.integers(0, n, n)
        d = auroc(scores_a[pick], positives[pick]) - auroc(scores_b[pick], positives[pick])
        if not math.isnan(d):
            differences.append(d)
    point = auroc(scores_a, positives) - auroc(scores_b, positives)
    if not differences:
        return {"difference": point, "lower": float("nan"), "upper": float("nan")}
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return {"difference": point, "lower": float(lower), "upper": float(upper)}


# ---------------------------------------------------------------------------
# Evaluation on the test splits
# ---------------------------------------------------------------------------

def _truth(record: dict, answer_vocab: Any) -> float:
    return float(record["answer"]) if record["type"] == COUNT else float(answer_vocab.encode(record["answer"]))


def _prediction(features: Features, j: int, is_count: bool) -> float:
    return float(features.pred_count[j]) if is_count else float(features.pred_class[j])


def _is_correct(prediction: float, truth: float, is_count: bool) -> bool:
    return round(prediction) == truth if is_count else prediction == truth


@dataclass
class QueryResults:
    """Per test question, grouped by type: everything the report needs."""
    correct: list[bool]
    agreement: list[float]
    random_agreement: list[float]
    vote_is_answer: list[bool]       # the neighbours' answer equals the model's
    vote_is_truth: list[bool]        # the neighbours' answer equals the true answer
    random_vote_is_answer: list[bool]
    vote_error: list[float]          # count only: |neighbour median - truth|
    model_error: list[float]         # count only: |prediction - truth|
    top_similarity: list[float]
    neighbours_right: list[float]    # share of neighbours the model answers right (training data)


def _empty_results() -> QueryResults:
    return QueryResults(*([] for _ in range(10)))


def score_queries(
    model: nn.Module, dataset: Any, rows: list[int], index: RetrievalIndex, train_truths: np.ndarray,
    train_right: np.ndarray, config: dict[str, Any], device: torch.device, rng: np.random.Generator, desc: str,
) -> dict[str, QueryResults]:
    records = dataset.qa_records
    features = empty_features(len(rows), model.classifier.in_features)
    compute_features(model, dataset, rows, features, config, device, desc)
    k = config["rag_k"]
    results: dict[str, QueryResults] = defaultdict(_empty_results)
    by_type: dict[str, list[int]] = defaultdict(list)
    for j, row in enumerate(rows):
        by_type[records[row]["type"]].append(j)
    for qtype, positions in by_type.items():
        is_count = qtype == COUNT
        sims, neighbours = index.search(features.vectors[positions], qtype, k, device)
        random_neighbours = index.random_rows(qtype, len(positions), k, rng)
        r = results[qtype]
        for i, j in enumerate(positions):
            prediction = _prediction(features, j, is_count)
            truth = _truth(records[rows[j]], dataset.answer_vocab)
            truths, random_truths = train_truths[neighbours[i]], train_truths[random_neighbours[i]]
            vote = neighbour_vote(truths, sims[i], is_count)
            random_vote = neighbour_vote(random_truths, np.ones(len(random_truths)), is_count)
            answer = round(prediction) if is_count else prediction
            r.correct.append(_is_correct(prediction, truth, is_count))
            r.agreement.append(agreement(prediction, truths, is_count))
            r.random_agreement.append(agreement(prediction, random_truths, is_count))
            r.vote_is_answer.append((round(vote) if is_count else vote) == answer)
            r.random_vote_is_answer.append((round(random_vote) if is_count else random_vote) == answer)
            r.vote_is_truth.append((round(vote) if is_count else vote) == truth)
            if is_count:
                r.vote_error.append(abs(vote - truth))
                r.model_error.append(abs(prediction - truth))
            r.top_similarity.append(float(sims[i, 0]))
            r.neighbours_right.append(float(train_right[neighbours[i]].mean()))
    return dict(results)


def _merge(parts: list[dict[str, QueryResults]]) -> dict[str, QueryResults]:
    merged: dict[str, QueryResults] = defaultdict(_empty_results)
    for part in parts:
        for qtype, r in part.items():
            for name in QueryResults.__dataclass_fields__:
                getattr(merged[qtype], name).extend(getattr(r, name))
    return dict(merged)


def error_warning(r: QueryResults, samples: int, seed: int) -> dict[str, float]:
    """AUROC of low agreement for flagging wrong answers, retrieval and random, and their difference."""
    wrong = ~np.array(r.correct, dtype=bool)
    retrieval, random = -np.array(r.agreement), -np.array(r.random_agreement)
    return {"retrieval": auroc(retrieval, wrong), "random": auroc(random, wrong),
            **bootstrap_difference(retrieval, random, wrong, samples, seed)}


def passes(warning: dict[str, float]) -> bool:
    return not math.isnan(warning["lower"]) and warning["lower"] > 0


def _mean(values: list[float] | list[bool]) -> float:
    return float(np.mean(values)) if values else float("nan")


def results_section(title: str, results: dict[str, QueryResults], types: list[str], config: dict[str, Any]) -> str:
    lines = [
        f"## {title}\n",
        "Error warning: AUROC of low neighbour agreement for flagging the model's wrong answers "
        "(0.5 = chance). Random: the same number of random same-type training questions.\n",
        "| Type | Questions | Model wrong | AUROC retrieval | AUROC random | Difference [95% CI] | Passes |",
        "|---|---|---|---|---|---|---|",
    ]
    for qtype in types:
        if qtype not in results:
            continue
        r = results[qtype]
        w = error_warning(r, config["rag_bootstrap_samples"], config["rag_seed"])
        lines.append(
            f"| {qtype} | {len(r.correct)} | {1 - _mean(r.correct):.1%} | {w['retrieval']:.3f} | {w['random']:.3f} "
            f"| {w['difference']:+.3f} [{w['lower']:+.3f}, {w['upper']:+.3f}] | {'yes' if passes(w) else 'no'} |"
        )
    lines += [
        "",
        "Neighbour answer: the neighbours' most common true answer (count: their median, rounded). "
        "Neighbours the model got right is on training data, so it is inflated by fitting.\n",
        "| Type | Model accuracy | Neighbour answer = true answer | Neighbour answer = model's | Same, random | "
        "Mean agreement | Same, random | Top-1 similarity | Neighbours the model got right |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for qtype in types:
        if qtype not in results:
            continue
        r = results[qtype]
        lines.append(
            f"| {qtype} | {_mean(r.correct):.1%} | {_mean(r.vote_is_truth):.1%} | {_mean(r.vote_is_answer):.1%} "
            f"| {_mean(r.random_vote_is_answer):.1%} | {_mean(r.agreement):.3f} | {_mean(r.random_agreement):.3f} "
            f"| {_mean(r.top_similarity):.3f} | {_mean(r.neighbours_right):.1%} |"
        )
    if COUNT in results and results[COUNT].vote_error:
        r = results[COUNT]
        lines.append(f"\nCount MAE: model {_mean(r.model_error):.3f}, neighbours' median count {_mean(r.vote_error):.3f}.")
    return "\n".join(lines) + "\n"


def run_retrieval(run_dir: Path, config: dict[str, Any], pilot: bool = False, index_dir: Path | None = None) -> str:
    """Builds (or reuses) the train index and returns the markdown report on the test splits."""
    from src.explain import question_pool

    config = dict(config)
    config["cache_images"] = False  # each tile is decoded once per pass anyway
    if not pilot:
        config["max_images_per_split"] = None
    started = time.monotonic()
    model, dataloaders, device = load_run(run_dir, config)
    model.eval()
    train = dataloaders["train"].dataset
    index_dir = index_dir or run_dir / ("retrieval_pilot" if pilot else "retrieval")
    checkpoint = hashlib.sha256((run_dir / "checkpoints" / "best.pt").read_bytes()).hexdigest()
    features = build_index(model, train, index_dir, config, device, checkpoint)
    build_minutes = (time.monotonic() - started) / 60

    types = [*config["question_types"], COUNT]
    train_records = train.qa_records
    train_truths = np.array([_truth(r, train.answer_vocab) for r in train_records])
    train_right = np.array([
        _is_correct(_prediction(features, j, r["type"] == COUNT), train_truths[j], r["type"] == COUNT)
        for j, r in enumerate(train_records)
    ])
    index = RetrievalIndex(features.vectors, [r["type"] for r in train_records], config["rag_search_chunk_rows"])
    rng = np.random.default_rng(config["rag_seed"])

    per_split: dict[str, dict[str, QueryResults]] = {}
    search_started = time.monotonic()
    queries = 0
    for split in [s for s in config["splits"] if s.startswith("test")]:
        dataset = dataloaders[split].dataset
        rows = question_pool(dataset.qa_records, config["rag_seed"])
        queries += len(rows)
        per_split[split] = score_queries(model, dataset, rows, index, train_truths, train_right, config,
                                         device, rng, f"score {split}")
    pooled = _merge(list(per_split.values()))
    passed = [t for t in types if t in pooled
              and passes(error_warning(pooled[t], config["rag_bootstrap_samples"], config["rag_seed"]))]

    sections = [
        f"# Retrieval results: {run_dir.name}" + (" (pilot subsets)" if pilot else "") + "\n\n"
        f"Index: {len(train_records):,} training QA pairs, {features.vectors.shape[1]}-wide fused vectors "
        f"({features.vectors.nbytes / 1e6:.0f} MB). k = {config['rag_k']}. Test questions: one per tile and "
        f"type ({queries:,} in all).\n",
        "**Rule, fixed in advance:** a type passes when the AUROC difference, test splits pooled, has a 95% "
        "lower bound above 0. Only passing types may use neighbour disagreement as an uncertainty warning.\n\n"
        f"**Passes on: {', '.join(passed) if passed else 'no type'}**"
        + (f"; fails on: {', '.join(t for t in types if t not in passed)}" if len(passed) < len(types) else "")
        + "\n",
        results_section("Test splits pooled (this decides)", pooled, types, config),
    ]
    sections += [results_section(f"{split} alone (does not decide)", r, types, config) for split, r in per_split.items()]
    search_minutes = (time.monotonic() - search_started) / 60
    sections.append(
        f"## Time\n\nIndex built or loaded in {build_minutes:.1f} min; {queries:,} questions encoded and "
        f"searched in {search_minutes:.1f} min, on {device.type}.\n"
    )
    return "\n".join(sections)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("config", type=Path, nargs="?", default=Path("configs/hr_config.yaml"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--index-dir", type=Path, help="default: <run_dir>/retrieval[_pilot]")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = run_retrieval(args.run_dir, config, args.pilot, args.index_dir)
    out_path = Path("reports") / f"rag_retrieval_{args.run_dir.name}{'_pilot' if args.pilot else ''}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote {out_path}")
    print(report)


if __name__ == "__main__":
    _main()

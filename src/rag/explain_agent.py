"""RAG explanations: the model's answer, its occlusion map and its retrieved neighbours, written up by an LLM.

The LLM explains this model's answer; it never answers the question itself and never sees the image.
The first sentence (the answer) is written by code, the LLM writes the rest, and code checks it: every
number must come from the facts, neighbours are cited only as #1-#5, no claims about objects or about
whether the answer is right, and the uncertainty and no-data notes appear exactly when the facts say so.
A failed check falls back to a fixed template, so an unchecked text is never shown.

Rules fixed before any explanation was generated:
  * Uncertainty warning. Classification: fewer than ``rag_warning_min_agree`` of the k neighbours have
    the model's answer as their true answer. Count: the rounded count is outside the range of the
    neighbours' true counts.
  * Blind rating: per question, the user rates two explanations (with and without the neighbours) in a
    random order. Retrieval improves the explanations if, among questions with a preference, the share
    preferring the one with neighbours has a 95% (Wilson) lower bound above 0.5.

    python -m src.rag.explain_agent <run_dir> configs/hr_config.yaml generate [--pilot] [--index-dir DIR]
    python -m src.rag.explain_agent <run_dir> configs/hr_config.yaml score --ratings ratings.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import yaml
from torch import nn

from src.data_pipeline import prepare_batch, vqa_collate_fn
from src.evaluate import load_run
from src.models.clip_baseline import answer_text
from src.rag.retrieval_index import (
    COUNT,
    RetrievalIndex,
    build_index,
    compute_features,
    empty_features,
)

ZONE_ROWS = ("top", "middle", "bottom")
ZONE_COLS = ("left", "centre", "right")
CONDITIONS = ("with_neighbours", "without_neighbours")


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

@dataclass
class Neighbour:
    rank: int
    img_id: int
    question: str
    answer: str  # readable true answer


@dataclass
class Facts:
    split: str
    row: int
    img_id: int
    question: str
    qtype: str
    answer: str                    # readable model answer (count: rounded)
    confidence: float | None       # predicted-class probability; None for count
    zones: list[str]               # where the top occlusion cells are, strongest first
    neighbours: list[Neighbour]
    agree: int                     # neighbours whose true answer is the model's (classification)
    count_range: tuple[float, float] | None
    warning: bool
    strip: bool
    truth: str                     # readable true answer: for rating only, never sent to the LLM
    correct: bool


def cell_zone(cell: int, grid: int) -> str:
    """The 3x3 zone of a grid cell, e.g. 'top left' or 'middle centre' (said as 'centre')."""
    row, col = divmod(cell, grid)
    name = f"{ZONE_ROWS[min(row * 3 // grid, 2)]} {ZONE_COLS[min(col * 3 // grid, 2)]}"
    return "centre" if name == "middle centre" else name


def top_zones(scores: torch.Tensor, grid: int, top: int) -> list[str]:
    """Zones of the ``top`` strongest cells, strongest first, without repeats."""
    zones: list[str] = []
    for cell in scores.argsort(descending=True)[:top].tolist():
        zone = cell_zone(cell, grid)
        if zone not in zones:
            zones.append(zone)
    return zones


def warning(prediction: float, neighbour_truths: np.ndarray, is_count: bool, min_agree: int) -> bool:
    if is_count:
        answer = round(prediction)
        return bool(answer < neighbour_truths.min() or answer > neighbour_truths.max())
    return int((neighbour_truths == prediction).sum()) < min_agree


def readable(answer: str | float, is_count: bool) -> str:
    return str(int(round(float(answer)))) if is_count else answer_text(str(answer))


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def facts_text(facts: Facts, with_neighbours: bool) -> str:
    """The facts block the LLM sees. The truth and whether the model is right are left out. The warning and
    the strip are stated only when they apply: the pilot showed a "no" line gets repeated back as filler."""
    lines = [f"Question: {facts.question}", f"Question type: {facts.qtype}"]
    if facts.confidence is None:
        lines.append(f'Model\'s answer: "{facts.answer}" (a count; the model gives no confidence for counts)')
    else:
        lines.append(f'Model\'s answer: "{facts.answer}" (confidence {facts.confidence:.0%})')
    lines.append(f"Grid cells whose change moved the answer most: {', '.join(facts.zones)}")
    if with_neighbours:
        k = len(facts.neighbours)
        lines.append(f"The {k} most similar training questions, by the model's own representation, with their true answers:")
        lines += [f'#{n.rank} "{n.question}" true answer: {n.answer}' for n in facts.neighbours]
        if facts.count_range is None:
            lines.append(f"Similar cases whose true answer is the model's answer: {facts.agree} of {k}")
        else:
            low, high = facts.count_range
            lines.append(f"Similar cases' true counts range from {_number(low)} to {_number(high)}")
        if facts.warning:
            lines.append("Uncertain: yes")
    if facts.strip:
        lines.append("Black no-data strip on this tile: yes")
    return "\n".join(lines)


def first_sentence(facts: Facts) -> str:
    """Written by code, so the stated answer is always the model's."""
    if facts.confidence is None:
        return f'The model answered "{facts.answer}".'
    return f'The model answered "{facts.answer}" with {facts.confidence:.0%} confidence.'


PROMPT_START = """You explain one answer of an aerial-image question-answering model to a non-expert.
The first sentence, stating the model's answer, is already written. Write only what follows it.

Rules:
- Use only the facts given. You have not seen the image: never guess what is in it, what the grid cells
  contain, or what features, cues or patterns the model used, learned or believes.
- Do not explain why the model answered as it did. Report the facts; do not add reasons.
- 1 to 3 sentences, at most {max_words} words, plain English. Short is fine when there is little to say.
- Do not state the answer again as your own claim, and do not say whether it is correct.
- Only use numbers that appear in the facts.
- The grid cells are where changing the image changed the answer most. Never say they show, contain,
  highlight or locate anything.
- If the facts mention a black no-data strip, say the tile has a "no-data strip" and that strong cells
  near it say nothing about what is there. If they do not mention one, never mention a strip or missing data."""

PROMPT_WITH_NEIGHBOURS = """
- The similar training cases were found after the model answered, by comparing its internal representation.
  They did not cause, support or influence the answer: never say the answer is based on, supported by,
  driven by or explained by them. You may say how many of them have the model's answer as their true answer.
- Refer to them only as #1 to #5, by rank. For each, say only what the facts give: its question and its true
  answer. Never say whether the model was right on them, and never mention similarity scores.
- Call their answers "true answers", never "correct" answers.
- If the facts say "Uncertain: yes", say the answer is uncertain and why, using the word "uncertain".
  Otherwise never use the word "uncertain"."""

PROMPT_WITHOUT_NEIGHBOURS = """
- You are given no training cases or examples. Never mention any, and never write "#" followed by a number."""


def system_prompt(config: dict[str, Any], with_neighbours: bool) -> str:
    rules = PROMPT_WITH_NEIGHBOURS if with_neighbours else PROMPT_WITHOUT_NEIGHBOURS
    return PROMPT_START.format(max_words=config["rag_max_words"]) + rules


def build_messages(facts: Facts, config: dict[str, Any], with_neighbours: bool) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(config, with_neighbours)},
        {"role": "user", "content": f"Facts:\n{facts_text(facts, with_neighbours)}\n\n"
                                    f"Already written: {first_sentence(facts)}\nContinue:"},
    ]


NUMBER = re.compile(r"\d+(?:\.\d+)?")


def check_explanation(body: str, facts: Facts, config: dict[str, Any], with_neighbours: bool) -> list[str]:
    """Every rule the LLM's text breaks; empty means it may be shown."""
    failures = []
    lower = body.lower()
    if not body.strip():
        return ["empty"]
    if len(body.split()) > config["rag_max_words"]:
        failures.append("too long")
    allowed = set(NUMBER.findall(facts_text(facts, with_neighbours))) | {str(i) for i in range(len(facts.neighbours) + 1)}
    unknown = sorted(set(NUMBER.findall(body)) - allowed)
    if unknown:
        failures.append(f"numbers not in the facts: {', '.join(unknown)}")
    cited = {int(n) for n in re.findall(r"#(\d+)", body)}
    if cited and (not with_neighbours or not cited <= set(range(1, len(facts.neighbours) + 1))):
        failures.append("cites a neighbour that was not given")
    banned = [p for p in config["rag_banned_phrases"] if re.search(p, lower)]
    if banned:
        failures.append(f"banned phrasing: {', '.join(banned)}")
    if with_neighbours and ("uncertain" in lower) != facts.warning:
        failures.append("uncertainty note " + ("missing" if facts.warning else "not called for"))
    if ("no-data" in lower or "strip" in lower) != facts.strip:
        failures.append("no-data note " + ("missing" if facts.strip else "not called for"))
    return failures


def template_body(facts: Facts, with_neighbours: bool) -> str:
    """The fallback text: fixed sentences built from the facts alone."""
    sentences = []
    if with_neighbours:
        k = len(facts.neighbours)
        if facts.count_range is None:
            sentences.append(f"Of the {k} training questions the model treats as most similar, {facts.agree} "
                             f"had this answer as their true answer (#1 is the closest).")
            if facts.warning:
                sentences.append("Because most of them disagree, treat this answer as uncertain.")
        else:
            low, high = facts.count_range
            spread = (f"all had a true count of {_number(low)}" if low == high
                      else f"had true counts from {_number(low)} to {_number(high)}")
            sentences.append(f"The {k} training questions the model treats as most similar {spread}.")
            if facts.warning:
                sentences.append("This count falls outside that range, so treat it as uncertain.")
    zones = facts.zones[0] if len(facts.zones) == 1 else f"{', '.join(facts.zones[:-1])} and {facts.zones[-1]}"
    sentences.append(f"Changing the cells at the {zones} of the tile changed the answer most. "
                     "That tells you what the answer depended on, not where anything is.")
    if facts.strip:
        sentences.append("The tile has a black no-data strip; strong cells next to it say nothing about what is there.")
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> tuple[str, int]:
        """The reply text and the total tokens used."""


class OpenAICompatibleClient:
    """Chat completions over HTTP for any OpenAI-compatible endpoint (Groq, Gemini's compatibility
    endpoint, ...). The key is read from an environment variable and never stored or logged."""

    def __init__(self, config: dict[str, Any]) -> None:
        key = os.environ.get(config["rag_llm_api_key_env"])
        if not key:
            raise RuntimeError(f"set the environment variable {config['rag_llm_api_key_env']} to your API key")
        self._key = key
        self.url = config["rag_llm_base_url"].rstrip("/") + "/chat/completions"
        self.payload = {"model": config["rag_llm_model"], "temperature": config["rag_llm_temperature"],
                        "max_tokens": config["rag_llm_max_tokens"], **(config.get("rag_llm_extra") or {})}
        self.min_interval = config["rag_llm_min_interval_s"]
        self.max_retries = config["rag_llm_max_retries"]
        self.timeout = config["rag_llm_timeout_s"]
        self._last = 0.0

    def _post(self, body: bytes) -> dict[str, Any]:
        request = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
            "User-Agent": "rsvqa-rag-explainer"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def complete(self, messages: list[dict[str, str]]) -> tuple[str, int]:
        body = json.dumps({**self.payload, "messages": messages}).encode()
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                reply = self._post(body)
                text = reply["choices"][0]["message"].get("content") or ""
                return text.strip(), int(reply.get("usage", {}).get("total_tokens", 0))
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise RuntimeError(f"LLM request failed: HTTP {error.code}") from None
                retry_after = error.headers.get("retry-after") if error.headers else None
                time.sleep(float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit()
                           else 2 ** attempt * 5)
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == self.max_retries:
                    raise RuntimeError(f"LLM request failed: {error}") from None
                time.sleep(2 ** attempt * 5)
        raise AssertionError("unreachable")


def explain(facts: Facts, client: LLMClient | None, config: dict[str, Any], with_neighbours: bool) -> dict[str, Any]:
    """The explanation shown to users, and where it came from: 'llm', or 'template' after a failed
    check or a failed request (or with no client at all)."""
    failures: list[str] = []
    tokens = 0
    body = ""
    if client is not None:
        try:
            body, tokens = client.complete(build_messages(facts, config, with_neighbours))
            failures = check_explanation(body, facts, config, with_neighbours)
        except RuntimeError as error:
            failures = [str(error)]
    source = "llm" if client is not None and not failures else "template"
    shown = body if source == "llm" else template_body(facts, with_neighbours)
    return {"text": f"{first_sentence(facts)} {shown}", "source": source, "failures": failures,
            "llm_text": body, "tokens": tokens}


# ---------------------------------------------------------------------------
# Building facts from the model
# ---------------------------------------------------------------------------

def _grid(model: nn.Module) -> int:
    return round(math.sqrt(model.encoder.pos_embed.size(1) - 1))  # pos_embed also holds CLS


def build_facts(
    model: nn.Module, dataset: Any, split: str, row: int, index: RetrievalIndex, train: Any,
    train_truths: np.ndarray, config: dict[str, Any], device: torch.device,
) -> Facts:
    from src.explain import no_data_cells, occlusion_map

    record = dataset.qa_records[row]
    is_count = record["type"] == COUNT
    batch = prepare_batch(vqa_collate_fn([dataset[row]]), device, dataset)
    grid = _grid(model)
    with torch.no_grad():
        outputs = model(batch)
        scores = occlusion_map(model, batch, outputs["class_logits"], outputs["count_pred"], grid * grid)[0]
    probabilities = outputs["class_logits"].softmax(-1)[0]
    predicted_class = int(probabilities.argmax())
    prediction = float(outputs["count_pred"][0]) if is_count else float(predicted_class)

    features = empty_features(1, model.classifier.in_features)
    compute_features(model, dataset, [row], features, config, device, "facts")
    _, rows = index.search(features.vectors, record["type"], config["rag_k"], device)
    neighbour_rows = rows[0]
    truths = train_truths[neighbour_rows]
    neighbours = [
        Neighbour(rank=i + 1, img_id=int(train.qa_records[r]["img_id"]), question=train.qa_records[r]["question"],
                  answer=readable(train.qa_records[r]["answer"], is_count))
        for i, r in enumerate(neighbour_rows.tolist())
    ]
    pixels = batch["image"] * dataset.image_std.to(device).view(3, 1, 1) + dataset.image_mean.to(device).view(3, 1, 1)
    strip = bool(no_data_cells(pixels, grid, config["xai_nodata_pixel_value"], config["xai_nodata_cell_fraction"]).any())
    truth = float(record["answer"]) if is_count else float(dataset.answer_vocab.encode(record["answer"]))
    answer = dataset.answer_vocab.idx_to_class[predicted_class] if not is_count else prediction
    return Facts(
        split=split, row=row, img_id=int(record["img_id"]), question=record["question"], qtype=record["type"],
        answer=readable(answer, is_count), confidence=None if is_count else float(probabilities[predicted_class]),
        zones=top_zones(scores, grid, config["rag_top_cells"]), neighbours=neighbours,
        agree=int((truths == prediction).sum()) if not is_count else 0,
        count_range=(float(truths.min()), float(truths.max())) if is_count else None,
        warning=warning(prediction, truths, is_count, config["rag_warning_min_agree"]), strip=strip,
        truth=readable(record["answer"], is_count),
        correct=(round(prediction) == truth) if is_count else (prediction == truth),
    )


# ---------------------------------------------------------------------------
# Warning statistics and the rating set
# ---------------------------------------------------------------------------

@dataclass
class PoolResult:
    rows: list[int] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    correct: list[bool] = field(default_factory=list)
    warned: list[bool] = field(default_factory=list)


def score_pool(
    model: nn.Module, dataset: Any, rows: list[int], index: RetrievalIndex, train_truths: np.ndarray,
    config: dict[str, Any], device: torch.device, desc: str,
) -> PoolResult:
    """Whether the model is right and whether the warning fires, for each question in ``rows``."""
    records = dataset.qa_records
    features = empty_features(len(rows), model.classifier.in_features)
    compute_features(model, dataset, rows, features, config, device, desc)
    result = PoolResult()
    by_type: dict[str, list[int]] = defaultdict(list)
    for j, row in enumerate(rows):
        by_type[records[row]["type"]].append(j)
    for qtype, positions in by_type.items():
        is_count = qtype == COUNT
        _, neighbours = index.search(features.vectors[positions], qtype, config["rag_k"], device)
        for i, j in enumerate(positions):
            record = records[rows[j]]
            prediction = float(features.pred_count[j]) if is_count else float(features.pred_class[j])
            truth = float(record["answer"]) if is_count else float(dataset.answer_vocab.encode(record["answer"]))
            result.rows.append(rows[j])
            result.types.append(qtype)
            result.correct.append(round(prediction) == truth if is_count else prediction == truth)
            result.warned.append(warning(prediction, train_truths[neighbours[i]], is_count, config["rag_warning_min_agree"]))
    return result


def warning_table(results: dict[str, PoolResult], types: list[str]) -> str:
    lines = [
        "| Set | Type | Questions | Warned | Model wrong when warned | Model wrong otherwise |",
        "|---|---|---|---|---|---|",
    ]
    sets = {**results, "pooled": _pool(list(results.values()))}
    for name, r in sets.items():
        for qtype in types:
            picked = [i for i, t in enumerate(r.types) if t == qtype]
            if not picked:
                continue
            warned = [i for i in picked if r.warned[i]]
            quiet = [i for i in picked if not r.warned[i]]
            wrong = lambda idx: f"{np.mean([not r.correct[i] for i in idx]):.1%} ({len(idx):,})" if idx else "—"
            lines.append(f"| {name} | {qtype} | {len(picked):,} | {len(warned) / len(picked):.1%} "
                         f"| {wrong(warned)} | {wrong(quiet)} |")
    return "\n".join(lines) + "\n"


def _pool(parts: list[PoolResult]) -> PoolResult:
    pooled = PoolResult()
    for part in parts:
        for name in ("rows", "types", "correct", "warned"):
            getattr(pooled, name).extend(getattr(part, name))
    return pooled


def rating_rows(result: PoolResult, per_group: int) -> list[int]:
    """Up to ``per_group`` questions per (type, right or wrong), in the pool's random order."""
    taken: dict[tuple[str, bool], int] = defaultdict(int)
    picked = []
    for row, qtype, correct in zip(result.rows, result.types, result.correct):
        if taken[(qtype, correct)] < per_group:
            taken[(qtype, correct)] += 1
            picked.append(row)
    return picked


RATING_FIELDS = ("id", "split", "question", "model_answer", "true_answer", "model_right", "text_1", "text_2",
                 "useful_1", "useful_2", "honest_1", "honest_2", "prefer")


def write_rating_sheet(items: list[dict[str, Any]], seed: int, sheet: Path, key: Path) -> None:
    """A CSV for blind rating (the two texts in a random order per question) and the key to unblind it."""
    rng = random.Random(seed)
    order = {}
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RATING_FIELDS)
        writer.writeheader()
        for item in items:
            first = rng.choice(CONDITIONS)
            second = CONDITIONS[1] if first == CONDITIONS[0] else CONDITIONS[0]
            order[item["id"]] = {"order": [first, second],
                                 "sources": {c: item[c]["source"] for c in CONDITIONS}}
            facts = item["facts"]
            writer.writerow({
                "id": item["id"], "split": facts["split"], "question": facts["question"],
                "model_answer": facts["answer"], "true_answer": facts["truth"],
                "model_right": "yes" if facts["correct"] else "no",
                "text_1": item[first]["text"], "text_2": item[second]["text"],
                "useful_1": "", "useful_2": "", "honest_1": "", "honest_2": "", "prefer": "",
            })
    key.write_text(json.dumps(order, indent=1))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _preference_line(label: str, wins: dict[str, int], same: int) -> str:
    decided = wins["with_neighbours"] + wins["without_neighbours"]
    if not decided:
        return f"| {label} | 0 | — | — | {same} |"
    low, high = wilson_interval(wins["with_neighbours"], decided)
    return (f"| {label} | {decided} | {wins['with_neighbours']} ({wins['with_neighbours'] / decided:.0%}) "
            f"| {low:.0%} to {high:.0%} | {same} |")


def score_ratings(sheet: Path, key: Path) -> str:
    """Unblinds a filled-in sheet. ``prefer`` is 1, 2 or 'same'; scores are 1 (poor) to 3 (good).

    Only pairs where both texts are the LLM's decide: a pair of LLM text against the template compares
    writers, not retrieval. All pairs, and pairs by source, are reported alongside."""
    key_data = json.loads(key.read_text())
    groups = ("both LLM", "with: LLM, without: template", "with: template, without: LLM", "both template")
    scores = {g: {c: {"useful": [], "honest": []} for c in CONDITIONS} for g in (*groups, "all")}
    wins = {g: {c: 0 for c in CONDITIONS} for g in (*groups, "all")}
    same = {g: 0 for g in (*groups, "all")}
    unrated = 0
    with sheet.open(newline="") as f:
        for row in csv.DictReader(f):
            entry = key_data[row["id"]]
            conditions, sources = entry["order"], entry["sources"]
            with_llm, without_llm = (sources[c] == "llm" for c in CONDITIONS)
            group = groups[0] if with_llm and without_llm else groups[1] if with_llm else \
                groups[2] if without_llm else groups[3]
            for g in (group, "all"):
                for slot, condition in zip(("1", "2"), conditions):
                    for measure in ("useful", "honest"):
                        value = row[f"{measure}_{slot}"].strip()
                        if value:
                            scores[g][condition][measure].append(float(value))
            prefer = row["prefer"].strip().lower()
            if prefer in ("1", "2"):
                for g in (group, "all"):
                    wins[g][conditions[int(prefer) - 1]] += 1
            elif prefer == "same":
                same[group] += 1
                same["all"] += 1
            else:
                unrated += 1
    deciding = wins["both LLM"]
    decided = deciding["with_neighbours"] + deciding["without_neighbours"]
    low, _ = wilson_interval(deciding["with_neighbours"], decided)
    improves = decided > 0 and low > 0.5
    lines = [
        "# Blind rating of RAG explanations\n",
        "**Rule, fixed before rating:** among pairs where both texts are written by the LLM and a preference "
        "is given, retrieval improves the explanations if the share preferring the one with neighbours has a "
        "95% Wilson lower bound above 0.5. Pairs involving the template compare writers, not retrieval, and "
        "do not decide.\n",
        (f"**Result: {'retrieval improves the explanations' if improves else 'no clear improvement from retrieval'}.**\n"
         if decided else "**No preferences given yet on pairs where both texts are the LLM's.**\n"),
        f"Not rated: {unrated}.\n",
        "| Pairs | With a preference | Preferred with neighbours | 95% CI | Same |",
        "|---|---|---|---|---|",
    ]
    lines += [_preference_line(g, wins[g], same[g]) for g in (*groups, "all")]
    mean = lambda values: f"{np.mean(values):.2f} (n={len(values)})" if values else "—"
    lines += ["", "| Pairs | Condition | Useful (1-3) | Honest about uncertainty (1-3) |", "|---|---|---|---|"]
    lines += [f"| {g} | {c.replace('_', ' ')} | {mean(scores[g][c]['useful'])} | {mean(scores[g][c]['honest'])} |"
              for g in ("both LLM", "all") for c in CONDITIONS]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

def run_generate(
    run_dir: Path, config: dict[str, Any], pilot: bool = False, index_dir: Path | None = None,
    client: LLMClient | None = None, out_dir: Path = Path("reports"),
) -> str:
    """Warning statistics over the test pool, then both explanations for the rating set."""
    from src.explain import question_pool
    import hashlib

    config = dict(config)
    config["cache_images"] = False
    if not pilot:
        config["max_images_per_split"] = None
    started = time.monotonic()
    model, dataloaders, device = load_run(run_dir, config)
    model.eval()
    train = dataloaders["train"].dataset
    index_dir = index_dir or run_dir / ("retrieval_pilot" if pilot else "retrieval")
    checkpoint = hashlib.sha256((run_dir / "checkpoints" / "best.pt").read_bytes()).hexdigest()
    features = build_index(model, train, index_dir, config, device, checkpoint)
    index = RetrievalIndex(features.vectors, [r["type"] for r in train.qa_records], config["rag_search_chunk_rows"])
    train_truths = np.array([float(r["answer"]) if r["type"] == COUNT else float(train.answer_vocab.encode(r["answer"]))
                             for r in train.qa_records])
    types = [*config["question_types"], COUNT]

    pools: dict[str, PoolResult] = {}
    items: list[dict[str, Any]] = []
    per_group = config["rag_pilot_rating_per_group" if pilot else "rag_rating_per_group"]
    llm_started = None
    for split in [s for s in config["splits"] if s.startswith("test")]:
        dataset = dataloaders[split].dataset
        pool = question_pool(dataset.qa_records, config["rag_seed"])
        pools[split] = score_pool(model, dataset, pool, index, train_truths, config, device, f"warning {split}")
        for row in rating_rows(pools[split], per_group):
            facts = build_facts(model, dataset, split, row, index, train, train_truths, config, device)
            llm_started = llm_started or time.monotonic()
            item = {"id": f"{split}-{row}", "facts": asdict(facts)}
            for condition in CONDITIONS:
                item[condition] = explain(facts, client, config, condition == "with_neighbours")
            items.append(item)

    name = run_dir.name + ("_pilot" if pilot else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"rag_explanations_{'pilot' if pilot else 'full'}.json").write_text(json.dumps(items, indent=1))
    sheet = out_dir / f"rag_rating_{name}.csv"
    write_rating_sheet(items, config["rag_seed"], sheet, run_dir / f"rag_rating_key_{'pilot' if pilot else 'full'}.json")
    return generate_report(name, items, pools, types, config, started, llm_started, sheet, blind=not pilot)


def generate_report(
    name: str, items: list[dict[str, Any]], pools: dict[str, PoolResult], types: list[str],
    config: dict[str, Any], started: float, llm_started: float | None, sheet: Path, blind: bool = False,
) -> str:
    """The report. When blind (the real rating set), the labelled texts are left out: they would show which
    text has the neighbours and whether the model was right before the rating is done."""
    sections = [
        f"# RAG explanations: {name}\n\n"
        f"LLM: `{config['rag_llm_model']}` at `{config['rag_llm_base_url']}`. Rating set: {len(items)} questions "
        f"(up to {config['rag_pilot_rating_per_group' if name.endswith('_pilot') else 'rag_rating_per_group']} per "
        "test split, type and right or wrong answer), each explained with and without the neighbours.\n",
        "## Uncertainty warning on the test pool\n\n"
        f"Rule, fixed in advance: classification warns when fewer than {config['rag_warning_min_agree']} of the "
        f"{config['rag_k']} neighbours have the model's answer as their true answer; count warns when the rounded "
        "count is outside the neighbours' range of true counts. One question per tile and type.\n\n"
        + warning_table(pools, types),
    ]
    lines = ["## Checks\n", "| Condition | LLM text shown | Fell back to the template | Tokens |", "|---|---|---|---|"]
    reasons: dict[str, int] = defaultdict(int)
    for condition in CONDITIONS:
        results = [item[condition] for item in items]
        shown = sum(r["source"] == "llm" for r in results)
        for r in results:
            for failure in r["failures"]:
                reasons[f"{condition.replace('_', ' ')}: {failure.split(':')[0]}"] += 1
        lines.append(f"| {condition.replace('_', ' ')} | {shown} of {len(results)} | {len(results) - shown} "
                     f"| {sum(r['tokens'] for r in results):,} |")
    if reasons:
        lines += ["", "Why texts fell back:", ""] + [f"- {reason}: {n}" for reason, n in sorted(reasons.items())]
    sections.append("\n".join(lines) + "\n")
    examples = ["## Explanations\n"]
    if blind:
        examples.append("Left out so the rating stays blind. After scoring, the labelled texts are in "
                        f"`rag_explanations_full.json` in the run folder.\n")
    for item in ([] if blind else items):
        f = item["facts"]
        examples.append(
            f"### {item['id']}: {f['question']}\n\nTrue answer: {f['truth']}; model {'right' if f['correct'] else 'wrong'}; "
            f"warning: {'yes' if f['warning'] else 'no'}; no-data strip: {'yes' if f['strip'] else 'no'}.\n\n"
            + "".join(f"- **{c.replace('_', ' ')}** ({item[c]['source']}"
                      + (f"; failed: {'; '.join(item[c]['failures'])}" if item[c]["failures"] else "")
                      + f"): {item[c]['text']}\n" for c in CONDITIONS)
        )
    sections.append("\n".join(examples))
    total = (time.monotonic() - started) / 60
    llm = (time.monotonic() - llm_started) / 60 if llm_started else 0.0
    sections.append(f"## Time\n\n{total:.1f} min in total, {llm:.1f} min of it explaining the rating set. "
                    f"Rating sheet: `{sheet}`.\n")
    return "\n".join(sections)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("stage", choices=("generate", "score"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--ratings", type=Path, help="score: the filled-in rating sheet")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    suffix = "_pilot" if args.pilot else ""
    if args.stage == "score":
        if not args.ratings:
            parser.error("score needs --ratings")
        report = score_ratings(args.ratings, args.run_dir / f"rag_rating_key_{'pilot' if args.pilot else 'full'}.json")
        out_path = Path("reports") / f"rag_rating_results_{args.run_dir.name}{suffix}.md"
    else:
        report = run_generate(args.run_dir, config, args.pilot, args.index_dir, OpenAICompatibleClient(config))
        out_path = Path("reports") / f"rag_explanations_{args.run_dir.name}{suffix}.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")
    print(report)


if __name__ == "__main__":
    _main()

"""Inspect a raw RSVQA download (LR or HR) before any pipeline code relies on it.

Stdlib only (Pillow optional), so it runs anywhere:
    python scripts/inspect_rsvqa.py data/raw/rsvqa_hr USGS_split
    python scripts/inspect_rsvqa.py data/raw/rsvqa_lr LR_split

Prints the file layout, per-split sizes, question types, answer distributions,
count statistics, tile overlap between splits and question lengths, then a
suggested config block. Verify the config values against this output.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def tokenize(question: str) -> list[str]:
    # must match src/data_pipeline.tokenize
    return question.rstrip("?").lower().split()


def load_records(raw_dir: Path, prefix: str, split: str) -> list[dict]:
    questions = json.loads((raw_dir / f"{prefix}_{split}_questions.json").read_text())["questions"]
    answers = json.loads((raw_dir / f"{prefix}_{split}_answers.json").read_text())["answers"]
    active_q = {q["id"]: q for q in questions if q["active"]}
    return [
        {"img_id": active_q[a["question_id"]]["img_id"], "question": active_q[a["question_id"]]["question"],
         "type": active_q[a["question_id"]]["type"], "answer": a["answer"]}
        for a in answers if a["active"]
    ]


def find_image_dirs(raw_dir: Path, max_depth: int = 3) -> list[tuple[Path, int, str]]:
    found = []
    for d in [raw_dir, *[p for p in raw_dir.rglob("*") if p.is_dir() and len(p.relative_to(raw_dir).parts) <= max_depth]]:
        exts = Counter(p.suffix.lower() for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if exts:
            ext, n = exts.most_common(1)[0]
            found.append((d, n, ext))
    return sorted(found, key=lambda x: -x[1])


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "n/a"


def main() -> None:
    raw_dir = Path(sys.argv[1])
    prefix = sys.argv[2]
    print(f"== Files in {raw_dir}")
    for p in sorted(raw_dir.iterdir()):
        size = p.stat().st_size / 1e6 if p.is_file() else 0
        print(f"  {p.name:45s} {'dir' if p.is_dir() else f'{size:9.1f} MB'}")

    splits = sorted(p.name[len(prefix) + 1:-len("_questions.json")] for p in raw_dir.glob(f"{prefix}_*_questions.json"))
    print(f"\n== Splits found: {splits}")
    records = {s: load_records(raw_dir, prefix, s) for s in splits}

    print("\n== Split sizes")
    for s in splits:
        types = Counter(r["type"] for r in records[s])
        print(f"  {s:12s} QA={len(records[s]):8d} tiles={len({r['img_id'] for r in records[s]}):6d} types={dict(types)}")

    print("\n== Tile overlap between splits (should all be 0)")
    ids = {s: {r["img_id"] for r in records[s]} for s in splits}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            print(f"  {a} & {b}: {len(ids[a] & ids[b])}")

    train = records.get("train", [])
    print("\n== Non-count answers in train, by question type")
    non_count = Counter(r["answer"] for r in train if r["type"] != "count")
    for qtype in sorted({r["type"] for r in train} - {"count"}):
        sub = [r["answer"] for r in train if r["type"] == qtype]
        top = Counter(sub).most_common(12)
        print(f"  {qtype} ({len(sub)} QA): " + ", ".join(f"{a!r} {pct(n, len(sub))}" for a, n in top))
    print(f"  -> distinct non-count answers (= vocab_size): {len(non_count)}: {sorted(non_count)}")

    counts = [int(r["answer"]) for r in train if r["type"] == "count" and str(r["answer"]).isdigit()]
    if counts:
        s = sorted(counts)
        n = len(s)
        print(f"\n== Count answers in train ({n} QA)")
        for lo, hi in [(0, 0), (1, 10), (11, 100), (101, 1000), (1001, 10**12)]:
            label = f"{lo}-{hi}" if hi < 10**12 else f">{lo - 1}"
            print(f"  {label:10s} {pct(sum(lo <= v <= hi for v in s), n)}")
        print(f"  median={statistics.median(s)} mean={statistics.mean(s):.1f} "
              f"p90={s[int(.9 * n)]} p99={s[int(.99 * n)]} max={s[-1]} distinct={len(set(s))}")

    lengths = sorted(len(tokenize(r["question"])) for r in train)
    if lengths:
        print(f"\n== Question length (tokens, train): max={lengths[-1]} p99={lengths[int(.99 * len(lengths))]}")

    print("\n== Image folders (most images first)")
    dirs = find_image_dirs(raw_dir)
    for d, n, ext in dirs[:5]:
        print(f"  {str(d.relative_to(raw_dir)) or '.':30s} {n:7d} files  ext={ext}")
    if dirs and train:
        d, _, ext = dirs[0]
        sample_ids = sorted({r["img_id"] for r in train})[:200]
        missing = [i for i in sample_ids if not (d / f"{i}{ext}").exists()]
        print(f"  first 200 train tile ids found in {d.name}/: {len(sample_ids) - len(missing)}"
              f"/{len(sample_ids)} (missing e.g. {missing[:3]})")
        try:
            from PIL import Image
            with Image.open(d / f"{sample_ids[0]}{ext}") as im:
                print(f"  sample image: size={im.size} mode={im.mode}")
        except Exception as e:  # Pillow missing or unreadable file
            print(f"  sample image not opened: {e}")
    elif not dirs:
        print("  no image folder found (images not extracted yet?)")

    print("\n== Suggested config values (verify before use)")
    print(f"split_file_prefix: {prefix}")
    if dirs:
        print(f"images_subdir: {dirs[0][0].relative_to(raw_dir)}")
        print(f"image_ext: {dirs[0][2]}")
    print(f"splits: {splits}")
    print(f"question_types: {sorted({r['type'] for r in train} - {'count'})}")
    print(f"vocab_size: {len(non_count)}")
    if lengths:
        print(f"max_question_len: {lengths[-1]}")


if __name__ == "__main__":
    main()

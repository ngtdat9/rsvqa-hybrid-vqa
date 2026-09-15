"""
Week 1 — Data inspection pass for RSVQA-LR.
Answers the BDAPP Lecture 03 "Key Questions to Ask" checklist against the
actual downloaded data, before any pipeline code gets written.
"""
import json
from collections import Counter
from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/raw/rsvqa_lr")
IMAGES_DIR = DATA_DIR / "images/Images_LR"
SPLITS = ["train", "val", "test"]


def load_split(split):
    with open(DATA_DIR / f"LR_split_{split}_questions.json") as f:
        questions = json.load(f)["questions"]
    with open(DATA_DIR / f"LR_split_{split}_answers.json") as f:
        answers = json.load(f)["answers"]
    return questions, answers


print("Images folder contents (first 5), to confirm the path is right:")
print(list(IMAGES_DIR.iterdir())[:5] if IMAGES_DIR.exists() else "IMAGES_DIR NOT FOUND")

print("\n" + "=" * 60)
print("1. HOW MANY SAMPLES?")
print("=" * 60)
print("NOTE: each split file holds the global question list (ids 0..N) truncated after")
print("that split's highest id, with all other splits' entries blanked to inactive stubs.")
print("The real sample count is the ACTIVE count; raw file length is padding-inflated.\n")
total_active = 0
total_raw = 0
for split in SPLITS:
    questions, answers = load_split(split)
    active_q = sum(1 for q in questions if q.get("active", True))
    active_a = sum(1 for a in answers if a.get("active", True))
    n_images = len({q["img_id"] for q in questions if q.get("active", True)})
    print(f"{split:5s}: {active_q:>6} active questions, {active_a:>6} active answers, "
          f"{n_images:>3} images  (raw file rows: {len(questions)})")
    total_active += active_q
    total_raw += len(questions)
print(f"\nTotal ACTIVE questions across splits: {total_active}  (paper reports 77,232)")
print(f"Sum of raw file rows:                {total_raw}  <- padding-inflated, not a real total")

image_files = list(IMAGES_DIR.glob("*.tif"))
print(f"Image files found: {len(image_files)}  (paper reports 772)")

print("\n" + "=" * 60)
print("2. WHAT ARE THE DATA TYPES / STRUCTURE?")
print("=" * 60)
train_q, train_a = load_split("train")
print("One question entry:")
print(json.dumps(train_q[0], indent=2))
print("One answer entry:")
print(json.dumps(train_a[0], indent=2))
if image_files:
    with Image.open(image_files[0]) as im:
        print(f"Sample image: {image_files[0].name}, size={im.size}, mode={im.mode}")

print("\n" + "=" * 60)
print("3. WHAT ARE THE CLASSES/LABELS? (answer vocabulary)")
print("=" * 60)
all_answers = set()
type_counts = Counter()
answers_by_type = {}
for split in SPLITS:
    questions, answers = load_split(split)
    type_by_qid = {q["id"]: q["type"] for q in questions if q.get("active", True)}
    for q in questions:
        if not q.get("active", True):
            continue
        type_counts[q["type"]] += 1
    for a in answers:
        if not a.get("active", True):
            continue
        all_answers.add(a["answer"])
        qtype = type_by_qid.get(a["question_id"])
        if qtype is not None:
            answers_by_type.setdefault(qtype, set()).add(a["answer"])
print(f"Distinct answers across all splits: {len(all_answers)}  (expected 9)")
print(sorted(all_answers))
print(f"Question type counts: {dict(type_counts)}")
print("\nAnswer vocabulary size by question type:")
for qtype, vocab in sorted(answers_by_type.items()):
    print(f"  {qtype:12s}: {len(vocab):5d} distinct answers")

print("\n" + "=" * 60)
print("4. INACTIVE STUB ENTRIES (expected padding, NOT missing data)")
print("=" * 60)
print("Rows shaped {'id': N, 'active': false} with no type/question/img_id. These are")
print("other splits' questions blanked out by the split-file format, not damaged records.")
print("Correct handling is to filter `active == True`; nothing to impute or drop.\n")
for split in SPLITS:
    questions, answers = load_split(split)
    inactive_q = sum(1 for q in questions if not q.get("active", True))
    inactive_a = sum(1 for a in answers if not a.get("active", True))
    empty_text = sum(1 for q in questions if not q.get("question", "").strip())
    active_q = len(questions) - inactive_q
    # empty text should be exactly the inactive rows -- any excess is a real problem
    unexplained = empty_text - inactive_q
    flag = "OK" if unexplained == 0 else f"WARNING: {unexplained} unexplained blanks"
    print(f"{split:5s}: {len(questions)} rows = {active_q} active + {inactive_q} stubs; "
          f"inactive answers={inactive_a}, empty question text={empty_text} ({flag})")

print("\n" + "=" * 60)
print("5. ARE THERE DUPLICATES?")
print("=" * 60)
for split in SPLITS:
    questions, _ = load_split(split)
    ids = [q["id"] for q in questions]
    status = "OK" if len(ids) == len(set(ids)) else "DUPLICATES FOUND"
    print(f"{split:5s}: {len(ids)} questions, {len(set(ids))} unique ids ({status})")
    exact_pairs = [(q["img_id"], q["question"]) for q in questions if q.get("active", True)]
    dupes = len(exact_pairs) - len(set(exact_pairs))
    print(f"       exact (image, question-text) duplicates: {dupes}")

print("\n" + "=" * 60)
print("6. CLASS DISTRIBUTION (train split, answer frequency by question type)")
print("=" * 60)
answers_by_id = {a["question_id"]: a["answer"] for a in train_a if a.get("active", True)}
dist_by_type = {}
for q in train_q:
    if not q.get("active", True):
        continue
    ans = answers_by_id.get(q["id"])
    dist_by_type.setdefault(q["type"], Counter())[ans] += 1
for qtype, counter in dist_by_type.items():
    total = sum(counter.values())
    print(f"\n{qtype} (n={total}):")
    for ans, n in counter.most_common():
        print(f"  {str(ans):25s} {n:6d}  ({100*n/total:.1f}%)")

print("\n" + "=" * 60)
print("7. ARE THERE CORRUPTED/LOW-QUALITY IMAGES?")
print("=" * 60)
corrupted = []
for path in image_files:
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as e:
        corrupted.append((path.name, str(e)))
print(f"Checked {len(image_files)} images, {len(corrupted)} failed to open")
for name, err in corrupted[:10]:
    print(f"  {name}: {err}")

print("\n" + "=" * 60)
print("8. DOES THE DATA MATCH THE PROBLEM? (spot check)")
print("=" * 60)
for q in train_q[:3]:
    ans = answers_by_id.get(q["id"], "MISSING")
    print(f"img {q['img_id']}: [{q['type']}] \"{q['question']}\" -> \"{ans}\"")
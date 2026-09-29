# Explainable Remote Sensing VQA — Project Plan

**Course:** Advanced Computer Vision (CNN, ViT, Hybrid CNN-Transformer, VLM, Multimodal, XAI) — also serves as the Big Data Analysis and Prediction Project (BDAPP) deliverable
**Timeline:** 8 weeks (2 months), solo
**Dataset:** RSVQA-HR — USGS High Resolution Orthoimagery, 15cm resolution aerial imagery, public download, no GEE required

## 1. Problem & Motivation

**Current situation:** Disaster-response teams, urban planners, and field analysts
regularly need answers to specific questions about a given satellite/aerial tile — is
there flooding, how many structures are present, what's the dominant land use.

**Problem:** Answering these questions today requires manually inspecting imagery
with GIS expertise. That's slow, and it gatekeeps the answer behind a specialist.

**Impact:** Delayed or expertise-gated answers slow down decisions that depend on
them — resource allocation, damage assessment, land-use monitoring.

**Opportunity:** A vision-language system that lets someone ask a plain-language
question about a tile and get an explainable answer removes the GIS-expertise
bottleneck for a well-defined class of questions.

**One-sentence goal (must pass the "can someone else understand it" test):**
Develop an explainable vision-language system that answers natural-language
questions about high-resolution aerial imagery, so non-experts aren't blocked by
manual, specialist-only image inspection.

## 2. Scope

| In scope | Out of scope |
|---|---|
| Closed-vocabulary questions (presence, count, comparison, area) on RSVQA-HR aerial tiles (512x512, 15cm resolution) | Free-text generation beyond the RAG explanation layer (Week 6) |
| Hybrid CNN-Transformer with a shared backbone and two output heads: classification for presence/comparison/rural_urban/area, regression (scalar) for count | Real-time or live satellite feeds |
| Zero-shot CLIP as the VLM comparison baseline | Full multi-temporal fire-spread modeling (that's the separate wildfire capstone) |
| Attention/saliency-based explainability (XAI) | Production-scale deployment |
| A demo UI showing the question → answer → explanation interaction | Full multi-agent orchestration/coordination systems |
| A data pipeline engineered for and trained at the full dataset scale (10,659 images, 1M+ QA pairs) | — |
| A lightweight multimodal RAG layer + LLM-generated explanations (retrieval plus single-agent reasoning) | — |

## 3. Dataset Quality Checklist

Scored against the course's 10-point "Goodness of the AI Dataset" framework
(Relevant, Sufficient, Representative, Diverse, Balanced, Accurate, Consistent,
Well-Annotated, Ethically Collected, Reproducible). Two honest gaps are called
out rather than hidden, since a critical self-assessment is itself part of the
grading lens.

| Criterion | Assessment |
|---|---|
| Relevant | Directly tests VQA on real aerial imagery, matching the project's stated goal |
| Sufficient | 10,659 images, 1.07M QA pairs — ample for training the hybrid classifier |
| Representative | Spans three distinct US metro regions (NYC/Long Island, Philadelphia, Portland) with varied urban/suburban/coastal land cover |
| Diverse | Question types span presence, count, comparison, and area; count is a regression target, the rest fall into a classification vocab (exact HR class count to be confirmed empirically the same way LR's was, per Week 1 finding below) |
| **Balanced — gap** | Answer-class distribution is very likely imbalanced (not verified pre-download). Mitigated in Week 2 by a stratified-group split (below) and monitored explicitly, not assumed away |
| **Accurate — caveat** | QA pairs are auto-generated from OpenStreetMap tags, not manually verified — accuracy is bounded by OSM completeness for these regions. State this as a known limitation, not a hidden one |
| Consistent | Fixed 512x512 tile format, uniform 15cm resolution across the whole dataset |
| **Well-Annotated — gap** | Labels come from automatic OSM-tag derivation, not expert annotation — a real difference from gold-standard, expert-validated datasets. Worth naming explicitly in the report |
| Ethically Collected | Public USGS orthoimagery + OSM data; no human subjects, no PII, no IRB requirement |
| Reproducible | Public benchmark with a published paper (Lobry et al.), standard splits already established in the literature |

**On "Early Fusion" (combining datasets for generalization):** considered
combining RSVQA-LR and RSVQA-HR as a single fused training set, but their answer
vocabularies differ (LR's non-count vocab is 4 classes — yes/no/rural/urban,
confirmed empirically from the raw data — vs. HR's larger vocab) and resolutions
differ (256px vs. 512px), so a direct fusion isn't straightforward without
vocabulary remapping. Each dataset is used sequentially instead — LR to validate
the pipeline, HR as the actual training/evaluation target — rather than fused.

## 4. Goal (technical)

Build a Hybrid CNN-Transformer visual question answering model for high-resolution
aerial imagery (RSVQA-HR: 10,659 images, 1,066,316 QA pairs) with a shared
CNN-Transformer backbone feeding two output heads — a classification head for
presence/comparison/rural_urban/area questions, and a regression head that
predicts count directly as a scalar rather than bucketing it into classes —
train it at full dataset scale, benchmark it against a pretrained VLM (CLIP,
zero-shot), and produce an explainability (XAI) analysis comparing attention-based
and saliency-based interpretations. With the extra time, add a lightweight
multimodal retrieval-augmented (RAG) explanation layer on top — retrieving similar
labeled examples and using an LLM to turn the model's answer and attention region
into a natural-language explanation. Engineer the data pipeline for full-dataset
scale from the start (chunked/streaming loading, not "load everything into a
dataframe") — this is what makes the project a genuine fit for a course named for
big data, not just a modeling exercise on a small benchmark.

## 5. Resource checklist

| Resource | Status |
|---|---|
| Time | 2 months — confirmed |
| Dataset access | RSVQA-HR, public, direct download — no special access needed |
| Dataset size | ~several GB (10,659 images at 512x512), 1,066,316 QA pairs (count questions regressed, not classified) |
| Compute | RTX 3070 Ti (8GB) — primary, for all development and most training |
| Compute fallback | Colab Pro ($9.99/mo, ~$20 across 2 months) — only if local VRAM bottlenecks on full-resolution or full-dataset runs |
| ML / DL / backend skills | Confirmed from prior discussion |
| ViT / Hybrid CNN-Transformer | New — budget extra learning time in Weeks 1-3 |
| Local storage | ~30-40GB free recommended (dataset + processed tensors + checkpoints + env) |
| OS for local dev | macOS — confirmed |
| Claude subscription or API access | Claude Pro — confirmed |

**Compute note:** 8GB VRAM is comfortable for pilot-subset training. Transformer
self-attention memory scales roughly with the square of patch count, so 512x512
inputs need more headroom than 256x256 would have — use mixed precision and
gradient checkpointing by default. Full-dataset training (Week 4) and RAG
retrieval-index building (Week 6) are the two points most likely to need the
Colab Pro fallback — premium GPU assignment there is priority access, not
guaranteed, so don't plan around reliably getting an A100.

## 6. Repository structure

```
rsvqa-project/
├── CLAUDE.md                  # project context for Claude Code (see Section 8)
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── model_config.yaml      # hyperparameters, paths — no magic numbers in code
├── data/
│   ├── raw/                   # untouched RSVQA-LR and RSVQA-HR downloads
│   └── processed/             # tokenized questions, answer vocab, split tensors
├── src/
│   ├── data_pipeline.py       # streaming/chunked loading — dataset doesn't fit casually in memory
│   ├── models/
│   │   ├── hybrid_encoder.py  # CNN stem + Transformer encoder
│   │   ├── vqa_head.py        # fusion + two heads: classification (config-driven vocab size) + count regression
│   │   └── clip_baseline.py   # zero-shot CLIP comparison
│   ├── rag/
│   │   ├── retrieval_index.py # embed + index training QA pairs for retrieval
│   │   └── explain_agent.py   # retrieval + LLM-generated natural-language explanation
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py             # attention rollout + Grad-CAM
├── notebooks/                 # exploration only — no pipeline logic lives here
├── experiments/                # saved checkpoints + run logs, gitignored
├── tests/
│   └── test_data_pipeline.py
└── reports/
    ├── figures/
    └── final_report.md        # include Problem/Motivation/Scope/Dataset Quality from this plan
```

Keep `src/` importable as a package (`__init__.py` in each folder) so scripts and
notebooks share the same code — no copy-pasted logic between the two.

## 7. Week-by-week plan

### Week 1 — Proof of concept on LR
- Download the full RSVQA-LR dataset (small, loads entirely in memory). Build
  the whole pipeline end-to-end on it — data loading, tokenization, a 4-class
  answer vocabulary (yes/no/rural/urban — confirmed empirically from the raw
  data; count questions are excluded from the vocab and regressed instead), a
  first pass of the Hybrid CNN-Transformer with its classification + count
  regression heads, training loop, basic evaluation — to prove the architecture
  and code are correct before HR's scale is added on top.
- Data inspection pass (sample count, class distribution, missing/corrupted
  files, duplicates) before writing any pipeline code — not after.
- Keep image size and vocabulary size as config parameters, not hardcoded, so
  the HR port is a config change, not a rewrite.
- Set up experiment logging (even simple structured logs) so results across
  the next 7 weeks are comparable.
- Deliverable: a fully working, tested LR pipeline with a baseline results table

### Week 2 — Port to HR, scalable pipeline
- Download RSVQA-HR, inspect tiles and QA pairs, including answer-class
  distribution — confirm the imbalance flagged in Section 3 rather than assume it.
- Build a streaming/chunked data loader designed for the full 1M+ pair scale
  from the start.
- Swap in the HR classification vocabulary (count questions still excluded and
  regressed, same as LR) and 512x512 image size via config.
- Implement the train/val/test split as a stratified-group split: group by
  image tile (mandatory — prevents the same tile's QA pairs leaking across
  splits) while approximately preserving per-class answer balance. Fit any
  preprocessing statistics (e.g. normalization) on the training split only,
  never the full dataset, per the course's leakage-prevention rule.
- Validate on a pilot subset of ~1,000-1,500 images (~100K QA pairs).
- Deliverable: `data_pipeline.py` fully tested on the HR pilot subset, proven
  able to stream the full dataset without loading it all into memory

### Week 3 — Hybrid CNN-Transformer development & tuning
- Iterate on the visual encoder: CNN stem depth, Transformer layer/head count,
  fusion strategy with the question embedding.
- Apply the course's regularization toolkit explicitly, not just whatever
  defaults a library provides: dropout, batch normalization, early stopping,
  and a scheduled learning rate — tuned on the pilot subset.
- Hyperparameter tuning (learning rate, batch size, augmentation) locally on
  the 3070 Ti (mixed precision, gradient checkpointing).
- Deliverable: best-performing architecture/config identified on the pilot subset

### Week 4 — Scale training to the full HR dataset
- Train the tuned architecture on the full 10,659-image, 1M+ QA pair dataset —
  local 3070 Ti first, Colab Pro fallback if you hit a memory or time wall.
- Deliverable: trained model on the full dataset, training curves, checkpoint

### Week 5 — VLM baseline + explainability
- Run CLIP zero-shot as the multimodal comparison baseline across the full test set.
- Implement attention rollout on the hybrid encoder and Grad-CAM-style saliency.
- Deliverable: full comparison table (your model vs. CLIP) across question
  types, XAI visualizations for representative examples

### Week 6 — Multimodal RAG + explanation layer
- Build a small retrieval index over training QA pairs/images (embedding-based).
- Use an LLM to combine the model's answer, the attended image region, and
  retrieved similar examples into a natural-language explanation — this is
  the project's Multimodal RAG / Agentic AI component, properly built rather
  than left as a stretch goal.
- Deliverable: working retrieval-augmented explanation demo

### Week 7 — Evaluation, ablations, error analysis
- Ablations: pilot-subset vs. full-dataset performance, with/without the RAG
  explanation layer, architecture variants from Week 3.
- Error analysis by question type and by answer-class frequency (does the
  model just default to the majority class on rare answers?) — a direct
  follow-up on the Balanced gap flagged in Section 3. For count questions,
  report regression metrics (MAE/RMSE) separately, plus accuracy-within-N
  (e.g. rounded prediction matches ground truth) to stay comparable with the
  paper's bucketed-classification numbers.
- Update the interactive UI demo to reflect the complete system (model + CLIP
  baseline + XAI + RAG explanation).
- Deliverable: ablation tables, error analysis writeup, updated UI demo

### Week 8 — Report, documentation, presentation prep
- Write `final_report.md` (open with Problem/Motivation/Scope from Sections
  1-2 and the Dataset Quality Checklist from Section 3 — including the
  Accurate/Well-Annotated/Balanced caveats stated plainly, not glossed over;
  include a one-line ethics statement: public imagery, no human subjects, no
  IRB required). Include methodology, results, ablations, and
  limitations/future work (e.g. full multi-agent orchestration, explicitly deferred).
- Polish repository documentation (README, CLAUDE.md).
- Prepare or update presentation materials for the final defense.
- Deliverable: final report, polished repo, presentation ready

## 8. Using Claude Code for this project

Claude Code is Anthropic's terminal-based coding agent — you point it at your project
directory and it reads, writes, and runs code with your approval.

### Install

Native installer (recommended — no dependencies, auto-updates):
```bash
# macOS / Linux / WSL
curl -fsSL https://claude.ai/install.sh | bash
```

**Authentication:** you already have Claude Pro, so `claude` will prompt you to log
in via browser the first time you run it — no separate API key needed.

### Set up the project

```bash
mkdir rsvqa-project && cd rsvqa-project
git init
claude
```

Inside the Claude Code session, ask it to scaffold the repo structure from Section 6 —
paste that tree directly into your first prompt.

### The CLAUDE.md file

Create a `CLAUDE.md` at the repo root. Claude Code reads this automatically at the
start of every session — put your project conventions here so you don't repeat
yourself each time:

```markdown
# Project context
Hybrid CNN-Transformer VQA on RSVQA-HR, with CLIP baseline, XAI comparison, and
a multimodal RAG explanation layer. See RSVQA_Project_Plan.md for the full plan,
including Problem/Motivation/Scope/Dataset Quality Checklist. Proof-of-concept
on RSVQA-LR first, then ported to RSVQA-HR and trained at full scale (10,659
images, 1M+ QA pairs).

# Conventions
- All hyperparameters in configs/model_config.yaml, never hardcoded
- Image size and answer-vocabulary size are config-driven, not hardcoded
  (this is what makes the LR-to-HR port a config change, not a rewrite)
- Count questions are a regression target (scalar output), not part of the
  classification vocab — the model has a shared backbone with a classification
  head (presence/comparison/rural_urban/area) and a separate count-regression head
- Type hints on all functions
- Every module in src/ gets a matching test in tests/
- Data pipeline must stream/chunk — never assume the full dataset fits in memory
- Splits are stratified-group: group by image tile (mandatory, prevents
  leakage), approximately balanced by answer class
- Any preprocessing statistic (normalization, etc.) is fit on the training
  split only — never on the full dataset before splitting
- Commit after each working milestone, not mid-edit
```

### Suggested workflow per week

1. Start each week's session by pointing Claude Code at the relevant plan section:
   *"Here's this week's goal from RSVQA_Project_Plan.md: [paste]. Let's start with
   the data pipeline module."*
2. Work in small, reviewable chunks — one module or function at a time, not
   "build the whole training loop" in one shot. Easier to catch mistakes early.
3. Ask it to write the test alongside the implementation, not after.
4. When something breaks, paste the full traceback rather than describing the
   error — Claude Code is much faster at root-causing from the actual stack trace.
5. Ask it to explain architecture decisions it makes (e.g. why a given CNN-stem +
   Transformer split, or a given memory-optimization choice) — useful both for
   your understanding and for your report.
6. Commit to git after each milestone (`git add -A && git commit -m "..."`) so you
   always have a working checkpoint to roll back to.
7. For long-running training jobs, run them directly (not through Claude Code's
   interactive loop) and bring results back for Claude Code to help you analyze.

### Where it fits vs. Colab

Use Claude Code locally for everything code-authoring: pipeline scripts, model
code, tests, explainability code, RAG components, and the report. Train locally
on the 3070 Ti by default. Only reach for Colab Pro for a specific full-scale
run that doesn't fit in 8GB — push code via git, pull it in Colab, run, and
bring results back. Keeps your repo as the single source of truth instead of
logic living only inside notebook cells.

# RSVQA Hybrid VQA

Explainable visual question answering on remote-sensing imagery: a hybrid
CNN-Transformer model benchmarked against zero-shot CLIP, with an
attention/saliency-based explainability (XAI) comparison and a lightweight
multimodal RAG explanation layer.

Proof-of-concept on **RSVQA-LR** (772 images, 77K QA pairs) first, then
ported to **RSVQA-HR** (10,659 images, 1M+ QA pairs) at full scale. See
[`RSVQA_Project_Plan.md`](RSVQA_Project_Plan.md) for the full project plan
(problem/motivation/scope, dataset quality checklist, week-by-week roadmap),
and [`CLAUDE.md`](CLAUDE.md) for the engineering conventions this repo
follows.

## Status

Week 1 (RSVQA-LR pipeline) is implemented and tested:

| Module | Status |
|---|---|
| `src/data_pipeline.py` | Done — raw loading, train-only vocab/norm-stat fitting, streaming `Dataset` |
| `src/models/hybrid_encoder.py` | Done — CNN stem + ViT-style Transformer encoder |
| `src/models/vqa_head.py` | Done — question encoder, fusion, classification + count-regression heads |
| `src/train.py` | Done — training loop, checkpointing, early stopping, JSON-lines logging |
| `src/evaluate.py` | Done — accuracy by question type, count MAE/RMSE, baseline results table |
| `src/models/clip_baseline.py`, `src/rag/`, `src/explain.py` | Not started (Weeks 5-6) |

Run `pytest tests/ -v` to verify (26 tests, all passing).

## Key design decisions

- **Count questions are a regression target**, not a classification bucket.
  `HybridVQAModel` has one shared backbone feeding two heads: a
  classification head (presence/comparison/rural_urban answers) and a
  scalar count-regression head. `vocab_size` in the config is the
  *classification* vocab only (4 classes for LR: `yes/no/rural/urban`) —
  count answers never enter that vocab.
- **Everything size-related is config-driven.** `image_size`, `vocab_size`,
  model dimensions, and training hyperparameters all live in
  `configs/model_config.yaml` — never hardcoded, so porting LR → HR is a
  config change, not a rewrite.
- **Train-only statistics.** The question vocabulary, answer vocabulary, and
  image normalization mean/std are all fit on the training split only, then
  reused (not recomputed) for val/test — prevents leakage.
- **Streaming-friendly from the start.** `RSVQADataset` loads images lazily
  in `__getitem__`, never eagerly into memory — same code path will handle
  HR's 10,659 images without a rewrite.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs a CPU build of torch/torchvision by default; see
the comment at the top of that file for the CUDA install command on a GPU
machine.

### Data

Download RSVQA-LR and place it at `data/raw/rsvqa_lr/` with this layout
(gitignored — not checked in):
```
data/raw/rsvqa_lr/
├── LR_split_{train,val,test}_{questions,answers,images}.json
├── all_questions.json, all_answers.json
└── images/Images_LR/*.tif
```

## Usage

**Run the test suite:**
```bash
pytest tests/ -v
```

**Train:**
```bash
python -m src.train
```
Reads `configs/model_config.yaml`, trains `HybridVQAModel` on RSVQA-LR,
writes checkpoints + a JSON-lines log to
`experiments/<experiment_name>_<timestamp>/`, and prints the best
checkpoint's path.

**Evaluate / produce a baseline results table:**
```bash
python -m src.evaluate experiments/<run_name>
```
Loads that run's `checkpoints/best.pt`, evaluates on the test split, and
writes a markdown table to `reports/baseline_results_<run_name>.md`.

### Training on a GPU (e.g. Google Colab)

The local dev machine is CPU/MPS-only, so a full-dataset run is slow there.
To train on a GPU instead:

1. Push this repo to GitHub, `git clone` it in a Colab notebook with a GPU
   runtime.
2. `data/raw/` is gitignored, so get the RSVQA-LR data there separately —
   e.g. zip it, upload to Google Drive, mount Drive in Colab, and unzip into
   `data/raw/rsvqa_lr`.
3. Bump `num_workers` and `batch_size` in `configs/model_config.yaml` to use
   the GPU's extra headroom, then run the training/evaluate commands above.
4. Copy `experiments/` and `reports/baseline_results_*.md` back out (e.g. to
   Drive, or download directly) before the Colab session ends — nothing
   there persists across sessions.

## Configuration

All hyperparameters live in `configs/model_config.yaml`:

| Key | Meaning |
|---|---|
| `image_size`, `max_question_len`, `batch_size`, `num_workers` | Data pipeline |
| `vocab_size` | Classification answer vocab size (excludes count) |
| `cnn_channels`, `d_model`, `num_transformer_layers`, `num_heads`, `ffn_dim`, `dropout`, `question_embed_dim`, `fusion_hidden_dim` | Model architecture |
| `count_loss_weight` | Weight on the count-regression loss term |
| `learning_rate`, `epochs`, `optimizer`, `early_stopping_patience` | Training |
| `experiments_dir`, `experiment_name` | Where runs are logged/checkpointed |

`question_vocab_size` is **not** a config key — it's a train-split artifact,
computed at runtime from `data/processed/vocab/question_vocab.json` and
passed into the model constructor explicitly (see `src/train.py:build_model`).

## Repository structure

```
src/
├── data_pipeline.py       # raw JSON/image loading -> training-ready batches
├── models/
│   ├── hybrid_encoder.py  # CNN stem + Transformer encoder
│   ├── vqa_head.py        # question encoder, fusion, dual heads, loss
│   └── clip_baseline.py   # zero-shot CLIP comparison (not started)
├── rag/                   # retrieval + LLM explanation layer (not started)
├── train.py
├── evaluate.py
└── explain.py             # attention rollout + Grad-CAM (not started)
tests/                     # one test module per src/ module
configs/model_config.yaml  # all hyperparameters
notebooks/                 # exploration only, no pipeline logic
experiments/                # checkpoints + run logs (gitignored)
reports/                   # data inspection notes, baseline results tables
```

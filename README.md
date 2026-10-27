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

**Week 1 (RSVQA-LR pipeline) is complete, trained, and evaluated.**
**Week 2 (RSVQA-HR port) groundwork is in place; the actual HR training run hasn't happened yet.**

| Module | Status |
|---|---|
| `src/data_pipeline.py` | Done — dataset-agnostic (LR and HR via config), train-only vocab/norm-stat fitting, streaming `Dataset`, pilot-subset sampling |
| `src/models/hybrid_encoder.py` | Done — CNN stem + ViT-style Transformer encoder |
| `src/models/vqa_head.py` | Done — question encoder, two fusion modes (concat / cross-attention), classification + count-regression heads |
| `src/train.py` | Done — training loop, checkpointing, early stopping, JSON-lines logging, config-path CLI arg |
| `src/evaluate.py` | Done — accuracy by question type, count MAE/RMSE, constant-baseline comparison, multi-split reports (HR has two test sets) |
| `configs/hr_config.yaml` | Drafted — several values (`images_subdir`, `image_ext`, `question_types`, `max_question_len`) still marked `VERIFY` pending a real HR download |
| `scripts/download_rsvqa_hr.py`, `scripts/inspect_rsvqa.py` | Done — resumable HR download, dataset inspection that suggests config values |
| `src/models/clip_baseline.py`, `src/rag/`, `src/explain.py` | Not started (Weeks 5-6) |

Run `pytest tests/ -v` to verify (66 tests, all passing — includes HR code paths tested against a small synthetic fake-HR dataset, so they don't require the real 14GB download).

The HR data was downloaded and inspected on 2026-10-22: 10,659 tiles (512×512 PNG), 4 splits with no
tile overlap, and question types presence / comp / area / count. Area answers turned out to be exact
values (`'1234m2'`), so they're grouped into 5 ranges (`area_buckets`); count stays regression. The
first HR pilot run (1,500 train tiles) is done; results are below.

The pilot took ~1 hour per epoch on a Colab T4 because data loading capped training at ~60
samples/s: every question re-decoded its tile's 512×512 PNG and normalized it on Colab's 2 CPU
cores. Two changes address that (see Key design decisions): tiles are decoded once into a cache
(`cache_images`), and images reach the GPU as uint8 to be normalized there. On synthetic HR-sized
tiles this took the loader from 269 to ~3,900 images/s on a laptop; the Colab speedup is still to be
measured.

**Next:** re-run the HR pilot with the faster loader to measure the new epoch time, then Week 3
tuning on the pilot subset.

### Week 1 baseline results (RSVQA-LR, 5 epochs, GPU)

See [`reports/baseline_results_week1_baseline_20261014-130407.md`](reports/baseline_results_week1_baseline_20261014-130407.md).

| Metric | Value |
|---|---|
| Classification accuracy (overall) | 73.2% |
| — comp | 65.0% |
| — rural_urban | 56.0% |
| — presence | 84.8% |
| Count MAE | 193.6 |
| Count RMSE | 732.5 |

**Known issue found here, since fixed:** the count-regression head barely
learned — `val_class_acc` climbed steadily across epochs (55.5% → 75.7%)
while `val_count_mae` stayed flat/noisy, despite count MSE dominating the
combined loss by several orders of magnitude over the classification
cross-entropy. Root cause was a loss-scaling problem (raw RSVQA counts run
into the thousands), not a capacity problem. Fixed via a config-driven
`count_loss_transform` (`log1p` by default — MSE on `log(1 + count)` instead
of the raw value; `none` reproduces this baseline for comparison). Also added
a **cross-attention fusion** mode (`fusion_type: cross_attention`, now the
default) where question words attend directly over image patch tokens,
replacing the original CLS-vector-concat approach (kept as `fusion_type:
concat` for the ablation).

### Week 2 LR re-run (log1p count loss + cross-attention, GPU)

Validation results from `week2_crossattn` (stopped early after 10 of 15 epochs; best checkpoint
at epoch 7). Test-set numbers not run yet.

| Metric (validation, best checkpoint) | Week 1 | Week 2 |
|---|---|---|
| Combined loss | ~750,000 | **1.52** |
| Classification accuracy | 74.2% | **78.3%** |
| Count MAE | 192.4 | **167.5** |
| Count MAE, always predicting the train median | — | 205.5 |

The loss is now on a sane scale and the count head beats the median baseline by ~18%, but count
predictions are still weak in absolute terms. LR counts are very skewed (median 6, max 16,640), so a
few huge-count tiles dominate MAE; HR counts are small (0–56), so this matters less there. Train loss
rose again after epoch 5 and val loss was noisy, consistent with a learning rate that's too high — the
HR config uses `3e-4` instead of `1e-3`.

### HR pilot results (1,500 train tiles, Colab T4)

See [`reports/baseline_results_hr_pilot_20261024-073555.md`](reports/baseline_results_hr_pilot_20261024-073555.md).
Early stopping ended training after 5 epochs; the best checkpoint is from epoch 2. Test sets are the
pilot's 300-tile subsets (~30,000 questions each), so treat these as early signals, not final results.

| Metric | test | test_phili (Philadelphia) | Majority / median baseline (train) |
|---|---|---|---|
| Classification accuracy (overall) | **82.7%** | 76.8% | ~63% |
| — presence | 89.3% | 81.7% | 59.7% (`no`) |
| — comp | 78.1% | 75.6% | 66.7% (`no`) |
| — area (5 ranges) | 81.4% | 70.8% | 60.7% (`0m2`) |
| Count MAE | **1.50** | 4.57 | 2.23 / 4.96 (median) |

- Every question type clearly beats always giving the most common answer; comp is the hardest.
- `test_phili`, from a city the model never saw, is ~6 points lower, and its count MAE barely beats
  the median baseline (4.57 vs 4.96): count generalizes poorly to the denser new city.
- The best checkpoint was chosen by combined validation loss, which picked epoch 2 although validation
  accuracy kept rising to 86.4% at epoch 4 (count error rose instead). How to pick the best checkpoint
  is a Week 3 question.

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
- **Each tile is decoded once, not once per question.** With `cache_images: true`
  (HR), `get_dataloaders` decodes each split's tiles into a memory-mapped
  `.npy` file under `<data_dir>/image_cache/` (~0.8 MB per HR tile on local disk,
  not RAM), reused across epochs and runs and rebuilt only when the tiles or
  `image_size` change. Every tile has ~100 questions, so this removes ~99% of
  decoding.
- **Images travel as uint8 and are normalized on the device.** Dataset items
  carry uint8 images; `prepare_batch` (used by training and evaluation) moves a
  batch to the GPU and normalizes it there with the train-split mean/std. This
  moves a quarter of the bytes between worker processes and takes the per-sample
  float work off the CPU.
- **Dataset layout is config-driven, not hardcoded to LR.** `split_file_prefix`,
  `images_subdir`, `image_ext`, `splits`, and `question_types` all live in
  config (`configs/model_config.yaml` for LR, `configs/hr_config.yaml` for
  HR) — the same `src/data_pipeline.py`/`src/evaluate.py` code serves both.
  `get_dataloaders` validates the configured `vocab_size` and each batch's
  `question_type`s against what the actual data contains, raising a clear
  error (not a silent mismatch) if they disagree.
- **A `data_dir`'s cached vocab/norm-stats are tagged with a manifest**
  (`raw_dir`, `split_file_prefix`, `images_subdir`, `image_ext`) recorded at
  build time — reusing a `data_dir` for a different raw dataset raises
  instead of silently serving stale artifacts built from the wrong data.
- **Cross-attention fusion** (default) has question words attend directly
  over image patch tokens, with retrievable per-word attention weights
  (`return_attention=True`) — the planned hook for Week 5's explainability
  work. The original concat+MLP fusion (CLS vector + mean-pooled question
  vector) is kept as `fusion_type: concat` for ablation.

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

**RSVQA-LR**: download it and place it at `data/raw/rsvqa_lr/` with this
layout (gitignored — not checked in):
```
data/raw/rsvqa_lr/
├── LR_split_{train,val,test}_{questions,answers,images}.json
├── all_questions.json, all_answers.json
└── images/Images_LR/*.tif
```

**RSVQA-HR**: `python scripts/download_rsvqa_hr.py data/raw/rsvqa_hr` (~14.4GB,
resumable, meant to be run on a box with real bandwidth — e.g. Colab). Then
`python scripts/inspect_rsvqa.py data/raw/rsvqa_hr USGS_split` to inspect the
actual file layout/answer vocabulary/question lengths and get a suggested
config block — use it to replace the `VERIFY`-tagged placeholders in
`configs/hr_config.yaml` before training on HR.

## Usage

**Run the test suite:**
```bash
pytest tests/ -v
```

**Train:**
```bash
python -m src.train                      # defaults to configs/model_config.yaml (LR)
python -m src.train configs/hr_config.yaml  # or train on HR instead
```
Trains `HybridVQAModel`, writes checkpoints + a JSON-lines log to
`experiments/<experiment_name>_<timestamp>/`, and prints the best
checkpoint's path.

**Evaluate / produce a results table:**
```bash
python -m src.evaluate experiments/<run_name>                      # LR
python -m src.evaluate experiments/<run_name> configs/hr_config.yaml  # HR
```
Loads that run's `checkpoints/best.pt` (model shape/hyperparameters come from
the checkpoint's own saved config, not the live YAML — so evaluating an older
checkpoint after editing the config can't silently mismatch), evaluates on
every split whose name starts with `test` (HR has two: `test` and the
Philadelphia `test_phili`), and writes a markdown report to
`reports/baseline_results_<run_name>.md`.

### Training on a GPU (e.g. Google Colab)

For HR, open [`notebooks/colab_hr_pilot.ipynb`](notebooks/colab_hr_pilot.ipynb) in Colab
(File → Open notebook → GitHub). It covers data setup from Drive, the area-range check, pilot
training with checkpoints written straight to Drive, and evaluation on both test sets. The steps
below describe the same process in general terms.

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

All hyperparameters live in `configs/model_config.yaml` (LR) /
`configs/hr_config.yaml` (HR):

| Key | Meaning |
|---|---|
| `image_size`, `max_question_len`, `batch_size`, `num_workers` | Data pipeline |
| `vocab_size` | Classification answer vocab size (excludes count) |
| `split_file_prefix`, `images_subdir`, `image_ext`, `splits` | Dataset file layout (differs between LR and HR) |
| `question_types` | The non-count question types to report classification accuracy for |
| `area_buckets` | HR only: range edges (m²) that exact area answers like `'1234m2'` are grouped into, making area a 5-class question; `null` for LR |
| `norm_stats_max_images` | Cap on train tiles sampled for image mean/std |
| `max_images_per_split`, `subset_seed` | Pilot-subset sampling (`null` for the full dataset) |
| `cache_images` | Decode each tile once into `<data_dir>/image_cache/` (on for HR, off for LR) |
| `cnn_channels`, `d_model`, `num_transformer_layers`, `num_heads`, `ffn_dim`, `dropout`, `question_embed_dim`, `fusion_hidden_dim` | Model architecture |
| `fusion_type` | `cross_attention` (default) or `concat` (Week 1 baseline, kept for ablation) |
| `count_loss_weight`, `count_loss_transform` | Count-regression loss weighting and scaling (`log1p` default, `none` for ablation) |
| `learning_rate`, `epochs`, `optimizer`, `early_stopping_patience` | Training |
| `experiments_dir`, `experiment_name` | Where runs are logged/checkpointed |

`question_vocab_size` is **not** a config key — it's a train-split artifact,
computed at runtime from `<data_dir>/vocab/question_vocab.json` and passed
into the model constructor explicitly (see `src/train.py:build_model`).

## Repository structure

```
src/
├── data_pipeline.py       # raw JSON/image loading -> training-ready batches (LR and HR)
├── models/
│   ├── hybrid_encoder.py  # CNN stem + Transformer encoder
│   ├── vqa_head.py        # question encoder, fusion (concat/cross-attention), dual heads, loss
│   └── clip_baseline.py   # zero-shot CLIP comparison (not started)
├── rag/                   # retrieval + LLM explanation layer (not started)
├── train.py
├── evaluate.py
└── explain.py             # attention rollout + Grad-CAM (not started)
scripts/
├── download_rsvqa_hr.py   # resumable HR download from Zenodo
└── inspect_rsvqa.py       # dataset inspection (LR or HR), suggests config values
tests/                     # one test module per src/ module, plus test_data_layout.py
                           # (HR code paths, tested against a tiny synthetic fake-HR dataset)
configs/
├── model_config.yaml      # LR hyperparameters
└── hr_config.yaml         # HR hyperparameters (some values VERIFY-tagged pending real data)
notebooks/                 # exploration + the Colab runner (colab_hr_pilot.ipynb); no pipeline logic
experiments/                # checkpoints + run logs (gitignored)
reports/                   # data inspection notes, baseline results tables
```

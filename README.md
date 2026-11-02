# RSVQA Hybrid VQA

Ask questions about satellite and aerial images, get answers. A hybrid CNN-Transformer built with PyTorch and trained on RSVQA-HR (10,659 images, 1M+ question-answer pairs).

**90.2% accuracy** on the test set, **0.91 count error** (median guess: 2.16), 80 tests passing.

## Results

Final model, complete test sets:

| Metric | Test (2,226 tiles) | New city, Philadelphia (1,056 tiles) |
|---|---|---|
| Accuracy (overall) | **90.2%** | 76.9% |
| Accuracy (presence) | 91.7% | 80.2% |
| Accuracy (comparison) | 90.9% | 79.6% |
| Accuracy (area) | 86.1% | 64.8% |
| Count MAE (median guess) | **0.91** (2.16) | 4.90 (5.25) |
| Count RMSE (median guess) | 2.34 (5.42) | 13.58 (14.10) |

1,500 vs 6,251 training tiles, complete test sets:

| Metric | Test, 1,500 | Test, 6,251 | Philadelphia, 1,500 | Philadelphia, 6,251 |
|---|---|---|---|---|
| Accuracy (overall) | 88.5% | 90.2% | 77.6% | 76.9% |
| Accuracy (presence) | 90.9% | 91.7% | 80.6% | 80.2% |
| Accuracy (comparison) | 88.4% | 90.9% | 79.8% | 79.6% |
| Accuracy (area) | 84.4% | 86.1% | 67.5% | 64.8% |
| Count MAE | 1.11 | 0.91 | 4.81 | 4.90 |
| Count RMSE | 2.86 | 2.34 | 13.29 | 13.58 |

## Status

| Module | Status |
|---|---|
| `src/data_pipeline.py` | Done |
| `src/models/hybrid_encoder.py` | Done |
| `src/models/vqa_head.py` | Done |
| `src/train.py` | Done |
| `src/evaluate.py` | Done |
| `configs/hr_config.yaml` | Done |
| `scripts/download_rsvqa_hr.py`, `scripts/inspect_rsvqa.py` | Done |
| `src/models/clip_baseline.py` (CLIP baseline) | Not started |
| `src/explain.py` (attention rollout, Grad-CAM) | Not started |
| `src/rag/` (RAG explanations) | Not started |

## Experiments

Final model, validation by epoch (6,251 training tiles):

| Epoch | Val accuracy | Val count MAE | Val score |
|---|---|---|---|
| 1 | 85.9% | 1.09 | 1.373 |
| 2 | 89.2% | 0.91 | 1.484 |
| 5 | 90.2% | 0.89 | 1.506 |
| 8 | 90.7% | 0.87 | 1.521 |
| 10 (best) | 90.7% | 0.87 | 1.522 |

Tuning, validation, 1,500 training tiles:

| Run | Peak LR | Epochs run | Best epoch | Best val accuracy | Val count MAE (best epoch) | Best val score |
|---|---|---|---|---|---|---|
| `hr_pilot_20261028-131854` (baseline) | 3e-4 | 10 | 10 | 89.7% | 1.04 | 1.443 |
| `tune_lr1e-3` | 1e-3 | 5 (early stop) | 2 | 81.6% (epoch 4) | 1.69 | 1.061 |
| `tune_lr1.5e-4` | 1.5e-4 | 10 | 10 | 89.2% | 1.02 | 1.447 |
| `tune_wide` | 1.5e-4 | 10 | 2 | 85.8% (epoch 3) | 1.47 | 1.208 |
| `tune_wide_clip` | 1.5e-4 | 10 | 2 | 85.6% (epoch 2) | 1.36 | 1.266 |
| `tune_lr1.5e-4_seed1` | 1.5e-4 | 10 | 7 | 89.2% | 1.02 | 1.450 |
| `tune_lr1.5e-4_seed2` | 1.5e-4 | 10 | 6 | 88.7% | 1.00 | 1.453 |

HR pilot run 1, 300-tile test subsets:

| Metric | Test | Philadelphia | Majority / median baseline (train) |
|---|---|---|---|
| Accuracy (overall) | **82.7%** | 76.8% | ~63% |
| Accuracy (presence) | 89.3% | 81.7% | 59.7% (`no`) |
| Accuracy (comparison) | 78.1% | 75.6% | 66.7% (`no`) |
| Accuracy (area) | 81.4% | 70.8% | 60.7% (`0m2`) |
| Count MAE | **1.50** | 4.57 | 2.23 / 4.96 (median) |

HR pilot runs 1 and 2, 300-tile test subsets:

| | Run 1 | Run 2 |
|---|---|---|
| Test accuracy (overall) | 82.7% | 87.0% |
| Test count MAE | 1.50 | 1.28 |
| Philadelphia accuracy (overall) | 76.8% | 83.3% |
| Philadelphia count MAE | 4.57 | 4.46 |

HR pilot runs 2 and 3, 300-tile test subsets:

| | Run 2 | Run 3 |
|---|---|---|
| Test accuracy (overall) | 87.0% | **88.5%** |
| Presence / comparison / area | 90.0 / 85.6 / 84.8% | 90.2 / 88.7 / 85.2% |
| Test count MAE | 1.28 | **1.14** |
| Philadelphia accuracy (overall) | **83.3%** | 81.3% |
| Philadelphia accuracy (area) | **76.1%** | 70.7% |
| Philadelphia count MAE | 4.46 | 4.41 |

RSVQA-LR proof of concept, week 1, test:

| Metric | Value |
|---|---|
| Accuracy (overall) | 73.2% |
| Accuracy (comparison) | 65.0% |
| Accuracy (rural/urban) | 56.0% |
| Accuracy (presence) | 84.8% |
| Count MAE | 193.6 |
| Count RMSE | 732.5 |

RSVQA-LR proof of concept, weeks 1 and 2, validation:

| Metric | Week 1 | Week 2 |
|---|---|---|
| Combined loss | ~750,000 | **1.52** |
| Accuracy | 74.2% | **78.3%** |
| Count MAE | 192.4 | **167.5** |
| Count MAE (median guess) | — | 205.5 |

Reports: [`reports/`](reports/)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_rsvqa_hr.py data/raw/rsvqa_hr
```

## Usage

```bash
pytest tests/ -v
python -m src.train configs/hr_config.yaml
python -m src.evaluate experiments/<run_name> configs/hr_config.yaml --full
```

Train on Colab: [`notebooks/colab_hr_pilot.ipynb`](notebooks/colab_hr_pilot.ipynb)

## Configuration

Settings live in `configs/hr_config.yaml` (HR) and `configs/model_config.yaml` (LR).

| Key | Meaning |
|---|---|
| `image_size`, `max_question_len`, `batch_size`, `num_workers` | Data pipeline |
| `vocab_size` | Classification answer vocab size (excludes count) |
| `split_file_prefix`, `images_subdir`, `image_ext`, `splits` | Dataset file layout |
| `question_types` | Non-count question types to report |
| `area_buckets` | HR area answer ranges (m²) |
| `norm_stats_max_images` | Train tiles sampled for image mean/std |
| `max_images_per_split`, `subset_seed` | Pilot-subset sampling |
| `cache_images` | Decode each tile once into a cache |
| `cnn_channels`, `d_model`, `num_transformer_layers`, `num_heads`, `ffn_dim`, `dropout`, `question_embed_dim`, `fusion_hidden_dim` | Model architecture |
| `fusion_type` | `cross_attention` or `concat` |
| `count_loss_weight`, `count_loss_transform` | Count loss weight and scaling |
| `learning_rate`, `weight_decay`, `epochs`, `optimizer`, `early_stopping_patience` | Training |
| `lr_schedule`, `warmup_fraction` | `cosine` or `constant` |
| `grad_clip_norm` | Optional gradient clipping |
| `seed` | Weight init, shuffle order, dropout |
| `mixed_precision` | float16 on CUDA |
| `selection_metric` | `score` or `val_loss` |
| `experiments_dir`, `experiment_name` | Run output location |

## Repository structure

```
src/
├── data_pipeline.py
├── models/
│   ├── hybrid_encoder.py
│   ├── vqa_head.py
│   └── clip_baseline.py
├── rag/
├── train.py
├── evaluate.py
└── explain.py
scripts/
tests/
configs/
notebooks/
experiments/
reports/
```

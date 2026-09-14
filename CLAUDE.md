# Project context
Hybrid CNN-Transformer VQA on RSVQA-HR, with CLIP baseline, XAI comparison, and
a multimodal RAG explanation layer. See RSVQA_Project_Plan.md for the full plan,
including Problem/Motivation/Scope/Dataset Quality Checklist. Proof-of-concept
on RSVQA-LR first, then ported to RSVQA-HR and trained at full scale (10,659
images, 1M+ QA pairs).

# Conventions
- All hyperparameters in configs/model_config.yaml, never hardcoded
- Image size and answer-vocabulary size are config-driven, not hardcoded
- Type hints on all functions
- Every module in src/ gets a matching test in tests/
- Data pipeline must stream/chunk — never assume the full dataset fits in memory
- Splits are stratified-group: group by image tile (mandatory, prevents
  leakage), approximately balanced by answer class
- Any preprocessing statistic (normalization, etc.) is fit on the training
  split only — never on the full dataset before splitting
- Commit after each working milestone, not mid-edit
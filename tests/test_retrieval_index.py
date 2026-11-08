import math
from pathlib import Path

import numpy as np
import pytest
import torch

import src.rag.retrieval_index as ri
from src.data_pipeline import get_dataloaders, prepare_batch, vqa_collate_fn
from src.rag.retrieval_index import (
    QueryResults,
    RetrievalIndex,
    agreement,
    auroc,
    bootstrap_difference,
    build_index,
    compute_features,
    empty_features,
    error_warning,
    neighbour_vote,
    passes,
    run_retrieval,
)
from src.train import build_model
from tests.test_data_layout import hr_config, write_fake_hr


def rag_config(tmp_path: Path) -> dict:
    config = hr_config(tmp_path)
    config.update(
        cnn_channels=[8, 16], d_model=32, num_transformer_layers=2, num_heads=2, ffn_dim=64, dropout=0.1,
        question_embed_dim=32, fusion_hidden_dim=32, fusion_type="cross_attention", mixed_precision=False,
        rag_k=3, rag_seed=0, rag_tile_batch_size=3, rag_question_batch_size=5, rag_search_chunk_rows=4,
        rag_bootstrap_samples=50,
    )
    return config


@pytest.fixture
def trained_run(tmp_path: Path) -> tuple[Path, dict]:
    write_fake_hr(tmp_path / "raw")
    config = rag_config(tmp_path)
    get_dataloaders(config)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    torch.manual_seed(0)
    torch.save({"model_state_dict": build_model(config).state_dict(), "config": config},
               run_dir / "checkpoints" / "best.pt")
    return run_dir, config


def test_features_match_the_full_forward_pass(trained_run):
    run_dir, config = trained_run
    dataset = get_dataloaders(config)["test"].dataset
    torch.manual_seed(0)
    model = build_model(config).eval()
    rows = list(range(len(dataset)))[::-1]  # any order: position j must hold record rows[j]
    out = empty_features(len(rows), config["fusion_hidden_dim"])
    compute_features(model, dataset, rows, out, config, torch.device("cpu"), "test")

    batch = prepare_batch(vqa_collate_fn([dataset[r] for r in rows]), None, dataset)
    with torch.no_grad():
        encoded = model.encoder(batch["image"])
        fused, _ = model.fuse(encoded, batch["question_tokens"], batch["question_len"])
        outputs = model(batch)
    expected = torch.nn.functional.normalize(fused, dim=-1).numpy()
    assert np.allclose(out.vectors.astype(np.float32), expected, atol=2e-3)
    is_count = ~batch["count_target"].isnan().numpy()
    assert np.array_equal(out.pred_class[~is_count], outputs["class_logits"].argmax(-1).numpy()[~is_count])
    assert (out.pred_class[is_count] == -1).all()
    assert np.allclose(out.pred_count[is_count], outputs["count_pred"].numpy()[is_count], atol=1e-4)
    assert np.isnan(out.pred_count[~is_count]).all()


def test_index_is_reused_only_when_records_and_checkpoint_match(trained_run, tmp_path, monkeypatch):
    run_dir, config = trained_run
    dataset = get_dataloaders(config)["train"].dataset
    model = build_model(config).eval()
    calls = []
    real = ri.compute_features
    monkeypatch.setattr(ri, "compute_features", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    first = build_index(model, dataset, tmp_path / "index", config, torch.device("cpu"), "ckpt-a")
    again = build_index(model, dataset, tmp_path / "index", config, torch.device("cpu"), "ckpt-a")
    assert len(calls) == 1
    assert np.array_equal(np.asarray(first.vectors), np.asarray(again.vectors))
    assert first.vectors.shape == (len(dataset), config["fusion_hidden_dim"])
    build_index(model, dataset, tmp_path / "index", config, torch.device("cpu"), "ckpt-b")
    assert len(calls) == 2


def unit(*rows: list[float]) -> np.ndarray:
    vectors = np.array(rows, dtype=np.float32)
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float16)


def test_search_stays_within_the_type_and_orders_by_similarity():
    vectors = unit([1, 0], [0.9, 0.1], [0, 1], [0.6, 0.4], [1, 0.05], [0.5, 0.5])
    types = ["presence", "presence", "presence", "count", "count", "presence"]
    index = RetrievalIndex(vectors, types, chunk_rows=2)  # chunks smaller than the type
    sims, rows = index.search(np.array([[1.0, 0.0]]), "presence", k=3)
    assert rows.tolist() == [[0, 1, 5]]
    assert np.all(np.diff(sims[0]) <= 0)
    assert math.isclose(sims[0, 0], 1.0, abs_tol=1e-3)
    _, count_rows = index.search(np.array([[1.0, 0.0]]), "count", k=5)  # only 2 count rows
    assert count_rows.tolist() == [[4, 3]]


def test_search_in_chunks_matches_brute_force():
    rng = np.random.default_rng(0)
    vectors = unit(*rng.normal(size=(50, 8)).tolist())
    queries = rng.normal(size=(7, 8)).astype(np.float32)
    _, chunked = RetrievalIndex(vectors, ["area"] * 50, chunk_rows=6).search(queries, "area", k=4)
    _, whole = RetrievalIndex(vectors, ["area"] * 50, chunk_rows=1000).search(queries, "area", k=4)
    brute = np.argsort(-(queries @ vectors.astype(np.float32).T), axis=1)[:, :4]
    assert np.array_equal(chunked, whole) and np.array_equal(chunked, brute)


def test_random_rows_are_distinct_same_type_and_seeded():
    index = RetrievalIndex(unit(*[[1, i] for i in range(10)]), ["comp", "area"] * 5, chunk_rows=4)
    a = index.random_rows("comp", 20, 3, np.random.default_rng(1))
    b = index.random_rows("comp", 20, 3, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert set(a.ravel()) <= {0, 2, 4, 6, 8}
    assert all(len(set(row)) == 3 for row in a)


def test_agreement_per_head():
    assert agreement(1.0, np.array([1.0, 0.0, 1.0, 1.0]), is_count=False) == 0.75
    # count: neighbours' median 3 against a prediction of 7
    assert math.isclose(agreement(7.0, np.array([1.0, 3.0, 9.0]), is_count=True), -abs(math.log1p(7) - math.log1p(3)))
    assert agreement(3.0, np.array([2.0, 3.0, 4.0]), is_count=True) == 0.0


def test_neighbour_vote_takes_the_majority_and_breaks_ties_by_similarity():
    assert neighbour_vote(np.array([0.0, 1.0, 1.0]), np.array([0.9, 0.5, 0.5]), False) == 1.0
    assert neighbour_vote(np.array([0.0, 1.0]), np.array([0.6, 0.9]), False) == 1.0
    assert neighbour_vote(np.array([2.0, 10.0, 4.0]), np.ones(3), True) == 4.0


def test_auroc_matches_hand_computed_values():
    positives = np.array([1, 1, 0, 0])
    assert auroc(np.array([4.0, 3.0, 2.0, 1.0]), positives) == 1.0
    assert auroc(np.array([1.0, 2.0, 3.0, 4.0]), positives) == 0.0
    assert auroc(np.ones(4), positives) == 0.5
    # pairs (pos, neg): (3,1) win, (3,4) loss, (2,1) win, (2,4) loss -> 0.5; plus a tie case
    assert auroc(np.array([3.0, 2.0, 1.0, 4.0]), positives) == 0.5
    assert auroc(np.array([2.0, 1.0, 1.0]), np.array([1, 1, 0])) == 0.75
    assert math.isnan(auroc(np.array([1.0, 2.0]), np.array([0, 0])))


def test_bootstrap_difference_is_zero_for_equal_scores_and_positive_for_a_better_one():
    rng = np.random.default_rng(0)
    positives = rng.random(200) < 0.3
    noise = rng.random(200)
    same = bootstrap_difference(noise, noise, positives, 100, 0)
    assert same == {"difference": 0.0, "lower": 0.0, "upper": 0.0}
    better = bootstrap_difference(positives + 0.1 * noise, noise, positives, 100, 0)
    assert better["difference"] > 0.3 and better["lower"] > 0


def results(correct: list[bool], agree: list[float], random_agree: list[float]) -> QueryResults:
    r = QueryResults(*([] for _ in QueryResults.__dataclass_fields__))
    r.correct, r.agreement, r.random_agreement = correct, agree, random_agree
    return r


def test_a_type_passes_only_when_retrieval_beats_random():
    correct = [True] * 30 + [False] * 30
    informative = [1.0] * 30 + [0.0] * 30
    flat = [0.5] * 60
    assert passes(error_warning(results(correct, informative, flat), 100, 0))
    assert not passes(error_warning(results(correct, flat, flat), 100, 0))
    assert not passes(error_warning(results([True] * 60, informative, flat), 100, 0))  # no errors: nan


def test_run_retrieval_writes_the_report_and_reuses_the_index(trained_run, monkeypatch):
    run_dir, config = trained_run
    report = run_retrieval(run_dir, config)
    assert "**Passes on:" in report
    assert "## Test splits pooled (this decides)" in report
    assert "## test alone (does not decide)" in report and "## test_phili alone (does not decide)" in report
    for qtype in ("presence", "comp", "area", "count"):
        assert f"| {qtype} |" in report
    assert "Count MAE: model" in report
    assert "| Type | Same question text | Mean similarity, neighbours | Mean similarity, random |" in report
    assert (run_dir / "retrieval" / "manifest.json").exists()

    monkeypatch.setattr(ri, "compute_features", lambda *a, **k: pytest.fail("index rebuilt"))
    monkeypatch.setattr(ri, "score_queries", lambda *a, **k: {})
    run_retrieval(run_dir, config)


def test_pilot_uses_its_own_index_folder(trained_run):
    run_dir, config = trained_run
    report = run_retrieval(run_dir, config, pilot=True)
    assert "(pilot subsets)" in report
    assert (run_dir / "retrieval_pilot" / "manifest.json").exists()


def test_neighbour_descriptions_same_question_and_similarity_against_random(trained_run, tmp_path):
    run_dir, config = trained_run
    loaders = get_dataloaders(config)
    train, test = loaders["train"].dataset, loaders["test"].dataset
    model = build_model(config).eval()
    features = build_index(model, train, tmp_path / "index", config, torch.device("cpu"), "ckpt")
    index = RetrievalIndex(features.vectors, [r["type"] for r in train.qa_records], config["rag_search_chunk_rows"])
    questions = np.array([r["question"] for r in train.qa_records], dtype=object)
    questions[0] = "a different question"  # one train question no longer matches its type's text
    results = ri.score_queries(model, test, list(range(len(test))), index, np.zeros(len(questions)),
                               np.ones(len(questions), dtype=bool), questions, config, torch.device("cpu"),
                               np.random.default_rng(0), "test")
    k = config["rag_k"]
    for qtype, r in results.items():
        # the top k are the k most similar rows, so no k random rows of the type can beat their mean
        assert all(n >= m - 1e-3 for n, m in zip(r.mean_similarity, r.random_similarity))
        expected = {1.0, 1 - 1 / k} if qtype == train.qa_records[0]["type"] else {1.0}
        assert set(np.round(r.same_question, 6)) <= {round(v, 6) for v in expected}

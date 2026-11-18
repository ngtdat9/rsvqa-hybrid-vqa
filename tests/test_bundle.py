import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data_pipeline import AnswerVocab
from src.demo.bundle import TrainTable, choose_demo_tiles, export_bundle
from src.rag.explain_agent import PoolResult
from tests.test_retrieval_index import rag_config, trained_run  # noqa: F401  (fixture)

HR_CONFIG = yaml.safe_load(Path("configs/hr_config.yaml").read_text())


class Echo:
    def complete(self, messages):
        return "Most of the similar cases have this answer as their true answer.", 10


def demo_settings(base: dict) -> dict:
    """The tiny test run's config plus the real RAG and demo settings, sized for the fake data."""
    extra = {k: v for k, v in HR_CONFIG.items() if k.startswith(("rag_", "demo_", "xai_nodata")) and k not in base}
    return {**base, **extra, "rag_k": 3, "rag_search_chunk_rows": 4, "rag_tile_batch_size": 3,
            "rag_question_batch_size": 5, "rag_llm_min_interval_s": 0, "demo_thumb_size": 8,
            "demo_warned_per_type": 1, "demo_wrong_tiles": 1, "demo_right_tiles": 1}


@pytest.fixture
def exported(trained_run, tmp_path: Path) -> tuple[Path, Path, dict, str]:
    run_dir, base = trained_run
    settings = demo_settings(base)
    bundle = tmp_path / "bundle"
    summary = export_bundle(run_dir, settings, run_dir / "retrieval", bundle, Echo())
    return bundle, run_dir / "retrieval", settings, summary


RECORDS = [
    {"img_id": 5, "question": "Is there a road?", "type": "presence", "answer": "yes"},
    {"img_id": 5, "question": "How many buildings are there?", "type": "count", "answer": "12"},
    {"img_id": 9, "question": "Is there a road?", "type": "presence", "answer": "no"},
    {"img_id": 9, "question": "What is the area covered by grass?", "type": "area", "answer": "11-100m2"},
]


def test_train_table_round_trips_the_records(tmp_path: Path):
    table = TrainTable.from_records(RECORDS)
    assert list(table.qa_records) == RECORDS and table.qa_records[3] == RECORDS[3] and len(table.qa_records) == 4
    assert table.type_list() == ["presence", "count", "presence", "area"]
    table.save(tmp_path)
    loaded = TrainTable.load(tmp_path)
    assert list(loaded.qa_records) == RECORDS
    assert loaded.questions == ["Is there a road?", "How many buildings are there?", "What is the area covered by grass?"]


def test_train_table_truths_match_the_rag_code():
    vocab = AnswerVocab.build(RECORDS)
    truths = TrainTable.from_records(RECORDS).truths(vocab)
    assert truths.tolist() == [vocab.encode("yes"), 12.0, vocab.encode("no"), vocab.encode("11-100m2")]


def test_demo_tiles_follow_the_rule_in_pool_order():
    # rows 0-7 on tiles 10-17; warned presence on 11 and 12, warned count on 13, wrong on 14 and 15, right elsewhere
    pool = PoolResult(rows=list(range(8)), types=["presence", "presence", "presence", "count", "count", "count",
                                                  "presence", "count"],
                      correct=[True, False, True, False, False, False, True, True],
                      warned=[False, True, True, True, False, False, False, False])
    tiles = choose_demo_tiles(pool, {row: 10 + row for row in range(8)}, warned_per_type=1, wrong=1, right=2)
    # first warned per type (11 presence, 13 count), first wrong unwarned (14), first two right not taken (10, 12)
    assert tiles == [11, 13, 14, 10, 12]
    assert choose_demo_tiles(pool, {row: 10 + row for row in range(8)}, 5, 0, 0) == [11, 12, 13]
    assert choose_demo_tiles(pool, {row: 10 + row for row in range(8)}, 0, 9, 0) == [14, 15]  # warned tiles excluded


def test_export_writes_a_complete_bundle(exported):
    bundle, index_dir, settings, summary = exported
    for name in ("checkpoint.pt", "config.json", "demo_records.json", "train_table.npz", "train_strings.json",
                 "cache.json", "manifest.json", "vocab/question_vocab.json", "vocab/image_norm_stats.json"):
        assert (bundle / name).exists(), name
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["index"] == json.loads((index_dir / "manifest.json").read_text())
    records = json.loads((bundle / "demo_records.json").read_text())
    tiles = {r["img_id"] for r in records}
    assert {r["split"] for r in records} == {"test", "test_phili"} and manifest["tiles"] == len(tiles)
    assert all((bundle / "tiles" / f"{t}.png").exists() for t in tiles)
    cache = json.loads((bundle / "cache.json").read_text())
    assert cache and all(item["img_id"] in tiles for item in cache)
    sources = [item["explanation"]["source"] for item in cache]
    assert "llm" in sources and set(sources) <= {"llm", "template"}  # warned ones fail the fake LLM's text
    assert all(len(item["occlusion"]) == len(cache[0]["occlusion"]) for item in cache)
    table = TrainTable.load(bundle)
    assert len(table.img_id) == manifest["index"]["rows"]
    assert len(list((bundle / "thumbs").glob("*.jpg"))) == len(set(table.img_id.tolist())) == manifest["thumbnails"]
    assert "| test | " in summary and "| test_phili | " in summary
    neighbour_tiles = {n["img_id"] for item in cache for n in item["facts"]["neighbours"]}
    assert neighbour_tiles <= set(table.img_id.tolist())  # every neighbour card has a thumbnail

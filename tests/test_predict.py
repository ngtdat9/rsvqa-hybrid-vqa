import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.demo.predict import Demo, heatmap_overlay, split_parts
from tests.test_bundle import Echo, demo_settings, exported  # noqa: F401  (fixture)
from tests.test_retrieval_index import rag_config, trained_run  # noqa: F401  (fixture)


def test_split_parts_labels_who_wrote_each_part():
    first, middle, fixed = "The model answered \"yes\".", "Four of five agree.", "Changing the cells changed it."
    assert split_parts(f"{first} {middle} {fixed}", "llm", first, fixed) == [
        ("code", first), ("llm", middle), ("code", fixed)]
    assert split_parts(f"{first} {middle} {fixed}", "template", first, fixed)[1] == ("template", middle)
    assert split_parts(f"{first} {fixed}", "code", first, fixed) == [("code", first), ("code", fixed)]


def test_heatmap_overlay_colours_the_strongest_cell_brightest():
    tile = Image.new("RGB", (64, 64), (100, 100, 100))
    scores = np.zeros(16)
    scores[5] = 2.0  # row 1, column 1 of a 4x4 grid
    out = np.asarray(heatmap_overlay(tile, scores, alpha=0.6))
    assert out.shape == (64, 64, 3)
    hot, cold = out[16:32, 16:32].mean(), out[:16, :16].mean()
    assert hot > cold
    flat = np.asarray(heatmap_overlay(tile, np.zeros(16), alpha=0.6))  # an all-zero map must not divide by 0
    assert np.isfinite(flat).all()


@pytest.fixture
def demo(exported) -> Demo:
    bundle, index_dir, settings, _ = exported
    return Demo(bundle, index_dir, settings, device=torch.device("cpu"), use_llm=False)


def test_demo_lists_tiles_and_questions_showcase_first(demo):
    tiles = demo.tiles()
    assert tiles and {t["split"] for t in tiles} == {"test", "test_phili"}
    assert all(t["path"].exists() and isinstance(t["strip"], bool) for t in tiles)
    questions = demo.questions(tiles[0]["img_id"])
    assert questions[0]["showcase"]
    assert [q["showcase"] for q in questions] == sorted((q["showcase"] for q in questions), reverse=True)


def test_showcase_answers_come_from_the_cache(demo):
    img_id = demo.tiles()[0]["img_id"]
    q = demo.questions(img_id)[0]
    answer = demo.ask(q["row"])
    item = demo.cache[(img_id, q["question"])]
    assert answer.precomputed and answer.text == item["explanation"]["text"]
    assert answer.parts[1][0] == item["explanation"]["source"] and " ".join(t for _, t in answer.parts) == answer.text
    assert len(answer.neighbours) == len(item["facts"]["neighbours"]) and answer.neighbours[0].thumb.exists()


def test_live_answers_match_the_precomputed_ones(demo):
    # answer every showcase question live and compare with what the export computed
    for (img_id, question), item in list(demo.cache.items()):
        row = next(q["row"] for q in demo.questions(img_id) if q["question"] == question)
        del demo.cache[(img_id, question)]
        live = demo.ask(row)
        assert not live.precomputed and set(live.seconds) == {"answer and search", "occlusion map", "explanation"}
        facts = item["facts"]
        assert (live.answer, live.warning, live.strip, live.truth) == (
            facts["answer"], facts["warning"], facts["strip"], facts["truth"])
        assert [n.img_id for n in live.neighbours] == [n["img_id"] for n in facts["neighbours"]]
        assert np.allclose(live.occlusion, item["occlusion"], atol=1e-4)
        assert live.parts[1][0] == "template"  # no LLM in the app here: the template paragraph


def test_demo_refuses_an_index_from_another_bundle(exported, tmp_path):
    bundle, index_dir, settings, _ = exported
    other = tmp_path / "other_index"
    other.mkdir()
    manifest = json.loads((index_dir / "manifest.json").read_text())
    (other / "manifest.json").write_text(json.dumps({**manifest, "checkpoint": "something else"}))
    with pytest.raises(ValueError, match="not the index"):
        Demo(bundle, other, settings, device=torch.device("cpu"), use_llm=False)

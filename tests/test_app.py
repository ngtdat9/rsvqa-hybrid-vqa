from pathlib import Path

import numpy as np
import pytest
import torch

gr = pytest.importorskip("gradio")  # demo-only dependency (requirements-demo.txt); absent on Colab

from src.demo.app import answer_html, build_ui, explanation_html, neighbour_items, question_choices, tile_items, truth_markdown
from src.demo.predict import Demo, DemoAnswer, NeighbourCard
from tests.test_bundle import Echo, demo_settings, exported  # noqa: F401  (fixture)
from tests.test_retrieval_index import rag_config, trained_run  # noqa: F401  (fixture)


def make_answer(**overrides) -> DemoAnswer:
    values = dict(split="test_phili", img_id=7, question="How many buildings are there?", qtype="count", answer="1",
                  confidence=None, truth="0", correct=False, warning=True, strip=True,
                  text='The model answered "1". Similar cases <all 0>. Changing the cells changed it.',
                  parts=[("code", 'The model answered "1".'), ("llm", "Similar cases <all 0>."),
                         ("code", "Changing the cells changed it.")],
                  occlusion=np.zeros(16), neighbours=[NeighbourCard(1, 3, "How many buildings are there?", "0",
                                                                    Path("/x/3.jpg")),
                                                     NeighbourCard(2, 4, "q", "0", None)],
                  precomputed=False, seconds={"occlusion map": 4.5})
    values.update(overrides)
    return DemoAnswer(**values)


def test_answer_card_shows_the_warning_and_how_it_was_computed():
    card = answer_html(make_answer())
    assert "1 object" in card and "Uncertain" in card and "no confidence for counts" in card
    assert "computed live on this laptop in occlusion map 4.5 s" in card and "No warning does not mean" in card
    quiet = answer_html(make_answer(qtype="presence", answer="yes", confidence=0.93, warning=False, precomputed=True))
    assert "No warning" in quiet and "confidence 93%" in quiet and "precomputed on the GPU" in quiet
    assert "objects" not in quiet


def test_truth_is_separate_from_the_answer_card():
    answer = make_answer()
    assert "0" not in answer_html(answer).split("answer-big")[1].split("</span>")[0]
    assert truth_markdown(answer) == "**True answer: 0** · the model was **wrong** · the tile has a black no-data strip"


def test_explanation_marks_who_wrote_what_only_when_asked():
    plain = explanation_html(make_answer(), False)
    shown = explanation_html(make_answer(), True)
    assert 'class="expl"' in plain and "legend" not in plain
    assert 'class="expl show"' in shown and "written by the LLM, checked by code" in shown
    assert "&lt;all 0&gt;" in plain  # text is escaped


def test_neighbour_items_skip_missing_thumbnails():
    assert neighbour_items(make_answer()) == [("/x/3.jpg", "#1 · true answer 0 · How many buildings are there?")]


@pytest.fixture
def demo(exported) -> Demo:
    bundle, index_dir, settings, _ = exported
    return Demo(bundle, index_dir, settings, device=torch.device("cpu"), use_llm=False)


def test_ui_builds_and_lists_the_bundle(demo):
    items = tile_items(demo, "test")
    assert items and all(Path(path).exists() for path, _ in items)
    img_id = demo.tiles()[0]["img_id"]
    choices = question_choices(demo, img_id)
    assert choices[0][0].startswith("★ ") and {row for _, row in choices} == {q["row"] for q in demo.questions(img_id)}
    assert isinstance(build_ui(demo), gr.Blocks)

import csv
import io
import json
import urllib.error
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import src.rag.explain_agent as ea
from src.rag.explain_agent import (
    Facts,
    Neighbour,
    OpenAICompatibleClient,
    cell_zone,
    check_explanation,
    explain,
    facts_text,
    first_sentence,
    rating_rows,
    PoolResult,
    run_generate,
    score_ratings,
    fixed_sentences,
    neighbour_template,
    top_zones,
    warning,
    wilson_interval,
    write_rating_sheet,
)
from tests.test_retrieval_index import rag_config, trained_run  # noqa: F401  (fixture)

HR_CONFIG = yaml.safe_load(Path("configs/hr_config.yaml").read_text())
LLM_KEYS = {k: v for k, v in HR_CONFIG.items() if k.startswith("rag_") or k.startswith("xai_nodata")}


def config() -> dict:
    return dict(LLM_KEYS)


def make_facts(**overrides) -> Facts:
    values = dict(
        split="test", row=7, img_id=3, question="Is there a road?", qtype="presence", answer="yes", confidence=0.93,
        zones=["top left", "centre"], neighbours=[Neighbour(i, 100 + i, "Is there a road?", "yes") for i in range(1, 6)],
        agree=4, count_range=None, warning=False, strip=False, truth="yes", correct=True,
    )
    values.update(overrides)
    return Facts(**values)


def count_facts(**overrides) -> Facts:
    values = dict(qtype="count", question="How many buildings are there?", answer="7", confidence=None, agree=0,
                  count_range=(2.0, 9.0), truth="6", correct=False,
                  neighbours=[Neighbour(i, 100 + i, "How many buildings are there?", str(c)) for i, c in
                              zip(range(1, 6), (2, 4, 5, 9, 3))])
    values.update(overrides)
    return make_facts(**values)


def test_zones_cover_the_grid_in_thirds():
    assert cell_zone(0, 16) == "top left"
    assert cell_zone(15, 16) == "top right"
    assert cell_zone(8 * 16 + 8, 16) == "centre"
    assert cell_zone(255, 16) == "bottom right"
    assert cell_zone(15 * 16, 16) == "bottom left"


def test_top_zones_are_strongest_first_without_repeats():
    scores = torch.zeros(16 * 16)
    scores[255], scores[0], scores[1] = 3.0, 2.0, 1.0  # two of the top three are top left
    assert top_zones(scores, 16, 3) == ["bottom right", "top left"]


def test_warning_rules():
    truths = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    assert warning(1.0, truths, False, 3)       # 2 of 5 agree
    assert not warning(0.0, truths, False, 3)   # 3 of 5 agree
    counts = np.array([2.0, 4.0, 9.0])
    assert not warning(8.6, counts, True, 3)    # rounds to 9, inside 2-9
    assert warning(9.6, counts, True, 3)        # rounds to 10
    assert warning(1.4, counts, True, 3)        # rounds to 1


def test_facts_text_hides_the_truth_and_leaves_the_cells_and_strip_to_code():
    facts = make_facts(truth="no", correct=False, strip=True)
    text = facts_text(facts)
    assert '#1 "Is there a road?" true answer: yes' in text and "4 of 5" in text
    assert "correct" not in text.lower() and 'answer: "no"' not in text
    assert "top left" not in text and "strip" not in text
    # a "no" line is left out: the pilot LLM echoed it back as filler
    assert "Uncertain" not in text
    assert "Uncertain: yes, because only 1 of the 5 similar cases" in facts_text(make_facts(warning=True, agree=1))
    assert "outside the similar cases' range" in facts_text(count_facts(warning=True))
    assert "range from 2 to 9" in facts_text(count_facts())
    assert "true counts: all 0" in facts_text(count_facts(count_range=(0.0, 0.0)))
    assert "all had a true count of 0" in neighbour_template(count_facts(count_range=(0.0, 0.0)))
    assert first_sentence(make_facts()) == 'The model answered "yes" with 93% confidence.'
    assert first_sentence(count_facts()) == 'The model answered "7".'


@pytest.mark.parametrize("facts", [
    make_facts(), make_facts(warning=True, agree=1), make_facts(strip=True), count_facts(), count_facts(warning=True),
    make_facts(answer="11 to 100 square meters", qtype="area", strip=True, warning=True, agree=2),
])
def test_the_neighbour_template_passes_every_check(facts):
    assert check_explanation(neighbour_template(facts), facts, config()) == []


@pytest.mark.parametrize("body, reason", [
    ("Similar cases #1 to #5 mostly agree, with 7 of them.", "numbers not in the facts: 7"),
    ("Case #6 agrees.", "cites a neighbour"),
    ("The grid cells at the top left highlight the road.", "banned phrasing"),
    ("The model saw a road at the top left.", "banned phrasing"),
    ("This answer is correct, as #1 agrees.", "banned phrasing"),
    ("Its similarity to #1 is high.", "banned phrasing"),
    # phrasings the pilot LLM used that the first checks let through
    ("Because none of the training cases conflicted, the model had no reason to doubt it.", "banned phrasing"),
    ("This suggests the confidence is based on strong precedent from those cases.", "banned phrasing"),
    ("The confidence is supported by five closely matching training examples.", "banned phrasing"),
    ("It sees a pattern similar to #1 through #5.", "banned phrasing"),
    ("Those areas hold visual cues the model associates with roads.", "banned phrasing"),
    ("These cells appear to contain the features the model uses.", "banned phrasing"),
    ("The explanation notes that #1 agrees.", "banned phrasing"),
    ("The model relied heavily on that part of the image.", "banned phrasing"),
    ("In #1 to #5 the correct counts were all zero.", "banned phrasing"),
    ("The answer is uncertain because the model's confidence is only 51%.", "banned phrasing"),
    ("The answer is marked as uncertain because the model indicated uncertainty.", "banned phrasing"),
    ("This answer is uncertain.", "uncertainty note not called for"),
    ("The tile has a no-data strip.", "mentions the cells or the strip"),
    ("#1 agrees, and the top left cells mattered most.", "mentions the cells or the strip"),
    ("word " * 91, "too long"),
    ("", "empty"),
])
def test_checks_catch_each_rule(body, reason):
    failures = check_explanation(body, make_facts(), config())
    assert any(f.startswith(reason) for f in failures), failures


def test_checks_require_the_uncertainty_note_when_warned():
    facts = make_facts(warning=True, agree=1)
    assert "uncertainty note missing" in check_explanation("Most of #1 to #5 disagree.", facts, config())
    assert check_explanation("Only 1 of the 5 similar cases agrees, so the answer is uncertain.", facts, config()) == []


def test_the_prompt_is_only_about_the_neighbours():
    system, user = (m["content"] for m in ea.build_messages(make_facts(strip=True), config()))
    assert "#1 to #5" in system and "uncertain" in system and "Do not mention grid cells" in system
    assert "#5" in user and "top left" not in user and "strip" not in user


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return self.replies.pop(0), 42


def test_the_two_conditions_differ_only_by_the_neighbour_paragraph():
    facts = make_facts(strip=True)
    client = FakeClient(["Four of the five closest training cases, #1 to #5, had this answer."])
    with_n = explain(facts, client, config(), True)
    without = explain(facts, client, config(), False)
    assert len(client.messages) == 1  # no LLM call without neighbours
    assert with_n["source"] == "llm" and with_n["tokens"] == 42 and without["source"] == "code"
    answer, fixed = first_sentence(facts), fixed_sentences(facts)
    assert without["text"] == f"{answer} {fixed}"
    assert with_n["text"] == f"{answer} Four of the five closest training cases, #1 to #5, had this answer. {fixed}"
    assert "no-data strip" in fixed and "top left and centre" in fixed


def test_explain_falls_back_to_the_template_on_a_failed_check_or_request():
    facts = make_facts()
    failed = explain(facts, FakeClient(["The heatmap highlights the road."]), config(), True)
    assert failed["source"] == "template" and failed["failures"]
    assert failed["text"] == f"{first_sentence(facts)} {neighbour_template(facts)} {fixed_sentences(facts)}"

    class Broken:
        def complete(self, messages):
            raise RuntimeError("LLM request failed: HTTP 401")

    broken = explain(facts, Broken(), config(), True)
    assert broken["source"] == "template" and broken["failures"] == ["LLM request failed: HTTP 401"]
    assert explain(facts, None, config(), True)["source"] == "template"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_client_reads_the_key_from_the_environment_and_retries_on_rate_limits(monkeypatch):
    settings = {**config(), "rag_llm_min_interval_s": 0, "rag_llm_api_key_env": "TEST_RAG_KEY"}
    monkeypatch.delenv("TEST_RAG_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TEST_RAG_KEY"):
        OpenAICompatibleClient(settings)
    monkeypatch.setenv("TEST_RAG_KEY", "secret")
    calls = []
    sleeps = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "busy", {"retry-after": "1.5"}, None)
        return _Response(json.dumps({"choices": [{"message": {"content": " Hi. "}}], "usage": {"total_tokens": 9}}).encode())

    monkeypatch.setattr(ea.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ea.time, "sleep", sleeps.append)
    client = OpenAICompatibleClient(settings)
    assert client.complete([{"role": "user", "content": "x"}]) == ("Hi.", 9)
    assert sleeps == [1.5] and len(calls) == 2
    sent = json.loads(calls[1].data)
    assert sent["model"] == settings["rag_llm_model"] and sent["temperature"] == 0 and sent["reasoning_effort"] == "low"
    assert calls[1].headers["Authorization"] == "Bearer secret"
    assert "secret" not in repr(vars(client).get("payload"))


def test_client_gives_up_on_errors_that_retrying_cannot_fix(monkeypatch):
    monkeypatch.setenv("TEST_RAG_KEY", "secret")
    settings = {**config(), "rag_llm_min_interval_s": 0, "rag_llm_api_key_env": "TEST_RAG_KEY"}

    def unauthorized(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "no", {}, None)

    monkeypatch.setattr(ea.urllib.request, "urlopen", unauthorized)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        OpenAICompatibleClient(settings).complete([])


def test_rating_rows_cap_each_type_and_outcome():
    pool = PoolResult(rows=list(range(8)), types=["presence"] * 4 + ["count"] * 4,
                      correct=[True, True, False, True, False, False, True, False], warned=[False] * 8)
    assert rating_rows(pool, 1) == [0, 2, 4, 6]
    assert rating_rows(pool, 2) == [0, 1, 2, 4, 5, 6]


def test_wilson_interval_matches_a_known_value():
    low, high = wilson_interval(15, 20)
    assert abs(low - 0.531) < 1e-3 and abs(high - 0.888) < 1e-3


def write_rows(sheet: Path, rows: list[dict]) -> None:
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_rating_sheet_is_blind_and_scoring_unblinds_it(tmp_path: Path):
    items = []
    for i in range(36):
        facts = make_facts(row=i, warning=i >= 30)
        with_source = "llm" if i < 20 or i >= 30 else "template"  # 20 main paragraphs by the LLM, 10 by the template
        items.append({"id": f"test-{i}", "set": "warned" if i >= 30 else "main", "facts": vars(facts) | {"neighbours": []},
                      "with_neighbours": {"text": f"WITH {i}", "source": with_source},
                      "without_neighbours": {"text": f"WITHOUT {i}", "source": "code"}})
    sheet, key = tmp_path / "sheet.csv", tmp_path / "key.json"
    write_rating_sheet(items, 0, sheet, key)
    rows = list(csv.DictReader(sheet.open()))
    assert "WITH" not in ",".join(rows[0].keys()) and "set" not in rows[0] and "warning" not in rows[0]
    assert {row["text_1"].split()[0] for row in rows} == {"WITH", "WITHOUT"}  # the order varies within a row
    positions = [int(row["id"].split("-")[1]) for row in rows]
    assert sorted(positions) == list(range(36)) and positions != list(range(36))  # warned rows not all at the end
    assert max(positions[:18]) >= 30

    # the rater prefers the text with neighbours every time, and rates it 3 against 1
    number = lambda row: int(row["id"].split("-")[1])
    for row in rows:
        with_slot = "1" if row["text_1"].startswith("WITH ") else "2"
        other = "2" if with_slot == "1" else "1"
        row.update({f"useful_{with_slot}": "3", f"useful_{other}": "1", f"honest_{with_slot}": "3",
                    f"honest_{other}": "2", "prefer": "same" if number(row) == 0 else with_slot})
    write_rows(sheet, rows)
    report = score_ratings(sheet, key)
    main, warned = report.split("## Warned set")
    assert "**Result: retrieval improves the explanations.**" in main
    assert "## Main set (30 questions)" in main
    by_llm = "| neighbours by LLM | 19 | 19 (100%) |"
    assert by_llm in main and main.split(by_llm)[1].split("\n")[0].endswith("| 1 |")
    assert "| neighbours by template | 10 | 10 (100%) |" in main and "| all | 29 | 29 (100%) |" in main
    assert "| neighbours by LLM | with neighbours | 3.00 (n=20) | 3.00 (n=20) |" in main
    assert "| all | without neighbours | 1.00 (n=30) | 2.00 (n=30) |" in main
    assert warned.startswith(" (6 questions; does not decide)") and "| neighbours by LLM | 6 | 6 (100%) |" in warned

    # preferences on the warned set alone never decide
    for row in rows:
        if number(row) < 30:
            row["prefer"] = ""
    write_rows(sheet, rows)
    assert "**No preferences given yet on the main set.**" in score_ratings(sheet, key)


def test_warned_set_takes_warned_questions_per_type_outside_the_main_set():
    pool = PoolResult(rows=list(range(8)), types=["presence"] * 4 + ["count"] * 4,
                      correct=[True] * 8, warned=[True, False, True, True, False, True, True, True])
    assert ea.warned_rows(pool, 2, exclude=set()) == [0, 2, 5, 6]
    assert ea.warned_rows(pool, 2, exclude={0, 5}) == [2, 3, 6, 7]
    assert ea.warned_rows(pool, 5, exclude=set()) == [0, 2, 3, 5, 6, 7]


def test_generate_runs_end_to_end_with_a_fake_llm(trained_run, tmp_path: Path):
    run_dir, base = trained_run
    settings = {**base, **{k: v for k, v in config().items() if k not in base}, "rag_k": 3,
                "rag_search_chunk_rows": 4, "rag_tile_batch_size": 3, "rag_question_batch_size": 5,
                "rag_rating_per_group": 1, "rag_warned_per_type": 1, "rag_llm_min_interval_s": 0}

    class Echo:
        def complete(self, messages):
            return "Most of the similar cases have this answer as their true answer.", 10

    report = run_generate(run_dir, settings, client=Echo(), out_dir=tmp_path / "reports")
    assert "## Uncertainty warning on the test pool" in report and "| pooled | presence |" in report
    assert "| main |" in report and "## Explanations" in report
    assert "Left out so the rating stays blind" in report and "True answer:" not in report
    items = json.loads((run_dir / "rag_explanations_full.json").read_text())
    assert items and all(item["facts"]["neighbours"] for item in items)
    assert {item["facts"]["split"] for item in items} == {"test", "test_phili"}
    assert "- Warned set:" in report and "| warned |" in report
    for item in items:  # the two texts differ only by the neighbour paragraph
        assert item["without_neighbours"]["source"] == "code"
        assert item["with_neighbours"]["text"].endswith(item["without_neighbours"]["text"].split(". ", 1)[1])
    for item in items:  # warned-set questions are warned and never repeat a main-set question
        assert item["set"] == "main" or item["facts"]["warning"]
    assert len({item["id"] for item in items}) == len(items)
    sheet = list(csv.DictReader((tmp_path / "reports" / f"rag_rating_{run_dir.name}.csv").open()))
    assert len(sheet) == len(items)
    assert (run_dir / "rag_rating_key_full.json").exists()
    pilot = run_generate(run_dir, settings, pilot=True, client=Echo(), out_dir=tmp_path / "reports")
    assert "True answer:" in pilot and "Left out" not in pilot

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
    template_body,
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


def test_facts_text_hides_the_truth_and_drops_neighbours_without_retrieval():
    facts = make_facts(truth="no", correct=False)
    with_n, without = facts_text(facts, True), facts_text(facts, False)
    assert '#1 "Is there a road?" true answer: yes' in with_n and "4 of 5" in with_n
    assert "#1" not in without and "Uncertain" not in without
    # a "no" line is left out: the pilot LLM echoed it back as filler ("there is no no-data strip")
    assert "Uncertain" not in with_n and "strip" not in with_n
    flagged = make_facts(warning=True, agree=1, strip=True)
    assert "Uncertain: yes" in facts_text(flagged, True) and "Uncertain" not in facts_text(flagged, False)
    assert all("no-data strip on this tile: yes" in facts_text(flagged, w) for w in (True, False))
    for text in (with_n, without):
        assert "correct" not in text.lower() and 'answer: "no"' not in text
    assert "range from 2 to 9" in facts_text(count_facts(), True)
    assert "all had a true count of 0" in template_body(count_facts(count_range=(0.0, 0.0)), True)
    assert first_sentence(make_facts()) == 'The model answered "yes" with 93% confidence.'
    assert first_sentence(count_facts()) == 'The model answered "7".'


@pytest.mark.parametrize("facts", [
    make_facts(), make_facts(warning=True, agree=1), make_facts(strip=True), count_facts(), count_facts(warning=True),
    make_facts(answer="11 to 100 square meters", qtype="area", strip=True, warning=True, agree=2),
])
@pytest.mark.parametrize("with_neighbours", [True, False])
def test_the_template_passes_every_check(facts, with_neighbours):
    assert check_explanation(template_body(facts, with_neighbours), facts, config(), with_neighbours) == []


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
    ("This answer is uncertain.", "uncertainty note not called for"),
    ("The tile has a no-data strip.", "no-data note not called for"),
    ("word " * 91, "too long"),
    ("", "empty"),
])
def test_checks_catch_each_rule(body, reason):
    failures = check_explanation(body, make_facts(), config(), True)
    assert any(f.startswith(reason) for f in failures), failures


def test_checks_require_the_warning_and_strip_notes_when_called_for():
    facts = make_facts(warning=True, agree=1, strip=True)
    failures = check_explanation("Most of #1 to #5 disagree.", facts, config(), True)
    assert "uncertainty note missing" in failures and "no-data note missing" in failures
    # without neighbours there is no warning to require, and mentioning uncertainty is allowed
    assert check_explanation("Changing the top left changed it most. The no-data strip says nothing about what is there; "
                             "the answer may be uncertain.", facts, config(), False) == []


def test_each_condition_gets_its_own_prompt():
    facts = make_facts()
    with_n = ea.build_messages(facts, config(), True)[0]["content"]
    without = ea.build_messages(facts, config(), False)[0]["content"]
    assert "#1 to #5" in with_n and "uncertain" in with_n
    # mentioning ranked cases made the pilot LLM invent them when none were given
    assert "#1" not in without and "uncertain" not in without.lower()
    assert "no training cases" in without


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return self.replies.pop(0), 42


def test_explain_shows_a_passing_llm_text_after_the_code_written_answer():
    client = FakeClient(["Four of the five closest training cases, #1 to #5, had this answer."])
    result = explain(make_facts(), client, config(), True)
    assert result["source"] == "llm" and result["tokens"] == 42
    assert result["text"] == 'The model answered "yes" with 93% confidence. Four of the five closest training cases, #1 to #5, had this answer.'
    prompt = client.messages[0][1]["content"]
    assert "Facts:" in prompt and "#5" in prompt and "true answer: yes" in prompt


def test_explain_falls_back_to_the_template_on_a_failed_check_or_request():
    failed = explain(make_facts(), FakeClient(["The heatmap highlights the road."]), config(), True)
    assert failed["source"] == "template" and failed["failures"]
    assert failed["text"].endswith(template_body(make_facts(), True))

    class Broken:
        def complete(self, messages):
            raise RuntimeError("LLM request failed: HTTP 401")

    broken = explain(make_facts(), Broken(), config(), True)
    assert broken["source"] == "template" and broken["failures"] == ["LLM request failed: HTTP 401"]
    assert explain(make_facts(), None, config(), True)["source"] == "template"


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


def test_rating_sheet_is_blind_and_scoring_unblinds_it(tmp_path: Path):
    items = []
    for i in range(30):
        facts = make_facts(row=i)
        without_source = "llm" if i < 20 else "template"  # 20 pairs both LLM, 10 LLM against the template
        items.append({"id": f"test-{i}", "facts": vars(facts) | {"neighbours": []},
                      "with_neighbours": {"text": f"WITH {i}", "source": "llm"},
                      "without_neighbours": {"text": f"WITHOUT {i}", "source": without_source}})
    sheet, key = tmp_path / "sheet.csv", tmp_path / "key.json"
    write_rating_sheet(items, 0, sheet, key)
    rows = list(csv.DictReader(sheet.open()))
    assert "WITH" not in ",".join(rows[0].keys())
    firsts = {row["text_1"].split()[0] for row in rows}
    assert firsts == {"WITH", "WITHOUT"}  # the order varies, so position gives nothing away

    # the rater prefers the text with neighbours every time, and rates it 3 against 1
    for row in rows:
        with_slot = "1" if row["text_1"].startswith("WITH ") else "2"
        other = "2" if with_slot == "1" else "1"
        row.update({f"useful_{with_slot}": "3", f"useful_{other}": "1", f"honest_{with_slot}": "3",
                    f"honest_{other}": "2", "prefer": with_slot})
    rows[0]["prefer"] = "same"
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = score_ratings(sheet, key)
    assert "**Result: retrieval improves the explanations.**" in report
    assert "| both LLM | 19 | 19 (100%) |" in report and report.split("| both LLM | 19 | 19 (100%) |")[1].split("\n")[0].endswith("| 1 |")
    assert "| with: LLM, without: template | 10 | 10 (100%) |" in report
    assert "| all | 29 | 29 (100%) |" in report
    assert "| both LLM | with neighbours | 3.00 (n=20) | 3.00 (n=20) |" in report
    assert "| all | without neighbours | 1.00 (n=30) | 2.00 (n=30) |" in report

    # preferences on LLM-against-template pairs alone never decide
    for row in rows[:20]:
        row["prefer"] = ""
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    assert "**No preferences given yet on pairs where both texts are the LLM's.**" in score_ratings(sheet, key)


def test_generate_runs_end_to_end_with_a_fake_llm(trained_run, tmp_path: Path):
    run_dir, base = trained_run
    settings = {**base, **{k: v for k, v in config().items() if k not in base}, "rag_k": 3,
                "rag_search_chunk_rows": 4, "rag_tile_batch_size": 3, "rag_question_batch_size": 5,
                "rag_rating_per_group": 1, "rag_llm_min_interval_s": 0}

    class Echo:
        def complete(self, messages):
            return "Changing those cells changed the answer most.", 10

    report = run_generate(run_dir, settings, client=Echo(), out_dir=tmp_path / "reports")
    assert "## Uncertainty warning on the test pool" in report and "| pooled | presence |" in report
    assert "| with neighbours |" in report and "## Explanations" in report
    assert "Left out so the rating stays blind" in report and "True answer:" not in report
    items = json.loads((run_dir / "rag_explanations_full.json").read_text())
    assert items and all(item["facts"]["neighbours"] for item in items)
    assert {item["facts"]["split"] for item in items} == {"test", "test_phili"}
    sheet = list(csv.DictReader((tmp_path / "reports" / f"rag_rating_{run_dir.name}.csv").open()))
    assert len(sheet) == len(items)
    assert (run_dir / "rag_rating_key_full.json").exists()
    pilot = run_generate(run_dir, settings, pilot=True, client=Echo(), out_dir=tmp_path / "reports")
    assert "True answer:" in pilot and "Left out" not in pilot

# RAG explanations: hr_full_20261031-090319

LLM: `openai/gpt-oss-120b` at `https://api.groq.com/openai/v1`. Each question is explained with and without the neighbours.

- Main set: 64 questions (up to 4 per test split, type and right or wrong answer). It decides.
- Warned set: 16 questions where the warning fired (up to 2 per test split and type, not in the main set). Reported on its own.

## Uncertainty warning on the test pool

Rule, fixed in advance: classification warns when fewer than 3 of the 5 neighbours have the model's answer as their true answer; count warns when the rounded count is outside the neighbours' range of true counts. One question per tile and type.

| Set | Type | Questions | Warned | Model wrong when warned | Model wrong otherwise |
|---|---|---|---|---|---|
| test | presence | 2,226 | 4.2% | 42.6% (94) | 7.5% (2,132) |
| test | comp | 2,226 | 3.6% | 37.5% (80) | 7.6% (2,146) |
| test | area | 2,226 | 9.9% | 52.5% (221) | 9.7% (2,005) |
| test | count | 2,226 | 1.4% | 87.1% (31) | 32.9% (2,195) |
| test_phili | presence | 1,056 | 6.6% | 44.3% (70) | 19.7% (986) |
| test_phili | comp | 1,056 | 6.5% | 29.0% (69) | 19.5% (987) |
| test_phili | area | 1,056 | 14.7% | 72.9% (155) | 27.2% (901) |
| test_phili | count | 1,056 | 0.4% | 75.0% (4) | 40.0% (1,052) |
| pooled | presence | 3,282 | 5.0% | 43.3% (164) | 11.3% (3,118) |
| pooled | comp | 3,282 | 4.5% | 33.6% (149) | 11.4% (3,133) |
| pooled | area | 3,282 | 11.5% | 60.9% (376) | 15.1% (2,906) |
| pooled | count | 3,282 | 1.1% | 85.7% (35) | 35.2% (3,247) |

## Checks

Code writes the answer, the cells and the strip note in both texts; the LLM writes only the neighbour paragraph, which falls back to a template when a check fails.

| Set | Neighbour paragraph by the LLM | Fell back to the template | Tokens |
|---|---|---|---|
| main | 60 of 64 | 4 | 48,661 |
| warned | 11 of 16 | 5 | 12,763 |

Why texts fell back:

- banned phrasing: 9
- too long: 1

## Explanations

Left out so the rating stays blind. After scoring, the labelled texts are in `rag_explanations_full.json` in the run folder.

## Time

11.5 min in total, 10.7 min of it explaining the rating set. Rating sheet: `reports/rag_rating_hr_full_20261031-090319.csv`.

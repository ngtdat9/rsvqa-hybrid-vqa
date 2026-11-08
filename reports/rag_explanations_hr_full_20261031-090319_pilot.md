# RAG explanations: hr_full_20261031-090319_pilot

LLM: `openai/gpt-oss-120b` at `https://api.groq.com/openai/v1`. Each question is explained with and without the neighbours.

- Main set: 16 questions (up to 1 per test split, type and right or wrong answer). It decides.
- Warned set: 8 questions where the warning fired (up to 1 per test split and type, not in the main set). Reported on its own.

## Uncertainty warning on the test pool

Rule, fixed in advance: classification warns when fewer than 3 of the 5 neighbours have the model's answer as their true answer; count warns when the rounded count is outside the neighbours' range of true counts. One question per tile and type.

| Set | Type | Questions | Warned | Model wrong when warned | Model wrong otherwise |
|---|---|---|---|---|---|
| test | presence | 300 | 3.3% | 10.0% (10) | 6.9% (290) |
| test | comp | 300 | 5.0% | 40.0% (15) | 9.5% (285) |
| test | area | 300 | 9.7% | 41.4% (29) | 7.0% (271) |
| test | count | 300 | 2.3% | 71.4% (7) | 29.4% (293) |
| test_phili | presence | 300 | 6.7% | 45.0% (20) | 17.1% (280) |
| test_phili | comp | 300 | 8.7% | 26.9% (26) | 23.0% (274) |
| test_phili | area | 300 | 13.0% | 66.7% (39) | 35.6% (261) |
| test_phili | count | 300 | 0.3% | 100.0% (1) | 37.8% (299) |
| pooled | presence | 600 | 5.0% | 33.3% (30) | 11.9% (570) |
| pooled | comp | 600 | 6.8% | 31.7% (41) | 16.1% (559) |
| pooled | area | 600 | 11.3% | 55.9% (68) | 21.1% (532) |
| pooled | count | 600 | 1.3% | 75.0% (8) | 33.6% (592) |

## Checks

Code writes the answer, the cells and the strip note in both texts; the LLM writes only the neighbour paragraph, which falls back to a template when a check fails.

| Set | Neighbour paragraph by the LLM | Fell back to the template | Tokens |
|---|---|---|---|
| main | 16 of 16 | 0 | 12,157 |
| warned | 5 of 8 | 3 | 6,494 |

Why texts fell back:

- banned phrasing: 3

## Explanations

### test-22575 (main set): Is a small commercial building present?

True answer: no; model right; warning: no; no-data strip: no.

- **with neighbours** (llm): The model answered "no" with 99% confidence. Among the most similar cases, #1 asked “Is a small commercial building present?” with a true answer no, and #2 asked the same question with a true answer no. All five of the similar training cases have the model’s answer as their true answer. Changing the cells at the top centre and centre of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "no" with 99% confidence. Changing the cells at the top centre and centre of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test-22388 (main set): Is a medium school present?

True answer: yes; model wrong; warning: no; no-data strip: no.

- **with neighbours** (llm): The model answered "no" with 82% confidence. Four of the five most similar training cases share the model’s answer. For example, case #1 asked “Is a medium school present in the image?” with a true answer of no, and case #3 asked “Is there a medium school?” also with a true answer of no. Changing the cells at the bottom right and top right of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "no" with 82% confidence. Changing the cells at the bottom right and top right of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test-2469 (main set): What is the area covered by small commercial buildings?

True answer: 0 square meters; model right; warning: no; no-data strip: no.

- **with neighbours** (llm): The model answered "0 square meters" with 99% confidence. Among the most similar training cases, #1 asks “What is the area covered by small commercial buildings?” with a true answer of 0 square meters, and #2 asks the same question with a true answer of 0 square meters; in total, 5 of the 5 similar cases have true answers that match the model’s answer. Changing the cells at the middle left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "0 square meters" with 99% confidence. Changing the cells at the middle left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test-6822 (main set): What is the area covered by buildings?

True answer: more than 1000 square meters; model wrong; warning: no; no-data strip: no.

- **with neighbours** (llm): The model answered "101 to 1000 square meters" with 76% confidence. Among the most similar training cases, #1 and #3 both asked “What is the area covered by buildings?” with the true answer 101 to 1000 square meters, and all five similar cases share this true answer. Changing the cells at the middle right and bottom left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "101 to 1000 square meters" with 76% confidence. Changing the cells at the middle right and bottom left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test-22946 (main set): Are there more commercial buildings than roads?

True answer: no; model right; warning: no; no-data strip: no.

- **with neighbours** (llm): The model answered "no" with 100% confidence. Among the most similar training cases, #1 asked “Are there more commercial buildings than roads?” with a true answer of no, and #2 asked the same question with a true answer of no; in fact, all five similar cases had the true answer “no.” Changing the cells at the middle left and top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "no" with 100% confidence. Changing the cells at the middle left and top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test-26969 (main set): Are there more buildings than pedestrians?

True answer: yes; model wrong; warning: no; no-data strip: no.

- **with neighbours** (llm): The model answered "no" with 65% confidence. Three of the five most similar training cases share the model’s answer. For example, case #1 asked “Are there more buildings than pedestrians?” with the true answer “no,” and case #4 asked the same question with the true answer “no.” Changing the cells at the bottom left, centre and top centre of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "no" with 65% confidence. Changing the cells at the bottom left, centre and top centre of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test-13935 (main set): How many rectangular buildings are there?

True answer: 13; model wrong; warning: no; no-data strip: no.

- **with neighbours** (llm): The model answered "11". Among the most similar training cases, #1 asked “How many rectangular buildings are there?” with a true answer of 10, and #2 asked the same question with a true answer of 16; none of the five similar cases have a true answer of 11. Changing the cells at the top right, middle right and bottom left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "11". Changing the cells at the top right, middle right and bottom left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test-8986 (main set): How many commercial buildings at the bottom of the  hospital are there?

True answer: 0; model right; warning: no; no-data strip: no.

- **with neighbours** (llm): The model answered "0". Among the most similar cases, #1 asked “How many commercial buildings on the right of the park are there?” with a true answer of 0, and #3 asked “How many commercial buildings at the bottom of the university are there?” also with a true answer of 0; in fact, all five similar training questions have true answers of 0. Changing the cells at the bottom centre and bottom right of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "0". Changing the cells at the bottom centre and bottom right of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test-16141 (warned set): Is there a road on the right of a  park?

True answer: yes; model wrong; warning: yes; no-data strip: no.

- **with neighbours** (llm): The model answered "no" with 51% confidence. The answer is uncertain because only 2 of the 5 similar cases have the model’s answer as their true answer. For example, case #1 asked “Is a park on the right of a road present?” with a true answer of no, and case #4 asked “Is there a road on the right of the library?” with a true answer of yes. Changing the cells at the bottom left and top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "no" with 51% confidence. Changing the cells at the bottom left and top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test-8395 (warned set): What is the area covered by large woods?

True answer: more than 1000 square meters; model right; warning: yes; no-data strip: no.

- **with neighbours** (template; failed: banned phrasing: \b(it|the model|model)\s+(sees|saw|looked|looks|noticed|notices|recogni[sz]ed|detected|identified|found|believes?|thinks?|learned|learnt|associates?)\b): The model answered "more than 1000 square meters" with 45% confidence. Of the 5 training questions the model treats as most similar, 1 had this answer as their true answer (#1 is the closest). Because most of them disagree, treat this answer as uncertain. Changing the cells at the top left and top centre of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "more than 1000 square meters" with 45% confidence. Changing the cells at the top left and top centre of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test-19136 (warned set): Is the number of water areas equal to the number of roads?

True answer: yes; model right; warning: yes; no-data strip: no.

- **with neighbours** (template; failed: banned phrasing: \b(it|the model|model)\s+(sees|saw|looked|looks|noticed|notices|recogni[sz]ed|detected|identified|found|believes?|thinks?|learned|learnt|associates?)\b): The model answered "yes" with 54% confidence. Of the 5 training questions the model treats as most similar, 2 had this answer as their true answer (#1 is the closest). Because most of them disagree, treat this answer as uncertain. Changing the cells at the bottom left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "yes" with 54% confidence. Changing the cells at the bottom left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test-6923 (warned set): How many residential buildings on the left of a  road are there?

True answer: 2; model wrong; warning: yes; no-data strip: no.

- **with neighbours** (llm): The model answered "1". The model’s answer is uncertain because its count of 1 lies outside the range of true counts (2 to 4) seen in the most similar training cases. For example, case #1 asked “How many residential buildings at the top of a park are there?” with a true answer of 4, and case #2 asked “How many residential buildings at the bottom of a road are there?” with a true answer of 2. Changing the cells at the middle left and bottom right of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "1". Changing the cells at the middle left and bottom right of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test_phili-13393 (main set): What is the area covered by pedestrians on the left of a  commercial building?

True answer: 0 square meters; model right; warning: no; no-data strip: yes.

- **with neighbours** (llm): The model answered "0 square meters" with 100% confidence. All five of the most similar training cases have the same true answer as the model’s answer. For example, case #1 asks “What is the area covered by pedestrians at the bottom of a commercial building?” with a true answer of 0 square meters, and case #2 asks “What is the area covered by pedestrians at the bottom of a commercial building in the image?” also with a true answer of 0 square meters. Changing the cells at the middle right and bottom right of the tile changed the answer most. That tells you what the answer depended on, not where anything is. The tile has a black no-data strip; strong cells next to it say nothing about what is there.
- **without neighbours** (code): The model answered "0 square meters" with 100% confidence. Changing the cells at the middle right and bottom right of the tile changed the answer most. That tells you what the answer depended on, not where anything is. The tile has a black no-data strip; strong cells next to it say nothing about what is there.

### test_phili-17143 (main set): What is the area covered by rectangular residential buildings?

True answer: more than 1000 square meters; model wrong; warning: no; no-data strip: no.

- **with neighbours** (llm): The model answered "0 square meters" with 89% confidence. Among the most similar training cases, #1 asked “What is the area covered by residential buildings?” with a true answer of 0 square meters, and #2 asked “What is the area covered by rectangular buildings?” also with a true answer of 0 square meters. In total, all five similar cases (5 of 5) have true answers that match the model’s answer. Changing the cells at the top left and bottom left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "0 square meters" with 89% confidence. Changing the cells at the top left and bottom left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test_phili-27477 (main set): Are there more buildings than water areas?

True answer: yes; model right; warning: no; no-data strip: no.

- **with neighbours** (llm): The model answered "yes" with 95% confidence. All five of the most similar training cases have the true answer “yes”. For instance, case #1 asks “Are there more buildings than water areas?” with true answer yes, and case #4 asks “Are there more residential buildings than water areas?” with true answer yes. Changing the cells at the bottom centre of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "yes" with 95% confidence. Changing the cells at the bottom centre of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test_phili-18482 (main set): Is the number of commercial buildings equal to the number of greenfields?

True answer: no; model wrong; warning: no; no-data strip: yes.

- **with neighbours** (llm): The model answered "yes" with 82% confidence. All five of the most similar training cases have the true answer “yes”. For example, case #1 asks “Is the number of commercial buildings equal to the number of cemeterys?” with a true answer of yes, and case #2 asks “Is the number of commercial buildings equal to the number of vineyards?” with a true answer of yes. Changing the cells at the bottom right and top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is. The tile has a black no-data strip; strong cells next to it say nothing about what is there.
- **without neighbours** (code): The model answered "yes" with 82% confidence. Changing the cells at the bottom right and top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is. The tile has a black no-data strip; strong cells next to it say nothing about what is there.

### test_phili-22680 (main set): Is a road on the right of a  small building present?

True answer: no; model right; warning: no; no-data strip: no.

- **with neighbours** (llm): The model answered "no" with 99% confidence. Among the most similar training cases, #1 asked “Is a stadium at the bottom of a small building present?” with a true answer of no, and #2 asked “Is there a small wood?” with a true answer of no. All five of the similar cases have the model’s answer as their true answers. Changing the cells at the bottom right and top right of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "no" with 99% confidence. Changing the cells at the bottom right and top right of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test_phili-12048 (main set): Is there a medium water area?

True answer: yes; model wrong; warning: no; no-data strip: yes.

- **with neighbours** (llm): The model answered "no" with 93% confidence. Among the five most similar training cases, all five have the model’s answer as their true answers. For example, case #1 asked “Is a medium water area present?” with true answer no, and case #2 asked “Is there a medium water area?” with true answer no. Changing the cells at the bottom right and middle right of the tile changed the answer most. That tells you what the answer depended on, not where anything is. The tile has a black no-data strip; strong cells next to it say nothing about what is there.
- **without neighbours** (code): The model answered "no" with 93% confidence. Changing the cells at the bottom right and middle right of the tile changed the answer most. That tells you what the answer depended on, not where anything is. The tile has a black no-data strip; strong cells next to it say nothing about what is there.

### test_phili-18696 (main set): How many medium buildings are there?

True answer: 7; model wrong; warning: no; no-data strip: no.

- **with neighbours** (llm): The model answered "0". Four of the five most similar training cases have true answers of 0, matching the model’s answer. For example, case #1 asks “How many medium buildings are there in the image?” with a true answer 0, while case #5 asks the same question but has a true answer 1. Changing the cells at the top left and middle right of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "0". Changing the cells at the top left and middle right of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test_phili-13982 (main set): What is the number of large water areas?

True answer: 0; model right; warning: no; no-data strip: yes.

- **with neighbours** (llm): The model answered "0". Among the most similar training cases, #1 asked “What is the number of large water areas?” with a true answer of 0, and #3 asked “What is the amount of large water areas in the image?” also with a true answer of 0; in fact, all five similar cases have true answers of 0. Changing the cells at the middle left and top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is. The tile has a black no-data strip; strong cells next to it say nothing about what is there.
- **without neighbours** (code): The model answered "0". Changing the cells at the middle left and top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is. The tile has a black no-data strip; strong cells next to it say nothing about what is there.

### test_phili-27515 (warned set): What is the area covered by residential buildings?

True answer: more than 1000 square meters; model wrong; warning: yes; no-data strip: no.

- **with neighbours** (template; failed: banned phrasing: \b(it|the model|model)\s+(sees|saw|looked|looks|noticed|notices|recogni[sz]ed|detected|identified|found|believes?|thinks?|learned|learnt|associates?)\b): The model answered "101 to 1000 square meters" with 55% confidence. Of the 5 training questions the model treats as most similar, 2 had this answer as their true answer (#1 is the closest). Because most of them disagree, treat this answer as uncertain. Changing the cells at the bottom right and top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "101 to 1000 square meters" with 55% confidence. Changing the cells at the bottom right and top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test_phili-15588 (warned set): Is the number of roads equal to the number of buildings?

True answer: no; model right; warning: yes; no-data strip: no.

- **with neighbours** (llm): The model answered "no" with 77% confidence. The model’s answer is uncertain because only 2 of the 5 similar cases share its answer as the true answer. For example, case #1 asks “Is the number of buildings equal to the number of roads?” with a true answer of no, and case #3 asks “Is the number of roads equal to the number of buildings?” with a true answer of no. Changing the cells at the middle left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "no" with 77% confidence. Changing the cells at the middle left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test_phili-7352 (warned set): Is there a parking on the left of a  road?

True answer: no; model wrong; warning: yes; no-data strip: no.

- **with neighbours** (llm): The model answered "yes" with 54% confidence. The model’s answer is uncertain because only 2 of the 5 similar cases share that answer as their true answer. For example, case #3 asked “Is there a parking on the right of a road?” with a true answer yes, and case #4 asked “Is a park on the left of a building present?” with a true answer yes. Changing the cells at the top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "yes" with 54% confidence. Changing the cells at the top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

### test_phili-6506 (warned set): What is the amount of buildings?

True answer: 71; model wrong; warning: yes; no-data strip: no.

- **with neighbours** (llm): The model answered "4". The model’s answer is uncertain because its count of 4 lies outside the range of true answers (6 to 12) seen in similar cases. Among the five most similar training questions, #1 asks “What is the amount of buildings?” with a true answer of 6, and #5 asks the same question with a true answer of 9; none of the similar cases have a true answer of 4. Changing the cells at the top centre and top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.
- **without neighbours** (code): The model answered "4". Changing the cells at the top centre and top left of the tile changed the answer most. That tells you what the answer depended on, not where anything is.

## Time

3.9 min in total, 3.1 min of it explaining the rating set. Rating sheet: `reports/rag_rating_hr_full_20261031-090319_pilot.csv`.

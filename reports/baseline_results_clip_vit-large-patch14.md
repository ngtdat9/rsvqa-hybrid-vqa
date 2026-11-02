# Results: zero-shot CLIP (openai/clip-vit-large-patch14)

## Sentence template, chosen on val

| Template | Val score | Val accuracy | Val count MAE |
|---|---|---|---|
| `{question} {answer}` | -4.4176 (chosen) | 0.4882 | 13.2596 |
| `Question: {question} Answer: {answer}` | -7.0243 | 0.3990 | 18.9117 |
| `An aerial image. Question: {question} Answer: {answer}` | -16.9051 | 0.4371 | 41.1814 |

## Split: test

| Metric | Value |
|---|---|
| Classification accuracy (overall) | 0.4947 |
| Classification accuracy (presence) | 0.5315 |
| Classification accuracy (comp) | 0.4957 |
| Classification accuracy (area) | 0.4273 |
| Count MAE | 14.1309 |
| Count RMSE | 20.3719 |
| Count MAE (always predict train median) | 2.1550 |
| Count RMSE (always predict train median) | 5.4177 |
| Accuracy, always the most common train answer (overall) | 0.6371 |
| Accuracy, always the most common train answer (presence) | 0.6082 |
| Accuracy, always the most common train answer (area) | 0.6240 |
| Accuracy, always the most common train answer (comp) | 0.6662 |

294,196 sentences encoded, 3.0 min

## Split: test_phili

| Metric | Value |
|---|---|
| Classification accuracy (overall) | 0.5495 |
| Classification accuracy (presence) | 0.5583 |
| Classification accuracy (comp) | 0.5737 |
| Classification accuracy (area) | 0.4795 |
| Count MAE | 11.3254 |
| Count RMSE | 17.3088 |
| Count MAE (always predict train median) | 5.2480 |
| Count RMSE (always predict train median) | 14.1048 |
| Accuracy, always the most common train answer (overall) | 0.6147 |
| Accuracy, always the most common train answer (presence) | 0.5725 |
| Accuracy, always the most common train answer (comp) | 0.6657 |
| Accuracy, always the most common train answer (area) | 0.5742 |

168,781 sentences encoded, 1.7 min

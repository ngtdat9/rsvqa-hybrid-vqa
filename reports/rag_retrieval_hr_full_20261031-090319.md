# Retrieval results: hr_full_20261031-090319

Index: 625,340 training QA pairs, 256-wide fused vectors (320 MB). k = 5. Test questions: one per tile and type (13,128 in all).

**Rule, fixed in advance:** a type passes when the AUROC difference, test splits pooled, has a 95% lower bound above 0. Only passing types may use neighbour disagreement as an uncertainty warning.

**Passes on: presence, comp, area, count**

## Test splits pooled (this decides)

Error warning: AUROC of low neighbour agreement for flagging the model's wrong answers (0.5 = chance). Random: the same number of random same-type training questions.

| Type | Questions | Model wrong | AUROC retrieval | AUROC random | Difference [95% CI] | Passes |
|---|---|---|---|---|---|---|
| presence | 3282 | 12.9% | 0.747 | 0.491 | +0.257 [+0.221, +0.294] | yes |
| comp | 3282 | 12.4% | 0.665 | 0.530 | +0.136 [+0.092, +0.174] | yes |
| area | 3282 | 20.4% | 0.790 | 0.632 | +0.158 [+0.130, +0.184] | yes |
| count | 3282 | 35.7% | 0.846 | 0.778 | +0.068 [+0.050, +0.087] | yes |

Neighbour answer: the neighbours' most common true answer (count: their median, rounded). Neighbours the model got right is on training data, so it is inflated by fitting.

| Type | Model accuracy | Neighbour answer = true answer | Neighbour answer = model's | Same, random | Mean agreement | Same, random | Top-1 similarity | Neighbours the model got right |
|---|---|---|---|---|---|---|---|---|
| presence | 87.1% | 86.4% | 95.0% | 54.9% | 0.916 | 0.528 | 1.000 | 91.7% |
| comp | 87.6% | 86.1% | 95.5% | 61.0% | 0.917 | 0.569 | 1.000 | 91.8% |
| area | 79.6% | 78.3% | 91.0% | 57.7% | 0.857 | 0.477 | 0.999 | 85.5% |
| count | 64.3% | 63.5% | 79.3% | 47.0% | -0.110 | -0.589 | 0.996 | 72.4% |

What the neighbours are (descriptive; cannot change the result above). Similarity is cosine between fused vectors; random is the same query against the random control rows.

| Type | Same question text | Mean similarity, neighbours | Mean similarity, random |
|---|---|---|---|
| presence | 70.2% | 1.000 | 0.837 |
| comp | 57.7% | 1.000 | 0.925 |
| area | 80.2% | 0.999 | 0.616 |
| count | 71.7% | 0.995 | 0.311 |

Count MAE: model 2.159, neighbours' median count 2.188.

## test alone (does not decide)

Error warning: AUROC of low neighbour agreement for flagging the model's wrong answers (0.5 = chance). Random: the same number of random same-type training questions.

| Type | Questions | Model wrong | AUROC retrieval | AUROC random | Difference [95% CI] | Passes |
|---|---|---|---|---|---|---|
| presence | 2226 | 8.9% | 0.782 | 0.556 | +0.226 [+0.172, +0.280] | yes |
| comp | 2226 | 8.7% | 0.706 | 0.518 | +0.188 [+0.130, +0.240] | yes |
| area | 2226 | 14.0% | 0.817 | 0.662 | +0.155 [+0.120, +0.190] | yes |
| count | 2226 | 33.6% | 0.842 | 0.826 | +0.017 [-0.004, +0.039] | no |

Neighbour answer: the neighbours' most common true answer (count: their median, rounded). Neighbours the model got right is on training data, so it is inflated by fitting.

| Type | Model accuracy | Neighbour answer = true answer | Neighbour answer = model's | Same, random | Mean agreement | Same, random | Top-1 similarity | Neighbours the model got right |
|---|---|---|---|---|---|---|---|---|
| presence | 91.1% | 90.4% | 95.8% | 54.3% | 0.931 | 0.524 | 1.000 | 93.2% |
| comp | 91.3% | 90.4% | 96.4% | 60.5% | 0.930 | 0.565 | 1.000 | 93.2% |
| area | 86.0% | 84.4% | 92.1% | 55.3% | 0.871 | 0.465 | 0.999 | 86.9% |
| count | 66.4% | 65.4% | 76.3% | 43.9% | -0.107 | -0.675 | 0.997 | 69.5% |

What the neighbours are (descriptive; cannot change the result above). Similarity is cosine between fused vectors; random is the same query against the random control rows.

| Type | Same question text | Mean similarity, neighbours | Mean similarity, random |
|---|---|---|---|
| presence | 76.0% | 1.000 | 0.828 |
| comp | 64.2% | 1.000 | 0.920 |
| area | 84.6% | 0.999 | 0.593 |
| count | 78.1% | 0.996 | 0.284 |

Count MAE: model 0.907, neighbours' median count 0.945.

## test_phili alone (does not decide)

Error warning: AUROC of low neighbour agreement for flagging the model's wrong answers (0.5 = chance). Random: the same number of random same-type training questions.

| Type | Questions | Model wrong | AUROC retrieval | AUROC random | Difference [95% CI] | Passes |
|---|---|---|---|---|---|---|
| presence | 1056 | 21.3% | 0.694 | 0.429 | +0.265 [+0.209, +0.320] | yes |
| comp | 1056 | 20.1% | 0.602 | 0.549 | +0.053 [-0.007, +0.112] | no |
| area | 1056 | 33.9% | 0.768 | 0.638 | +0.130 [+0.090, +0.167] | yes |
| count | 1056 | 40.2% | 0.855 | 0.703 | +0.152 [+0.119, +0.184] | yes |

Neighbour answer: the neighbours' most common true answer (count: their median, rounded). Neighbours the model got right is on training data, so it is inflated by fitting.

| Type | Model accuracy | Neighbour answer = true answer | Neighbour answer = model's | Same, random | Mean agreement | Same, random | Top-1 similarity | Neighbours the model got right |
|---|---|---|---|---|---|---|---|---|
| presence | 78.7% | 77.9% | 93.4% | 56.2% | 0.885 | 0.537 | 1.000 | 88.6% |
| comp | 79.9% | 77.2% | 93.5% | 62.1% | 0.888 | 0.577 | 1.000 | 88.8% |
| area | 66.1% | 65.4% | 88.6% | 62.7% | 0.827 | 0.503 | 0.998 | 82.7% |
| count | 59.8% | 59.5% | 85.5% | 53.4% | -0.117 | -0.408 | 0.994 | 78.3% |

What the neighbours are (descriptive; cannot change the result above). Similarity is cosine between fused vectors; random is the same query against the random control rows.

| Type | Same question text | Mean similarity, neighbours | Mean similarity, random |
|---|---|---|---|
| presence | 57.8% | 1.000 | 0.856 |
| comp | 43.8% | 1.000 | 0.936 |
| area | 70.9% | 0.998 | 0.663 |
| count | 58.3% | 0.992 | 0.368 |

Count MAE: model 4.798, neighbours' median count 4.807.

## Time

Index built or loaded in 1.8 min; 13,128 questions encoded and searched in 1.5 min, on cuda.

# Retrieval results: hr_full_20261031-090319 (pilot subsets)

Index: 150,061 training QA pairs, 256-wide fused vectors (77 MB). k = 5. Test questions: one per tile and type (2,400 in all).

**Rule, fixed in advance:** a type passes when the AUROC difference, test splits pooled, has a 95% lower bound above 0. Only passing types may use neighbour disagreement as an uncertainty warning.

**Passes on: presence, comp, area, count**

## Test splits pooled (this decides)

Error warning: AUROC of low neighbour agreement for flagging the model's wrong answers (0.5 = chance). Random: the same number of random same-type training questions.

| Type | Questions | Model wrong | AUROC retrieval | AUROC random | Difference [95% CI] | Passes |
|---|---|---|---|---|---|---|
| presence | 600 | 13.0% | 0.752 | 0.451 | +0.301 [+0.211, +0.386] | yes |
| comp | 600 | 17.2% | 0.634 | 0.508 | +0.126 [+0.043, +0.212] | yes |
| area | 600 | 25.0% | 0.751 | 0.576 | +0.175 [+0.114, +0.235] | yes |
| count | 600 | 34.2% | 0.876 | 0.771 | +0.104 [+0.065, +0.145] | yes |

Neighbour answer: the neighbours' most common true answer (count: their median, rounded). Neighbours the model got right is on training data, so it is inflated by fitting.

| Type | Model accuracy | Neighbour answer = true answer | Neighbour answer = model's | Same, random | Mean agreement | Same, random | Top-1 similarity | Neighbours the model got right |
|---|---|---|---|---|---|---|---|---|
| presence | 87.0% | 85.3% | 95.0% | 56.3% | 0.908 | 0.526 | 1.000 | 91.2% |
| comp | 82.8% | 80.3% | 93.2% | 63.7% | 0.908 | 0.565 | 1.000 | 91.8% |
| area | 75.0% | 72.7% | 90.8% | 56.2% | 0.853 | 0.476 | 0.998 | 85.1% |
| count | 65.8% | 66.8% | 80.0% | 47.8% | -0.109 | -0.539 | 0.994 | 74.2% |

Count MAE: model 2.555, neighbours' median count 2.548.

## test alone (does not decide)

Error warning: AUROC of low neighbour agreement for flagging the model's wrong answers (0.5 = chance). Random: the same number of random same-type training questions.

| Type | Questions | Model wrong | AUROC retrieval | AUROC random | Difference [95% CI] | Passes |
|---|---|---|---|---|---|---|
| presence | 300 | 7.0% | 0.724 | 0.603 | +0.121 [-0.052, +0.282] | no |
| comp | 300 | 11.0% | 0.713 | 0.502 | +0.211 [+0.057, +0.355] | yes |
| area | 300 | 10.3% | 0.809 | 0.630 | +0.179 [+0.047, +0.316] | yes |
| count | 300 | 30.3% | 0.863 | 0.843 | +0.020 [-0.039, +0.086] | no |

Neighbour answer: the neighbours' most common true answer (count: their median, rounded). Neighbours the model got right is on training data, so it is inflated by fitting.

| Type | Model accuracy | Neighbour answer = true answer | Neighbour answer = model's | Same, random | Mean agreement | Same, random | Top-1 similarity | Neighbours the model got right |
|---|---|---|---|---|---|---|---|---|
| presence | 93.0% | 90.3% | 96.7% | 54.0% | 0.930 | 0.512 | 1.000 | 93.3% |
| comp | 89.0% | 88.0% | 95.0% | 57.3% | 0.933 | 0.538 | 1.000 | 94.3% |
| area | 89.7% | 86.0% | 91.7% | 51.3% | 0.879 | 0.446 | 0.999 | 87.6% |
| count | 69.7% | 71.0% | 75.3% | 43.3% | -0.104 | -0.664 | 0.996 | 70.1% |

Count MAE: model 0.849, neighbours' median count 0.823.

## test_phili alone (does not decide)

Error warning: AUROC of low neighbour agreement for flagging the model's wrong answers (0.5 = chance). Random: the same number of random same-type training questions.

| Type | Questions | Model wrong | AUROC retrieval | AUROC random | Difference [95% CI] | Passes |
|---|---|---|---|---|---|---|
| presence | 300 | 19.0% | 0.755 | 0.391 | +0.364 [+0.258, +0.472] | yes |
| comp | 300 | 23.3% | 0.571 | 0.533 | +0.038 [-0.059, +0.136] | no |
| area | 300 | 39.7% | 0.733 | 0.605 | +0.128 [+0.054, +0.199] | yes |
| count | 300 | 38.0% | 0.891 | 0.730 | +0.161 [+0.103, +0.219] | yes |

Neighbour answer: the neighbours' most common true answer (count: their median, rounded). Neighbours the model got right is on training data, so it is inflated by fitting.

| Type | Model accuracy | Neighbour answer = true answer | Neighbour answer = model's | Same, random | Mean agreement | Same, random | Top-1 similarity | Neighbours the model got right |
|---|---|---|---|---|---|---|---|---|
| presence | 81.0% | 80.3% | 93.3% | 58.7% | 0.885 | 0.539 | 1.000 | 89.0% |
| comp | 76.7% | 72.7% | 91.3% | 70.0% | 0.883 | 0.593 | 0.999 | 89.3% |
| area | 60.3% | 59.3% | 90.0% | 61.0% | 0.827 | 0.507 | 0.997 | 82.7% |
| count | 62.0% | 62.7% | 84.7% | 52.3% | -0.115 | -0.414 | 0.992 | 78.3% |

Count MAE: model 4.262, neighbours' median count 4.273.

## Time

Index built or loaded in 1.1 min; 2,400 questions encoded and searched in 0.3 min, on cuda.

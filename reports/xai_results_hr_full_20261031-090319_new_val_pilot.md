# XAI results: hr\_full\_20261031-090319 (pilot subsets)

Splits: val. Deleted cells are replaced with: blur.

**Map for the RAG layer: occlusion**

- `gradient_attention`: does not beat random on: area, comp  
- `occlusion`: qualifies, mean standardized effect 0.841

## Against the random map (test splits pooled)

Per question, the effect averaged over deletion sizes, minus the random map's.

| Map | Type | Mean difference | 95% lower bound | Standardized |
| :---- | :---- | :---- | :---- | :---- |
| gradient\_attention | presence | 0.0365 | 0.0067 | 0.367 |
| gradient\_attention | comp | 0.0287 | \-0.0031 | 0.255 |
| gradient\_attention | area | 0.0125 | \-0.0229 | 0.098 |
| gradient\_attention | count | 0.0667 | 0.0228 | 0.421 |
| occlusion | presence | 0.1301 | 0.0889 | 0.944 |
| occlusion | comp | 0.1763 | 0.1268 | 1.008 |
| occlusion | area | 0.1290 | 0.0914 | 0.952 |
| occlusion | count | 0.0903 | 0.0358 | 0.460 |

## Against a random block (secondary, test splits pooled)

The same, against one compact random patch of the same size. Added after the pilot, so it does not affect the choice of map; it shows whether scattered deletion made the random map a harsher baseline.

| Map | Type | Mean difference | 95% lower bound | Standardized |
| :---- | :---- | :---- | :---- | :---- |
| gradient\_attention | presence | 0.0293 | \-0.0164 | 0.192 |
| gradient\_attention | comp | 0.0472 | 0.0049 | 0.315 |
| gradient\_attention | area | 0.0384 | 0.0026 | 0.297 |
| gradient\_attention | count | 0.1244 | 0.0719 | 0.657 |
| occlusion | presence | 0.1229 | 0.0734 | 0.742 |
| occlusion | comp | 0.1948 | 0.1292 | 0.841 |
| occlusion | area | 0.1549 | 0.1053 | 0.865 |
| occlusion | count | 0.1480 | 0.0756 | 0.567 |

## Checks

Mean rank correlation; a map fails at or above the limit.

| Map | Trained vs random weights (limit 0.5) | Two questions, same tile (limit 0.9) | All-constant maps |
| :---- | :---- | :---- | :---- |
| gradient\_attention | 0.001 | 0.407 | 0 of 191 |
| occlusion | 0.029 | 0.151 | 0 of 191 |

## Agreement between maps

| Pair | Rank correlation | Overlap of top 20% (IoU) |
| :---- | :---- | :---- |
| gradient\_attention vs occlusion | 0.045 | 0.129 |

## Deletion test: val

Effect: drop in the predicted class's probability, or for count the absolute change in log1p(count). Averages over questions; the sample is balanced between right and wrong answers.

| Type | Questions (right) | Map | 10% | 20% | 30% | 50% | Mean | Answer changed (mean) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| presence | 43 (25) | gradient\_attention | 0.0387 | 0.0574 | 0.0664 | 0.0970 | 0.0649 | 14.5% |
| presence | 43 (25) | occlusion | 0.1387 | 0.1509 | 0.1678 | 0.1765 | 0.1585 | 23.3% |
| presence | 43 (25) | random | 0.0085 | 0.0076 | 0.0228 | 0.0746 | 0.0283 | 8.7% |
| presence | 43 (25) | random\_block | 0.0025 | 0.0194 | 0.0456 | 0.0746 | 0.0355 | 6.4% |
| comp | 48 (25) | gradient\_attention | 0.0556 | 0.0819 | 0.1076 | 0.1434 | 0.0971 | 19.3% |
| comp | 48 (25) | occlusion | 0.1962 | 0.2372 | 0.2806 | 0.2649 | 0.2447 | 30.2% |
| comp | 48 (25) | random | 0.0215 | 0.0468 | 0.0700 | 0.1354 | 0.0684 | 14.6% |
| comp | 48 (25) | random\_block | 0.0192 | 0.0307 | 0.0513 | 0.0987 | 0.0500 | 10.9% |
| area | 50 (25) | gradient\_attention | 0.0140 | 0.0573 | 0.0634 | 0.0927 | 0.0568 | 20.5% |
| area | 50 (25) | occlusion | 0.1605 | 0.1755 | 0.1911 | 0.1664 | 0.1734 | 42.0% |
| area | 50 (25) | random | 0.0239 | 0.0362 | 0.0410 | 0.0763 | 0.0444 | 12.5% |
| area | 50 (25) | random\_block | \-0.0028 | 0.0068 | 0.0230 | 0.0468 | 0.0184 | 8.0% |
| count | 50 (25) | gradient\_attention | 0.1126 | 0.1631 | 0.2129 | 0.3563 | 0.2113 | 40.0% |
| count | 50 (25) | occlusion | 0.1402 | 0.2038 | 0.2539 | 0.3416 | 0.2349 | 39.5% |
| count | 50 (25) | random | 0.0749 | 0.1132 | 0.1362 | 0.2540 | 0.1446 | 36.0% |
| count | 50 (25) | random\_block | 0.0312 | 0.0585 | 0.0833 | 0.1745 | 0.0869 | 24.5% |

## Speed

1.432 s for one answer and its maps, on cuda.

6.6 min in total.
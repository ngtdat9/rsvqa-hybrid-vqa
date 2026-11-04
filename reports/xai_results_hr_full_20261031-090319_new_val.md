# XAI results: hr\_full\_20261031-090319

Splits: val. Deleted cells are replaced with: blur.

**Map for the RAG layer: occlusion**

- `gradient_attention`: qualifies, mean standardized effect 0.283  
- `occlusion`: qualifies, mean standardized effect 0.765

## Against the random map (scored splits pooled; this decides the choice)

Per question, the effect averaged over deletion sizes, minus the random map's.

| Map | Type | Mean difference | 95% lower bound | Standardized |
| :---- | :---- | :---- | :---- | :---- |
| gradient\_attention | presence | 0.0226 | 0.0125 | 0.245 |
| gradient\_attention | comp | 0.0164 | 0.0067 | 0.182 |
| gradient\_attention | area | 0.0159 | 0.0068 | 0.176 |
| gradient\_attention | count | 0.1185 | 0.0989 | 0.531 |
| occlusion | presence | 0.1144 | 0.0985 | 0.790 |
| occlusion | comp | 0.1222 | 0.1045 | 0.743 |
| occlusion | area | 0.1341 | 0.1192 | 0.899 |
| occlusion | count | 0.1597 | 0.1374 | 0.629 |

## Against a random block (secondary, scored splits pooled)

The same, against one compact random patch of the same size. Added after the pilot, so it does not affect the choice of map; it shows whether scattered deletion made the random map a harsher baseline.

| Map | Type | Mean difference | 95% lower bound | Standardized |
| :---- | :---- | :---- | :---- | :---- |
| gradient\_attention | presence | 0.0242 | 0.0116 | 0.210 |
| gradient\_attention | comp | 0.0312 | 0.0182 | 0.258 |
| gradient\_attention | area | 0.0290 | 0.0175 | 0.251 |
| gradient\_attention | count | 0.1633 | 0.1375 | 0.554 |
| occlusion | presence | 0.1160 | 0.0986 | 0.733 |
| occlusion | comp | 0.1369 | 0.1153 | 0.682 |
| occlusion | area | 0.1472 | 0.1296 | 0.833 |
| occlusion | count | 0.2045 | 0.1752 | 0.612 |

## Checks

Mean rank correlation; a map fails at or above the limit.

| Map | Trained vs random weights (limit 0.5) | Two questions, same tile (limit 0.9) | All-constant maps |
| :---- | :---- | :---- | :---- |
| gradient\_attention | \-0.015 | 0.352 | 0 of 1538 |
| occlusion | 0.023 | 0.101 | 2 of 1538 |

## Agreement between maps

| Pair | Rank correlation | Overlap of top 20% (IoU) |
| :---- | :---- | :---- |
| gradient\_attention vs occlusion | 0.065 | 0.135 |

## Deletion test: val

Effect: drop in the predicted class's probability, or for count the absolute change in log1p(count). Averages over questions; the sample is balanced between right and wrong answers.

| Type | Questions (right) | Map | 10% | 20% | 30% | 50% | Mean | Answer changed (mean) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| presence | 321 (250) | gradient\_attention | 0.0199 | 0.0366 | 0.0468 | 0.0828 | 0.0465 | 9.0% |
| presence | 321 (250) | occlusion | 0.1006 | 0.1338 | 0.1502 | 0.1686 | 0.1383 | 21.6% |
| presence | 321 (250) | random | 0.0038 | 0.0127 | 0.0251 | 0.0541 | 0.0239 | 5.5% |
| presence | 321 (250) | random\_block | 0.0049 | 0.0127 | 0.0204 | 0.0514 | 0.0223 | 4.9% |
| comp | 332 (250) | gradient\_attention | 0.0271 | 0.0428 | 0.0559 | 0.0897 | 0.0539 | 9.0% |
| comp | 332 (250) | occlusion | 0.1276 | 0.1598 | 0.1696 | 0.1813 | 0.1596 | 19.1% |
| comp | 332 (250) | random | 0.0076 | 0.0205 | 0.0380 | 0.0836 | 0.0374 | 6.6% |
| comp | 332 (250) | random\_block | 0.0069 | 0.0130 | 0.0238 | 0.0468 | 0.0226 | 4.1% |
| area | 385 (250) | gradient\_attention | 0.0173 | 0.0389 | 0.0670 | 0.1200 | 0.0608 | 12.9% |
| area | 385 (250) | occlusion | 0.1384 | 0.1767 | 0.1950 | 0.2059 | 0.1790 | 29.4% |
| area | 385 (250) | random | 0.0129 | 0.0267 | 0.0449 | 0.0952 | 0.0449 | 9.9% |
| area | 385 (250) | random\_block | 0.0055 | 0.0160 | 0.0309 | 0.0747 | 0.0318 | 8.4% |
| count | 500 (250) | gradient\_attention | 0.1474 | 0.2374 | 0.3190 | 0.4815 | 0.2963 | 39.1% |
| count | 500 (250) | occlusion | 0.2089 | 0.2908 | 0.3645 | 0.4858 | 0.3375 | 40.1% |
| count | 500 (250) | random | 0.0626 | 0.1135 | 0.1857 | 0.3495 | 0.1778 | 33.3% |
| count | 500 (250) | random\_block | 0.0530 | 0.0891 | 0.1389 | 0.2510 | 0.1330 | 30.9% |

## Speed

1.443 s for one answer and its maps, on cuda.

42.4 min in total.
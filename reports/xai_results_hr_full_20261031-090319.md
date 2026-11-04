# XAI results: hr_full_20261031-090319

**Map for the RAG layer: none qualifies; the maps must not be presented as evidence**

- `cross_attention`: does not beat random on: area, comp, count, presence
- `rollout`: barely changes with the question (correlation 0.967)
- `grad_cam`: does not beat random on: area, comp, presence

## Against the random map (test splits pooled)

Per question, the effect averaged over deletion sizes, minus the random map's.

| Map | Type | Mean difference | 95% lower bound | Standardized |
|---|---|---|---|---|
| cross_attention | presence | -0.0034 | -0.0112 | -0.028 |
| cross_attention | comp | -0.0240 | -0.0321 | -0.193 |
| cross_attention | area | -0.0056 | -0.0137 | -0.043 |
| cross_attention | count | -0.0037 | -0.0132 | -0.024 |
| rollout | presence | 0.0200 | 0.0099 | 0.128 |
| rollout | comp | 0.0120 | 0.0023 | 0.080 |
| rollout | area | 0.0199 | 0.0109 | 0.137 |
| rollout | count | 0.0749 | 0.0619 | 0.357 |
| grad_cam | presence | -0.0291 | -0.0387 | -0.196 |
| grad_cam | comp | -0.0172 | -0.0266 | -0.118 |
| grad_cam | area | -0.0151 | -0.0238 | -0.107 |
| grad_cam | count | 0.0217 | 0.0123 | 0.144 |

## Against a random block (secondary, test splits pooled)

The same, against one compact random patch of the same size. Added after the pilot, so it does not affect the choice of map; it shows whether scattered deletion made the random map a harsher baseline.

| Map | Type | Mean difference | 95% lower bound | Standardized |
|---|---|---|---|---|
| cross_attention | presence | 0.0253 | 0.0158 | 0.171 |
| cross_attention | comp | 0.0016 | -0.0085 | 0.010 |
| cross_attention | area | 0.0233 | 0.0142 | 0.160 |
| cross_attention | count | 0.0511 | 0.0411 | 0.317 |
| rollout | presence | 0.0487 | 0.0386 | 0.310 |
| rollout | comp | 0.0376 | 0.0252 | 0.198 |
| rollout | area | 0.0488 | 0.0392 | 0.315 |
| rollout | count | 0.1297 | 0.1139 | 0.509 |
| grad_cam | presence | -0.0004 | -0.0082 | -0.003 |
| grad_cam | comp | 0.0084 | -0.0023 | 0.051 |
| grad_cam | area | 0.0138 | 0.0059 | 0.108 |
| grad_cam | count | 0.0765 | 0.0645 | 0.395 |

## Checks

Mean rank correlation; a map fails at or above the limit.

| Map | Trained vs random weights (limit 0.5) | Two questions, same tile (limit 0.9) | All-constant maps |
|---|---|---|---|
| cross_attention | 0.001 | 0.847 | 0 of 3829 |
| rollout | 0.077 | 0.967 | 0 of 3829 |
| grad_cam | 0.084 | 0.134 | 314 of 3829 |

## Agreement between maps

| Pair | Rank correlation | Overlap of top 20% (IoU) |
|---|---|---|
| cross_attention vs rollout | 0.471 | 0.282 |
| cross_attention vs grad_cam | 0.014 | 0.155 |
| rollout vs grad_cam | 0.018 | 0.165 |

## Deletion test: test

Effect: drop in the predicted class's probability, or for count the absolute change in log1p(count). Averages over questions; the sample is balanced between right and wrong answers.

| Type | Questions (right) | Map | 10% | 20% | 30% | 50% | Mean | Answer changed (mean) |
|---|---|---|---|---|---|---|---|---|
| presence | 449 (250) | cross_attention | 0.0184 | 0.0269 | 0.0339 | 0.0682 | 0.0368 | 11.6% |
| presence | 449 (250) | rollout | 0.0412 | 0.0647 | 0.0862 | 0.1281 | 0.0800 | 17.4% |
| presence | 449 (250) | grad_cam | 0.0074 | 0.0247 | 0.0398 | 0.0882 | 0.0400 | 11.1% |
| presence | 449 (250) | random | 0.0116 | 0.0212 | 0.0364 | 0.0806 | 0.0375 | 10.9% |
| presence | 449 (250) | random_block | 0.0084 | 0.0111 | 0.0247 | 0.0474 | 0.0229 | 7.9% |
| comp | 443 (250) | cross_attention | 0.0158 | 0.0166 | 0.0285 | 0.0536 | 0.0286 | 10.2% |
| comp | 443 (250) | rollout | 0.0457 | 0.0674 | 0.0881 | 0.1365 | 0.0844 | 17.4% |
| comp | 443 (250) | grad_cam | 0.0208 | 0.0474 | 0.0790 | 0.1215 | 0.0672 | 14.3% |
| comp | 443 (250) | random | 0.0289 | 0.0398 | 0.0574 | 0.1019 | 0.0570 | 13.1% |
| comp | 443 (250) | random_block | 0.0043 | 0.0110 | 0.0230 | 0.0565 | 0.0237 | 6.9% |
| area | 500 (250) | cross_attention | 0.0215 | 0.0336 | 0.0485 | 0.0767 | 0.0451 | 14.8% |
| area | 500 (250) | rollout | 0.0396 | 0.0678 | 0.0941 | 0.1517 | 0.0883 | 21.8% |
| area | 500 (250) | grad_cam | 0.0192 | 0.0416 | 0.0659 | 0.1178 | 0.0611 | 15.0% |
| area | 500 (250) | random | 0.0112 | 0.0287 | 0.0516 | 0.1005 | 0.0480 | 13.5% |
| area | 500 (250) | random_block | 0.0104 | 0.0232 | 0.0392 | 0.0771 | 0.0375 | 11.9% |
| count | 500 (250) | cross_attention | 0.0749 | 0.1113 | 0.1597 | 0.2513 | 0.1493 | 33.1% |
| count | 500 (250) | rollout | 0.1658 | 0.2566 | 0.3355 | 0.4929 | 0.3127 | 40.6% |
| count | 500 (250) | grad_cam | 0.0901 | 0.1565 | 0.2359 | 0.3908 | 0.2183 | 35.1% |
| count | 500 (250) | random | 0.0686 | 0.1172 | 0.1713 | 0.3434 | 0.1751 | 33.8% |
| count | 500 (250) | random_block | 0.0475 | 0.0790 | 0.1204 | 0.2213 | 0.1170 | 30.0% |

## Deletion test: test_phili

Effect: drop in the predicted class's probability, or for count the absolute change in log1p(count). Averages over questions; the sample is balanced between right and wrong answers.

| Type | Questions (right) | Map | 10% | 20% | 30% | 50% | Mean | Answer changed (mean) |
|---|---|---|---|---|---|---|---|---|
| presence | 475 (250) | cross_attention | 0.0434 | 0.0479 | 0.0591 | 0.0570 | 0.0518 | 15.2% |
| presence | 475 (250) | rollout | 0.0636 | 0.0642 | 0.0461 | 0.0518 | 0.0564 | 15.4% |
| presence | 475 (250) | grad_cam | -0.0115 | -0.0094 | -0.0054 | 0.0211 | -0.0013 | 8.2% |
| presence | 475 (250) | random | 0.0475 | 0.0587 | 0.0591 | 0.0658 | 0.0578 | 15.2% |
| presence | 475 (250) | random_block | 0.0024 | 0.0089 | 0.0170 | 0.0345 | 0.0157 | 7.9% |
| comp | 462 (250) | cross_attention | 0.0089 | 0.0099 | 0.0011 | -0.0014 | 0.0046 | 9.6% |
| comp | 462 (250) | rollout | 0.0449 | 0.0177 | 0.0116 | 0.0127 | 0.0217 | 11.4% |
| comp | 462 (250) | grad_cam | -0.0242 | -0.0255 | -0.0200 | -0.0064 | -0.0190 | 7.4% |
| comp | 462 (250) | random | 0.0307 | 0.0355 | 0.0239 | 0.0073 | 0.0244 | 11.1% |
| comp | 462 (250) | random_block | -0.0055 | -0.0010 | 0.0085 | 0.0231 | 0.0063 | 5.6% |
| area | 500 (250) | cross_attention | 0.0530 | 0.0603 | 0.0741 | 0.0832 | 0.0677 | 19.2% |
| area | 500 (250) | rollout | 0.0731 | 0.0705 | 0.0751 | 0.0834 | 0.0755 | 20.2% |
| area | 500 (250) | grad_cam | 0.0135 | 0.0222 | 0.0370 | 0.0581 | 0.0327 | 14.9% |
| area | 500 (250) | random | 0.0573 | 0.0717 | 0.0815 | 0.0936 | 0.0760 | 17.0% |
| area | 500 (250) | random_block | 0.0132 | 0.0216 | 0.0306 | 0.0497 | 0.0288 | 11.6% |
| count | 500 (250) | cross_attention | 0.1116 | 0.1410 | 0.1615 | 0.1911 | 0.1513 | 24.0% |
| count | 500 (250) | rollout | 0.1216 | 0.1341 | 0.1434 | 0.1813 | 0.1451 | 22.6% |
| count | 500 (250) | grad_cam | 0.0905 | 0.1235 | 0.1413 | 0.1766 | 0.1330 | 21.5% |
| count | 500 (250) | random | 0.0966 | 0.1235 | 0.1449 | 0.1667 | 0.1329 | 21.6% |
| count | 500 (250) | random_block | 0.0459 | 0.0623 | 0.0874 | 0.1298 | 0.0814 | 14.6% |

## Speed

0.009 s for one answer and its maps, on cuda.

10.0 min in total.

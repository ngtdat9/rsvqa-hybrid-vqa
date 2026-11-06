# XAI follow-up: hr_full_20261031-090319

Splits: test, test_phili. Deleted cells are replaced with: blur. Same selection of questions as the test run. This does not change the choice of map; it limits what may be claimed for it.

## How common no-data strips are

A no-data cell has at least 25% of its pixels black (at most 0.01 in every channel). A strip tile has at least one, so a large dark shadow could also count: check the strip figures.

| Split | Type | Questions | On strip tiles | Share | No-data cells per strip tile |
|---|---|---|---|---|---|
| test | presence | 449 | 11 | 2.4% | 16.5 |
| test | comp | 443 | 5 | 1.1% | 16.0 |
| test | area | 500 | 12 | 2.4% | 18.9 |
| test | count | 500 | 9 | 1.8% | 16.6 |
| test_phili | presence | 475 | 155 | 32.6% | 39.9 |
| test_phili | comp | 462 | 166 | 35.9% | 39.9 |
| test_phili | area | 500 | 162 | 32.4% | 38.1 |
| test_phili | count | 500 | 155 | 31.0% | 37.6 |

## Accuracy on strip tiles and other tiles

Over one question per tile and type (count: rounded prediction equals the truth), not the sample below, and not the full test sets. A difference here does not show that strips cause errors: strip tiles may also differ in other ways.

| Split | Type | Tiles | Questions | Accuracy |
|---|---|---|---|---|
| test | presence | strip tiles | 48 | 85.4% |
| test | presence | other tiles | 2178 | 91.2% |
| test | comp | strip tiles | 48 | 97.9% |
| test | comp | other tiles | 2178 | 91.2% |
| test | area | strip tiles | 48 | 83.3% |
| test | area | other tiles | 2178 | 86.1% |
| test | count | strip tiles | 48 | 60.4% |
| test | count | other tiles | 2178 | 66.5% |
| test_phili | presence | strip tiles | 362 | 79.8% |
| test_phili | presence | other tiles | 694 | 78.1% |
| test_phili | comp | strip tiles | 362 | 76.8% |
| test_phili | comp | other tiles | 694 | 81.6% |
| test_phili | area | strip tiles | 362 | 69.9% |
| test_phili | area | other tiles | 694 | 64.1% |
| test_phili | count | strip tiles | 362 | 62.7% |
| test_phili | count | other tiles | 694 | 58.4% |

## Occlusion without the strip tiles

- pooled: beats random on every type
- test: beats random on every type
- test_phili: beats random on every type

Per question, the effect averaged over deletion sizes, minus the random map's. "only strips" can rest on few questions.

| Split | Questions kept | Type | Questions | Mean difference | 95% lower bound | Standardized |
|---|---|---|---|---|---|---|
| pooled | all | presence | 924 | 0.1224 | 0.1122 | 0.776 |
| pooled | all | comp | 905 | 0.1427 | 0.1307 | 0.773 |
| pooled | all | area | 1000 | 0.1194 | 0.1103 | 0.812 |
| pooled | all | count | 1000 | 0.0999 | 0.0877 | 0.506 |
| pooled | without strips | presence | 758 | 0.1256 | 0.1142 | 0.785 |
| pooled | without strips | comp | 734 | 0.1535 | 0.1397 | 0.809 |
| pooled | without strips | area | 826 | 0.1252 | 0.1151 | 0.847 |
| pooled | without strips | count | 836 | 0.1135 | 0.0993 | 0.541 |
| pooled | only strips | presence | 166 | 0.1081 | 0.0859 | 0.740 |
| pooled | only strips | comp | 171 | 0.0968 | 0.0738 | 0.631 |
| pooled | only strips | area | 174 | 0.0920 | 0.0711 | 0.654 |
| pooled | only strips | count | 164 | 0.0307 | 0.0169 | 0.340 |
| test | all | presence | 449 | 0.1172 | 0.1031 | 0.772 |
| test | all | comp | 443 | 0.1494 | 0.1316 | 0.782 |
| test | all | area | 500 | 0.1265 | 0.1138 | 0.874 |
| test | all | count | 500 | 0.1448 | 0.1231 | 0.585 |
| test | without strips | presence | 438 | 0.1162 | 0.1021 | 0.768 |
| test | without strips | comp | 438 | 0.1507 | 0.1328 | 0.787 |
| test | without strips | area | 488 | 0.1268 | 0.1140 | 0.876 |
| test | without strips | count | 491 | 0.1459 | 0.1240 | 0.587 |
| test | only strips | presence | 11 | 0.1543 | 0.0516 | 0.888 |
| test | only strips | comp | 5 | 0.0288 | -0.0101 | 0.648 |
| test | only strips | area | 12 | 0.1115 | 0.0267 | 0.744 |
| test | only strips | count | 9 | 0.0822 | -0.0431 | 0.429 |
| test_phili | all | presence | 475 | 0.1274 | 0.1127 | 0.781 |
| test_phili | all | comp | 462 | 0.1364 | 0.1201 | 0.765 |
| test_phili | all | area | 500 | 0.1124 | 0.0993 | 0.754 |
| test_phili | all | count | 500 | 0.0551 | 0.0452 | 0.487 |
| test_phili | without strips | presence | 320 | 0.1383 | 0.1196 | 0.811 |
| test_phili | without strips | comp | 296 | 0.1575 | 0.1361 | 0.841 |
| test_phili | without strips | area | 338 | 0.1229 | 0.1066 | 0.807 |
| test_phili | without strips | count | 345 | 0.0673 | 0.0544 | 0.548 |
| test_phili | only strips | presence | 155 | 0.1048 | 0.0821 | 0.728 |
| test_phili | only strips | comp | 166 | 0.0988 | 0.0752 | 0.637 |
| test_phili | only strips | area | 162 | 0.0906 | 0.0690 | 0.646 |
| test_phili | only strips | count | 155 | 0.0277 | 0.0150 | 0.342 |

## Where the top 20% of cells fall on strip tiles

Share of a map's top cells in or beside no-data cells, against that area's share of the tile.

| Split | Map | Questions | Top cells in or beside no-data | Share of the tile |
|---|---|---|---|---|
| test | cross_attention | 37 | 22.2% | 12.7% |
| test | rollout | 37 | 29.4% | 12.7% |
| test | grad_cam | 37 | 16.5% | 12.7% |
| test | occlusion | 37 | 17.8% | 12.7% |
| test | random | 37 | 12.3% | 12.7% |
| test | random_block | 37 | 9.6% | 12.7% |
| test_phili | cross_attention | 638 | 27.8% | 21.7% |
| test_phili | rollout | 638 | 59.2% | 21.7% |
| test_phili | grad_cam | 638 | 22.6% | 21.7% |
| test_phili | occlusion | 638 | 52.2% | 21.7% |
| test_phili | random | 638 | 21.7% | 21.7% |
| test_phili | random_block | 638 | 19.2% | 21.7% |

## Edge control: tiles without strips

Strips lie along edges. If a map's top 20% of cells favour the outermost ring on tiles without strips too, part of the pull towards strips is a pull towards edges.

| Split | Map | Questions | Top cells on the outer ring | Share of the tile |
|---|---|---|---|---|
| test | cross_attention | 1855 | 24.7% | 23.4% |
| test | rollout | 1855 | 45.9% | 23.4% |
| test | grad_cam | 1855 | 24.2% | 23.4% |
| test | occlusion | 1855 | 26.6% | 23.4% |
| test | random | 1855 | 23.4% | 23.4% |
| test | random_block | 1855 | 15.9% | 23.4% |
| test_phili | cross_attention | 1299 | 26.1% | 23.4% |
| test_phili | rollout | 1299 | 52.2% | 23.4% |
| test_phili | grad_cam | 1299 | 23.0% | 23.4% |
| test_phili | occlusion | 1299 | 30.0% | 23.4% |
| test_phili | random | 1299 | 23.4% | 23.4% |
| test_phili | random_block | 1299 | 15.9% | 23.4% |

## All four maps under blur deletion

The first three were first tested with mean-colour deletion. Their two correlation checks do not depend on deletion, so they stand as first reported.

- `cross_attention`: does not beat random on: presence, comp
- `rollout`: beats random on every type
- `grad_cam`: does not beat random on: presence, comp, area
- `occlusion`: beats random on every type

## Against the random map, blur: pooled

Effect averaged over deletion sizes, minus the random map's.

| Map | Type | Mean difference | 95% lower bound | Standardized |
|---|---|---|---|---|
| cross_attention | presence | 0.0005 | -0.0070 | 0.004 |
| cross_attention | comp | -0.0158 | -0.0232 | -0.139 |
| cross_attention | area | 0.0089 | 0.0008 | 0.068 |
| cross_attention | count | 0.0136 | 0.0037 | 0.085 |
| rollout | presence | 0.0249 | 0.0163 | 0.188 |
| rollout | comp | 0.0160 | 0.0075 | 0.123 |
| rollout | area | 0.0232 | 0.0150 | 0.176 |
| rollout | count | 0.0804 | 0.0701 | 0.482 |
| grad_cam | presence | -0.0011 | -0.0071 | -0.012 |
| grad_cam | comp | 0.0040 | -0.0038 | 0.034 |
| grad_cam | area | 0.0028 | -0.0032 | 0.029 |
| grad_cam | count | 0.0287 | 0.0205 | 0.217 |
| occlusion | presence | 0.1224 | 0.1122 | 0.776 |
| occlusion | comp | 0.1427 | 0.1307 | 0.773 |
| occlusion | area | 0.1194 | 0.1103 | 0.812 |
| occlusion | count | 0.0999 | 0.0877 | 0.506 |

## Against the random map, blur: test

Effect averaged over deletion sizes, minus the random map's.

| Map | Type | Mean difference | 95% lower bound | Standardized |
|---|---|---|---|---|
| cross_attention | presence | 0.0034 | -0.0070 | 0.030 |
| cross_attention | comp | -0.0204 | -0.0314 | -0.174 |
| cross_attention | area | 0.0070 | -0.0029 | 0.062 |
| cross_attention | count | -0.0296 | -0.0445 | -0.175 |
| rollout | presence | 0.0355 | 0.0240 | 0.284 |
| rollout | comp | 0.0279 | 0.0157 | 0.213 |
| rollout | area | 0.0300 | 0.0199 | 0.262 |
| rollout | count | 0.1009 | 0.0837 | 0.514 |
| grad_cam | presence | 0.0100 | 0.0022 | 0.118 |
| grad_cam | comp | 0.0198 | 0.0078 | 0.154 |
| grad_cam | area | 0.0156 | 0.0076 | 0.171 |
| grad_cam | count | 0.0401 | 0.0265 | 0.257 |
| occlusion | presence | 0.1172 | 0.1031 | 0.772 |
| occlusion | comp | 0.1494 | 0.1316 | 0.782 |
| occlusion | area | 0.1265 | 0.1138 | 0.874 |
| occlusion | count | 0.1448 | 0.1231 | 0.585 |

## Against the random map, blur: test_phili

Effect averaged over deletion sizes, minus the random map's.

| Map | Type | Mean difference | 95% lower bound | Standardized |
|---|---|---|---|---|
| cross_attention | presence | -0.0023 | -0.0130 | -0.019 |
| cross_attention | comp | -0.0114 | -0.0214 | -0.104 |
| cross_attention | area | 0.0109 | -0.0021 | 0.074 |
| cross_attention | count | 0.0568 | 0.0448 | 0.415 |
| rollout | presence | 0.0148 | 0.0023 | 0.107 |
| rollout | comp | 0.0045 | -0.0071 | 0.035 |
| rollout | area | 0.0164 | 0.0035 | 0.112 |
| rollout | count | 0.0600 | 0.0488 | 0.470 |
| grad_cam | presence | -0.0117 | -0.0206 | -0.119 |
| grad_cam | comp | -0.0110 | -0.0210 | -0.101 |
| grad_cam | area | -0.0101 | -0.0187 | -0.102 |
| grad_cam | count | 0.0173 | 0.0083 | 0.169 |
| occlusion | presence | 0.1274 | 0.1127 | 0.781 |
| occlusion | comp | 0.1364 | 0.1201 | 0.765 |
| occlusion | area | 0.1124 | 0.0993 | 0.754 |
| occlusion | count | 0.0551 | 0.0452 | 0.487 |

## Against a random block, blur (secondary, pooled)

The same, against one compact random patch.

| Map | Type | Mean difference | 95% lower bound | Standardized |
|---|---|---|---|---|
| cross_attention | presence | -0.0002 | -0.0085 | -0.001 |
| cross_attention | comp | -0.0212 | -0.0298 | -0.161 |
| cross_attention | area | 0.0191 | 0.0105 | 0.137 |
| cross_attention | count | 0.0302 | 0.0213 | 0.209 |
| rollout | presence | 0.0242 | 0.0147 | 0.164 |
| rollout | comp | 0.0105 | -0.0010 | 0.060 |
| rollout | area | 0.0333 | 0.0242 | 0.228 |
| rollout | count | 0.0971 | 0.0843 | 0.471 |
| grad_cam | presence | -0.0018 | -0.0088 | -0.017 |
| grad_cam | comp | -0.0014 | -0.0112 | -0.009 |
| grad_cam | area | 0.0129 | 0.0063 | 0.121 |
| grad_cam | count | 0.0454 | 0.0348 | 0.267 |
| occlusion | presence | 0.1217 | 0.1109 | 0.723 |
| occlusion | comp | 0.1373 | 0.1240 | 0.673 |
| occlusion | area | 0.1296 | 0.1196 | 0.802 |
| occlusion | count | 0.1166 | 0.1006 | 0.453 |

## All-constant maps

| Map | All-constant |
|---|---|
| cross_attention | 0 of 3829 |
| rollout | 0 of 3829 |
| grad_cam | 314 of 3829 |
| occlusion | 3 of 3829 |

## Deletion test: test

Effect: drop in the predicted class's probability, or for count the absolute change in log1p(count). Averages over questions; the sample is balanced between right and wrong answers.

| Type | Questions (right) | Map | 10% | 20% | 30% | 50% | Mean | Answer changed (mean) |
|---|---|---|---|---|---|---|---|---|
| presence | 449 (250) | cross_attention | 0.0167 | 0.0197 | 0.0259 | 0.0544 | 0.0292 | 8.3% |
| presence | 449 (250) | rollout | 0.0323 | 0.0517 | 0.0661 | 0.0950 | 0.0613 | 13.5% |
| presence | 449 (250) | grad_cam | 0.0028 | 0.0215 | 0.0366 | 0.0822 | 0.0358 | 10.4% |
| presence | 449 (250) | occlusion | 0.1109 | 0.1417 | 0.1565 | 0.1624 | 0.1429 | 25.7% |
| presence | 449 (250) | random | 0.0080 | 0.0147 | 0.0240 | 0.0562 | 0.0257 | 8.2% |
| presence | 449 (250) | random_block | 0.0117 | 0.0191 | 0.0278 | 0.0535 | 0.0280 | 8.5% |
| comp | 443 (250) | cross_attention | 0.0119 | 0.0108 | 0.0180 | 0.0444 | 0.0213 | 7.5% |
| comp | 443 (250) | rollout | 0.0348 | 0.0563 | 0.0746 | 0.1130 | 0.0697 | 14.8% |
| comp | 443 (250) | grad_cam | 0.0220 | 0.0435 | 0.0700 | 0.1104 | 0.0615 | 12.4% |
| comp | 443 (250) | occlusion | 0.1641 | 0.1952 | 0.2046 | 0.2004 | 0.1911 | 29.3% |
| comp | 443 (250) | random | 0.0122 | 0.0252 | 0.0363 | 0.0933 | 0.0417 | 9.7% |
| comp | 443 (250) | random_block | 0.0052 | 0.0213 | 0.0365 | 0.0705 | 0.0334 | 6.9% |
| area | 500 (250) | cross_attention | 0.0190 | 0.0317 | 0.0437 | 0.0649 | 0.0398 | 14.7% |
| area | 500 (250) | rollout | 0.0286 | 0.0503 | 0.0713 | 0.1009 | 0.0628 | 17.4% |
| area | 500 (250) | grad_cam | 0.0131 | 0.0333 | 0.0543 | 0.0931 | 0.0485 | 12.8% |
| area | 500 (250) | occlusion | 0.1349 | 0.1596 | 0.1697 | 0.1731 | 0.1593 | 32.8% |
| area | 500 (250) | random | 0.0058 | 0.0176 | 0.0340 | 0.0740 | 0.0328 | 10.9% |
| area | 500 (250) | random_block | 0.0076 | 0.0170 | 0.0292 | 0.0686 | 0.0306 | 11.1% |
| count | 500 (250) | cross_attention | 0.0665 | 0.0969 | 0.1345 | 0.2526 | 0.1376 | 31.0% |
| count | 500 (250) | rollout | 0.1342 | 0.2133 | 0.2974 | 0.4275 | 0.2681 | 37.8% |
| count | 500 (250) | grad_cam | 0.0741 | 0.1437 | 0.2324 | 0.3793 | 0.2074 | 32.4% |
| count | 500 (250) | occlusion | 0.1856 | 0.2655 | 0.3330 | 0.4641 | 0.3120 | 38.0% |
| count | 500 (250) | random | 0.0519 | 0.1091 | 0.1760 | 0.3319 | 0.1672 | 31.3% |
| count | 500 (250) | random_block | 0.0510 | 0.0916 | 0.1445 | 0.2464 | 0.1334 | 28.1% |

## Deletion test: test_phili

Effect: drop in the predicted class's probability, or for count the absolute change in log1p(count). Averages over questions; the sample is balanced between right and wrong answers.

| Type | Questions (right) | Map | 10% | 20% | 30% | 50% | Mean | Answer changed (mean) |
|---|---|---|---|---|---|---|---|---|
| presence | 475 (250) | cross_attention | 0.0035 | 0.0046 | 0.0124 | 0.0183 | 0.0097 | 10.6% |
| presence | 475 (250) | rollout | 0.0172 | 0.0265 | 0.0276 | 0.0356 | 0.0267 | 13.9% |
| presence | 475 (250) | grad_cam | -0.0088 | -0.0051 | -0.0010 | 0.0160 | 0.0003 | 8.5% |
| presence | 475 (250) | occlusion | 0.1226 | 0.1455 | 0.1484 | 0.1409 | 0.1394 | 25.9% |
| presence | 475 (250) | random | 0.0055 | 0.0038 | 0.0139 | 0.0247 | 0.0120 | 7.2% |
| presence | 475 (250) | random_block | 0.0034 | 0.0064 | 0.0137 | 0.0210 | 0.0111 | 6.7% |
| comp | 462 (250) | cross_attention | -0.0059 | -0.0099 | -0.0067 | -0.0020 | -0.0061 | 7.8% |
| comp | 462 (250) | rollout | 0.0085 | 0.0091 | 0.0077 | 0.0137 | 0.0097 | 9.4% |
| comp | 462 (250) | grad_cam | -0.0133 | -0.0115 | -0.0071 | 0.0089 | -0.0057 | 6.4% |
| comp | 462 (250) | occlusion | 0.1271 | 0.1472 | 0.1502 | 0.1423 | 0.1417 | 21.7% |
| comp | 462 (250) | random | 0.0038 | -0.0008 | 0.0047 | 0.0135 | 0.0053 | 5.8% |
| comp | 462 (250) | random_block | 0.0038 | 0.0135 | 0.0259 | 0.0523 | 0.0239 | 4.8% |
| area | 500 (250) | cross_attention | 0.0235 | 0.0383 | 0.0450 | 0.0509 | 0.0394 | 19.6% |
| area | 500 (250) | rollout | 0.0321 | 0.0433 | 0.0500 | 0.0543 | 0.0449 | 19.9% |
| area | 500 (250) | grad_cam | 0.0050 | 0.0097 | 0.0195 | 0.0398 | 0.0185 | 12.3% |
| area | 500 (250) | occlusion | 0.1250 | 0.1448 | 0.1517 | 0.1423 | 0.1410 | 28.3% |
| area | 500 (250) | random | 0.0122 | 0.0240 | 0.0313 | 0.0467 | 0.0285 | 12.6% |
| area | 500 (250) | random_block | 0.0017 | 0.0061 | 0.0123 | 0.0218 | 0.0105 | 10.6% |
| count | 500 (250) | cross_attention | 0.0919 | 0.1296 | 0.1537 | 0.1701 | 0.1363 | 20.0% |
| count | 500 (250) | rollout | 0.1009 | 0.1311 | 0.1514 | 0.1748 | 0.1395 | 20.8% |
| count | 500 (250) | grad_cam | 0.0575 | 0.0847 | 0.1046 | 0.1406 | 0.0969 | 17.8% |
| count | 500 (250) | occlusion | 0.0999 | 0.1273 | 0.1448 | 0.1664 | 0.1346 | 21.9% |
| count | 500 (250) | random | 0.0419 | 0.0631 | 0.0878 | 0.1256 | 0.0796 | 14.3% |
| count | 500 (250) | random_block | 0.0379 | 0.0672 | 0.0880 | 0.1271 | 0.0801 | 14.3% |

75.4 min in total.

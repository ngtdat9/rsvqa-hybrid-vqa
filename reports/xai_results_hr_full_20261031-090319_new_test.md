# XAI results: hr\_full\_20261031-090319

Splits: test, test\_phili. Deleted cells are replaced with: blur.

**Map for the RAG layer: occlusion**

- `occlusion`: qualifies, mean standardized effect 0.716

## Against the random map (scored splits pooled; this decides the choice)

Per question, the effect averaged over deletion sizes, minus the random map's.

| Map | Type | Mean difference | 95% lower bound | Standardized |
| :---- | :---- | :---- | :---- | :---- |
| occlusion | presence | 0.1186 | 0.1088 | 0.779 |
| occlusion | comp | 0.1454 | 0.1334 | 0.791 |
| occlusion | area | 0.1196 | 0.1103 | 0.796 |
| occlusion | count | 0.0956 | 0.0837 | 0.500 |

## Against a random block (secondary, scored splits pooled)

The same, against one compact random patch of the same size. Added after the pilot, so it does not affect the choice of map; it shows whether scattered deletion made the random map a harsher baseline.

| Map | Type | Mean difference | 95% lower bound | Standardized |
| :---- | :---- | :---- | :---- | :---- |
| occlusion | presence | 0.1216 | 0.1108 | 0.728 |
| occlusion | comp | 0.1376 | 0.1241 | 0.664 |
| occlusion | area | 0.1223 | 0.1122 | 0.750 |
| occlusion | count | 0.1208 | 0.1046 | 0.461 |

## Against the random map: test alone (does not decide the choice)

The same as the pooled table, for this split only.

| Map | Type | Mean difference | 95% lower bound | Standardized |
| :---- | :---- | :---- | :---- | :---- |
| occlusion | presence | 0.1150 | 0.1011 | 0.766 |
| occlusion | comp | 0.1530 | 0.1351 | 0.797 |
| occlusion | area | 0.1257 | 0.1128 | 0.852 |
| occlusion | count | 0.1374 | 0.1166 | 0.579 |

## Against the random map: test\_phili alone (does not decide the choice)

The same as the pooled table, for this split only.

| Map | Type | Mean difference | 95% lower bound | Standardized |
| :---- | :---- | :---- | :---- | :---- |
| occlusion | presence | 0.1220 | 0.1081 | 0.791 |
| occlusion | comp | 0.1381 | 0.1221 | 0.786 |
| occlusion | area | 0.1134 | 0.1000 | 0.742 |
| occlusion | count | 0.0538 | 0.0437 | 0.467 |

## Checks

Mean rank correlation; a map fails at or above the limit.

| Map | Trained vs random weights (limit 0.5) | Two questions, same tile (limit 0.9) | All-constant maps |
| :---- | :---- | :---- | :---- |
| occlusion | 0.024 | 0.114 | 3 of 3829 |

## Deletion test: test

Effect: drop in the predicted class's probability, or for count the absolute change in log1p(count). Averages over questions; the sample is balanced between right and wrong answers.

| Type | Questions (right) | Map | 10% | 20% | 30% | 50% | Mean | Answer changed (mean) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| presence | 449 (250) | occlusion | 0.1109 | 0.1416 | 0.1565 | 0.1616 | 0.1427 | 25.7% |
| presence | 449 (250) | random | 0.0098 | 0.0142 | 0.0259 | 0.0606 | 0.0276 | 8.5% |
| presence | 449 (250) | random\_block | 0.0081 | 0.0167 | 0.0277 | 0.0565 | 0.0272 | 8.9% |
| comp | 443 (250) | occlusion | 0.1641 | 0.1952 | 0.2046 | 0.2003 | 0.1911 | 29.3% |
| comp | 443 (250) | random | 0.0065 | 0.0240 | 0.0415 | 0.0805 | 0.0381 | 9.8% |
| comp | 443 (250) | random\_block | 0.0095 | 0.0211 | 0.0352 | 0.0732 | 0.0347 | 7.5% |
| area | 500 (250) | occlusion | 0.1349 | 0.1596 | 0.1697 | 0.1731 | 0.1593 | 32.8% |
| area | 500 (250) | random | 0.0052 | 0.0186 | 0.0354 | 0.0752 | 0.0336 | 10.7% |
| area | 500 (250) | random\_block | 0.0089 | 0.0236 | 0.0390 | 0.0735 | 0.0363 | 12.0% |
| count | 500 (250) | occlusion | 0.1856 | 0.2655 | 0.3330 | 0.4641 | 0.3120 | 38.0% |
| count | 500 (250) | random | 0.0637 | 0.1175 | 0.1860 | 0.3314 | 0.1746 | 32.0% |
| count | 500 (250) | random\_block | 0.0510 | 0.0892 | 0.1344 | 0.2425 | 0.1293 | 30.1% |

## Deletion test: test\_phili

Effect: drop in the predicted class's probability, or for count the absolute change in log1p(count). Averages over questions; the sample is balanced between right and wrong answers.

| Type | Questions (right) | Map | 10% | 20% | 30% | 50% | Mean | Answer changed (mean) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| presence | 475 (250) | occlusion | 0.1226 | 0.1455 | 0.1485 | 0.1409 | 0.1394 | 25.9% |
| presence | 475 (250) | random | 0.0062 | 0.0126 | 0.0185 | 0.0323 | 0.0174 | 8.9% |
| presence | 475 (250) | random\_block | 0.0013 | 0.0122 | 0.0116 | 0.0229 | 0.0120 | 6.7% |
| comp | 462 (250) | occlusion | 0.1271 | 0.1472 | 0.1502 | 0.1423 | 0.1417 | 21.7% |
| comp | 462 (250) | random | 0.0029 | 0.0014 | 0.0006 | 0.0095 | 0.0036 | 6.3% |
| comp | 462 (250) | random\_block | 0.0046 | 0.0125 | 0.0252 | 0.0461 | 0.0221 | 5.9% |
| area | 500 (250) | occlusion | 0.1250 | 0.1448 | 0.1517 | 0.1424 | 0.1410 | 28.3% |
| area | 500 (250) | random | 0.0132 | 0.0230 | 0.0289 | 0.0450 | 0.0275 | 11.9% |
| area | 500 (250) | random\_block | 0.0076 | 0.0149 | 0.0234 | 0.0320 | 0.0195 | 11.6% |
| count | 500 (250) | occlusion | 0.0999 | 0.1273 | 0.1448 | 0.1664 | 0.1346 | 21.9% |
| count | 500 (250) | random | 0.0422 | 0.0672 | 0.0872 | 0.1267 | 0.0808 | 15.2% |
| count | 500 (250) | random\_block | 0.0363 | 0.0590 | 0.0835 | 0.1241 | 0.0757 | 13.9% |

## Speed

1.480 s for one answer and its maps, on cuda.

96.9 min in total.
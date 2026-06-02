# tsfresh Feature Extraction and Selection

## 결론부터

tsfresh는 시계열에서 많은 feature를 자동 추출하고, target과 통계적으로 관련 있어 보이는 feature만 p-value와 multiple testing correction으로 선택합니다.

## FRESH 알고리즘

1. raw time series를 feature matrix로 변환합니다.
2. 각 feature와 target의 관련성을 통계 검정합니다.
3. feature가 많아서 생기는 false discovery를 multiple testing correction으로 통제합니다.

## p-value 해석

p-value가 작다는 것은 target과 관련 있어 보인다는 뜻이지, 고장의 원인이라는 뜻은 아닙니다.

## train-only 원칙

`select_features`는 label을 보기 때문에 train set에서만 수행해야 합니다.

위험한 방식:

```text
전체 데이터 feature extraction
→ 전체 label로 feature selection
→ train/test split
```

안전한 방식:

```text
unit/run split
→ train에서 feature selection
→ 선택 feature를 validation/test에 적용
```

# UCR / UEA Time-Series Classification Benchmark

## 결론부터

UCR Archive는 시계열 분류 알고리즘을 공정하게 비교하기 위한 공통 시험지입니다.

## 핵심 역할

| 역할 | 의미 |
|---|---|
| 표준 dataset | 모델을 같은 문제에서 비교 |
| fixed train/test split | 재현성 확보 |
| baseline | 1-NN Euclidean, 1-NN DTW 등과 비교 |
| benchmark 문화 | cherry-picking 방지 |

## 설비 진단과의 차이

UCR fixed split은 재현성에는 좋지만, 설비 진단에서는 unit/run/time/condition split이 더 중요합니다.

## 핵심 주의

- UCR accuracy가 높다고 실제 설비 현장 성능이 보장되지는 않습니다.
- 설비 데이터는 결측, drift, class imbalance, domain shift가 큽니다.
- window random split은 인접 window leakage를 만들 수 있습니다.

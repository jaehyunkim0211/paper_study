# Transformer forecasting 실무 원칙

Transformer 기반 시계열 모델은 긴 history를 볼 수 있지만 계산량, covariate leakage, window split leakage에 특히 주의해야 합니다.

## 핵심 원칙

- Self-attention은 기본적으로 O(L²) 비용입니다.
- 긴 sequence에서는 sparse attention, decomposition, frequency domain, patching 등 효율화가 필요합니다.
- Future-known covariate와 future-unknown covariate를 구분해야 합니다.
- Point forecast만으로는 예지보전에 부족할 수 있어 uncertainty 보완이 필요합니다.

## 실무 split

- 같은 설비 미래 예측: 과거 train, 미래 test
- 새 설비 일반화: unit holdout
- 새 운전 조건: condition holdout
- RUL: bearing/engine unit split
- 이상탐지: 정상 train, 미래 event test

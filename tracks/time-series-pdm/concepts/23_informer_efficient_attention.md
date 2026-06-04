# Informer: efficient attention

Informer는 long sequence time-series forecasting을 위해 ProbSparse attention, self-attention distilling, generative-style decoder를 제안한 모델입니다.

## 핵심

- Vanilla attention은 O(L²) 비용입니다.
- ProbSparse attention은 active query에 계산을 집중합니다.
- Self-attention distilling은 encoder 길이를 줄입니다.
- Generative-style decoder는 긴 미래 horizon을 한 번에 출력합니다.

## 설비 활용

온도, 압력, 전류 RMS, 진동 RMS 같은 긴 저주파 센서의 장기 예측에 사용할 수 있습니다.

## 주의

Sparse attention이 작은 초기 이상 신호를 놓칠 수 있으므로 recall, missed detection, lead time을 확인해야 합니다. 미래 실제 부하를 covariate로 넣으면 leakage입니다.

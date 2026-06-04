# 시계열 Foundation Model과 TimeGPT

TimeGPT는 대규모 다양한 시계열로 사전학습한 foundation model을 zero-shot 또는 fine-tuning 방식으로 활용하는 흐름의 출발점입니다.

## 핵심

- Zero-shot forecasting: 내 데이터로 학습하지 않고 바로 예측합니다.
- Fine-tuning: 사전학습 모델을 내 설비 데이터에 추가 적응시킵니다.
- Prediction interval: conformal prediction 등으로 불확실성을 제공합니다.
- Anomaly detection: 실제값이 예측구간 밖으로 벗어나면 이상 후보입니다.

## 설비 활용

온도, 압력, 전류 RMS, 진동 RMS, health index 예측의 빠른 baseline으로 유용합니다. Raw vibration fault classification보다는 feature sequence forecasting에 더 적합합니다.

## 주의

Black-box 성격이 강하므로 local baseline과 비교해야 합니다. 미래 실제 부하, 미래 실제 전류, 고장 여부 같은 future-unknown covariate를 넣으면 leakage입니다. Prediction interval calibration을 반드시 검증해야 합니다.

# Forecasting Benchmark Metrics: M4, sMAPE, MASE, OWA

## 결론부터

Forecasting 모델은 복잡한지보다 단순 baseline보다 실제로 얼마나 나은지로 평가해야 합니다.

## sMAPE

```math
sMAPE = (1/h) sum 2|y_t - yhat_t| / (|y_t| + |yhat_t|)
```

오차를 실제값과 예측값의 scale로 나눠 비교합니다.

## MASE

```math
MASE = model_error / naive_error
```

내 모델이 naive 또는 seasonal naive baseline 대비 얼마나 나은지 봅니다.

## OWA

```math
OWA = 0.5 * (sMAPE_model/sMAPE_Naive2 + MASE_model/MASE_Naive2)
```

M4 Competition의 종합 지표입니다. 1보다 작으면 Naive2보다 좋다는 뜻입니다.

## 설비 예측 적용

- last value, moving average, seasonal naive, ARIMA, XGBoost를 baseline으로 둡니다.
- 딥러닝/Transformer가 이 baseline을 올바른 split에서 이기는지 확인합니다.
- 예측 residual을 이상탐지 score로 바꿀 때는 false alarm, missed detection, detection delay를 봅니다.

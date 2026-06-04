# 23편: TimeGPT-1

## 결론부터

**시계열 예측도 LLM처럼 대규모 다양한 시계열 데이터로 사전학습한 foundation model을 만들 수 있고, 새로운 시계열에 대해 zero-shot 또는 fine-tuning 방식으로 빠르게 예측할 수 있습니다.**

## 이 논문을 한 문장으로 요약하면

TimeGPT-1은 다양한 도메인의 대규모 시계열로 사전학습한 Transformer 기반 시계열 foundation model이며, 개별 데이터셋마다 모델을 처음부터 학습하지 않고도 forecasting, prediction interval, anomaly detection에 활용할 수 있음을 보여준 논문입니다.

## 왜 이 논문이 필요했나?

기존 forecasting pipeline은 데이터 처리, 모델 선택, hyperparameter tuning, 학습이 필요합니다. 설비마다 온도, 전류, 압력, 진동 RMS의 주기와 scale이 달라 매번 모델을 만들기 어렵습니다. TimeGPT는 이미 다양한 시계열 패턴을 학습한 foundation model을 가져와 zero-shot 또는 fine-tuning으로 빠르게 예측하는 흐름을 제안합니다.

## 핵심 개념

### 1. Foundation model

대규모 시계열 데이터로 사전학습한 뒤 여러 forecasting task에 전이해서 쓰는 모델입니다.

```text
대규모 시계열 사전학습
→ 온도 예측 / 압력 예측 / 이상탐지 / health feature 예측
```

### 2. Zero-shot forecasting

내 설비 데이터로 모델 parameter를 다시 학습하지 않고, 과거 history를 입력해 바로 미래를 예측합니다. 빠른 baseline으로 유용합니다.

### 3. Fine-tuning

사전학습된 모델을 내 공장 또는 내 설비 데이터에 조금 더 맞추는 과정입니다. 하지만 train/validation/test split이 먼저이며, test를 보고 fine-tuning하면 leakage입니다.

### 4. Transformer 기반 encoder-decoder

TimeGPT는 시계열용 Transformer 모델입니다. 기존 LLM을 그대로 가져온 것이 아니라 시계열 예측 오류를 줄이도록 학습된 모델로 이해해야 합니다.

### 5. 대규모 사전학습 데이터

TimeGPT 논문은 finance, economics, weather, IoT sensor, energy, sales, transport 등 다양한 domain의 대규모 시계열 data point로 학습했다고 설명합니다. 따라서 다양한 trend, seasonality, noise, outlier 패턴을 배웠을 가능성이 있습니다.

### 6. Prediction interval

예측값 하나뿐 아니라 conformal prediction 기반 prediction interval을 제공합니다. 설비 예지보전에서는 upper quantile이 안전 기준을 넘는지 보는 것이 중요합니다.

### 7. Anomaly detection

예측구간 밖으로 실제값이 벗어나거나 예측 오차가 크면 anomaly 후보입니다. 단, setpoint 변경, 운전 mode 변경, sensor issue를 함께 확인해야 합니다.

### 8. Exogenous variables

Load, speed, setpoint, shift, ambient temperature 등을 넣을 수 있습니다. 그러나 미래 실제 부하, 미래 실제 전류, 고장 여부처럼 prediction 시점에 모르는 feature를 넣으면 leakage입니다.

### 9. Black-box 성격

편리하지만 내부 feature와 decision 근거가 제한적으로만 보일 수 있습니다. 도메인 feature, 운전 로그, 정비 로그와 함께 검증해야 합니다.

## 작은 예시

모터 온도 history를 입력해 5분 뒤까지 예측합니다.

| time | forecast | lower | upper |
|---|---:|---:|---:|
| 09:05 | 71.4 | 71.0 | 71.9 |
| 09:06 | 71.8 | 71.2 | 72.5 |
| 09:07 | 72.1 | 71.3 | 73.0 |

Upper bound가 안전 기준에 가까워지면 보수적 경보를 고려할 수 있습니다. 실제값이 upper 밖으로 크게 벗어나면 이상 후보입니다.

## 실무에서 어떻게 써야 하나?

TimeGPT는 빠른 forecasting/anomaly baseline으로 좋습니다. Last value, seasonal naive, ARIMA/STL, TCN, N-BEATS, PatchTST, DeepAR와 같은 split에서 비교해야 합니다.

Raw 고주파 vibration fault classification에는 직접 쓰기보다 RMS, kurtosis, envelope peak, FFT band energy, health index 같은 feature sequence를 예측하는 것이 더 실무적입니다.

Exogenous variable availability table을 만들어 static, past-known, future-known, future-unknown을 구분해야 합니다.

## 데이터 분할

Foundation model이어도 split이 먼저입니다. Window random split, 전체 scaling, 같은 설비/run 혼합, future-unknown covariate 사용, test threshold tuning, RUL cycle random split은 성능을 부풀립니다.

## 평가 지표

Forecasting은 MAE, RMSE, sMAPE, MASE 또는 seasonal naive 대비 상대오차를 봅니다. Prediction interval은 coverage와 calibration을 봅니다. 이상탐지는 precision, recall, F1, false alarm, missed detection, detection delay, event-level recall을 봅니다. RUL은 MAE, RMSE, NASA scoring function, early/late penalty를 봅니다.

## 한계

Black-box API 모델로 쓰는 경우가 많아 내부 해석이 제한적입니다. Zero-shot이 항상 local model을 이기는 것은 아닙니다. Covariate leakage에 취약합니다. Prediction interval calibration을 반드시 검증해야 합니다. Raw vibration classifier라기보다 forecasting foundation model입니다.

## 이전 논문들과 연결

| 논문 | 차이 |
|---|---|
| DeepAR | 내 데이터에서 global probabilistic model 학습 |
| PatchTST | 내 데이터로 patch Transformer 학습 |
| TimeGPT | 사전학습된 foundation model을 zero-shot/fine-tuning으로 사용 |

## 오늘 공부용 요약

TimeGPT는 시계열 foundation model 흐름의 출발점입니다. 설비 데이터에서는 온도, 압력, 전류 RMS, 진동 RMS, health index 예측의 빠른 baseline으로 쓸 수 있습니다. 하지만 data split, covariate leakage, prediction interval calibration, false alarm/missed detection 검증이 반드시 필요합니다.

## 다음 논문 예고

다음은 **24편: TimesFM / A Decoder-Only Foundation Model for Time-Series Forecasting**입니다.

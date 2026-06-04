# 16편: DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks

## 결론부터

**시계열 예측을 개별 센서나 개별 설비마다 따로 모델링하지 말고, 여러 관련 시계열을 하나의 global RNN 모델로 함께 학습하면 더 안정적인 확률 예측을 만들 수 있습니다.**

## 이 논문을 한 문장으로 요약하면

DeepAR는 많은 관련 시계열을 함께 학습하는 autoregressive LSTM 기반 probabilistic forecasting 모델로, 미래값 하나를 점으로 예측하는 대신 미래값의 확률분포와 sample path를 생성합니다.

## 왜 이 논문이 필요했나?

전통 forecasting은 설비 A 모델, 설비 B 모델처럼 개별 시계열마다 모델을 따로 맞추는 경우가 많습니다. 그러나 같은 종류 모터나 펌프가 수백 대 있다면 공통 패턴이 있습니다. DeepAR는 여러 시계열을 하나의 global model로 학습해 공통 패턴을 공유합니다.

설비 예지보전에서는 새 설비 history가 짧거나, 여러 설비의 온도/전류/진동 RMS를 함께 예측해야 할 때 유용합니다.

## 핵심 개념

### 1. Global model

```text
설비 A, B, C, D의 센서 시계열 전체
→ 하나의 DeepAR 모델
→ 각 설비별 미래 예측 분포
```

개별 ARIMA 모델보다 유사 설비의 패턴을 공유할 수 있습니다.

### 2. Probabilistic forecasting

점 예측은 “10분 뒤 온도 75도”라고 말합니다. 확률 예측은 “10분 뒤 온도는 72~78도 범위일 가능성이 큼”이라고 말합니다. 예지보전에서는 upper quantile, prediction interval, 위험 확률이 중요합니다.

### 3. Autoregressive

다음 값을 예측할 때 직전 target 값을 입력으로 씁니다. 예측 구간에서는 실제 미래값이 없으므로 모델이 sample한 값을 다음 step 입력으로 사용합니다.


h_{i,t} = h(h_{i,t-1}, z_{i,t-1}, x_{i,t}, \Theta)


### 4. Conditioning range와 prediction range

Conditioning range는 실제 과거값을 볼 수 있는 구간입니다. Prediction range는 미래 예측 구간입니다. Prediction range의 실제값이나 실제 미래 covariate를 쓰면 leakage입니다.

### 5. Covariates

시간대, 요일, 설비 타입, 계획된 setpoint, load schedule 같은 변수를 넣을 수 있습니다. 하지만 미래 실제 부하, 미래 실제 전류, 고장 여부는 prediction 시점에 모르므로 넣으면 안 됩니다.

### 6. Likelihood

DeepAR는 target이 특정 분포에서 나온다고 보고 분포 parameter를 예측합니다. 온도/압력은 Gaussian 또는 Student-t, count는 negative binomial, binary event는 Bernoulli 등을 고려할 수 있습니다.

### 7. Sampling

예측분포에서 여러 sample path를 만들고 P10/P50/P90 같은 quantile을 계산합니다.

### 8. Calibration

90% 예측구간이면 실제값이 대략 90% 들어와야 합니다. Coverage와 calibration이 좋지 않으면 risk 판단이 위험합니다.

## 작은 예시

모터 A/B/C의 온도와 부하가 있습니다.

| motor | load | temperature |
|---|---:|---:|
| A | 40 | 70.0 |
| B | 40 | 68.0 |
| C | 70 | 82.0 |

DeepAR는 여러 motor 전체에서 “부하 증가 후 온도 상승” 패턴을 학습하고, 각 motor의 미래 온도 분포를 예측합니다.

## 실무에서 어떻게 써야 하나?

여러 관련 설비가 있을 때 baseline으로 사용합니다. 단일 설비 하나만 있으면 ARIMA, TCN, N-BEATS가 더 자연스러울 수 있습니다. Prediction interval을 경보 기준으로 쓰고, covariate leakage를 엄격히 막아야 합니다.

## 데이터 분할

같은 설비의 미래 예측은 과거 train, 미래 test입니다. 새 설비 일반화는 unit holdout입니다. RUL은 bearing/engine unit split입니다. Window random split은 금지입니다.

## 한계

관련 시계열이 많을 때 강합니다. Autoregressive sampling은 horizon이 길어질수록 오차가 누적될 수 있습니다. Likelihood 선택이 중요합니다. Covariate leakage에 취약합니다. 고장 유형 classification에는 직접적인 모델이 아닙니다.

## 이전 논문들과 연결

| 모델 | 이상탐지 신호 |
|---|---|
| ARIMA | residual |
| Kalman Filter | innovation |
| TCN | forecast error |
| DeepAR | prediction interval 밖 실제값 / NLL |

## 오늘 공부용 요약

DeepAR는 여러 관련 시계열을 global RNN으로 학습하고, point forecast가 아니라 예측분포를 냅니다. 설비 예지보전에서는 온도, 전류 RMS, 진동 RMS, 압력의 probabilistic forecasting에 유용합니다.

## 다음 논문 예고

다음은 **17편: N-BEATS**입니다. Backcast와 forecast를 반복적으로 분리하는 fully connected forecasting 모델입니다.

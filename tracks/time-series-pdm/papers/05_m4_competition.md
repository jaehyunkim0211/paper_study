# 5편: The M4 Competition: 100,000 Time Series and 61 Forecasting Methods

## 결론부터

**이 논문의 핵심 결론은, 시계열 예측 모델은 “이론적으로 멋져 보이는가”보다 “다양한 실제 시계열에서 단순한 baseline과 공정하게 비교했을 때 정말 잘 맞히는가”로 평가해야 한다는 것입니다.**

## 이 논문을 한 문장으로 요약하면

**M4 Competition은 100,000개의 시계열과 61개의 예측 방법을 비교해, 단일 모델의 화려함보다 benchmark, baseline, ensemble, hybrid method, 재현 가능한 평가가 forecasting 연구와 실무에 더 중요하다는 것을 보여준 대규모 예측 대회입니다.**

## 왜 이 논문이 필요했나?

Part 1에서 ARIMA, STL, Kalman Filter, DTW를 배웠습니다. 하지만 한 가지 질문이 남습니다.

**그래서 어떤 예측 모델이 실제로 잘 되나?**

논문만 읽다 보면 수식이 복잡한 모델, 딥러닝 모델, 최신 모델이 더 좋아 보입니다. 하지만 forecasting에서는 seasonal naive, exponential smoothing, ARIMA, Theta, ensemble 같은 단순하거나 전통적인 방법이 복잡한 모델보다 더 잘 되는 경우가 많습니다.

M4가 필요한 이유는 바로 여기에 있습니다. **예측 모델은 말로 설득하는 것이 아니라, 다양한 데이터에서 같은 기준으로 붙여봐야 합니다.**

설비 예측이나 예지보전에서도 같은 태도가 필요합니다. “새로운 Transformer 기반 RUL 모델이 최고 성능”이라고 주장해도 다음을 확인해야 합니다.

| 확인 질문 | 왜 중요한가 |
|---|---|
| 어떤 baseline과 비교했나? | 단순 LSTM, CNN, XGBoost, ARIMA보다 정말 나은가? |
| train/test split이 올바른가? | engine unit 단위 split인가, window random split인가? |
| 지표가 적절한가? | RMSE만 봤는가, NASA score도 봤는가? |
| 여러 조건에서 평가했나? | 특정 운전 조건에만 맞춘 것은 아닌가? |
| 재현 가능한가? | 데이터, 코드, 하이퍼파라미터가 공개되어 있는가? |

## 핵심 개념 1. Forecasting competition은 실전 시험장이다

M4는 하나의 모델을 제안하는 논문이 아니라, 여러 모델을 같은 조건에서 비교합니다.

| 일반 모델 논문 | M4 Competition |
|---|---|
| “우리 모델이 좋다”를 주장 | 여러 모델을 같은 데이터에서 비교 |
| 특정 데이터셋 몇 개 사용 | 100,000개 시계열 사용 |
| 평가 기준이 논문마다 다를 수 있음 | 동일한 평가 지표 사용 |
| 모델 구조 중심 | 실전 예측 성능 중심 |

## 핵심 개념 2. M4 데이터셋

M4 데이터셋은 여러 frequency의 시계열로 구성됩니다.

| Frequency | 시계열 개수 | Forecast horizon | Seasonality |
|---|---:|---:|---:|
| Yearly | 23,000 | 6 | 1 |
| Quarterly | 24,000 | 8 | 4 |
| Monthly | 48,000 | 18 | 12 |
| Weekly | 359 | 13 | 1 |
| Daily | 4,227 | 14 | 1 |
| Hourly | 414 | 48 | 24 |

M4는 대규모 benchmark이지만 설비 진단 benchmark는 아닙니다. 설비 데이터는 다변량 센서, 고주파 진동, 운전 조건 변화, 정비 이벤트, 고장 희소성 같은 추가 문제가 있습니다.

## 핵심 개념 3. Forecast horizon

Forecast horizon은 얼마나 먼 미래를 예측할 것인지입니다.

| 문제 | horizon 예시 | 실무 의미 |
|---|---:|---|
| 온도 예측 | 10분 후 | 과열 조기 경보 |
| 압력 예측 | 5분 후 | 공정 안정성 감시 |
| 전류 예측 | 1시간 후 | 부하 이상 감지 |
| vibration trend 예측 | 1일 후 | 열화 추세 확인 |
| RUL 예측 | 고장까지 남은 cycle | 정비 일정 결정 |

Horizon이 길어질수록 예측은 어려워집니다.

## 핵심 개념 4. Baseline

복잡한 모델이 baseline을 이기지 못하면 실무에서 쓸 이유가 약합니다.

| Baseline | 설명 |
|---|---|
| Naive | 미래값을 마지막 관측값과 같다고 예측 |
| Seasonal Naive | 미래값을 직전 같은 계절 위치 값과 같다고 예측 |
| Exponential Smoothing | 최근 값에 더 큰 가중치를 두는 예측 |
| ARIMA | 과거 값과 과거 오차 기반 예측 |
| Theta | M-Competition 계열에서 강력했던 통계적 방법 |

설비 데이터에서도 last value, moving average, seasonal naive, ARIMA, XGBoost를 먼저 봐야 합니다.

## 핵심 개념 5. sMAPE

\[
sMAPE = \frac{1}{h}\sum_{t=1}^{h}
\frac{2|y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t|}
\]

sMAPE는 예측이 실제값에서 얼마나 벗어났는지를 값의 크기를 감안해 보는 지표입니다. 값이 0에 가까울 때 불안정할 수 있어 단독으로 쓰면 부족합니다.

## 핵심 개념 6. MASE

\[
MASE =
\frac{
\frac{1}{h}\sum_{t=1}^{h}|y_t - \hat{y}_t|
}{
\frac{1}{n-m}\sum_{t=m+1}^{n}|y_t - y_{t-m}|
}
\]

MASE는 내 모델의 오차를 naive 또는 seasonal naive baseline의 오차와 비교합니다.

| MASE | 의미 |
|---:|---|
| < 1 | naive baseline보다 좋음 |
| > 1 | naive baseline보다 나쁨 |

## 핵심 개념 7. OWA

\[
OWA =
\frac{1}{2}
\left(
\frac{sMAPE_{model}}{sMAPE_{Naive2}}
+
\frac{MASE_{model}}{MASE_{Naive2}}
\right)
\]

OWA는 모델이 Naive2보다 얼마나 나은지 sMAPE와 MASE 두 관점에서 평균낸 M4의 종합 지표입니다.

## 핵심 개념 8. Winning method: hybrid가 강했다

M4에서 유명한 결과는 winning method가 exponential smoothing과 LSTM/RNN 계열을 결합한 ES-RNN 계열 hybrid였다는 점입니다.

**딥러닝이 이겼지만, 통계 모델을 버리고 이긴 것이 아닙니다.**

설비 데이터에서도 pure end-to-end 모델보다 다음 조합이 강할 수 있습니다.

| 전통 기법 | ML/DL 결합 |
|---|---|
| FFT feature | CNN/XGBoost 분류 |
| envelope spectrum | bearing fault classifier |
| STL residual | anomaly detector |
| Kalman innovation | threshold 또는 ML score |
| ARIMA residual | residual-based anomaly detection |

## 작은 예시 데이터

| 시간 | 실제 | Naive 예측 | ARIMA 예측 | LSTM 예측 |
|---|---:|---:|---:|---:|
| 09:06 | 71.0 | 70.9 | 71.0 | 71.2 |
| 09:07 | 71.2 | 70.9 | 71.2 | 71.3 |
| 09:08 | 71.5 | 70.9 | 71.3 | 71.4 |
| 09:09 | 71.6 | 70.9 | 71.5 | 71.5 |

| 모델 | MAE |
|---|---:|
| Naive | 0.425 |
| ARIMA | 0.075 |
| LSTM | 0.125 |

이 예시에서는 LSTM도 좋지만 ARIMA가 더 좋습니다. 모델 이름만 보고 판단하면 안 됩니다.

## 실무에서 어떻게 써야 하나?

1. 딥러닝 전에 baseline 표를 반드시 만듭니다.
2. train/test split을 먼저 설계합니다.
3. 예측 지표와 운영 지표를 분리해서 봅니다.
4. 여러 설비/센서를 함께 학습할 때는 global model을 고려하되 unit holdout을 봅니다.
5. 데이터 버전, split, preprocessing, feature, threshold, 평가 코드를 남깁니다.

| 위험한 방식 | 문제 |
|---|---|
| 전체 시계열 랜덤 row split | 미래 정보가 train에 섞임 |
| sliding window 생성 후 랜덤 split | 인접 window가 train/test에 섞임 |
| 전체 데이터 scaling 후 split | test 분포 정보가 train 전처리에 들어감 |

## 한계

1. M4는 설비 진단 benchmark가 아닙니다.
2. M4의 대부분은 yearly/quarterly/monthly 저빈도 데이터입니다.
3. Forecasting accuracy와 maintenance value는 다릅니다.
4. 대회 성능은 현장 일반화와 다를 수 있습니다.
5. 복잡한 모델은 재현성과 운영성이 떨어질 수 있습니다.

## 이전 논문들과 연결해서 이해하기

| 관점 | ARIMA | STL | Kalman Filter | DTW | M4 |
|---|---|---|---|---|---|
| 핵심 질문 | 예측할 수 있나? | 분해할 수 있나? | 숨은 상태를 추정할 수 있나? | 시간축 shift에도 비슷한가? | 다양한 데이터에서 실제로 잘 되는가? |
| 설비 활용 | 센서 예측 baseline | 정상 주기 제거 | 상태 추정 | 패턴 비교 | 모델 선택과 평가 설계 |

## 오늘 공부용 요약

1. M4는 100,000개 시계열과 61개 forecasting method를 비교한 대규모 benchmark입니다.
2. 복잡한 모델보다 공정한 평가와 baseline 비교가 중요합니다.
3. sMAPE, MASE, OWA는 scale과 baseline 대비 성능을 보려는 지표입니다.
4. winning method는 순수 딥러닝이 아니라 통계 모델과 딥러닝의 hybrid였습니다.
5. 설비 예측에서도 last value, seasonal naive, ARIMA, XGBoost 같은 baseline을 먼저 만들어야 합니다.
6. 시계열/설비 진단에서는 모델보다 데이터 분할이 먼저입니다.

## 다음 논문 예고

다음은 **6편: The UCR Time Series Classification Archive**입니다.

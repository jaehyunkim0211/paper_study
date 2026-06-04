# 17편: N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting

## 결론부터

**RNN/CNN/Transformer 없이 fully connected block만 깊게 쌓고, 과거를 설명하는 backcast와 미래를 예측하는 forecast를 반복적으로 분리하면 강력하면서도 일부 해석 가능한 딥러닝 forecasting 모델을 만들 수 있습니다.**

## 이 논문을 한 문장으로 요약하면

N-BEATS는 과거 구간을 설명하는 backcast와 미래 구간을 예측하는 forecast를 동시에 내는 block을 여러 개 쌓아, 최종 forecast를 부분 예측들의 합으로 만드는 univariate point forecasting 모델입니다.

## 왜 이 논문이 필요했나?

DeepAR는 probabilistic RNN 모델이었고, TCN은 causal convolution 모델이었습니다. N-BEATS는 RNN도 CNN도 attention도 쓰지 않고 fully connected block만으로 M3, M4, TOURISM 같은 forecasting benchmark에서 강한 성능을 보였습니다. 또한 interpretable 구조에서는 trend와 seasonality basis를 사용해 STL과 비슷한 해석 관점을 제공합니다.

## 핵심 개념

### 1. Point forecasting

N-BEATS는 기본적으로 예측분포가 아니라 미래값 vector를 예측하는 point forecasting 모델입니다.

### 2. Lookback window와 horizon

```text
과거 t개 센서값 → 미래 H개 센서값 예측
```

온도/압력/전류 RMS 같은 저주파 센서 예측에 자연스럽습니다.

### 3. Backcast

Backcast는 block이 입력 과거 구간에서 설명할 수 있는 부분입니다. 입력에서 backcast를 빼고 남은 residual을 다음 block으로 넘깁니다.

### 4. Forecast

각 block은 미래에 대한 partial forecast도 냅니다. 최종 forecast는 모든 block의 forecast를 더한 값입니다.


\hat{y} = \sum_{\ell} \hat{y}_{\ell}


### 5. Doubly residual stacking

Backward residual은 입력에서 backcast를 빼는 방향이고, forward residual은 partial forecast를 누적하는 방향입니다.

### 6. Basis expansion

Block은 forecast 값을 직접 하나씩 내는 것이 아니라 basis coefficient를 만들고 basis function과 조합합니다.


\hat{y}_{\ell} = g_{\ell}^{f}(\theta_{\ell}^{f})


### 7. Generic N-BEATS

시계열 지식을 명시적으로 넣지 않고 basis를 학습합니다. 유연하지만 해석은 어렵습니다.

### 8. Interpretable N-BEATS

Trend stack과 seasonality stack을 사용합니다. Trend는 polynomial basis, seasonality는 Fourier basis로 표현합니다.

## 작은 예시

온도 데이터가 완만히 상승합니다.

| block | backcast로 설명한 패턴 | forecast |
|---|---|---|
| Block 1 | 장기 상승 trend | 미래 상승 |
| Block 2 | 작은 주기 흔들림 | 계절 보정 |
| Block 3 | residual 보정 | 작은 보정 |

최종 예측은 block별 forecast의 합입니다.

## 실무에서 어떻게 써야 하나?

N-BEATS는 온도, 압력, 전류 RMS, 진동 RMS, health index 같은 저주파 센서의 point forecasting baseline으로 좋습니다. Raw vibration waveform에는 직접 적용하기보다 RMS, kurtosis, envelope peak, FFT band energy 같은 feature sequence를 예측하는 것이 더 자연스럽습니다.

ARIMA, ETS, STL+forecast, TCN, DeepAR, PatchTST와 같은 split에서 비교해야 합니다.

## 이상탐지와 RUL 연결

N-BEATS 예측값과 실제값의 차이를 anomaly score로 쓸 수 있습니다. RUL에서는 health index를 forecasting하고 threshold 도달 시점을 RUL 후보로 볼 수 있습니다. RUL에서는 반드시 bearing/engine unit split을 해야 합니다.

## 한계

기본 N-BEATS는 univariate point forecasting입니다. 다변량 센서 관계와 예측 불확실성은 별도 설계가 필요합니다. Interpretable 구조도 물리 해석을 보장하지 않습니다. Data leakage에 취약합니다.

## 오늘 공부용 요약

N-BEATS는 backcast/forecast block을 반복해 과거 residual을 제거하고 미래 forecast를 누적하는 모델입니다. STL과 연결되는 trend/seasonality 해석을 일부 제공하지만, 실무에서는 baseline 비교와 split 설계가 먼저입니다.

## 다음 논문 예고

다음은 **18편: TimesNet**입니다. 1D 시계열을 period 기반 2D tensor로 접어 intra/inter-period variation을 보는 모델입니다.

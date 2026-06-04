# 15편: An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling

## 결론부터

**시계열·시퀀스 모델링에서 RNN/LSTM을 무조건 기본 선택으로 볼 필요는 없고, causal dilated convolution을 쓰는 TCN이 더 단순하면서도 긴 과거 정보를 잘 활용하는 강력한 출발점이 될 수 있습니다.**

## 이 논문을 한 문장으로 요약하면

TCN은 미래 정보를 보지 않는 causal convolution, 멀리 떨어진 과거를 효율적으로 보는 dilated convolution, 깊게 쌓아도 안정적인 residual connection을 결합한 sequence model입니다.

## 왜 이 논문이 필요했나?

설비 데이터에는 고정 window 분류만 있는 것이 아닙니다. 10분 뒤 온도 예측, 예측 오차 기반 이상탐지, 매 시점 정상/위험 labeling, RUL 예측처럼 시간 순서를 유지한 문제가 많습니다. 전통적으로 이런 문제는 RNN/LSTM/GRU가 기본 선택처럼 여겨졌지만, RNN은 순차 계산이 필요하고 긴 history를 실제로 잘 기억하지 못할 수 있습니다.

TCN은 이 문제를 convolution으로 해결합니다. 시간축 convolution을 쓰되 미래를 보지 않게 causal하게 만들고, dilation으로 멀리 있는 과거를 효율적으로 봅니다.

## 핵심 개념

### 1. Causal sequence modeling

시점 t의 출력을 예측할 때는 x0부터 xt까지만 써야 합니다. 실시간 설비 모니터링에서 미래 센서값은 알 수 없습니다.

### 2. Causal convolution

일반 convolution은 앞뒤를 같이 볼 수 있지만, causal convolution은 현재와 과거만 봅니다.

```text
예측 t: x[t], x[t-1], x[t-2]만 사용
```

### 3. Dilated convolution

Dilation은 띄엄띄엄 과거를 보는 방법입니다.

```text
kernel size 3, dilation 2:
x[t], x[t-2], x[t-4]
```

짧은 kernel으로도 긴 과거를 볼 수 있습니다.

### 4. Receptive field

현재 예측을 위해 모델이 볼 수 있는 과거 범위입니다. 냉각 성능 저하는 수십 분~수시간 history가 필요하고, RUL은 수십~수백 cycle history가 필요할 수 있습니다.

### 5. Dilated causal convolution 수식


F(s) = \sum_{i=0}^{k-1} f(i) x_{s-d i}


미래 값이 아니라 현재와 과거만 사용합니다.

### 6. Residual block

깊게 쌓아도 학습이 안정되도록 residual connection을 씁니다.


output = Activation(x + F(x))


### 7. 병렬화

RNN은 t=1, t=2, t=3을 순차 계산하지만, TCN은 convolution 구조라 같은 layer의 여러 시점을 병렬 계산할 수 있습니다.

## 작은 예시

모터 온도와 전류 RMS로 다음 온도를 예측한다고 하겠습니다.

| 입력 sequence | target |
|---|---|
| 09:00~09:03 온도/전류 | 09:04 온도 |
| 09:01~09:04 온도/전류 | 09:05 온도 |

TCN은 짧은 dilation으로 최근 변화, 큰 dilation으로 긴 부하/온도 trend를 봅니다. 예측 오차가 갑자기 커지면 이상 후보입니다.

## 실무에서 어떻게 써야 하나?

TCN은 forecasting, anomaly detection, RUL, degradation tracking의 강력한 baseline입니다. ARIMA, Kalman innovation, N-BEATS, DeepAR, PatchTST와 비교해야 합니다. Receptive field를 고장 물리 시간과 맞춰야 합니다.

Causal architecture와 proper split은 다릅니다. TCN이 causal이어도 전체 scaling, window random split, RUL cycle random split을 하면 leakage가 생깁니다.

## 평가 지표

Forecasting은 MAE, RMSE, sMAPE, MASE를 봅니다. Fault classification은 precision, recall, F1, confusion matrix를 봅니다. Anomaly detection은 false alarm, missed detection, detection delay, event-level recall을 봅니다. RUL은 MAE, RMSE, NASA scoring function, early/late penalty를 봅니다.

## 한계

TCN은 receptive field 밖의 과거를 보지 못합니다. 논문은 산업 설비 진단 논문이 아니므로 설비 데이터에서는 별도 검증이 필요합니다. Causal convolution은 미래 누수를 막지만 preprocessing leakage는 막지 못합니다. 고주파 raw vibration에는 domain preprocessing이 여전히 중요합니다.

## 이전 논문들과 연결

| 논문 | 연결 |
|---|---|
| InceptionTime | multi-scale CNN 분류 |
| TCN | causal dilation으로 sequence 예측 |
| ARIMA/Kalman | 예측 오차/innovation 이상탐지 |
| TCN | 딥러닝 예측 오차 이상탐지 |

## 오늘 공부용 요약

TCN은 RNN의 대안으로 causal dilated convolution을 사용합니다. 설비 예측, 이상탐지, RUL에 자연스럽습니다. 핵심은 receptive field를 물리 시간과 맞추고, 미래 정보가 들어가지 않도록 split과 preprocessing을 설계하는 것입니다.

## 다음 논문 예고

다음은 **16편: DeepAR**입니다. 여러 관련 시계열을 global RNN으로 확률 예측하는 모델입니다.

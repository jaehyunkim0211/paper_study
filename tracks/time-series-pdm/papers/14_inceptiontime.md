# 14편: InceptionTime: Finding AlexNet for Time Series Classification

## 결론부터

**시계열 분류에서도 이미지 분야의 Inception처럼 여러 길이의 convolution filter를 동시에 적용하면, 짧은 충격과 긴 cycle을 함께 볼 수 있는 강력한 multi-scale CNN을 만들 수 있습니다.**

## 이 논문을 한 문장으로 요약하면

InceptionTime은 짧은 패턴과 긴 패턴을 동시에 보는 Inception module을 여러 층 쌓고, 이런 Inception network 5개를 ensemble해서 시계열 분류에서 높은 정확도와 확장성을 동시에 달성한 모델입니다.

## 왜 이 논문이 필요했나?

FCN/ResNet은 raw time series를 CNN으로 직접 학습할 수 있음을 보여줬습니다. 그러나 설비 고장은 하나의 시간 스케일에만 나타나지 않습니다. 베어링 진동에는 순간 충격, 충격 반복 간격, 장기 진동 증가가 함께 있고, 모터 전류에는 짧은 waveform distortion, startup curve, 긴 부하 trend가 함께 있습니다.

InceptionTime은 이 문제를 해결하려고 짧은 filter, 중간 filter, 긴 filter를 병렬로 적용합니다. 즉, 한 모델 안에서 여러 시간 스케일의 pattern detector를 동시에 둡니다.

## 핵심 개념

### 1. Multi-scale pattern

고장은 짧은 peak, 중간 cycle, 긴 trend에 동시에 나타날 수 있습니다. 짧은 filter만 쓰면 긴 cycle을 놓칠 수 있고, 긴 filter만 쓰면 짧은 충격이 희석될 수 있습니다.

### 2. Inception module

구조는 다음처럼 이해하면 됩니다.

```text
입력 시계열
├─ Conv1D(kernel=10)
├─ Conv1D(kernel=20)
├─ Conv1D(kernel=40)
└─ MaxPooling + 1x1 Conv
→ concat
```

각 branch는 서로 다른 시간 스케일을 봅니다.

### 3. Bottleneck layer

1x1 convolution으로 채널 수를 줄여 긴 convolution의 계산량과 overfitting을 줄입니다. 다변량 센서에서는 진동 x/y/z, 전류, 온도, 압력 같은 채널을 압축한 뒤 여러 filter length를 적용하는 것으로 볼 수 있습니다.

### 4. Residual connection

깊은 Inception module을 쌓아도 학습이 안정되도록 shortcut을 사용합니다.


output = F(x) + x


### 5. Receptive field

Receptive field는 모델이 한 번에 참고할 수 있는 원본 시간 범위입니다. 설비 문제마다 필요한 receptive field가 다릅니다. 순간 spike는 작아도 되지만, startup 전체나 RUL trend는 긴 receptive field가 필요합니다.

### 6. Ensemble of 5

InceptionTime은 같은 architecture를 서로 다른 초기값으로 5번 학습하고 예측 확률을 평균냅니다. 작은 데이터에서 딥러닝 결과가 흔들리는 것을 줄이는 효과가 있습니다.

## 작은 예시

베어링 진동 window에 짧은 충격이 반복된다고 하겠습니다.

| filter 길이 | 잡는 정보 |
|---:|---|
| 짧음 | 한 번의 충격 peak |
| 중간 | 충격 주변 waveform |
| 김 | 충격 반복 간격과 context |

Inception module은 이 세 가지를 병렬로 봅니다. 외륜 결함처럼 짧은 충격과 반복 주기가 모두 중요한 문제에 자연스럽습니다.

## 실무에서 어떻게 써야 하나?

InceptionTime은 multi-scale CNN baseline입니다. MiniRocket, FCN/ResNet, catch22, tsfresh+XGBoost, HIVE-COTE와 같은 split에서 비교해야 합니다. Raw vibration, envelope signal, band-pass signal, FFT magnitude, RMS/kurtosis sequence를 각각 입력으로 비교하는 것이 좋습니다.

Window length는 물리 시간과 맞춰야 합니다. 베어링은 회전 주기와 결함 충격 반복, 모터 startup은 startup phase 전체, 압력은 valve cycle을 고려해야 합니다.

## 데이터 분할 주의

1초 window, 0.5초 stride로 자른 뒤 random split하면 같은 run의 인접 window가 train/test에 섞입니다. InceptionTime은 여러 scale에서 인접 window 유사성을 잘 잡기 때문에 성능이 비정상적으로 높아질 수 있습니다. Unit/run/condition/time split이 먼저입니다.

## 평가 지표

Accuracy뿐 아니라 precision, recall, F1, confusion matrix, false alarm, missed detection을 봐야 합니다. 이상탐지로 쓰면 detection delay와 event-level recall, RUL로 쓰면 MAE, RMSE, NASA scoring function, early/late penalty를 봐야 합니다.

## 한계

Ensemble이라 학습 비용이 큽니다. 데이터가 적으면 overfitting 가능성이 있습니다. raw waveform만으로는 FFT/envelope 같은 물리 지식을 놓칠 수 있습니다. UCR 성능이 현장 일반화를 보장하지 않습니다.

## 이전 논문들과 연결

| 논문 | 차이 |
|---|---|
| FCN/ResNet | 단일 CNN baseline |
| InceptionTime | 여러 길이 filter를 병렬 적용 |
| ROCKET | random convolution feature |
| InceptionTime | learned multi-scale convolution feature |

## 오늘 공부용 요약

InceptionTime은 여러 시간 스케일을 동시에 보는 CNN입니다. 베어링 충격, 압력 cycle, startup 전류처럼 짧은 패턴과 긴 문맥이 함께 중요한 설비 데이터에 잘 맞습니다. 그러나 split이 잘못되면 multi-scale leakage를 더 잘 학습할 수 있습니다.

## 다음 논문 예고

다음은 **15편: TCN**입니다. RNN 대신 causal dilated convolution으로 sequence modeling을 하는 방법을 봅니다.

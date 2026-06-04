# 13편: Time Series Classification from Scratch with Deep Neural Networks

## 결론부터

**시계열 분류에서도 사람이 직접 만든 feature나 복잡한 전처리 없이 raw time series를 CNN 계열 딥러닝 모델에 바로 넣어도 강력한 baseline을 만들 수 있습니다.**

## 이 논문을 한 문장으로 요약하면

Wang, Yan, Oates의 이 논문은 MLP, FCN, ResNet을 UCR 시계열 분류 데이터셋에 적용해, 특히 FCN과 ResNet이 feature engineering 없이도 기존 전통 TSC 방법들과 경쟁할 수 있음을 보여준 딥러닝 기반 TSC의 대표 baseline 논문입니다.

## 왜 이 논문이 필요했나?

Part 2에서는 Shapelet, tsfresh, catch22, BOSS, ROCKET, HIVE-COTE처럼 raw time series를 어떤 표현으로 바꾸는 방법을 공부했습니다. 이 논문은 질문을 바꿉니다. 사람이 RMS, kurtosis, FFT, shapelet 같은 feature를 만들지 않고, 모델이 raw time series에서 직접 feature를 배우면 안 될까?

설비 진단에서는 이 질문이 매우 중요합니다. 베어링 진동 waveform, 모터 전류 waveform, 압력 cycle을 사람이 일일이 feature로 바꾸는 대신 CNN이 정상/고장 패턴을 직접 학습할 수 있기 때문입니다. 다만 feature를 모델이 배우게 한다고 해서 데이터 분할 문제가 사라지는 것은 아닙니다. 오히려 딥러닝은 잘못된 split의 누수를 더 잘 외울 수 있습니다.

## 핵심 개념

### 1. End-to-end learning

전통 ML은 보통 다음 흐름입니다.

```text
raw vibration → RMS / kurtosis / FFT / envelope feature → XGBoost
```

End-to-end learning은 다음 흐름입니다.

```text
raw vibration → FCN / ResNet → 정상 / 고장 분류
```

즉, feature extraction과 classification을 한 모델 안에서 같이 학습합니다.

### 2. MLP

MLP는 시계열을 시간 구조가 있는 신호가 아니라 긴 벡터처럼 봅니다. 구현은 쉽지만 local pattern이나 시간축 shift에 약합니다. 설비 진단에서는 강력한 최종 모델이라기보다 단순 딥러닝 baseline입니다.

### 3. 1D convolution

1D CNN filter는 작은 패턴 탐지기입니다.


y_t = \sum_{i=0}^{k-1} w_i x_{t+i} + b


베어링 진동에서는 짧은 peak, 상승 후 하강, 충격성 waveform, 반복 패턴을 감지할 수 있습니다. ROCKET은 convolution filter를 무작위로 만들었고, FCN/ResNet은 label에 맞게 filter를 학습합니다.

### 4. FCN

FCN은 다음 구조입니다.

```text
raw time series
→ Conv1D + BatchNorm + ReLU
→ Conv1D + BatchNorm + ReLU
→ Conv1D + BatchNorm + ReLU
→ Global Average Pooling
→ Softmax
```

논문에서 FCN의 kernel size는 8, 5, 3이고 filter 수는 128, 256, 128 구조로 제시됩니다.

### 5. Global Average Pooling

GAP는 마지막 feature map을 시간축으로 평균내어 parameter 수를 줄이고 overfitting을 완화합니다. 또한 Class Activation Map, CAM을 가능하게 해 모델이 어느 시간 구간을 보고 판단했는지 확인할 수 있습니다.

### 6. ResNet

ResNet은 residual shortcut을 사용합니다.


output = F(x) + x


깊은 CNN에서도 gradient가 흐르기 쉽게 만들고, 정상 waveform에 고장으로 인해 추가되는 변화만 학습하는 식으로 이해할 수 있습니다.

### 7. Softmax와 cross-entropy

마지막 layer는 class별 logit을 softmax 확률로 바꾸고 categorical cross-entropy로 학습합니다. 설비 진단에서는 softmax threshold를 validation에서 정하고, test에서 고정해야 합니다.

### 8. CAM

CAM은 모델이 어느 시간 구간을 보고 특정 class로 판단했는지 보여줍니다. 베어링 외륜 결함이라면 CAM이 충격 peak 주변을 밝게 표시하는지 확인할 수 있습니다. 하지만 CAM은 설명의 출발점일 뿐이며, FFT/envelope 분석과 정비 로그로 검증해야 합니다.

## 작은 예시

가상의 진동 window가 있습니다.

| window | sequence | label |
|---|---|---|
| W1 | `[0.1, 0.1, 0.2, 0.1, 0.1]` | 정상 |
| W2 | `[0.1, 0.2, 0.1, 1.5, 3.0, 1.4]` | 외륜 결함 |

FCN은 raw sequence에 convolution filter를 적용해 짧은 peak와 상승-peak-하강 pattern을 학습합니다. CAM을 보면 W2의 peak 주변이 외륜 결함 판단에 크게 기여할 수 있습니다.

## 실무에서 어떻게 써야 하나?

FCN/ResNet은 딥러닝 TSC baseline입니다. 반드시 1-NN DTW, BOSS, catch22/catch24 + RandomForest, tsfresh + XGBoost, MiniRocket, HIVE-COTE와 비교해야 합니다. raw vibration만 넣고 끝내지 말고, band-pass signal, envelope signal, FFT magnitude, RMS/kurtosis sequence와도 비교해야 합니다.

가장 중요한 것은 split입니다. 긴 진동 run을 1초 window, 0.5초 stride로 잘라 window 단위 랜덤 split하면 인접 window가 train/test에 섞입니다. FCN은 local pattern을 잘 학습하므로 거의 같은 raw signal 조각을 외울 수 있습니다.

권장 split은 설비 unit 단위, run 단위, load/speed condition holdout, 시간 순서 split, RUL에서는 engine/bearing unit split입니다.

## 평가 지표

설비 진단 분류에서는 accuracy만 보면 안 됩니다. precision, recall, F1, confusion matrix, false alarm, missed detection을 함께 봐야 합니다. 이상탐지로 연결할 때는 detection delay, threshold sensitivity, event-level recall을 봐야 합니다. RUL에서는 MAE, RMSE, NASA scoring function, early/late prediction penalty를 봐야 합니다.

## 한계

데이터가 적으면 overfitting이 쉽습니다. raw input만으로 물리 해석이 부족할 수 있습니다. UCR benchmark 성능이 현장 일반화를 보장하지 않습니다. 다변량 센서와 운전 조건을 별도로 고려해야 합니다. 딥러닝 모델은 baseline 비교 없이 쓰면 위험합니다.

## 이전 논문들과 연결

| 이전 논문 | 연결 |
|---|---|
| ROCKET | convolution filter를 무작위로 쓰는 방법 |
| FCN/ResNet | convolution filter를 label에 맞게 학습 |
| HIVE-COTE | 여러 전통 표현을 ensemble |
| FCN/ResNet | raw time series에서 learned representation을 생성 |

## 오늘 공부용 요약

FCN/ResNet은 raw 시계열을 직접 학습하는 딥러닝 TSC baseline입니다. CNN filter는 설비 waveform의 local pattern을 감지합니다. CAM은 판단 근거 구간을 보여줄 수 있습니다. 그러나 window random split, 전체 scaling, test threshold tuning은 성능을 크게 부풀릴 수 있습니다.

## 다음 논문 예고

다음은 **14편: InceptionTime**입니다. 여러 길이의 convolution filter를 동시에 적용해 multi-scale pattern을 보는 모델입니다.

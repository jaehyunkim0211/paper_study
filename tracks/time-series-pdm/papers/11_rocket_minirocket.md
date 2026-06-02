# 11편: ROCKET / MiniRocket

## ROCKET: Exceptionally Fast and Accurate Time Series Classification Using Random Convolutional Kernels
## MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification

## 결론부터

**이 논문의 핵심 결론은, 시계열 분류에서 CNN처럼 convolution을 쓰되 kernel을 학습하지 않고 무작위로 많이 뿌린 다음, 그 결과를 단순 선형 분류기에 넣어도 매우 빠르고 강력한 성능을 낼 수 있다는 것입니다.**

## 이 논문을 한 문장으로 요약하면

**ROCKET은 시계열에 수천~수만 개의 random convolution kernel을 적용해 feature를 만들고, MiniRocket은 이 아이디어를 거의 deterministic하고 훨씬 빠른 방식으로 정리해 실무 baseline으로 쓰기 좋게 만든 방법입니다.**

## 왜 이 논문이 필요했나?

Shapelet, tsfresh, catch22, BOSS는 각각 좋은 전통 TSC 방법이지만 약점이 있습니다.

| 방법 | 강점 | 약점 |
|---|---|---|
| Shapelet | 해석 가능 | 후보 탐색 비용이 큼 |
| tsfresh | feature 탐색 폭이 넓음 | feature가 많고 leakage/overfitting 위험 |
| catch22 | 빠르고 간단함 | 표현력이 부족할 수 있음 |
| BOSS | 반복 패턴에 강함 | symbolic 변환과 parameter 선택 필요 |

ROCKET은 CNN처럼 다양한 패턴을 잡되, CNN처럼 오래 학습하지 않고 빠르게 feature만 만들 수 없을까라는 질문에서 출발합니다.

## 핵심 개념 1. Convolution은 작은 패턴 탐지기다

```text
x = [0.1, 0.1, 0.2, 1.5, 3.0, 1.4, 0.2, 0.1]
kernel = [1, 2, 1]
```

kernel을 시계열 위로 밀면서 “이 모양과 비슷한 부분이 있는가?”를 봅니다.

\[
y_t = \sum_{i=0}^{k-1} w_i x_{t+i}
\]

CNN에서는 \(w_i\)를 학습하고, ROCKET에서는 무작위로 정합니다.

## 핵심 개념 2. Random convolution kernel

ROCKET kernel에서 random하게 정하는 요소는 다음입니다.

| 요소 | 의미 | 설비 진단 해석 |
|---|---|---|
| length | kernel 길이 | 짧은 충격을 볼지, 긴 cycle을 볼지 |
| weights | kernel 모양 | 상승/하강/peak/진동 형태 |
| bias | threshold 역할 | 어느 정도 반응해야 양성으로 볼지 |
| dilation | kernel 사이 간격 | 빠른 패턴과 느린 패턴을 모두 보기 |
| padding | 가장자리 처리 | window 시작/끝 패턴도 볼지 |

## 핵심 개념 3. Dilation

dilation은 띄엄띄엄 보는 convolution입니다.

| dilation | 잡기 쉬운 패턴 |
|---:|---|
| 작음 | 짧은 충격, 고주파 진동 |
| 중간 | 회전 cycle, 기어 mesh 관련 패턴 |
| 큼 | startup curve, 압력 cycle, 열화 trend |

## 핵심 개념 4. Max와 PPV

convolution output이 다음과 같다고 합시다.

```text
[-0.5, -0.2, 0.1, 2.8, 3.1, 0.4, -0.1]
```

| feature | 의미 |
|---|---|
| max | 이 kernel이 가장 강하게 반응한 정도 |
| PPV | output 중 양수인 값의 비율 |

\[
PPV = \frac{1}{n}\sum_{t=1}^{n} \mathbf{1}(y_t > 0)
\]

max는 한 번의 강한 이벤트, PPV는 패턴이 얼마나 자주 나타나는지를 봅니다.

## 핵심 개념 5. Feature extractor + 선형 분류기

```text
raw time series
→ random convolution kernels
→ max/PPV features
→ linear classifier
→ class prediction
```

ROCKET의 힘은 복잡한 classifier가 아니라 transform에서 나옵니다.

## 핵심 개념 6. MiniRocket

| ROCKET | MiniRocket |
|---|---|
| random kernel length `{7, 9, 11}` | kernel length 9 고정 |
| random weights | `{−1, 2}` 형태의 고정 kernel subset |
| random bias | convolution output에서 선택 |
| random dilation | 입력 길이에 따라 고정 |
| max + PPV | PPV 중심 |
| 약 20,000 features | 약 10,000 features |

MiniRocket은 빠르고 튜닝이 적어서 실무 baseline으로 매우 좋습니다.

## 핵심 개념 7. CNN과 비교

| 관점 | CNN | ROCKET / MiniRocket |
|---|---|---|
| kernel | 학습함 | random 또는 fixed |
| 학습 방식 | backpropagation | transform 후 선형 분류기 |
| 계산 비용 | 상대적으로 큼 | 매우 빠름 |
| 데이터 요구량 | 많을수록 좋음 | 작은 데이터에서도 baseline 가능 |
| 실무 위치 | 최종 deep model 후보 | 강력한 빠른 baseline |

## 작은 예시 데이터

| window | K1_max | K1_PPV | K2_max | K2_PPV | label |
|---|---:|---:|---:|---:|---|
| N1 | 0.3 | 0.20 | 0.2 | 0.10 | 정상 |
| N2 | 0.4 | 0.25 | 0.3 | 0.10 | 정상 |
| F1 | 2.8 | 0.55 | 4.9 | 0.45 | 고장 |
| F2 | 2.5 | 0.50 | 4.5 | 0.40 | 고장 |

원본 시계열이 convolution feature vector로 바뀌고, RidgeClassifier나 Logistic Regression에 들어갑니다.

## 실무에서 어떻게 써야 하나?

1. MiniRocket을 강력한 기본 baseline으로 둡니다.
2. raw waveform, filtered vibration, envelope signal, RMS sequence 등 여러 입력 표현에서 비교합니다.
3. multivariate sensor에는 multivariate MiniRocket 또는 센서별 feature concat을 씁니다.
4. transform과 scaler는 train에서만 fit합니다.
5. 고장 분류에서는 accuracy만 보지 않습니다.

| 지표 | 실무 해석 |
|---|---|
| precision | false alarm 비용 |
| recall | missed detection 위험 |
| F1 | 불균형 데이터에서 균형 |
| confusion matrix | 고장 유형별 오진 확인 |
| lead time | 예지보전 가치 |

## 데이터 분할 주의

MiniRocket은 강력한 feature extractor이므로 인접 window leakage가 있으면 성능이 비정상적으로 높아질 수 있습니다.

| 위험 | 설명 |
|---|---|
| window random split | 인접 window가 train/test에 섞임 |
| same run split | 같은 run fingerprint 학습 |
| same unit split | 특정 설비의 고유 신호 학습 |
| 전체 scaling | test 분포 정보 누수 |

## 이상탐지와 RUL 연결

정상 window의 MiniRocket feature 분포를 학습하고, 새 window가 벗어나면 anomaly score로 쓸 수 있습니다. RUL에서는 cycle window를 MiniRocket feature로 바꾼 뒤 RUL regressor에 넣을 수 있습니다. RUL에서는 unit split과 MAE, RMSE, NASA scoring function, early/late penalty가 중요합니다.

## 한계

1. 해석 가능성이 domain feature보다 낮습니다.
2. split이 잘못되면 높은 성능이 쉽게 나옵니다.
3. 절대 amplitude 정보를 놓칠 수 있어 RMS, max, crest factor, FFT feature를 추가해야 할 수 있습니다.
4. 고주파 raw signal에서는 window 설계가 중요합니다.
5. 센서 간 물리 관계를 명시적으로 설명하지 않습니다.
6. 이상탐지/RUL은 별도 설계가 필요합니다.

## 이전 논문들과 연결해서 이해하기

| 관점 | tsfresh | catch22 | BOSS | ROCKET/MiniRocket |
|---|---|---|---|---|
| 표현 | 통계/신호 feature | 22개 대표 feature | SFA word histogram | convolution max/PPV feature |
| 분류기 | RF/XGBoost | RF/XGBoost | 1-NN | Ridge/Logistic |
| 장점 | 넓은 탐색 | 빠르고 해석 가능 | 반복 pattern에 강함 | 매우 빠르고 강력 |
| 단점 | leakage 위험 | 표현력 제한 | 순서 일부 손실 | 물리 해석 어려움 |

## 오늘 공부용 요약

1. ROCKET은 random convolution kernel로 시계열을 feature vector로 변환합니다.
2. Max와 PPV가 핵심 feature입니다.
3. MiniRocket은 ROCKET을 거의 deterministic하고 훨씬 빠르게 만든 방법입니다.
4. MiniRocket은 CNN/Transformer 전에 반드시 비교해야 할 강력한 baseline입니다.
5. window random split과 전체 scaling은 여전히 위험합니다.

## 다음 논문 예고

다음은 **12편: HIVE-COTE 2.0**입니다.

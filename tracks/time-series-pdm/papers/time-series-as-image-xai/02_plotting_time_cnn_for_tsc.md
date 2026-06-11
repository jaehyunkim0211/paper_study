논문 URL: https://arxiv.org/abs/2102.04179

# 2편: Plotting Time: On the Usage of CNNs for Time Series Classification

## 결론부터

이 논문의 핵심 결론은 **시계열을 복잡한 변환 없이 기본 line plot 이미지로 바꿔 CNN에 넣어도 일부 시계열 분류 문제에서 꽤 강한 성능을 낼 수 있다**는 것이다.

실무적으로는 이 논문을 “plot CNN이 모든 TSC에서 최고”라고 읽으면 안 된다. 더 안전한 해석은 다음이다.

> 매우 단순한 plot image + shallow CNN도 일부 데이터셋에서 강하게 작동한다. 따라서 시계열-as-image 접근은 실험해볼 가치가 있다. 하지만 plot scale, rendering artifact, split leakage를 반드시 검증해야 한다.

## 이 논문을 한 문장으로 요약하면

시계열을 matplotlib 기본 line plot 이미지로 저장하고, 그 이미지를 얕은 CNN에 넣어 분류하는 매우 단순한 방법을 제안했으며, 두 개의 real-world dataset과 98개 UCR dataset에서 시계열-as-image 접근의 가능성을 실험한 논문이다.

## 왜 이 논문이 필요했나?

기존 시계열 분류는 대부분 원본 숫자 배열을 직접 다루었다.

| 접근 | 예시 |
|---|---|
| 거리 기반 | Euclidean, DTW |
| feature 기반 | tsfresh, catch22, RMS, FFT, kurtosis |
| dictionary 기반 | BOSS, WEASEL |
| convolution feature | ROCKET, MiniRocket |
| 딥러닝 | FCN, ResNet, InceptionTime |
| ensemble | HIVE-COTE, TS-CHIEF |

이 논문은 질문을 바꾼다.

> 사람이 시계열 plot을 보고 패턴을 이해한다면, CNN도 plot 이미지에서 패턴을 배울 수 있지 않을까?

설비 진단에서도 이 질문은 자연스럽다.

| 원본 숫자 배열 | 사람이 보는 plot |
|---|---|
| `[0.1, 0.12, 0.11, 0.8, 1.5, 0.6]` | 중간에 큰 진동 peak가 보임 |

현장 엔지니어는 숫자 배열보다 plot을 더 잘 이해한다. 이 논문은 그 직관을 CNN 입력으로 사용한다.

## 핵심 개념 1. 기본 line plot을 그대로 쓴다

이 논문은 GAF, recurrence plot, spectrogram 같은 복잡한 변환을 사용하지 않는다.

| 복잡한 변환 | 사용 여부 |
|---|---|
| Gramian Angular Field | 사용하지 않음 |
| Recurrence Plot | 사용하지 않음 |
| Spectrogram | 사용하지 않음 |
| Markov Transition Field | 사용하지 않음 |
| 기본 line plot | 사용함 |

각 시계열을 matplotlib 기본 설정으로 plot하고, 432×288 크기의 이미지로 만든다. 축과 숫자도 포함한다.

설비 예시:

```text
온도 sequence
[70.0, 70.1, 70.3, 70.5, 70.8]

→ line plot image
→ CNN
```

## 핵심 개념 2. 왜 image로 바꾸나?

논문이 보는 이유는 크게 두 가지다.

| 이유 | 설명 |
|---|---|
| 해석 | 사람이 보는 plot과 가까움 |
| 모델 활용 | CNN 같은 computer vision model을 바로 사용 가능 |

시계열 plot에는 다음 visual pattern이 담긴다.

| 시계열 패턴 | 이미지 패턴 |
|---|---|
| 급상승 | 위로 치솟는 line |
| spike | 뾰족한 peak |
| oscillation | 반복 wave pattern |
| flatline | 수평선 |
| drift | 천천히 기울어진 선 |

## 핵심 개념 3. Image normalization이 중요하다

논문에서 흥미로운 점은 image normalization이다. 기본 plot image는 흰 배경이 대부분이고 선이 조금만 있다. CNN이 흰 배경에 압도될 수 있다.

논문은 pixel 값을 rescale하고 sample-wise standard normalization을 적용해, 배경보다 curve와 axis 숫자 같은 정보 영역이 더 잘 드러나게 한다.

실무 교훈:

> 시계열-as-image 모델에서는 plot 생성과 image preprocessing 자체가 모델의 일부다.

문서화해야 할 항목:

| 항목 | 이유 |
|---|---|
| image size | pattern 해상도 |
| line thickness | 작은 spike 표시 |
| y축 scale | amplitude 정보 |
| x축 scale | duration 정보 |
| 축 포함 여부 | artifact 위험 |
| normalization | 학습 안정성 |
| DPI/rendering | pixel artifact |

## 핵심 개념 4. Multivariate 처리 방식

논문은 multivariate time series를 두 방식으로 처리한다.

### 방식 A. 여러 series를 같은 plot에 겹쳐 그리기

```text
temperature + current + vibration
→ 한 plot에 여러 line
→ CNN
```

장점은 단순하다. 단점은 scale이 다른 센서가 서로 가릴 수 있다.

### 방식 B. Separate plot + siamese CNN

```text
temperature plot → CNN head
current plot → CNN head
vibration plot → CNN head
concat → classifier
```

센서별 feature를 따로 추출할 수 있지만 계산 비용이 커진다.

설비에서는 scale 차이가 크기 때문에 별도 plot이나 센서별 normalization을 신중히 고려해야 한다.

## 핵심 개념 5. 길이가 다른 시계열도 같은 image size로 처리 가능

원본 숫자 모델은 길이가 다르면 padding, truncation, resampling이 필요하다. 그러나 plot image 방식은 길이가 달라도 같은 크기의 이미지로 저장할 수 있다.

```text
sample A 길이 100 → 432×288 image
sample B 길이 150 → 432×288 image
sample C 길이 80  → 432×288 image
```

설비 예시:

| 문제 | 길이 차이 |
|---|---|
| motor startup | 설비마다 startup 시간이 다름 |
| batch process | batch 길이가 다름 |
| 이상 event | event 지속 시간이 다름 |
| run-to-failure | 설비마다 failure time이 다름 |

하지만 같은 image width로 압축하면 실제 duration 정보가 왜곡될 수 있다.

## 실험 결과 요약

### OPTOX

OPTOX는 fluorescence induction curve로 contaminant class를 식별하는 real-world dataset이다.

| 모델 | Test accuracy |
|---|---:|
| RF | 0.7461 |
| XGBoost | 0.8076 |
| ROCKET | 0.9705 |
| ANN | 0.9089 |
| 1D CNN | 0.9641 |
| CNN Log | 0.9765 |
| CNN | 0.9732 |

CNN Log가 가장 좋았다. 중요한 교훈은 **plot 변환에도 도메인 지식이 중요하다**는 것이다. OPTOX에서는 log scale이 domain convention에 가까웠고 성능도 좋았다.

설비 예시:

| 신호 | scale 고려 |
|---|---|
| vibration spectrum | log scale이 유용할 수 있음 |
| envelope spectrum | 결함 주파수 band 확대 필요 |
| 온도 | 절대 y축 scale 중요 |
| 압력 | cycle phase alignment 중요 |
| startup current | 시작점 정렬 중요 |

### FISIO

FISIO는 HR과 ventilation 기반 cardio-respiratory 데이터다. 논문은 multivariate input이 항상 좋은 것은 아니라고 보고한다. BMI task에서는 HR만 쓴 CNN이 가장 좋은 median result를 보였다.

설비 교훈:

> 센서가 많다고 항상 좋지 않다. 불필요한 센서는 confounding factor가 될 수 있다.

실무에서는 반드시 ablation을 한다.

```text
vibration only
current only
temperature only
vibration + current
vibration + current + temperature
```

### UCR 98개

Plot image CNN은 전체적으로 TS-CHIEF, HIVE-COTE, ROCKET, InceptionTime보다 강하다고 말할 수는 없다. 그러나 98개 중 6개 dataset에서 SOTA를 match/beat했다.

올바른 해석:

| 잘못된 해석 | 올바른 해석 |
|---|---|
| plot CNN이 TSC SOTA다 | 전체적으로 SOTA는 아니지만 탐구 가치가 있다 |
| 이미지로 바꾸면 무조건 좋다 | dataset에 따라 작동한다 |
| MiniRocket보다 좋다 | 반드시 같은 split에서 비교해야 한다 |

## 실무에서 어떻게 써야 하나?

내가 실무자라면 이 방법을 다음 위치에 둔다.

```text
주 모델 후보:
MiniRocket / InceptionTime / feature+XGBoost / TCN

보조·비교 모델:
plot image CNN

설명:
Grad-CAM / feature map / case review
```

### 추천 적용 시나리오

| 상황 | plot CNN 적합성 |
|---|---|
| 사람이 plot으로 class를 잘 구분할 수 있음 | 높음 |
| 센서 길이가 서로 다름 | 중간~높음 |
| 시각적 dashboard가 중요 | 높음 |
| raw 고주파 vibration | 주의 |
| frequency feature가 중요 | spectrogram/FFT image도 비교 필요 |

## 실무 주의점

### 1. y축 scale 문제

sample별 자동 y축 scale을 쓰면 amplitude 정보가 사라질 수 있다.

```text
정상: [0.1, 0.2, 0.1]
고장: [1.0, 2.0, 1.0]
```

sample별 자동 scale이면 두 plot이 비슷한 모양이 될 수 있다. 베어링 고장처럼 amplitude 증가가 핵심이면 치명적이다.

### 2. 다변량 센서 plot 겹침

온도 70~90과 진동 RMS 0.1~1.5를 같은 y축에 그리면 진동 변화가 거의 보이지 않을 수 있다. 센서별 subplot이나 separate image가 더 나을 수 있다.

### 3. window random split 금지

```text
긴 bearing run
→ 1초 window, 0.5초 stride
→ plot image 생성
→ random split
```

인접 image가 거의 같으므로 test 성능이 과대평가된다.

## 한계

1. Plot artifact를 학습할 수 있다.
2. 전체적으로 SOTA라고 보기는 어렵다.
3. multivariate 설비 데이터에 대한 검증은 제한적이다.
4. raw 고주파 진동에는 직접 적용이 어려울 수 있다.
5. 이상탐지/RUL 운영 지표는 다루지 않는다.
6. plot 생성 방식이 바뀌면 model input distribution이 바뀐다.

## 이전 논문들과 연결

| 논문 | 연결 |
|---|---|
| XAI for TSC | plot image에 Grad-CAM/LIME을 붙일 수 있음 |
| Shapelet | 중요한 부분 pattern이 class를 구분한다는 생각 |
| BOSS | 시계열을 다른 representation으로 바꿈 |
| ROCKET | convolution으로 pattern 감지 |
| InceptionTime | CNN 기반 TSC |
| TimesNet | 시계열을 2D 구조로 바꾸는 큰 방향 |
| PatchTST | 한 점보다 구간이 의미 단위라는 생각 |

## 오늘 공부용 요약

- 이 논문은 시계열을 기본 line plot image로 바꿔 shallow CNN으로 분류한다.
- 복잡한 이미지 변환이 아니라 matplotlib 기본 plot을 사용한다.
- image normalization이 매우 중요하다.
- multivariate 시계열은 같은 plot에 겹치거나 separate plot + siamese CNN으로 처리한다.
- OPTOX에서는 CNN Log가 가장 좋은 성능을 보였다.
- FISIO에서는 센서를 많이 쓰는 것이 항상 좋지 않았다.
- UCR 98개에서는 전체 SOTA는 아니지만 일부 dataset에서 강했다.
- 실무에서는 plot image CNN을 강력 baseline과 함께 비교해야 한다.
- 모델보다 split이 먼저다.

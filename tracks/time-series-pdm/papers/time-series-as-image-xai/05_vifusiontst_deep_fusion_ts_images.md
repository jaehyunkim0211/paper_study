논문 URL: https://arxiv.org/abs/2506.22498

# 5편: ViFusionTST: Deep Fusion of Time-Series Image Representations from Load Signals for Early Bed-Exit Prediction

## 결론부터

이 논문의 핵심 결론은 **시계열을 하나의 이미지 표현으로만 바꾸지 말고, 원본 파형을 보존하는 line plot과 동역학 구조를 드러내는 RP·MTF·GAF texture map을 따로 처리한 뒤 cross-attention으로 융합하면 짧은 전조 행동을 더 잘 감지할 수 있다**는 것이다.

실무적으로는 다음처럼 읽으면 좋다.

> ViFusionTST는 시계열-as-image 접근의 다음 단계다. 단순 plot 하나를 CNN/ViT에 넣는 것이 아니라, 서로 다른 이미지 표현이 담고 있는 정보를 분리해서 학습하고, 모델이 sample마다 어떤 표현을 더 믿을지 cross-attention으로 자동 조절하게 만든다.

## 이 논문을 한 문장으로 요약하면

침대 다리 아래의 저가 load cell에서 얻은 load signal을 여러 이미지 표현, 즉 RGB line plot, recurrence plot, Markov transition field, Gramian angular field로 바꾼 뒤, dual-stream Swin Transformer와 cross-attention fusion으로 bed-exit 전조를 분류하는 모델이다.

## 왜 이 논문이 필요했나?

이전 시계열-as-image 논문들은 대체로 하나의 image representation에 의존했다.

| 표현 | 잘 보는 것 | 약점 |
|---|---|---|
| line plot | peak, slope, local fluctuation | recurrence/state transition 구조는 약함 |
| RP | 반복 상태, recurrence | raw waveform detail은 약함 |
| MTF | 상태 전이 | 원본 amplitude 직관성 낮음 |
| GAF | global pairwise relation | local peak 해석 약함 |

ViFusionTST는 이 표현들이 상호보완적이라고 본다.

설비 진단으로 바꾸면:

| 표현 | 설비 신호에서 잘 보이는 것 |
|---|---|
| line plot | 진동 peak, 온도 상승, 압력 spike, 전류 startup shape |
| RP | 반복 상태, motif, 유사 pattern 재방문 |
| MTF | 낮은 압력 → 높은 압력 같은 상태 전이 |
| GAF | 전체 시계열의 global pairwise relation |
| fusion model | 각 표현의 장점을 sample별로 조합 |

## 핵심 개념 1. Bed-exit prediction은 전조 감지 문제다

이 논문은 환자가 침대에서 이미 나간 뒤 감지하는 것이 아니라, 침대에서 나가기 전 transition behavior를 감지하려 한다.

설비 예지보전과 매우 닮았다.

| bed-exit 문제 | 설비 예지보전 |
|---|---|
| 환자가 나가기 전 감지 | 설비가 고장나기 전 감지 |
| transition period | degradation / warning period |
| routine repositioning과 구분 | 정상 부하 변화와 고장 전조 구분 |
| caregiver 조기 alert | 정비팀 조기 alert |
| 늦으면 낙상 위험 | 늦으면 설비 파손·품질 문제 |

즉, 핵심은 event 이후 분류가 아니라 **event 이전의 미묘한 transition을 잡는 것**이다.

## 핵심 개념 2. 센서 구성

논문은 침대 다리 아래 load cell을 사용한다. 기록되는 주요 신호는 다음이다.

| 신호 | 설명 |
|---|---|
| load | load cell이 직접 측정한 vertical force |
| vibration | load signal에서 band-pass filtering으로 파생 |
| bed occupancy status | threshold로 빈 침대/점유 상태 구분 |
| in-bed duration | 장기 점유 시간 feature |

설비로 비유하면:

| bed monitoring | 설비 monitoring |
|---|---|
| load cell | strain gauge, torque sensor, current sensor |
| derived vibration | band-pass filtered vibration, RMS, envelope |
| occupancy status | operation mode, on/off status |
| in-bed duration | running time, maintenance age |

## 핵심 개념 3. Line plot stream

Line plot stream은 load, vibration, occupancy, in-bed duration 네 variable을 subplot으로 그린 RGB line plot을 사용한다.

Line plot은 다음 정보를 잘 보존한다.

| line plot에서 보이는 것 | bed-exit 해석 | 설비 해석 |
|---|---|---|
| peak | 갑작스러운 움직임 | 진동 충격, 압력 spike |
| slope | 점진적 이동 | 온도 상승, RMS 증가 |
| fluctuation | 자세 변화 | 불안정 운전, hunting |
| flatline | 안정 상태 | 정상 안정, sensor stuck 가능 |
| transition shape | 침대 밖으로 나가기 전 움직임 | 고장 전조 pattern |

## 핵심 개념 4. RP, MTF, GAF texture stream

### Recurrence Plot, RP

RP는 시계열이 과거의 어떤 상태와 비슷한 상태를 다시 방문하는지를 이미지로 표현한다.

설비 예:

| RP가 잘 보는 것 | 예시 |
|---|---|
| 반복 cycle | 압력 valve cycle |
| motif 반복 | 진동 충격 반복 |
| 정상 상태 재방문 | 안정 운전 |
| recurrence 구조 붕괴 | 이상 상태 |

### Markov Transition Field, MTF

MTF는 값을 몇 개의 상태 bin으로 나누고, 상태 간 전이 확률을 시간 쌍 matrix로 표현한다.

설비 예:

```text
압력 상태: Low → Medium → High → Medium
```

정상 cycle과 abnormal transition을 구분하는 데 도움이 된다.

### Gramian Angular Field, GAF

GAF는 시계열 값을 angle로 바꾼 뒤, 모든 시간 쌍의 angular relation을 이미지로 만든다. 전체 shape와 global temporal relation을 보기 좋다.

## 핵심 개념 5. 왜 RP/MTF/GAF는 load 하나에만 적용했나?

논문은 line plot에는 여러 variable을 넣지만, RP/MTF/GAF는 가장 informative한 load signal에만 적용한다. RP/MTF/GAF가 기본적으로 univariate 변환이기 때문이다.

설비 응용에서는 다음처럼 바꿀 수 있다.

| 문제 | texture map 대상 후보 |
|---|---|
| 베어링 결함 | envelope peak, vibration RMS |
| 모터 전기 이상 | current RMS 또는 harmonic energy |
| 압력 이상 | pressure signal |
| 온도 과열 | temperature residual |
| RUL | health index |

## 핵심 개념 6. Dual-stream Swin Transformer

구조는 다음과 같다.

```text
Stream 1:
RGB line plot image
→ Swin-Tiny encoder
→ line plot feature

Stream 2:
RP + MTF + GAF 3-channel texture image
→ Swin-Tiny encoder
→ texture feature

Fusion:
cross-attention
→ joint representation
→ classification
```

Line plot stream은 local waveform pattern에 강하고, texture stream은 higher-order dynamics에 강하다.

## 핵심 개념 7. Cross-attention fusion

단순 concat은 두 modality를 그냥 붙인다.

```text
line feature + texture feature → concat → classifier
```

Cross-attention은 sample마다 어떤 정보가 더 중요한지 동적으로 조절한다.

| 상황 | 더 중요할 수 있는 표현 |
|---|---|
| 명확한 peak | line plot |
| 반복 motif 변화 | RP |
| 상태 전이 변화 | MTF |
| global shape 변화 | GAF |
| 복합 이상 | fusion 필요 |

논문 ablation에서 cross-attention fusion은 단순 concatenation보다 F1, recall, AUPRC를 크게 개선했다.

## 실험 결과

데이터는 장기요양시설 95개 침대에서 6개월 동안 수집되었다. 57개 침대의 579일치 sensor reading을 수동 검토해 라벨링했고, transition period 1,211개와 대응하는 non-active period를 구성했다.

주요 결과:

| 모델 | F1 | Recall | Precision | Accuracy | AUPRC |
|---|---:|---:|---:|---:|---:|
| TCN | 0.688 | 0.738 | 0.644 | 0.821 | 0.721 |
| TimesNet | 0.690 | 0.631 | 0.759 | 0.848 | 0.750 |
| Mamba | 0.701 | 0.631 | 0.787 | 0.855 | 0.787 |
| iTransformer | 0.730 | 0.707 | 0.754 | 0.860 | 0.807 |
| ViTST | 0.734 | 0.679 | 0.797 | 0.868 | 0.832 |
| ViFusionTST | 0.794 | 0.827 | 0.763 | 0.885 | 0.868 |

ViFusionTST는 특히 recall이 높다. 이는 전조 event를 놓치지 않는 데 중요하다.

## Ablation 핵심

### Fusion 방법

| Fusion method | F1 | Recall | AUPRC |
|---|---:|---:|---:|
| Early Concatenation | 0.658 | 0.582 | 0.761 |
| Mid Concatenation | 0.764 | 0.720 | 0.847 |
| Cross Fusion | 0.794 | 0.827 | 0.868 |

Cross-attention fusion이 가장 좋았다.

### 입력 modality

| Input | F1 | Recall | AUPRC |
|---|---:|---:|---:|
| Line Plot | 0.771 | 0.824 | 0.835 |
| RP/MTF/GAF | 0.659 | 0.640 | 0.732 |
| Both | 0.794 | 0.827 | 0.868 |

Line plot이 강하지만 texture map을 fusion하면 더 좋아진다.

### Vision encoder

Swin-Tiny가 Swin-Base, ViT-Large 등 더 큰 모델보다 더 좋았다. 시계열 이미지가 자연 이미지보다 단순할 수 있으므로 작은 모델이 더 잘 맞을 수 있다는 교훈을 준다.

## 설비 진단에 적용한다면

예: 모터 베어링 고장 전조 감지

### Stream 1. Line plot

```text
vibration RMS
kurtosis
envelope peak
current RMS
temperature
speed/load
```

### Stream 2. Texture map

핵심 health signal을 고른다.

```text
envelope peak sequence 또는 vibration RMS sequence
→ RP
→ MTF
→ GAF
```

### Fusion

```text
line plot Swin encoder
texture map Swin encoder
cross-attention fusion
정상 / early warning / critical 분류
```

## 실무 주의점

### 1. Split

위험한 방식:

```text
긴 설비 run
→ 3시간 lookback window 생성
→ image representation 생성
→ window random split
```

인접 window image가 거의 같아져 성능이 과대평가된다.

안전한 방식:

| 목적 | split |
|---|---|
| 새 설비 일반화 | unit holdout |
| 새 run 일반화 | run holdout |
| 새 운전 조건 | condition holdout |
| 시간 예측 | 과거 train, 미래 test |
| RUL | bearing/engine unit split |

### 2. 어떤 신호에 RP/MTF/GAF를 적용할지 선택해야 한다

| 설비 문제 | 후보 신호 |
|---|---|
| 베어링 결함 | envelope peak, vibration RMS |
| 모터 전기 이상 | current harmonic energy |
| 압력 이상 | pressure signal |
| 온도 과열 | temperature residual |
| RUL | health index |

### 3. 운영 지표

classification score만으로는 부족하다.

| 지표 | 의미 |
|---|---|
| recall | 고장 전조 missed detection 방지 |
| precision | false alarm 감소 |
| AUPRC | class imbalance 대응 |
| false alarm per day | 운영 가능성 |
| lead time | 예지보전 가치 |
| detection delay | 늦은 경보 방지 |

## 한계

1. Bed-exit 특화 문제라 설비에 바로 일반화하면 안 된다.
2. line plot, RP, MTF, GAF 생성 비용과 설계가 복잡하다.
3. 어떤 신호에 texture map을 적용할지 도메인 지식이 필요하다.
4. 큰 vision backbone이 항상 좋은 것은 아니다.
5. split 방식이 충분히 엄격하지 않으면 성능이 과대평가될 수 있다.
6. explainability 분석은 별도로 추가해야 한다.

## 이전 논문들과 연결

| 논문 | 연결 |
|---|---|
| Plotting Time | line plot image가 강력한 baseline |
| XAI for TSC | line plot에 Grad-CAM/LIME 가능 |
| ViTST | line graph image + Swin Transformer |
| TSSI | multivariate screenshot image + CNN |
| ViFusionTST | line plot + RP/MTF/GAF + cross-attention fusion |

## 오늘 공부용 요약

- ViFusionTST는 early bed-exit prediction을 위한 time-series-as-image fusion model이다.
- line plot은 raw waveform detail을 보존한다.
- RP는 recurrence structure를 본다.
- MTF는 state transition을 본다.
- GAF는 global angular relation을 본다.
- line plot stream과 texture stream을 dual Swin encoder로 처리한다.
- cross-attention fusion이 단순 concat보다 훨씬 좋았다.
- 설비에서는 고장 전조 transition을 잡는 데 응용할 수 있다.
- 모델보다 split이 먼저다.

논문 URL: https://arxiv.org/abs/2502.08869

# 6편: Harnessing Vision Models for Time Series Analysis: A Survey

## 결론부터

이 survey의 핵심 결론은 **시계열을 숫자 sequence나 language token으로만 다룰 필요는 없고, line plot·heatmap·spectrogram·GAF·RP 같은 이미지 표현으로 바꾼 뒤 CNN, ViT, LVM, VLM, LMM을 활용하는 하나의 큰 연구 흐름이 형성되고 있다**는 것이다.

실무적으로는 다음처럼 읽는 것이 좋다.

> 시계열-as-image 접근은 “그림으로 바꾸면 좋아진다”는 단순한 이야기가 아니다. 어떤 이미지 표현을 쓰는지, 어떤 vision model을 쓰는지, scale·channel·image size·복원 방식·split을 어떻게 설계하는지에 따라 성패가 갈린다.

## 이 논문을 한 문장으로 요약하면

시계열을 이미지로 변환하는 방법과, 변환된 이미지를 vision model로 처리하는 방법을 이중 taxonomy로 정리하고, 전처리·후처리·미래 연구 방향까지 다룬 vision model 기반 시계열 분석 survey다.

## 왜 이 논문이 필요했나?

앞에서 본 논문들은 개별 방법이었다.

| 논문 | 핵심 |
|---|---|
| Plotting Time | line plot image + CNN |
| ViTST | irregular MTS → line graph grid image → Swin |
| TSSI | screenshot image channel → CNN |
| ViFusionTST | line plot + RP/MTF/GAF fusion |
| MLLM4TS | 숫자 시계열 + plot image + LLM fusion |

이 survey는 이들을 큰 지도 안에 넣어준다.

```text
시계열 전처리
→ 시계열을 이미지로 변환
→ 이미지 전처리
→ vision model로 분석
→ 후처리
```

설비 데이터에서는 다음처럼 대응된다.

```text
센서 결측, 이상값, scale, 운전 mode 정리
→ 온도/압력/전류/진동 RMS를 line plot, heatmap, spectrogram 등으로 변환
→ image size, channel 수, normalization 조정
→ CNN/ResNet/ViT/Swin/LMM으로 분석
→ class, anomaly score, forecast, RUL stage로 해석
```

## Taxonomy 1. Time Series to Image Transformation

시계열을 어떤 이미지로 바꿀 것인가?

| 이미지 변환 | 직관 | 설비 예시 |
|---|---|---|
| Line Plot | 사람이 보는 선 그래프 | 온도 trend, 진동 peak |
| Heatmap | 값 matrix를 색으로 표현 | 센서 × 시간 상태 |
| Spectrogram | 시간-주파수 변화 | 진동, 전류 harmonic |
| GAF | 시점 간 angular correlation | global health pattern |
| RP | 유사 상태 재방문 | 반복 cycle, motif |
| MTF/기타 | 상태 전이, bitmap, delay embedding | 압력 상태 전이, multi-view fusion |

## 핵심 개념 1. Line Plot

Line plot은 가장 직관적인 시계열 이미지다.

```text
x축: time
y축: value
선: 시간에 따른 값 변화
```

장점:

| 장점 | 설명 |
|---|---|
| 사람이 이해하기 쉬움 | 현장 dashboard에 좋음 |
| peak/slope/trend가 잘 보임 | 고장 전조 설명 가능 |
| XAI와 연결 쉬움 | Grad-CAM, LIME |

한계:

| 위험 | 설명 |
|---|---|
| y축 scale | amplitude 보존/손실 |
| x축 scale | duration 왜곡 |
| 축/label/grid | artifact 학습 |
| line thickness | 작은 spike 희석 |
| image size | 고주파 정보 손실 |

## 핵심 개념 2. Heatmap

Heatmap은 다변량 시계열을 sensor × time matrix로 표현한다.

```text
세로축: sensor
가로축: time
색: value magnitude
```

설비 예:

| sensor | t1 | t2 | t3 |
|---|---:|---:|---:|
| temperature | 70 | 71 | 72 |
| current | 10 | 12 | 13 |
| vibration | 0.2 | 0.4 | 1.0 |

Heatmap의 핵심 위험은 variate order다. 관련 센서를 가까이 배치해야 CNN local filter가 관계를 배우기 쉽다.

좋은 배치 예:

```text
전류, 전압, 전력
온도, 냉각수
압력, 유량
진동 x, y, z
```

## 핵심 개념 3. Spectrogram

Spectrogram은 주파수 성분이 시간에 따라 어떻게 변하는지 보여준다.

설비에서 매우 중요하다.

| 신호 | spectrogram이 유용한 이유 |
|---|---|
| 진동 | 결함 주파수, transient shock |
| 전류 | harmonic, sideband, startup transient |
| 압력 | pulsation, valve cycle 변화 |
| acoustic emission | 고장 event의 주파수 변화 |
| motor startup | 시간에 따른 frequency 변화 |

설정이 중요하다.

| 선택 | 영향 |
|---|---|
| window size | 시간 해상도 vs 주파수 해상도 |
| wavelet type | transient 감지 |
| frequency range | 결함 주파수 포함 여부 |
| log scale | 작은 성분 강조 |
| sampling rate | aliasing 방지 |

## 핵심 개념 4. GAF

GAF는 시계열 값을 angle로 변환한 뒤 모든 시간 쌍의 angular relation을 이미지로 만든다.

장점:

| 장점 | 설명 |
|---|---|
| global shape 표현 | 전체 health trend |
| 시점 간 관계 | 초기와 후반 상태 관계 |
| CNN/ViT 입력 가능 | image matrix 생성 |

한계:

| 한계 | 설명 |
|---|---|
| 비용 | 보통 T×T matrix |
| 긴 sequence 부담 | window 단위 적용 필요 |
| 해석성 | line plot보다 직관성 낮음 |

## 핵심 개념 5. RP

RP는 비슷한 상태가 언제 다시 나타나는지 보여준다.

설비 예:

| RP가 잘 보는 것 | 예시 |
|---|---|
| 반복 cycle | 압력 valve cycle |
| motif 반복 | 진동 충격 반복 |
| 정상 운전 pattern | 비슷한 상태 재방문 |
| 이상 pattern | recurrence 구조 붕괴 |

RP는 반복 pattern과 abnormal dynamics를 보기 좋지만, thresholding에 따른 정보 손실이 있을 수 있다.

## Taxonomy 2. Imaged Time Series Modeling

이미지로 변환한 뒤 어떤 모델을 쓸 것인가?

| 모델링 방식 | 예시 |
|---|---|
| Conventional vision models | CNN, ResNet, VGG, Inception |
| Large Vision Models | ViT, Swin, BEiT, MAE |
| Large Multimodal Models | LLaVA, GPT-4o, Gemini, Claude |
| Task-specific heads | classification, forecasting, anomaly, recovery |

## 핵심 개념 6. Conventional vision models

CNN/ResNet은 작은 데이터와 classification에 실용적이다.

| 장점 | 설명 |
|---|---|
| 구현 쉬움 | PyTorch/torchvision |
| 학습 비용 낮음 | LVM보다 가벼움 |
| Grad-CAM 가능 | 설명성 |
| spectrogram에 강함 | 진동/전류 분석 |

설비에서는 spectrogram 기반 fault classification에 ResNet18/50이 좋은 출발점이다.

## 핵심 개념 7. Large Vision Models

ViT, Swin, MAE 같은 대규모 vision model은 image patch와 pretraining을 활용한다.

| 데이터 | LVM 후보 |
|---|---|
| spectrogram/scalogram | ViT, Swin |
| line graph grid | Swin |
| screenshot MTSC | ResNet/Swin |
| masked sensor representation | MAE-style pretraining |

하지만 큰 모델이 항상 좋은 것은 아니다. 시계열 이미지는 자연 이미지보다 단순할 수 있고 데이터가 작으면 overfitting이 생긴다.

## 핵심 개념 8. LMM/VLM

VLM/LMM은 plot image와 text prompt를 함께 입력으로 받을 수 있다.

예:

```text
진동 RMS plot + 정비 로그 + 질문
→ “어느 구간이 이상한가?”
```

설비 활용:

| 사용 | 예시 |
|---|---|
| 경보 설명 | “14:32부터 진동이 급상승했습니다.” |
| plot QA | “이 압력 plot에서 이상 구간은 어디인가?” |
| 정비 리포트 초안 | 센서 plot과 로그 요약 |
| human-in-the-loop | 작업자에게 설명 가능한 인터페이스 |

하지만 hallucination, scale 오독, data security 문제가 있으므로 최종 자동판정 단독 사용은 위험하다.

## 전처리와 후처리의 핵심

### Normalization

| 방식 | 장점 | 위험 |
|---|---|---|
| sample별 normalization | shape 비교에 좋음 | amplitude 정보 손실 |
| train min/max | leakage 방지 | test extreme clipping |
| sensor spec range | 물리 해석 가능 | spec 관리 필요 |
| z-score | scale 차이 완화 | 고장 amplitude 손실 가능 |

### Image alignment

vision model은 보통 input size와 channel 수를 요구한다.

| 문제 | 예시 |
|---|---|
| 센서 10개 channel | RGB pre-trained ResNet과 안 맞음 |
| heatmap 크기 불일치 | interpolation 필요 |
| 긴 sequence 압축 | 해상도 손실 |
| 작은 spike | resize 과정에서 사라짐 |
| ViT positional embedding | input size mismatch |

### Time series recovery

classification은 image에서 class만 내면 되지만 forecasting은 numerical value를 복원해야 한다.

| 이미지 표현 | forecasting 적합성 |
|---|---|
| heatmap | 값 복원 쉬움 |
| line plot | 복원 까다로움 |
| GAF | 일부 inverse 가능 |
| RP | thresholding으로 복원 어려움 |
| screenshot binary image | 직접 value recovery 어려움 |

## 실무 적용 가이드

### 1. 표현은 목적별로 고른다

| 문제 | 우선 표현 |
|---|---|
| 온도 trend | line plot, heatmap |
| 압력 cycle | line plot, RP, MTF |
| 베어링 raw vibration | spectrogram, scalogram, envelope spectrum |
| 모터 전류 | spectrogram, harmonic image |
| 다변량 sensor 상태 | heatmap, TSSI, ViTST grid |
| 불규칙 센서 log | line graph grid |
| 설명 dashboard | line plot + heatmap + Grad-CAM |

### 2. 모델은 데이터 크기와 표현에 맞춘다

| 상황 | 모델 |
|---|---|
| 데이터 작음 | small CNN, ResNet18 |
| 데이터 중간 | ResNet50, EfficientNet, Swin-Tiny |
| 데이터 많음 | ViT/Swin/MAE fine-tuning |
| irregular multivariate | ViTST |
| screenshot MTSC | TSSI |
| multi-view | ViFusionTST |
| explanation 중심 | Grad-CAM, VLM |

### 3. baseline과 비교한다

| baseline | 이유 |
|---|---|
| rule threshold | 현장 기준 |
| catch22/tsfresh + XGBoost | tabular feature 강력 |
| MiniRocket | TSC 강력 baseline |
| InceptionTime | raw sequence CNN |
| HIVE-COTE | 성능 상한선 |
| TCN/N-BEATS/PatchTST | forecasting/anomaly baseline |
| ARIMA/STL/Kalman | residual baseline |

## split 주의

위험한 방식:

```text
긴 설비 run
→ sliding window
→ image 변환
→ random train/test split
```

인접 window가 거의 같은 image가 된다.

안전한 방식:

| 목적 | split |
|---|---|
| 새 설비 일반화 | unit holdout |
| 새 run 일반화 | run holdout |
| 새 운전 조건 | condition holdout |
| 같은 설비 미래 | time split |
| RUL | bearing/engine unit split |
| 이상탐지 | 정상 train, 미래 test |

## 한계

1. Survey라서 새 모델 성능을 직접 제시하는 논문은 아니다.
2. 시계열-as-image가 항상 좋은 것은 아니다.
3. 이미지 변환 과정에서 정보 손실과 artifact가 생긴다.
4. 설비 신호처리 지식을 대체하지 않는다.
5. LMM/VLM은 hallucination과 reproducibility 문제가 있다.
6. data leakage를 자동으로 막아주지 않는다.

## 오늘 공부용 요약

- 이 survey는 vision model 기반 시계열 분석의 큰 지도를 제공한다.
- 주요 이미지 표현은 line plot, heatmap, spectrogram, GAF, RP, 기타 multi-view 방법이다.
- 모델은 CNN/ResNet 같은 conventional model, ViT/Swin 같은 LVM, GPT-4o/LLaVA 같은 LMM/VLM으로 나뉜다.
- 설비 데이터에서는 진동/전류에는 spectrogram, 온도/압력에는 line plot/heatmap, 다변량 센서에는 heatmap/TSSI/ViTST가 유용할 수 있다.
- forecasting에서는 image에서 numerical time series를 복원하는 후처리가 중요하다.
- 모델보다 split이 먼저다.

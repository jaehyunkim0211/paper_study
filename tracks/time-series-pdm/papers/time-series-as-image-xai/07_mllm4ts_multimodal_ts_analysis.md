논문 URL: https://arxiv.org/abs/2510.07513

# 7편: MLLM4TS: Leveraging Vision and Multimodal Language Models for General Time-Series Analysis

## 결론부터

이 논문의 핵심 결론은 **시계열을 숫자 sequence로만 LLM에 억지로 맞추지 말고, 원본 숫자 시계열 branch와 line plot 이미지 branch를 함께 사용하면 classification, anomaly detection, forecasting을 하나의 multimodal framework로 더 잘 처리할 수 있다**는 것이다.

실무적으로는 다음처럼 이해하면 좋다.

> MLLM4TS는 “시계열 plot을 MLLM에 그냥 보여주자”가 아니라, 숫자 시계열의 세밀한 시간 정보와 이미지 plot의 전역적·채널 간 시각 정보를 각각 embedding한 뒤, GPT-2 같은 language backbone 안에서 융합하는 구조다.

## 이 논문을 한 문장으로 요약하면

MLLM4TS는 multivariate time series를 원본 숫자 patch와 color-coded line plot image라는 두 modality로 동시에 표현하고, CLIP vision encoder와 GPT-2 language model을 결합해 classification, anomaly detection, forecasting을 모두 처리하려는 general-purpose multimodal time-series framework다.

## 왜 이 논문이 필요했나?

기존 흐름은 크게 둘로 나뉜다.

| 흐름 | 한계 |
|---|---|
| 숫자 시계열 모델 | cross-channel visual context와 사람이 보는 plot pattern을 직접 쓰기 어려움 |
| 시계열-as-image 모델 | 원본 수치 detail이 손실될 수 있음 |
| LLM 기반 시계열 모델 | continuous numerical data와 discrete token 사이의 modality gap |

MLLM4TS는 이 세 흐름을 결합한다.

```text
원본 숫자 시계열
+ color-coded line plot image
+ pretrained language backbone
→ general time-series analysis
```

설비 진단에서는 다음 문제와 연결된다.

| 실무 문제 | MLLM4TS식 접근 |
|---|---|
| 여러 센서가 동시에 변함 | line plot image로 cross-channel visual context 제공 |
| 숫자 sequence가 길고 복잡함 | patch tokenizer로 숫자 branch 구성 |
| plot에는 사람이 보는 pattern이 있음 | vision branch로 전역 패턴 반영 |
| task마다 모델이 다름 | task-specific head만 바꿔 공통 backbone 사용 |

## 핵심 개념 1. 숫자 branch + 이미지 branch

전체 구조는 다음이다.

```text
원본 multivariate time series
 ├─ 숫자 branch: time-series tokenizer → time-series embedding
 └─ 이미지 branch: color-coded line plot → CLIP vision encoder → plot embedding

time-series embedding + plot embedding
→ GPT-2 language backbone
→ task-specific head
→ classification / anomaly detection / forecasting
```

| branch | 보는 정보 | 설비 예시 |
|---|---|---|
| 숫자 branch | 세밀한 시간 값, local detail | 진동 RMS의 정확한 변화, 온도 slope |
| 이미지 branch | 전체 plot shape, channel 간 visual relation | 전류·온도·진동이 함께 변하는 형태 |
| GPT-2 backbone | multimodal token sequence 처리 | 두 표현을 결합해 task 판단 |

## 핵심 개념 2. Color-coded horizontal line plot

각 channel을 고유한 색의 line plot으로 그리고, horizontal layout으로 composite image를 만든다.

예:

| color | sensor |
|---|---|
| red | temperature |
| blue | current RMS |
| green | vibration RMS |
| purple | pressure |

Horizontal layout의 장점은 x축 time alignment가 명확하다는 점이다.

```text
같은 x 위치 = 같은 시간대
```

설비 예:

| 같은 시간 구간 | 시각적 정보 |
|---|---|
| 14:00~14:10 | 전류 상승, 온도 변화 없음 |
| 14:10~14:20 | 전류 유지, 온도 상승, 진동 peak |
| 14:20~14:30 | 압력 spike, 유량 감소 |

## 핵심 개념 3. Time-series tokenizer

숫자 branch에서는 시계열을 patch로 나눈다.

```text
원본 sensor sequence
→ RevIN normalization
→ non-overlapping patches
→ linear projection
→ embedding
```

PatchTST와 비슷하게, 한 점이 아니라 구간 patch를 token처럼 사용한다.

설비 주의점:

RevIN이나 normalization은 distribution shift에는 도움이 되지만, 진동 amplitude 증가처럼 absolute scale이 고장 정보인 경우 이를 약화시킬 수 있다. raw scale feature나 domain feature를 별도로 보존하는 것이 좋다.

## 핵심 개념 4. CLIP vision encoder

Image branch는 CLIP-ViT-L-14를 사용한다. CLIP은 image-text alignment를 학습한 모델이므로, GPT-2 같은 language backbone과 결합하기 더 자연스럽다는 것이 논문의 해석이다.

실무적으로는:

| 장점 | 한계 |
|---|---|
| line, color, spatial layout 인식 가능 | 설비 물리 주파수를 안다는 뜻은 아님 |
| language backbone과 embedding 결합에 유리 | raw 고주파 signal은 plot에서 손실 가능 |
| pretrained representation 활용 | computation cost 증가 |

## 핵심 개념 5. Plot projection module

CLIP visual embedding은 GPT-2 embedding과 바로 맞지 않는다. 그래서 linear projection으로 visual embedding을 language model embedding space에 맞춘다.

```text
CLIP visual embedding
→ plot projection
→ GPT-2가 처리할 수 있는 embedding
```

이는 modality gap을 줄이는 adapter 역할이다.

## 핵심 개념 6. Temporal-aware visual patch alignment

MLLM4TS의 중요한 아이디어다.

line plot image에서 가로축은 time이다. 따라서 같은 horizontal index의 visual patch들은 같은 time segment에 대응한다.

```text
온도 plot의 x=10
전류 plot의 x=10
진동 plot의 x=10
압력 plot의 x=10
→ 같은 time segment
```

모델은 같은 시간대의 visual patches를 group으로 묶고, 숫자 time-series patch와 temporal resolution을 맞춘다.

설비 예:

```text
숫자 patch t4~t5:
temperature 71~72.5, current 13, vibration 0.6~1.2

visual patch t4~t5:
red line 상승, blue line 유지, green line peak

→ 같은 time segment로 align
```

## 핵심 개념 7. Early fusion

논문은 early fusion과 late fusion을 비교한다.

| 방식 | 설명 |
|---|---|
| Early fusion | time-series embedding과 plot embedding을 LLM 전에 합침 |
| Late fusion | LLM 처리 후 나중에 합침 |

Early fusion이 더 좋았다.

설비 해석:

> 같은 시간대의 숫자 변화와 plot 모양을 초반부터 함께 묶으면, 전류 상승+진동 peak+온도 지연 상승 같은 low-level cross-modal pattern을 더 잘 볼 수 있다.

## 핵심 개념 8. GPT-2 backbone과 selective fine-tuning

Language backbone은 GPT-2다. 전체를 fine-tuning하지 않고, self-attention block과 FFN은 frozen하며 positional embedding과 layer normalization 등을 선택적으로 fine-tuning한다.

실무 교훈:

> 더 큰 LLM이 항상 좋은 것은 아니다. 시계열 데이터가 작고 noisy하면 작은 backbone + 좋은 modality alignment가 더 실용적일 수 있다.

## 핵심 개념 9. Task-specific heads

MLLM4TS는 같은 backbone 위에 task head를 붙인다.

| task | head | score/loss |
|---|---|---|
| classification | linear + softmax | cross-entropy |
| anomaly detection | reconstruction | MSE anomaly score |
| forecasting | future value prediction | MSE |

설비에서는 다음으로 대응된다.

| task | 설비 예시 |
|---|---|
| classification | 정상/고장 유형 분류 |
| anomaly detection | 정상 패턴 재구성 실패 → 이상 |
| forecasting | 미래 온도·압력·진동 RMS 예측 |

## 실험 결과 요약

### Classification

UEA 10개 multivariate classification dataset 평균 accuracy:

| 모델 | 평균 accuracy |
|---|---:|
| XGBoost | 66.0 |
| ROCKET | 72.5 |
| TimesNet | 73.6 |
| UniTS | 75.0 |
| OFA | 72.2 |
| MLLM4TS | 76.7 |

MLLM4TS는 평균 기준 가장 높았다.

### Anomaly detection

TSB-AD-M benchmark 평균 VUS-PR:

| 모델 | VUS-PR |
|---|---:|
| PCA | 0.310 |
| CNN | 0.313 |
| OmniAnomaly | 0.312 |
| LSTMAD | 0.307 |
| USAD | 0.304 |
| AutoEncoder | 0.295 |
| OFA | 0.296 |
| MLLM4TS | 0.349 |

MLLM4TS는 전체 평균에서 좋은 결과를 보였다. 다만 모든 domain에서 항상 압도적인 것은 아니다.

### Forecasting

Weather, Solar, ETTh1, ECL, Traffic에서 경쟁력 있는 결과를 보였지만, PatchTST, iTransformer, VisionTS 등 forecasting 전용 모델을 모든 dataset에서 이긴 것은 아니다.

실무 해석:

> MLLM4TS는 general framework로는 흥미롭지만, forecasting만 목표라면 PatchTST, N-BEATS, TCN, DeepAR, TimeGPT 같은 전용 baseline과 반드시 비교해야 한다.

## Ablation 교훈

| 비교 | 결과 | 실무 해석 |
|---|---|---|
| horizontal vs grid | horizontal이 더 좋음 | time alignment가 중요 |
| CLIP vs ResNet | CLIP이 classification에서 더 좋음 | multimodal 결합에는 vision-language encoder가 유리 |
| early vs late fusion | early fusion이 더 좋음 | 낮은 수준에서 modality 결합 중요 |
| patch size sensitivity | MLLM4TS가 덜 민감 | visual context가 patch 선택을 보완 |
| 큰 LLM | 항상 이점 없음 | 큰 모델보다 alignment/설계가 중요 |
| channel pruning | 대표 channel만 plot하는 것이 유리할 수 있음 | high-dimensional sensor plot은 clutter 위험 |

## 실무 적용 가이드

### 1. Raw vibration보다 feature sequence에 먼저 적용

베어링 진단 예:

```text
raw vibration
→ RMS
→ kurtosis
→ envelope peak
→ BPFO/BPFI band energy
→ temperature
→ current RMS
→ MLLM4TS
```

raw waveform line plot은 고주파 결함 정보를 잃을 수 있다.

### 2. channel color mapping 고정

```text
red = temperature
blue = current
green = vibration
```

train/test/deployment에서 절대 바뀌면 안 된다.

### 3. representative channel selection은 train에서만

센서가 300개라면 모두 plot하지 않고 대표 channel을 고를 수 있다. 하지만 전체 데이터 correlation으로 고르면 test 정보가 들어갈 수 있다. train에서만 선택해야 한다.

## 설비 workflow 예시

```text
1. unit/run/time/condition split 설계
2. raw sensor 품질 점검
3. vibration에서 RMS, kurtosis, envelope peak, FFT band energy 생성
4. train 기준 scaling/RevIN/domain scale 정책 결정
5. representative channel 선택은 train에서만 수행
6. color-coded horizontal plot image 생성
7. numeric feature sequence patch 생성
8. MLLM4TS 학습 또는 fine-tuning
9. MiniRocket, XGBoost, TCN, PatchTST, TimeGPT와 비교
10. classification: precision, recall, F1, confusion matrix 평가
11. anomaly: false alarm, missed detection, detection delay 평가
12. forecasting: MAE, RMSE, MASE 평가
13. attention/plot analysis로 case review
14. 현장 정비 로그와 비교
```

## 한계

1. Vision branch 때문에 계산 비용이 증가한다.
2. irregular sampling은 아직 직접 다루지 않는다.
3. line plot image가 모든 정보를 보존하지 않는다.
4. forecasting 전용 모델을 항상 이기지는 않는다.
5. anomaly detection benchmark 지표와 운영 지표는 다르다.
6. scaling, plotting, channel pruning, split 과정에서 leakage 위험이 크다.

## split 주의

위험한 방식:

```text
긴 설비 run
→ sliding window 생성
→ line plot image 생성
→ window random split
```

안전한 방식:

| 목적 | split |
|---|---|
| 새 설비 일반화 | unit holdout |
| 새 run 일반화 | run holdout |
| 새 운전 조건 | load/speed condition holdout |
| 미래 예측 | time split |
| RUL | bearing/engine unit split |
| 이상탐지 | 정상 train, 미래 event test |

## 이전 논문들과 연결

| 논문 | 연결 |
|---|---|
| ViTST | line graph image + vision transformer |
| TSSI | multivariate line/screenshot image |
| ViFusionTST | line plot + texture map fusion |
| TimeGPT | 시계열 foundation model |
| PatchTST | numeric patch tokenization |
| MLLM4TS | numeric patch + plot image + LLM fusion |

## 오늘 공부용 요약

- MLLM4TS는 숫자 시계열 branch와 plot image branch를 함께 쓰는 multimodal framework다.
- image branch는 color-coded horizontal line plot과 CLIP visual encoder를 사용한다.
- 숫자 branch는 RevIN, patching, linear projection을 사용한다.
- temporal-aware visual patch alignment로 visual patch와 numeric time segment를 맞춘다.
- early fusion이 late fusion보다 좋았다.
- GPT-2 backbone을 selective fine-tuning한다.
- classification, anomaly detection, forecasting head를 모두 지원한다.
- 설비에서는 raw vibration보다 RMS, kurtosis, envelope peak, FFT band energy 같은 feature sequence에 적용하는 것이 실무적이다.
- 모델보다 split이 먼저다.

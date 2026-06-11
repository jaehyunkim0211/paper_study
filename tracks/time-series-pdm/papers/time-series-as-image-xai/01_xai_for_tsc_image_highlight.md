논문 URL: https://arxiv.org/abs/2311.17110

# 1편: XAI for Time-Series Classification Leveraging Image Highlight Methods

## 결론부터

이 논문의 핵심 결론은 **시계열을 원본 숫자 배열로만 설명하려 하지 말고, 시계열을 2D plot 이미지로 변환한 뒤 LIME·Grad-CAM 같은 이미지 XAI 기법을 적용하면 딥러닝 시계열 분류 모델의 판단 근거를 사람이 더 직관적으로 볼 수 있다**는 것이다.

실무적으로는 이 논문을 “시계열을 이미지로 바꾸면 무조건 성능이 좋아진다”는 주장으로 읽기보다, **설비 진단 모델의 예측을 현장 엔지니어에게 설명하기 위한 보조 XAI 방법**으로 읽는 것이 좋다.

## 이 논문을 한 문장으로 요약하면

원본 시계열 배열은 Dense teacher model에 넣고, 같은 시계열을 2D line plot 이미지로 바꾼 입력은 CNN student model에 넣은 뒤, CNN 쪽에 LIME과 Grad-CAM을 적용해 **모델이 plot의 어느 부분을 보고 분류했는지**를 시각적으로 보여주는 teacher-student 기반 XAI 시계열 분류 접근이다.

## 왜 이 논문이 필요했나?

시계열 분류 모델은 정확도가 높아도 설명이 어렵다. 예를 들어 베어링 진동 데이터를 보고 모델이 “외륜 결함”이라고 판단했을 때, 현장 엔지니어는 보통 다음을 묻는다.

> 그래서 신호의 어느 구간 때문에 외륜 결함이라고 판단했나?

일반적인 CNN, Transformer, MiniRocket, XGBoost 모델은 class를 잘 맞힐 수 있지만, 판단 근거를 현장 사용자에게 바로 설명하기 어렵다. 이 논문은 시계열을 사람이 볼 수 있는 plot 이미지로 변환하면, 이미지 XAI 기법을 그대로 적용할 수 있다는 점에 착안한다.

설비 진단에서는 다음 질문이 중요하다.

| 현장 질문 | 필요한 설명 |
|---|---|
| 왜 고장이라고 했나? | 어느 시간 구간, 어떤 peak, 어떤 pattern 때문인가 |
| 정상 부하 변화와 다른가? | load/speed/setpoint와 비교 |
| 센서 오류는 아닌가? | spike, dropout, flatline 여부 |
| 경보를 믿어도 되나? | false alarm 가능성 |
| 어떤 조치를 해야 하나? | 점검, 감시 강화, 정지 여부 |

이 논문은 이 중 첫 번째 질문, 즉 **시계열 plot의 어느 부분이 분류 판단에 중요했는가**를 다룬다.

## 핵심 아이디어

전체 구조는 다음처럼 이해하면 된다.

```text
원본 시계열
 ├─ 숫자 배열 그대로 → Dense teacher model → 분류
 └─ line plot 이미지로 변환 → CNN student model → LIME/Grad-CAM 설명
```

| 표현 | 모델 | 역할 |
|---|---|---|
| 숫자 배열 | Dense network | 원본 시계열 기반 teacher |
| 2D plot image | CNN | 사람이 볼 수 있는 이미지 설명을 만들 student |
| image heatmap | LIME / Grad-CAM | 중요한 plot 영역 시각화 |

## 핵심 개념 1. 시계열을 2D plot 이미지로 바꾸기

이 논문은 GAF, RP, MTF 같은 복잡한 이미지 변환을 쓰는 것이 아니라, x축을 time, y축을 value로 하는 line plot을 사용한다.

예를 들어 진동 RMS가 다음과 같다고 하자.

```text
[0.10, 0.11, 0.12, 0.50, 1.20, 0.40, 0.15]
```

plot으로 그리면 중간에 큰 peak가 보인다. CNN은 이 이미지를 보고 고장 여부를 분류한다.

설비 관점에서 plot은 다음을 직관적으로 보여준다.

| 센서 | plot에서 보이는 것 |
|---|---|
| 온도 | 서서히 상승하는 trend |
| 압력 | 반복 cycle과 spike |
| 전류 RMS | startup 과도 구간, 부하 변화 |
| 진동 RMS | 고장 전 peak 증가 |
| raw vibration | 충격 waveform, noise |

단, plot 이미지는 y축 scale, line thickness, image resolution, 축 표시 여부에 민감하다. 모델이 진짜 신호 물리를 배운 것인지 plot style artifact를 배운 것인지 반드시 확인해야 한다.

## 핵심 개념 2. Teacher-student architecture

Teacher-student 구조는 다음처럼 이해할 수 있다.

| 역할 | 설명 |
|---|---|
| Teacher | 원본 숫자 배열을 보는 더 강한 모델 |
| Student | plot 이미지를 보는 CNN 모델 |
| Distillation | teacher의 지식을 student에 전달하려는 구조 |
| XAI | student CNN에 LIME/Grad-CAM 적용 |

설비 예시로 바꾸면 다음과 같다.

```text
Teacher:
원본 센서 숫자 sequence를 보고 고장 분류

Student:
같은 sequence의 plot 이미지를 보고 고장 분류

XAI:
student CNN이 plot의 어느 부분을 보고 고장이라고 했는지 highlight
```

## 핵심 개념 3. LIME

LIME은 model-agnostic 설명 방법이다. 이미지 LIME에서는 이미지를 여러 super-pixel로 나누고, 일부 super-pixel을 가리거나 바꾼 뒤 예측값이 얼마나 변하는지 본다.

진동 plot에서 중간 peak 영역을 가렸을 때 고장 확률이 크게 떨어진다면, 그 영역은 고장 판단에 중요하다고 볼 수 있다.

| LIME highlight | 해석 |
|---|---|
| 초반 정상 구간 강조 | 초기 패턴이 class 판단에 중요 |
| 중간 peak 강조 | 충격성 이벤트가 고장 판단 근거 |
| 후반 trend 강조 | 열화 trend가 판단 근거 |
| plot 외곽/축 강조 | artifact 학습 의심 |

## 핵심 개념 4. Grad-CAM

Grad-CAM은 CNN 내부 feature map과 gradient를 이용해 class 판단에 중요한 image region을 heatmap으로 보여준다.

설비 예시:

| Grad-CAM이 밝은 구간 | 가능한 해석 |
|---|---|
| 진동 peak 주변 | 베어링 충격성 이벤트 가능 |
| 온도 급상승 구간 | 과열 전조 가능 |
| 압력 cycle peak | valve/control 문제 가능 |
| plot 축/빈 공간 | image artifact 의심 |

Grad-CAM은 인과 설명이 아니라 post-hoc 시각화다. 따라서 FFT, envelope, RMS, kurtosis, 운전 조건, 정비 로그와 함께 검증해야 한다.

## 실험 데이터셋

논문은 UCR archive의 Wafer와 FordA를 사용했다.

| 데이터셋 | 의미 | 설비와의 연결 |
|---|---|---|
| Wafer | semiconductor fabrication sensor, normal/abnormal | 공정 센서 이상탐지와 유사 |
| FordA | automotive subsystem engine noise, binary symptom | 모터/엔진 소음·진동 진단과 유사 |

Wafer는 이미 거의 포화 성능이 가능한 dataset이고, FordA는 engine noise 기반 binary classification이라 설비 진단과 더 직접적으로 연결된다.

## 핵심 결과

| Dataset | Dense | CNN | Proposed |
|---|---:|---:|---:|
| FordA accuracy | 0.84 | 0.73 | 0.85 |
| Wafer accuracy | 1.00 | 1.00 | 1.00 |

FordA에서는 proposed model이 Dense보다 0.01 높고 CNN-only보다 높았다. 하지만 개선폭은 작다. 따라서 이 논문의 핵심 가치는 정확도 개선보다는 **설명 가능성**이다.

또한 proposed model은 training time이 증가하는 trade-off가 있다. 실무에서는 “성능 개선이 작지만 설명성이 필요한가?”를 따져야 한다.

## 실무에서 어떻게 써야 하나?

내가 실무자라면 이 방법을 최종 고장 분류 모델 단독으로 쓰기보다, **설명 보조 모듈**로 사용한다.

```text
주 모델:
MiniRocket / InceptionTime / XGBoost / TCN 등

설명 모듈:
time-series plot image + CNN + Grad-CAM/LIME
```

예시 pipeline:

```text
1. unit/run split
2. raw vibration에서 RMS, kurtosis, envelope peak, FFT band energy 계산
3. MiniRocket/InceptionTime/XGBoost로 고장 분류
4. 같은 window의 raw waveform/RMS/envelope plot 생성
5. CNN image model로 plot classification
6. Grad-CAM/LIME heatmap 생성
7. 주 모델 예측과 heatmap 근거를 dashboard에 표시
```

## 실무 주의점

### 1. Plot artifact를 학습할 수 있다

| artifact | 문제 |
|---|---|
| y축 숫자 | class별 scale 차이를 axis number로 학습 |
| x축 label | 길이/duration artifact 학습 |
| line thickness | rendering style 학습 |
| grid/background | 배경 패턴 학습 |
| image resolution | 세부 신호 손실 |

### 2. Split이 가장 중요하다

위험한 방식:

```text
긴 설비 run
→ sliding window 생성
→ plot image 생성
→ window 단위 random train/test split
```

인접 window는 거의 같은 그림이므로 test 성능이 심하게 과대평가될 수 있다.

안전한 방식:

| 목적 | split |
|---|---|
| 새 설비 일반화 | unit-based split |
| 새 run 일반화 | run-based split |
| 새 운전 조건 | condition holdout |
| 실시간 예측 | time-based split |
| RUL | bearing/engine unit split |
| 이상탐지 | 정상 train, 미래 event test |

### 3. Accuracy만 보면 안 된다

설비 진단에서는 다음을 함께 봐야 한다.

| 문제 | 지표 |
|---|---|
| 고장 분류 | precision, recall, F1, confusion matrix |
| 운영 경보 | false alarm per day/week |
| 고장 놓침 | missed detection |
| 조기 경보 | lead time |
| 이상탐지 | detection delay, event-level recall |
| RUL | MAE, RMSE, NASA scoring function, early/late penalty |

## 한계

1. Plot image가 원본 시계열을 완벽히 보존하지 않는다.
2. LIME/Grad-CAM은 인과 설명이 아니다.
3. 실험 dataset이 2개뿐이라 일반화 주장은 조심해야 한다.
4. Training time이 증가한다.
5. 설비 운영 지표인 false alarm, missed detection, lead time은 논문에서 직접 다루지 않는다.
6. 다변량 센서와 물리 feature를 직접 다루는 구조는 아니다.

## 이전 논문들과 연결

| 이전 논문 | 연결 |
|---|---|
| Shapelet | 중요한 부분 구간을 찾는다는 점에서 연결 |
| BOSS | 시계열을 다른 representation으로 바꿔 분류한다는 점에서 연결 |
| ROCKET/MiniRocket | convolution feature는 강하지만 설명성은 약함 |
| FCN/ResNet | CAM으로 중요한 시점 확인 가능 |
| InceptionTime | raw time series CNN 분류와 연결 |
| TimesNet | 1D 시계열을 2D 구조로 바꾸는 방향과 연결 |
| PatchTST | 한 점보다 구간/patch가 의미 단위라는 생각과 연결 |

## 오늘 공부용 요약

- 이 논문은 time-series classification에 image XAI를 적용하려는 논문이다.
- 핵심은 시계열을 2D line plot 이미지로 바꾸고 CNN student model에 LIME/Grad-CAM을 적용하는 것이다.
- Dense teacher는 원본 time series array를 보고, CNN student는 plot image를 본다.
- FordA에서는 proposed model이 Dense보다 약간 높은 accuracy를 보였다.
- 핵심 가치는 정확도 개선보다 시각적 설명 가능성이다.
- 설비 진단에서는 최종 모델 단독보다 dashboard용 설명 보조 모듈로 쓰는 것이 좋다.
- Heatmap은 실제 고장 원인이 아니라 모델이 참고한 영역일 뿐이다.
- 시계열/설비 진단에서는 모델보다 split이 먼저다.

논문 URL: https://arxiv.org/abs/2303.12799

# 3편: Time Series as Images: Vision Transformer for Irregularly Sampled Time Series

## 결론부터

이 논문의 핵심 결론은 **불규칙하게 샘플링된 다변량 시계열을 복잡한 보간·마스킹·전용 모델 없이, 여러 개의 line graph가 들어 있는 하나의 이미지로 바꾸고 사전학습된 Vision Transformer에 넣으면 강력한 분류 성능과 결측 센서에 대한 높은 강건성을 얻을 수 있다**는 것이다.

실무적으로는 이 논문을 다음처럼 읽는 것이 좋다.

> 센서 timestamp가 불규칙하고, 결측이 많고, 일부 센서가 test 시점에 빠질 수 있는 경우, 시계열을 line graph grid image로 바꿔 pre-trained vision backbone을 fine-tuning하는 방식은 매우 실용적인 후보가 될 수 있다.

## 이 논문을 한 문장으로 요약하면

ViTST는 불규칙 샘플링된 여러 센서 시계열을 센서별 line graph로 그리고, 이 작은 plot들을 grid로 배열해 하나의 RGB 이미지로 만든 뒤, ImageNet-21K로 사전학습된 Swin Transformer를 fine-tuning해서 시계열 분류를 수행하는 방법이다.

## 왜 이 논문이 필요했나?

현실의 설비·의료·IoT 데이터는 정규 샘플링되지 않는 경우가 많다.

| 이상적인 데이터 | 현실 데이터 |
|---|---|
| 모든 센서가 1초마다 기록 | 센서마다 timestamp가 다름 |
| 결측 없음 | sensor dropout, 통신 끊김 |
| 모든 channel 길이 같음 | 설비별/run별 길이 다름 |
| 모든 변수 동시 관측 | 시점마다 관측된 sensor가 다름 |
| fixed tensor | sparse irregular log에 가까움 |

기존 LSTM, TCN, Transformer 기반 모델은 보통 regular interval, fully observed, fixed-size input을 가정하는 경우가 많다. 이를 맞추려면 보간, padding, masking, aggregation이 필요하다.

ViTST는 이 문제를 다르게 본다.

> 관측 시간과 값을 그대로 line graph로 그리면 irregular sampling이 이미지 안에 자연스럽게 표현된다.

설비 예시:

```text
온도: 09:00, 09:03, 09:07
전류: 09:01, 09:02, 09:08
진동: 09:00, 09:05

→ 센서별 line graph
→ grid image
→ Swin Transformer
→ 정상/이상/고장 class
```

## 핵심 개념 1. Irregularly sampled time series

불규칙 샘플링 시계열은 관측 시간 간격이 일정하지 않은 시계열이다.

예:

| timestamp | temperature |
|---|---:|
| 09:00 | 70.0 |
| 09:03 | 70.5 |
| 09:08 | 71.2 |
| 09:20 | 73.0 |

설비에서 이런 일이 생기는 이유는 많다.

| 원인 | 예시 |
|---|---|
| 센서 sampling rate 차이 | 온도 1분, 진동 RMS 1초 |
| 통신 누락 | 일부 timestamp 결측 |
| event-driven logging | threshold 넘을 때만 기록 |
| 수동 점검 데이터 | 작업자 입력이 불규칙 |
| 설비별 설정 차이 | logging 주기 다름 |

## 핵심 개념 2. 센서별 line graph

ViTST는 각 variable을 별도 line graph로 그린다.

예:

| 센서 | 관측값 |
|---|---|
| 온도 | `(0,70), (3,71), (8,73)` |
| 전류 | `(1,10), (2,12), (7,11)` |
| 압력 | `(0,5.0), (5,5.8), (8,5.4)` |
| 진동 RMS | `(2,0.2), (6,0.9)` |

각 센서 plot에서 x축은 timestamp, y축은 value다. 실제 관측 point를 marker로 표시하고, 점들을 직선으로 연결한다.

이 방식의 장점:

| 숫자 tensor 방식 | line graph 방식 |
|---|---|
| 결측을 채워야 함 | 결측은 긴 간격/빈 구간으로 표현 |
| 시간 간격 정보를 별도 feature로 넣어야 함 | x축 위치가 시간 간격을 표현 |
| 센서별 scale 처리 필요 | 센서별 subplot으로 scale 분리 |
| mask tensor 필요 | 관측 marker로 실제 point 표시 가능 |

## 핵심 개념 3. 여러 line graph를 grid image로 배열

센서별 plot을 하나의 grid image로 만든다.

예: 센서 4개

```text
┌────────┬────────┐
│ 온도   │ 전류   │
├────────┼────────┤
│ 압력   │ 진동   │
└────────┴────────┘
```

논문에서는 P19/P12는 6×6 grid, PAM은 4×5 grid를 사용했다. 각 grid cell은 64×64다.

실무에서 중요한 점:

> value scale은 train set 기준으로 정해야 한다.

전체 데이터, 특히 test 구간까지 보고 y축 scale을 정하면 미래 test 정보가 이미지 변환에 들어갈 수 있다.

## 핵심 개념 4. Swin Transformer

ViTST는 기본 vision backbone으로 Swin Transformer를 사용한다.

Swin Transformer는 이미지를 patch로 나누고, local window 안에서 self-attention을 수행한 뒤, window를 shift해서 다른 영역과 정보를 섞는다.

설비 grid image에서:

| Swin 처리 | 해석 |
|---|---|
| local window attention | 하나의 센서 plot 내부 temporal dynamics |
| shifted window attention | 서로 다른 센서 plot 간 관계 |
| hierarchical feature | local pattern에서 global pattern으로 확장 |

예:

```text
전류 plot의 급상승 patch
+ 온도 plot의 지연 상승 patch
+ 진동 plot의 peak patch
→ cross-variable pattern
```

## 핵심 개념 5. Pre-trained vision transformer

ViTST는 ImageNet-21K로 사전학습된 Swin Transformer를 fine-tuning한다.

핵심 주장은 다음이다.

> 자연 이미지에서 배운 line, edge, shape, texture 인식 능력이 synthetic time-series line graph image에도 전이될 수 있다.

하지만 이것이 설비 고장 물리를 안다는 뜻은 아니다. 모델은 plot의 시각적 pattern을 잘 볼 수 있을 뿐이다. 실제 고장 해석은 도메인 feature와 함께 해야 한다.

## 핵심 개념 6. Static feature와 text encoder

의료 dataset에는 weight, height, ICU type 같은 static feature가 있다. 논문은 이런 static feature를 sentence template으로 바꾸고 RoBERTa text encoder로 embedding한 뒤 image embedding과 concat한다.

설비에도 static feature가 많다.

| static feature | 설비 예시 |
|---|---|
| equipment type | 모터, 펌프, 압축기 |
| capacity | 정격 용량 |
| sensor location | drive-end bearing, x/y/z |
| installation age | 설치 후 경과 시간 |
| maintenance history | 최근 정비 유형 |

단, 미래 정보는 넣으면 안 된다. 정비 후 결과나 고장 판정 같은 정보는 leakage다.

## 핵심 개념 7. Leave-sensors-out

ViTST의 중요한 실험은 일부 센서가 test에서 빠지는 상황이다.

현실에서 센서는 고장나거나 통신이 끊길 수 있다.

| 현실 문제 | leave-sensors-out 실험과 연결 |
|---|---|
| 온도 센서 고장 | 온도 subplot 없음 |
| 진동 센서 통신 끊김 | 진동 variable missing |
| 새 설비에 일부 센서만 있음 | variable set 차이 |
| edge device sensor dropout | test-time missing variable |

논문은 절반의 variable이 drop되어도 ViTST가 강한 성능을 유지했다고 보고한다. 설비에서는 이 실험을 반드시 별도로 해봐야 한다.

## 실험 데이터와 결과

| dataset | samples | variables | avg obs | classes | missing ratio |
|---|---:|---:|---:|---:|---:|
| P19 | 38,803 | 34 | 401 | 2 | 94.9% |
| P12 | 11,988 | 36 | 233 | 2 | 88.4% |
| PAM | 5,333 | 17 | 4,048 | 8 | 60.0% |

ViTST는 P19, P12, PAM에서 GRU-D, SeFT, mTAND, IP-Net, Raindrop 등 irregular time-series baseline보다 강한 성능을 보였다고 보고된다.

또한 regular multivariate UEA dataset에서도 경쟁력 있는 결과를 보였다.

## 실무에서 어떻게 써야 하나?

ViTST는 다음 상황에서 특히 검토할 만하다.

| 상황 | 적합성 |
|---|---|
| 센서별 sampling rate가 다름 | 높음 |
| 결측이 많음 | 높음 |
| 센서 수가 많고 heterogeneous | 중간~높음 |
| static metadata가 중요 | 중간 |
| raw vibration 고장 분류 | 낮음~중간 |
| sensor dropout robustness 필요 | 높음 |

추천 pipeline:

```text
1. 설비 unit/run/time/condition split 설계
2. 센서별 (time, value) tuple 구성
3. train 기준 value scale 결정
4. 센서별 line graph 생성
5. grid layout과 variable order 고정
6. Swin/ViT fine-tuning
7. MiniRocket, XGBoost, TCN/GRU-D/Raindrop baseline과 비교
8. sensor dropout robustness 테스트
9. attention map 확인
10. false alarm/missed detection/lead time 평가
```

## 실무 주의점

### 1. 이미지 변환이 정보를 왜곡할 수 있다

| 문제 | 설명 |
|---|---|
| image resolution | 고주파 정보 손실 |
| interpolation | 실제 관측되지 않은 값을 선으로 연결 |
| marker/line style | style artifact 학습 가능 |
| y-axis scale | amplitude 왜곡 |
| variable order | grid 위치 bias |

### 2. raw vibration에는 직접 적용이 애매하다

베어링 raw vibration에는 FFT, envelope, order tracking이 중요하다. ViTST에는 raw waveform보다 다음 feature sequence를 line graph로 넣는 것이 더 실무적일 수 있다.

```text
raw vibration
→ RMS / kurtosis / envelope peak / FFT band energy
→ irregular feature logs
→ ViTST grid image
```

### 3. Attention map은 인과 설명이 아니다

Attention이 진동 peak를 본다고 해서 그것이 실제 고장 원인이라는 뜻은 아니다. FFT, envelope, RMS, kurtosis, 운전 조건, 정비 로그와 함께 봐야 한다.

### 4. Split이 먼저다

위험한 방식:

```text
긴 설비 run
→ 여러 window 생성
→ grid image 생성
→ random train/test split
```

인접 image가 거의 같기 때문에 성능이 과대평가된다.

## 이전 논문들과 연결

| 논문 | 연결 |
|---|---|
| Plotting Time | line plot image 사용 |
| XAI for TSC | plot image에 XAI 적용 |
| ViTST | line graph grid + pre-trained Swin |
| PatchTST | patch token과 vision patch 사고 연결 |
| TimesNet | 1D 시계열을 2D 구조로 보는 흐름 |
| TSSI | 다른 방식의 multivariate image encoding |

## 오늘 공부용 요약

- ViTST는 irregularly sampled multivariate time series를 line graph image로 변환한다.
- 센서별 line graph를 grid로 배열해 하나의 이미지로 만든다.
- 기본 backbone은 ImageNet-21K pre-trained Swin Transformer다.
- irregular sampling과 missing sensor를 복잡한 imputation 없이 이미지로 표현한다.
- leave-sensors-out setting에서 높은 강건성을 보였다.
- 설비 데이터에서는 irregular sensor log, sensor dropout, heterogeneous sensor set이 있을 때 유용하다.
- raw 고주파 vibration에는 바로 쓰기보다 health feature sequence를 이미지화하는 것이 더 실무적이다.
- 모델보다 split이 먼저다.

논문 URL: https://doi.org/10.1016/j.cie.2025.111393

# 4편: TSSI: Time Series as Screenshot Images for Multivariate Time Series Classification Using Convolutional Neural Networks

## 결론부터

이 논문의 핵심 결론은 **다변량 시계열을 복잡한 수학적 이미지 변환으로 바꾸지 않고도, 센서별 시계열 plot의 screenshot 같은 binary image로 만들고 이를 multi-channel image로 합치면 ResNet 같은 검증된 CNN을 활용해 강력한 multivariate time-series classification을 할 수 있다**는 것이다.

실무적으로는 TSSI를 다음처럼 이해하면 좋다.

> 센서 하나하나의 시간 패턴은 binary screenshot image로 보존하고, 여러 센서 간 관계는 channel 방향으로 쌓아 CNN이 보게 하는 방법이다.

## 이 논문을 한 문장으로 요약하면

TSSI는 multivariate time series의 각 univariate channel을 시계열 plot을 찍은 binary screenshot image로 바꾼 뒤, 이 이미지들을 channel 방향으로 concat해 하나의 multi-channel image를 만들고, ResNet18/ResNet50 같은 CNN으로 분류하는 MTSC 방법이다.

## 왜 이 논문이 필요했나?

설비 데이터는 보통 단일 센서가 아니다.

| 설비 | 센서 예시 |
|---|---|
| 모터 | 전류, 전압, 온도, 진동 x/y/z |
| 펌프 | 압력, 유량, 전류, 진동 |
| 압축기 | 압력, 온도, 유량, 전류, 진동 |
| 베어링 | 진동 x/y/z, 온도, 회전수 |
| 생산 설비 | 온도, 압력, 속도, torque, load |

다변량 시계열에서는 두 가지 정보가 중요하다.

| 정보 | 뜻 | 설비 예시 |
|---|---|---|
| intra-series temporal correlation | 한 센서 내부의 시간 패턴 | 진동 RMS가 특정 구간에서 급상승 |
| inter-series correlation | 센서들 사이의 관계 | 전류 상승 후 온도 상승, 압력 증가 + 유량 감소 |

기존 GAF, MTF, RP 같은 방법은 수치 matrix를 만들지만, TSSI는 더 직관적인 screenshot image를 만든다.

## 핵심 개념 1. Time Series as Screenshot Images

TSSI는 이름 그대로 시계열을 screenshot image처럼 만든다.

예를 들어 진동 RMS sequence가 있다.

```text
[0.10, 0.11, 0.12, 0.80, 1.50, 0.60, 0.20]
```

이를 plot처럼 보면 중간에 peak가 있다. TSSI는 선이 지나가는 pixel을 binary image로 만든다.

```text
선 위 pixel = 1 또는 255
배경 pixel = 0
```

입문자용 설명:

> 시계열 선 그래프를 흑백 이미지로 찍는다.

## 핵심 개념 2. Intra-series temporal correlation 보존

한 센서 안에서 값이 어떻게 이어지는지가 intra-series pattern이다.

예: 압력 cycle

```text
[5.0, 5.5, 6.0, 5.4, 5.1, 5.6, 6.1, 5.5]
```

line screenshot은 다음을 보존한다.

| pattern | 설비 해석 |
|---|---|
| 상승 후 하강 | valve cycle |
| spike | 압력 이상 |
| flatline | sensor stuck 또는 안정 상태 |
| repeated peak | 반복 이벤트 |
| trend | 막힘/부하 변화 |

## 핵심 개념 3. Inter-series correlation은 channel concat으로 처리

센서마다 binary image를 만든다.

```text
temperature → image 1
current     → image 2
vibration   → image 3
pressure    → image 4
```

그리고 channel 방향으로 쌓는다.

```text
multi-channel image = [temperature image,
                       current image,
                       vibration image,
                       pressure image]
```

RGB 이미지가 Red/Green/Blue channel을 갖는 것과 비슷하다.

```text
RGB image  = [R, G, B]
TSSI image = [sensor1, sensor2, sensor3, ...]
```

CNN은 여러 channel을 동시에 보면서 센서 간 관계를 배울 수 있다.

| 센서 관계 | 설비 해석 |
|---|---|
| 전류 상승 + 온도 지연 상승 | 부하 증가 후 열 반응 |
| 진동 peak + 온도 상승 | 마찰/베어링 문제 |
| 압력 증가 + 유량 감소 | 막힘 가능성 |
| 전류 변동 + 진동 변동 | 기계·전기 복합 이상 |

## 핵심 개념 4. Grid image가 아니라 channel stack

ViTST는 센서 plot을 grid에 배치한다.

```text
온도 | 전류
압력 | 진동
```

TSSI는 센서 plot을 channel로 쌓는다.

```text
height × width × num_sensors
```

| 방식 | 장점 | 단점 |
|---|---|---|
| grid image | 사람이 보기 직관적 | layout/variable order 영향 |
| channel concat | CNN multi-channel convolution과 자연스럽게 연결 | channel 수가 많으면 RGB pretraining 활용 어려움 |

## 핵심 개념 5. Binary image의 장단점

| 장점 | 설명 |
|---|---|
| 단순함 | 선이 지나간 위치만 표시 |
| 배경과 신호 구분 | CNN이 shape를 보기 쉬움 |
| 해석성 | 사람이 봐도 이해 가능 |
| CNN 친화적 | edge/line pattern 학습 가능 |

| 단점 | 설명 |
|---|---|
| intensity 정보 손실 | 선 위치 외의 density 정보 제한 |
| 해상도 민감 | 작은 spike가 희석될 수 있음 |
| scale 민감 | y축 scale에 따라 이미지 위치가 바뀜 |
| interpolation 영향 | 실제 관측 사이를 선으로 이어 artifact 가능 |

## 핵심 개념 6. Scale은 train 기준이어야 한다

TSSI에서 y축 scale은 이미지 좌표가 된다. 따라서 scale을 어떻게 잡느냐가 매우 중요하다.

위험한 방식:

```text
전체 train+test min/max로 image scale 결정
```

안전한 방식:

```text
train split 먼저
→ train min/max 또는 domain limit 결정
→ validation/test에는 같은 scale 적용
```

설비에서는 train min/max만 쓰면 test extreme fault가 clipping될 수 있다. domain limit 또는 sensor spec range를 같이 고려하는 것이 좋다.

## 핵심 개념 7. ResNet 같은 검증된 CNN을 쓸 수 있다

TSSI는 이미지로 변환한 뒤 ResNet18, ResNet50 같은 CNN backbone을 사용할 수 있다.

장점:

| 장점 | 설명 |
|---|---|
| 구현 쉬움 | PyTorch/torchvision 활용 |
| 검증된 backbone | ResNet 계열 안정적 |
| Grad-CAM 가능 | 설명성 연결 |
| architecture tuning 부담 감소 | encoding에 집중 가능 |

주의점:

일반 ResNet은 RGB 3-channel에 맞춰져 있다. TSSI는 센서 수만큼 channel이 된다. 센서가 10개면 10-channel input이므로 첫 convolution layer를 수정해야 한다.

## 작은 예시: 4개 센서 TSSI

| time | temperature | current | vibration | pressure |
|---:|---:|---:|---:|---:|
| 1 | 70.0 | 10.0 | 0.20 | 5.0 |
| 2 | 70.2 | 10.2 | 0.21 | 5.1 |
| 3 | 70.4 | 10.3 | 0.23 | 5.2 |
| 4 | 71.0 | 11.5 | 0.80 | 5.3 |
| 5 | 72.0 | 12.0 | 1.50 | 5.6 |
| 6 | 72.5 | 11.8 | 0.90 | 5.5 |

TSSI image:

```text
channel 1 = temperature screenshot
channel 2 = current screenshot
channel 3 = vibration screenshot
channel 4 = pressure screenshot
```

CNN이 볼 수 있는 패턴:

| 패턴 | 해석 |
|---|---|
| vibration peak + current 상승 | 부하/마찰 관련 가능 |
| temperature 상승 지속 | 열 축적 |
| pressure 변화 없음 | 유체 계통 문제 가능성 낮음 |
| multi-sensor 동시 변화 | 실제 상태 변화 가능성 증가 |

## 실무에서 어떻게 써야 하나?

TSSI는 다음 문제에 적합하다.

| 문제 | 적합성 |
|---|---|
| multivariate sensor fault classification | 높음 |
| 운전 mode classification | 높음 |
| health stage classification | 중간~높음 |
| raw vibration fault classification | 비교 후보 |
| forecasting | 낮음 |
| RUL regression | 직접적이지 않음 |

추천 pipeline:

```text
1. unit/run split 설계
2. raw sensor와 domain feature sequence 생성
3. train 기준 scale 결정
4. TSSI multi-channel image 생성
5. ResNet18/50 학습
6. MiniRocket, InceptionTime, feature+XGBoost와 비교
7. Grad-CAM으로 channel/time 중요 영역 확인
8. false alarm/missed detection 사례 분석
```

## 설비 진단 응용

베어링 데이터라면 raw waveform만 channel로 넣기보다 다음 feature를 함께 쓰는 것이 좋다.

| channel | 의미 |
|---|---|
| raw vibration plot | 원본 waveform |
| RMS sequence plot | 에너지 trend |
| kurtosis sequence plot | 충격성 |
| envelope peak plot | 베어링 결함 signature |
| FFT band energy plot | frequency band trend |

```text
raw vibration
→ RMS/kurtosis/envelope/FFT band feature
→ 각 feature를 screenshot image로 변환
→ multi-channel TSSI
→ ResNet
```

## 한계

1. screenshot image가 원본 정보를 완벽히 보존하지 않는다.
2. binary화와 해상도 때문에 작은 이상 신호가 사라질 수 있다.
3. channel 수가 많아지면 CNN 구조가 까다로워진다.
4. 센서 간 시간 지연 관계를 명시적으로 모델링하지 않는다.
5. raw 고주파 vibration에는 specialized baseline과 비교해야 한다.
6. classification 중심 방법이지 forecasting/RUL regression 전용은 아니다.
7. data leakage를 자동으로 막지 못한다.

## split 주의

위험한 방식:

```text
긴 설비 run
→ sliding window 생성
→ TSSI image 변환
→ window random train/test split
```

인접 window image가 거의 같아진다.

안전한 split:

| 목적 | split |
|---|---|
| 새 설비 일반화 | unit split |
| 새 run 일반화 | run split |
| 새 운전 조건 | condition holdout |
| 같은 설비 미래 | time split |
| RUL | bearing/engine unit split |
| 이상탐지 | 정상 train, 미래 test |

## 이전 논문들과 연결

| 논문 | 차이 |
|---|---|
| Plotting Time | 단일 line plot image + CNN |
| ViTST | 센서별 line graph grid + Swin |
| TSSI | 센서별 screenshot binary image를 channel concat + CNN |
| ViFusionTST | line plot + RP/MTF/GAF fusion |

## 오늘 공부용 요약

- TSSI는 MTSC를 위한 time-series-to-image encoding 방법이다.
- 각 univariate channel을 binary screenshot image로 바꾼다.
- 각 sensor image를 channel 방향으로 concat한다.
- CNN/ResNet이 intra-series와 inter-series pattern을 학습한다.
- scale은 train 기준 또는 domain 기준이어야 한다.
- 설비에서는 raw sensor뿐 아니라 RMS, kurtosis, envelope peak, FFT band energy를 channel로 추가하는 방식이 좋다.
- classification에는 유용하지만 forecasting/RUL regression에는 직접적이지 않다.
- 모델보다 split이 먼저다.

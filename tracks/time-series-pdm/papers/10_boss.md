# 10편: The BOSS is concerned with time series classification in the presence of noise

## 결론부터

**이 논문의 핵심 결론은, 시계열을 원본 값 그대로 비교하지 말고 “짧은 구간 → Fourier 기반 symbolic word → word histogram”으로 바꾸면, 노이즈와 시간 위치 어긋남에 더 강한 시계열 분류기를 만들 수 있다는 것입니다.**

## 이 논문을 한 문장으로 요약하면

**BOSS는 시계열을 여러 개의 짧은 window로 자른 뒤, 각 window를 Symbolic Fourier Approximation, 즉 SFA word로 바꾸고, 그 word들의 빈도 histogram을 비교해 분류하는 dictionary-based time-series classification 방법입니다.**

## 왜 이 논문이 필요했나?

BOSS는 시계열을 자연어 문서처럼 봅니다.

| 문서 분류 | BOSS 시계열 분류 |
|---|---|
| 문서를 단어 빈도로 표현 | 시계열을 SFA word 빈도로 표현 |
| 특정 단어가 class를 설명 | 특정 pattern word가 고장 class를 설명 |
| bag-of-words | bag-of-SFA-symbols |

베어링 고장 신호는 전체 waveform 전체가 고장처럼 보이는 것이 아니라, 짧은 충격 패턴이 반복적으로 나타나는 경우가 많습니다. BOSS는 그 반복적인 짧은 패턴을 단어로 바꾸고, 단어 빈도로 정상/고장을 구분합니다.

## 핵심 개념 1. Dictionary-based classifier

| 짧은 진동 패턴 | symbolic word |
|---|---|
| 잔잔한 낮은 진동 | `aaaa` |
| 약간 상승 후 하강 | `abca` |
| 충격성 peak | `dcba` |
| 고주파 흔들림 | `ccbd` |

하나의 시계열은 word count vector가 됩니다.

| word | 정상 시계열 빈도 | 고장 시계열 빈도 |
|---|---:|---:|
| `aaaa` | 12 | 3 |
| `abca` | 5 | 2 |
| `dcba` | 0 | 9 |
| `ccbd` | 1 | 6 |

## 핵심 개념 2. Sliding window

긴 시계열을 짧은 window로 자릅니다.

```text
T = [0.1, 0.2, 0.1, 1.5, 3.0, 1.4, 0.2, 0.1]
```

window 길이 4:

| 위치 | subsequence |
|---:|---|
| 1 | `[0.1, 0.2, 0.1, 1.5]` |
| 2 | `[0.2, 0.1, 1.5, 3.0]` |
| 3 | `[0.1, 1.5, 3.0, 1.4]` |
| 4 | `[1.5, 3.0, 1.4, 0.2]` |

## 핵심 개념 3. DFT

BOSS는 window마다 Fourier 계수를 계산합니다.

\[
X_k = \sum_{n=0}^{N-1} x_n e^{-i 2\pi kn/N}
\]

DFT는 “이 짧은 window가 어떤 주파수 성분들로 구성되어 있는지”를 봅니다. 설비 진동에서는 베어링 결함, 기어 결함, 모터 전류 이상이 주파수 영역에서 더 잘 보일 수 있습니다.

## 핵심 개념 4. SFA

SFA는 Fourier coefficient를 symbolic word로 바꿉니다.

| 값 범위 | symbol |
|---|---|
| 매우 낮음 | `a` |
| 낮음 | `b` |
| 높음 | `c` |
| 매우 높음 | `d` |

```text
window → DFT coefficients → discretisation → "dcba"
```

## 핵심 개념 5. MCB

MCB, Multiple Coefficient Binning은 Fourier coefficient 숫자를 symbol로 바꾸기 위한 경계값을 학습하는 방법입니다. 이 경계값은 반드시 train 데이터에서 학습해야 합니다. 전체 데이터로 binning하면 test 분포 정보가 들어갈 수 있습니다.

## 핵심 개념 6. Bag-of-words histogram

```text
["aaaa", "aaaa", "abca", "dcba", "aaaa", "dcba"]
```

| word | count |
|---|---:|
| `aaaa` | 3 |
| `abca` | 1 |
| `dcba` | 2 |

BOSS는 word의 정확한 위치보다 word 빈도를 봅니다.

## 핵심 개념 7. Numerosity reduction

overlap이 큰 sliding window에서는 같은 word가 연속 반복될 수 있습니다.

```text
원래: ["aaaa", "aaaa", "aaaa", "abca", "abca", "dcba"]
reduction 후: ["aaaa", "abca", "dcba"]
```

이렇게 하면 같은 신호 조각을 여러 번 세는 문제를 줄일 수 있습니다. 단, train/test leakage를 해결해주지는 않습니다.

## 핵심 개념 8. BOSS distance

\[
dist(B_1, B_2)
=
\sum_{a \in B_1,\, B_1(a)>0}
\left[B_1(a) - B_2(a)\right]^2
\]

query histogram에 나타난 word들을 중심으로 상대 시계열에도 그 word들이 비슷하게 나타나는지 봅니다.

## 핵심 개념 9. 1-NN classifier

test 시계열을 BOSS histogram으로 바꾸고, train histogram과 거리를 계산해 가장 가까운 train label을 따릅니다.

## 핵심 개념 10. BOSS ensemble

여러 window length와 parameter 조합을 탐색하고 좋은 BOSS model을 ensemble로 사용합니다. 설비 고장마다 시간 스케일이 다르기 때문입니다.

| 고장/이상 | 적절한 시간 스케일 |
|---|---|
| 베어링 충격 | 매우 짧은 window |
| 기어 결함 | 회전/mesh cycle |
| 모터 startup 이상 | 수백 ms~수초 |
| 압력 hunting | 제어 cycle |
| 온도 이상 | 수분~수십분 |

## 작은 예시 데이터

| window | label | SFA word sequence |
|---|---|---|
| N1 | 정상 | `aaaa, aaaa, aabb, aabb, abca` |
| N2 | 정상 | `aaaa, aabb, aabb, abca, abca` |
| F1 | 외륜 결함 | `abca, dcba, dcbb, dcba, cbaa` |
| F2 | 외륜 결함 | `bcda, dcba, dcbb, dcba, dcbb` |

고장 시계열에는 `dcba`, `dcbb` 같은 word가 많이 나타납니다.

## 실무에서 어떻게 써야 하나?

1. BOSS는 dictionary-based TSC baseline으로 씁니다.
2. raw vibration에 바로 적용할 수 있지만 band-pass, envelope, FFT band energy sequence와도 비교합니다.
3. window length는 물리 시간 스케일과 연결해야 합니다.
4. normalization 여부는 센서 물리 의미에 따라 validation에서 결정합니다.
5. train/test split이 먼저입니다.

| 위험한 방식 | 문제 |
|---|---|
| 전체 run을 window로 자른 뒤 random split | 인접 window 누수 |
| 전체 데이터로 MCB binning | test 분포 정보 누수 |
| test 성능으로 window length 선택 | model selection leakage |

## 이상탐지와 RUL 연결

정상 histogram library와 멀어지면 이상 후보로 볼 수 있고, 알려진 고장 histogram과 가까워지면 해당 fault 후보로 볼 수 있습니다. RUL에서는 fault-word ratio, distance to healthy prototype, histogram drift slope 등을 feature로 만들 수 있습니다.

## 한계

1. histogram은 word 순서를 버립니다.
2. Fourier 기반 symbolic 표현이 모든 고장에 충분하지는 않습니다.
3. parameter 선택이 중요합니다.
4. BOSS distance는 엄밀한 metric이 아닙니다.
5. 다변량 센서에는 직접 확장이 필요합니다.
6. 데이터 누수에 취약합니다.

## 오늘 공부용 요약

1. BOSS는 Bag-of-SFA-Symbols입니다.
2. sliding window를 SFA word로 바꾸고 word histogram으로 표현합니다.
3. SFA는 DFT 기반 symbolic representation입니다.
4. MCB breakpoints는 train 기준으로만 학습해야 합니다.
5. BOSS는 반복 고장 pattern word에 강하지만 순서 정보는 일부 잃습니다.
6. 설비 진단에서는 window random split을 피해야 합니다.

## 다음 논문 예고

다음은 **11편: ROCKET / MiniRocket**입니다.

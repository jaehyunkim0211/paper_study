# 7편: Time Series Shapelets: A New Primitive for Data Mining

## 결론부터

**이 논문의 핵심 결론은, 시계열 전체를 비교하지 말고 “분류를 가장 잘 설명하는 짧은 부분 패턴”을 찾으면 더 해석 가능하고 빠른 시계열 분류기를 만들 수 있다는 것입니다.**

## 이 논문을 한 문장으로 요약하면

**Shapelet은 특정 class를 가장 잘 대표하는 짧은 subsequence이며, 이 논문은 시계열을 통째로 비교하는 1-NN/DTW 방식의 한계를 줄이기 위해 “결정적인 부분 패턴”을 찾아 decision tree처럼 분류하는 방법을 제안합니다.**

## 왜 이 논문이 필요했나?

DTW는 전체 시계열을 비교합니다. 하지만 설비 진단에서는 전체 구간이 중요한 것이 아닐 수 있습니다. 정말 중요한 것은 “고장을 구분하는 짧은 충격 패턴”일 수 있습니다.

| 전체 비교의 문제 | Shapelet 관점 |
|---|---|
| 전체 waveform에는 정상 구간이 대부분임 | 고장 class를 구분하는 짧은 충격 구간을 찾음 |
| 부하 변화 때문에 전체 amplitude가 달라짐 | 특정 국소 패턴만 비교 |
| 센서 노이즈가 전체 거리 계산을 방해함 | 분류에 유용한 부분만 사용 |
| DTW/1-NN은 설명이 어려움 | “이 부분 패턴 때문에 고장으로 봤다”고 설명 가능 |

## 핵심 개념 1. Shapelet은 짧지만 결정적인 부분 패턴이다

| window | label | sequence |
|---|---|---|
| W1 | 정상 | `[0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1]` |
| W2 | 정상 | `[0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1]` |
| W3 | 고장 | `[0.1, 0.2, 0.1, 1.5, 3.0, 1.4, 0.2]` |
| W4 | 고장 | `[0.2, 0.1, 1.4, 2.8, 1.5, 0.2, 0.1]` |

고장을 잘 구분하는 짧은 패턴은 `[1.5, 3.0, 1.4]`일 수 있습니다. 이 패턴이 고장 window에는 가깝게 나타나고 정상 window에는 멀게 나타난다면 shapelet 후보입니다.

## 핵심 개념 2. Subsequence

길이 \(m\)인 시계열 \(T\)에서 시작 위치 \(p\), 길이 \(l\)인 subsequence는 다음입니다.

\[
S = t_p, t_{p+1}, \dots, t_{p+l-1}
\]

Shapelet은 연속된 구간입니다. 고장 신호는 전체 window에 균등하게 퍼져 있지 않고 특정 짧은 구간에 나타날 수 있습니다.

## 핵심 개념 3. SubsequenceDist

\[
SubsequenceDist(T, S) = \min_{S' \in S_T^{|S|}} Dist(S, S')
\]

긴 시계열 \(T\) 안에서 shapelet \(S\)와 가장 비슷한 위치를 찾고, 그 가장 작은 거리를 사용합니다. 고장 패턴이 window 내 정확히 같은 위치에 나타나지 않아도 됩니다.

## 핵심 개념 4. Shapelet은 split rule처럼 작동한다

| window | 실제 label | shapelet까지 거리 |
|---|---|---:|
| W1 | 정상 | 4.8 |
| W2 | 정상 | 5.2 |
| W3 | 정상 | 4.5 |
| W4 | 고장 | 0.3 |
| W5 | 고장 | 0.5 |
| W6 | 고장 | 0.7 |

`거리 < 1.0 → 고장`, `거리 ≥ 1.0 → 정상`이라는 규칙으로 분류할 수 있습니다.

## 핵심 개념 5. Information Gain

좋은 shapelet은 거리 기준으로 dataset을 나눴을 때 한쪽에는 고장, 다른 쪽에는 정상만 모이게 합니다.

\[
I(D) = -p(A)\log p(A) - p(B)\log p(B)
\]

\[
Gain(sp) = I(D) - \hat{I}(D)
\]

Information gain이 클수록 class 혼합도를 크게 줄이는 좋은 shapelet입니다.

## 핵심 개념 6. 계산량

가능한 모든 subsequence를 후보로 만들고, 각 후보를 모든 time series와 비교해야 하므로 계산량이 큽니다. raw vibration처럼 길이가 긴 데이터에서는 후보 길이를 물리 지식으로 제한해야 합니다.

## 핵심 개념 7. 해석 가능성

Shapelet은 이렇게 설명할 수 있습니다.

> “이 window는 정상보다 고장 shapelet과 가까웠습니다. 특히 0.42초 부근의 충격 패턴이 외륜 결함 class에서 자주 나타나는 패턴과 유사합니다.”

## 작은 예시 데이터

| window | label | shapelet `[1.5, 3.0, 1.4]`까지 최소 거리 |
|---|---|---:|
| N1 | 정상 | 4.2 |
| N2 | 정상 | 4.1 |
| N3 | 정상 | 4.3 |
| F1 | 고장 | 0.0 |
| F2 | 고장 | 0.3 |
| F3 | 고장 | 0.2 |

threshold 1.0이면 완벽히 분류됩니다.

## 실무에서 어떻게 써야 하나?

1. 설명 가능한 고장 signature를 찾는 데 유용합니다.
2. raw vibration에 바로 쓰기보다 RMS, envelope, FFT band energy sequence에서 찾는 것도 좋습니다.
3. 후보 길이는 물리 지식으로 제한합니다.
4. train/test split을 먼저 정하고 train set에서만 shapelet을 찾아야 합니다.
5. 선택된 shapelet은 waveform과 정비 이력으로 검증해야 합니다.

| 위험한 방식 | 문제 |
|---|---|
| 전체 데이터에서 shapelet 후보 탐색 | test label 정보가 shapelet 선택에 반영됨 |
| 전체 window를 랜덤 split | 인접 window가 train/test에 섞임 |
| test 성능을 보며 shapelet 길이 선택 | test leakage |

## 한계

1. Shapelet discovery는 계산량이 큽니다.
2. 너무 짧은 shapelet은 noise를 고장 패턴으로 착각할 수 있습니다.
3. 하나의 shapelet이 전체 class를 대표하지 못할 수 있습니다.
4. normalization이 amplitude 고장 정보를 지울 수 있습니다.
5. data leakage에 매우 취약합니다.
6. 다변량 센서에는 직접 확장이 필요합니다.

## 이전 논문들과 연결해서 이해하기

| 관점 | DTW | UCR Archive | Shapelet |
|---|---|---|---|
| 핵심 질문 | 두 시계열 전체를 어떻게 맞출까? | TSC 모델을 어떻게 공정하게 비교할까? | class를 가르는 짧은 부분 패턴은 무엇인가? |
| 비교 단위 | 전체 시계열 | benchmark instance | subsequence |
| 장점 | time shift에 강함 | 공정 비교 가능 | 해석 가능성 높음 |

## 오늘 공부용 요약

1. Shapelet은 class를 가장 잘 구분하는 짧은 subsequence입니다.
2. SubsequenceDist는 긴 시계열 안에서 가장 잘 맞는 위치와의 거리입니다.
3. Shapelet은 거리 threshold를 기준으로 decision tree split처럼 작동합니다.
4. 좋은 shapelet은 information gain이 높습니다.
5. 설비 진단에서는 베어링 충격, 모터 startup 변형, 압력 cycle 이상 같은 고장 signature 탐색에 유용합니다.
6. shapelet은 반드시 train set에서만 찾아야 합니다.

## 다음 논문 예고

다음은 **8편: Time Series Feature Extraction on basis of Scalable Hypothesis Tests**, 즉 **tsfresh** 논문입니다.

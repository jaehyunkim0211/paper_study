# 4편: Making Time-series Classification More Accurate Using Learned Constraints

## 결론부터

**이 논문의 핵심 결론은, 시계열 분류에서는 두 신호를 같은 시간 인덱스끼리만 비교하면 안 되고, “어디까지 시간축을 늘이거나 줄여서 맞춰도 되는지”를 데이터로 학습하면 DTW 분류의 정확도와 속도를 동시에 개선할 수 있다는 것입니다.**

## 이 논문을 한 문장으로 요약하면

**Ratanamahatana와 Keogh는 Dynamic Time Warping, 즉 DTW의 warping path에 임의 모양의 제약을 학습하는 R-K Band를 제안해, 무작정 많이 휘어 맞추는 DTW보다 더 정확하고 빠른 시계열 분류를 만들 수 있음을 보였습니다.**

## 왜 이 논문이 필요했나?

설비 진단 분류에서는 고장 패턴이 항상 정확히 같은 시점에 나타나지 않습니다.

| 원인 | 패턴이 시간축에서 어긋나는 이유 |
|---|---|
| 회전수 변동 | 충격 간격이 조금 늘어나거나 줄어듦 |
| 부하 변동 | 진동 peak가 조금 늦거나 빠르게 나타남 |
| trigger 기준 차이 | window 시작점이 매번 조금 다름 |
| 센서 sampling 차이 | 같은 이벤트가 인덱스상 다른 위치에 기록됨 |
| 운전 cycle 차이 | 같은 공정 단계가 조금 빠르거나 느리게 진행됨 |

Euclidean distance는 같은 인덱스끼리 비교합니다. 그러나 같은 베어링 고장 충격 패턴이 한 window에서는 50번째 sample, 다른 window에서는 53번째 sample에 나타났다면, 이를 완전히 다른 신호처럼 보면 안 됩니다.

## 핵심 개념 1. Euclidean distance는 같은 시간 위치끼리 비교한다

두 신호가 있습니다.

| sample | 신호 A | 신호 B |
|---:|---:|---:|
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 3 | 5 | 0 |
| 4 | 0 | 5 |
| 5 | 0 | 0 |

두 신호 모두 peak가 한 번 있지만, A는 3번째, B는 4번째 sample에 있습니다. Euclidean distance는 같은 위치끼리 비교하므로 peak가 한 칸 밀린 것만으로도 거리가 커집니다.

## 핵심 개념 2. DTW는 시간축을 조금 늘이거나 줄여서 비교한다

DTW는 두 시계열을 고무줄처럼 조금 늘이거나 줄여서 가장 자연스럽게 맞춰본 뒤 거리를 계산합니다.

| A의 sample | B의 sample | 해석 |
|---:|---:|---|
| 1 | 1 | 둘 다 0 |
| 2 | 2 | 둘 다 0 |
| 3 | 4 | A의 peak와 B의 peak를 맞춤 |
| 4 | 5 | 둘 다 0 근처 |

설비 데이터에서는 같은 고장 충격이 조금 빠르거나 늦게 나타나는 경우, batch 공정 시간이 조금 늘어나는 경우, 모터 startup 패턴이 조금 길어지는 경우에 DTW가 유용합니다.

## 핵심 개념 3. Warping path

DTW는 두 시계열 \(Q=q_1,...,q_n\), \(C=c_1,...,c_m\)의 모든 점 조합을 거리 matrix로 만들고, 그 matrix를 통과하는 최적 경로를 찾습니다.

| 개념 | 쉬운 설명 |
|---|---|
| 거리 matrix | 모든 sample 조합의 차이를 적어둔 표 |
| warping path | 두 시계열을 어떤 순서로 맞출지 정한 선 |
| optimal path | 전체 차이가 가장 작아지는 정렬 방법 |
| DTW distance | 최적 정렬을 했을 때의 누적 차이 |

## 핵심 개념 4. DTW의 수식

DTW는 비용이 가장 적게 드는 길을 찾는 문제입니다.

\[
d(q_i, c_j) = (q_i - c_j)^2
\]

\[
\gamma(i,j) = d(q_i,c_j) + \min
\left\{
\gamma(i-1,j-1),
\gamma(i-1,j),
\gamma(i,j-1)
\right\}
\]

현재 칸까지 오는 세 가지 방법 중 누적 비용이 가장 작은 길을 선택하고 현재 칸의 비용을 더합니다.

## 핵심 개념 5. DTW에는 제약이 필요하다

DTW가 너무 자유로우면 병적인 정렬이 생길 수 있습니다. 짧은 충격 하나를 긴 구간 전체와 맞추는 식의 물리적으로 말이 안 되는 정렬이 가능합니다.

| 제약 | 의미 | 설비 데이터 해석 |
|---|---|---|
| boundary condition | 시작은 시작끼리, 끝은 끝끼리 맞춤 | cycle 시작/끝 유지 |
| continuity condition | 경로가 갑자기 점프하지 않음 | 시간 흐름이 끊기지 않음 |
| monotonic condition | 시간은 뒤로 가지 않음 | 미래 sample을 과거 sample에 맞추지 않음 |
| slope constraint | 너무 급격히 늘이거나 줄이지 않음 | 짧은 충격 하나를 긴 구간 전체와 맞추지 않음 |
| adjustment window | 대각선에서 너무 멀리 벗어나지 않음 | 시간 shift 허용 범위 제한 |

## 핵심 개념 6. Sakoe-Chiba Band

Sakoe-Chiba Band는 대각선 주변 일정 폭 안에서만 warping을 허용합니다.

| band width | 해석 |
|---:|---|
| 0 | 전혀 시간 shift를 허용하지 않음 |
| 작음 | 약간의 phase shift만 허용 |
| 큼 | 큰 시간 지연이나 cycle 길이 차이까지 허용 |
| 너무 큼 | 물리적으로 말이 안 되는 정렬 가능 |

## 핵심 개념 7. 넓은 band가 항상 좋은 것은 아니다

너무 넓게 허용하면 같은 class 내부의 패턴은 잘 맞출 수 있지만, 다른 class와도 억지로 잘 맞춰질 수 있습니다. 정상 진동과 고장 진동이 너무 자유롭게 맞춰지면 false alarm이나 오분류가 늘 수 있습니다.

## 핵심 개념 8. R-K Band

R-K Band는 구간마다 다른 warping 허용 폭을 학습합니다.

| 구간 | 시간축 변동성 | 권장 band |
|---|---|---|
| 시작 구간 | 거의 고정 | 좁음 |
| 중간 가속 구간 | 설비마다 속도가 다름 | 넓음 |
| 끝 구간 | 안정적 | 좁음 |

모든 고장 유형이 같은 방식으로 시간축 변동을 보이지 않기 때문에, class별 R-K Band도 가능합니다.

## 작은 예시 데이터: 베어링 진동 고장 분류

| window | label | sequence |
|---|---|---|
| W1 | 정상 | `[0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1]` |
| W2 | 베어링 결함 | `[0.1, 0.2, 0.3, 1.2, 3.0, 1.1, 0.3, 0.2]` |
| T1 | 베어링 결함 | `[0.1, 0.2, 1.1, 2.8, 1.2, 0.3, 0.2, 0.1]` |

T1은 W2와 비슷하지만 peak 위치가 한 sample 앞당겨져 있습니다. Euclidean distance는 차이를 크게 볼 수 있고, DTW는 peak와 감쇠 구간을 맞춰 같은 고장 패턴으로 볼 수 있습니다.

## 실무에서 어떻게 써야 하나?

1. DTW는 데이터가 적을 때 강력한 baseline으로 써야 합니다.
2. raw vibration waveform보다는 startup 전류 sequence, batch 압력 curve, 진동 RMS trend처럼 정렬 가능한 구간에 특히 유용합니다.
3. band width는 물리 지식과 validation으로 결정해야 합니다.
4. train/test split이 먼저입니다.

| 위험한 방식 | 왜 문제인가 |
|---|---|
| 긴 진동 run을 sliding window로 자른 뒤 window 랜덤 split | 같은 설비, 같은 run, 인접 window가 양쪽에 섞임 |
| 같은 bearing의 같은 fault run이 train/test에 모두 포함 | 모델이 고장 일반화가 아니라 fingerprint를 외움 |
| 전체 데이터에서 z-normalization 후 split | test 분포 정보가 train 전처리에 섞임 |

## 평가 지표

설비 진단 분류에서는 accuracy만 보면 안 됩니다.

| 지표 | 의미 |
|---|---|
| precision | 고장이라고 한 것 중 실제 고장 비율 |
| recall | 실제 고장 중 잡아낸 비율 |
| F1 | precision과 recall의 균형 |
| confusion matrix | 어떤 클래스를 헷갈렸는지 |
| false alarm | 정상인데 경보 |
| missed detection | 고장인데 정상 |
| lead time | 고장 전에 얼마나 빨리 감지했는지 |

## 한계

1. 기본 DTW는 계산량이 큽니다.
2. 너무 많이 warping하면 물리적으로 말이 안 되는 정렬이 생깁니다.
3. amplitude scaling과 offset에 민감할 수 있습니다.
4. 다변량 센서 관계를 직접 잘 다루지는 않습니다.
5. R-K Band 학습은 overfitting 위험이 있습니다.

## 이전 논문들과 연결해서 이해하기

| 관점 | ARIMA | STL | Kalman Filter | DTW |
|---|---|---|---|---|
| 핵심 질문 | 과거로 미래를 예측할 수 있나? | 추세/계절성/잔차로 나눌 수 있나? | noisy 관측으로 숨은 상태를 추정할 수 있나? | 두 시계열이 시간축이 어긋나도 비슷한가? |
| 설비 예시 | 온도/압력 예측 | 정상 주기 제거 | 센서 fusion | 진동/전류 패턴 정렬 |

## 오늘 공부용 요약

1. Euclidean distance는 시간축 shift에 약합니다.
2. DTW는 두 시계열을 비선형으로 정렬해 더 자연스러운 거리를 계산합니다.
3. DTW에는 물리적으로 가능한 정렬만 허용하는 제약이 필요합니다.
4. R-K Band는 구간별로 다른 warping 허용 폭을 학습합니다.
5. 설비 진단에서는 window random split을 피해야 합니다.

## 다음 논문 예고

다음은 **5편: The M4 Competition: 100,000 Time Series and 61 Forecasting Methods**입니다.

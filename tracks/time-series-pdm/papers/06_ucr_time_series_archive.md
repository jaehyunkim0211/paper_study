# 6편: The UCR Time Series Archive / UCR Time Series Classification Archive

## 결론부터

**이 논문의 핵심 결론은, 시계열 분류 연구는 “내 모델이 좋아 보인다”가 아니라 “공개 benchmark, 고정된 split, 단순 baseline, 통계적 비교, 재현 가능한 코드” 위에서 검증되어야 한다는 것입니다.**

## 이 논문을 한 문장으로 요약하면

**UCR Time Series Archive는 시계열 분류 알고리즘을 공정하게 비교하기 위한 대표 benchmark이며, 2018년 확장판은 128개 univariate dataset을 소개하면서 cherry-picking, train/test split, normalization, baseline, 통계적 유의성, 재현성 문제를 함께 정리한 실험 방법론 논문입니다.**

## 왜 이 논문이 필요했나?

Part 1에서는 한 시계열 내부의 시간 구조를 이해했습니다. Part 2부터는 질문이 바뀝니다.

**시계열 하나를 보고, 이것이 어떤 class인지 맞히는 문제**입니다.

| 입력 시계열 | 분류 label |
|---|---|
| 진동 waveform 1초 구간 | 정상 |
| 진동 waveform 1초 구간 | 베어링 외륜 결함 |
| 전류 waveform 2초 구간 | rotor bar fault |
| 압력 cycle 30초 구간 | 밸브 이상 |
| 온도 trend 1시간 구간 | 냉각 성능 저하 |

예전에는 논문마다 다른 dataset, split, preprocessing, baseline으로 성능을 주장했습니다. UCR Archive는 모두가 같은 시험지를 풀 수 있게 해줍니다.

## 핵심 개념 1. Time-Series Classification, TSC

TSC는 **시간 순서가 있는 신호 하나를 보고 class label을 맞히는 것**입니다.

| sequence | label |
|---|---|
| `[0.02, 0.03, -0.01, 0.80, -0.75, ...]` | 베어링 결함 |
| `[0.01, 0.02, 0.00, -0.01, 0.02, ...]` | 정상 |

TSC에서는 각 time point가 label을 갖는 것이 아니라, 하나의 시계열 조각 전체가 label을 갖습니다.

## 핵심 개념 2. UCR Archive는 공통 시험지다

| 역할 | 의미 |
|---|---|
| 표준 dataset 제공 | 모든 연구자가 같은 문제를 풀 수 있음 |
| train/test split 제공 | 결과 재현이 쉬움 |
| baseline 제공 | 새 모델이 단순 방법보다 나은지 확인 |
| 다양한 도메인 제공 | ECG, 모션, 센서 등 여러 형태 |
| 연구 비교 기준 제공 | 논문 간 성능 비교 가능 |

M4가 forecasting benchmark라면 UCR은 time-series classification benchmark입니다.

## 핵심 개념 3. Univariate와 Multivariate

| 유형 | 입력 형태 | 예시 |
|---|---|---|
| Univariate TSC | 센서 1개 시계열 | 진동 RMS 하나로 정상/고장 분류 |
| Multivariate TSC | 여러 센서 시계열 | 진동 + 온도 + 전류 + 압력으로 고장 유형 분류 |

UCR 2018 archive는 주로 univariate입니다. 설비 진단은 multivariate가 많으므로 UEA multivariate archive와 실제 설비 split도 함께 고려해야 합니다.

## 핵심 개념 4. 고정 train/test split

UCR의 fixed split은 재현성을 위한 장치입니다. 같은 dataset, 같은 split, 같은 1-NN Euclidean으로 결과가 다르면 구현 문제를 의심할 수 있습니다.

하지만 UCR의 fixed split이 현장 설비 데이터에도 좋은 split이라는 뜻은 아닙니다.

| 상황 | 권장 split |
|---|---|
| 여러 설비가 있음 | 설비 unit 단위 split |
| 여러 run이 있음 | run 단위 split |
| 여러 부하 조건이 있음 | load/speed condition holdout |
| 하나의 긴 run만 있음 | 시간 순서 기반 split |
| RUL 데이터 | engine/bearing unit 단위 split |

## 핵심 개념 5. Baseline: 1-NN Euclidean과 1-NN DTW

UCR의 전통 baseline은 1-Nearest Neighbor입니다.

| 거리 | 의미 |
|---|---|
| Euclidean Distance | 같은 시간 index끼리 비교 |
| Dynamic Time Warping | 시간축이 조금 밀려도 맞춰서 비교 |
| Constrained DTW | 물리적으로 가능한 범위 안에서만 warping 허용 |

새 모델이 복잡하더라도 1-NN DTW보다 못하면 정말 좋은 모델인지 다시 생각해야 합니다.

## 핵심 개념 6. z-normalization

z-normalization은 평균을 빼고 표준편차로 나누는 처리입니다. 절대 높이보다 모양이 중요한 문제에서는 scale 차이를 줄여줍니다.

| 상황 | 도움이 되는 경우 | 위험한 경우 |
|---|---|---|
| 진동 waveform | sensor gain 차이 제거 | 고장 amplitude 증가 정보 손실 |
| 전류 startup | 모터 크기 차이 완화 | 과전류 절대 크기 손실 |
| 온도 trend | 설비별 offset 제거 | 절대 온도 초과 정보 손실 |

전처리 기준은 반드시 train에서만 계산해야 합니다.

## 핵심 개념 7. Cherry-picking

Cherry-picking은 잘되는 dataset이나 조건만 골라 모델을 좋아 보이게 만드는 문제입니다.

설비 진단에서도 특정 load, 특정 speed, 특정 bearing에서만 99.9% accuracy를 보이고 다른 조건은 숨기는 식으로 나타날 수 있습니다.

## 핵심 개념 8. 통계적 비교

여러 classifier를 비교할 때 평균 accuracy 하나로 끝내면 안 됩니다. dataset별 win/tie/loss, average rank, statistical test, critical difference diagram 등을 함께 봐야 합니다.

설비 진단에서는 accuracy뿐 아니라 precision, recall, F1, confusion matrix, false alarm, missed detection을 반드시 봐야 합니다.

## 작은 예시: split 문제

| window | 설비 | run | 구간 | label |
|---|---|---|---|---|
| W1 | A | Run 1 | 00:00.0~00:01.0 | 정상 |
| W2 | A | Run 1 | 00:00.5~00:01.5 | 정상 |
| W3 | A | Run 1 | 00:01.0~00:02.0 | 정상 |
| W4 | A | Run 2 | 00:00.0~00:01.0 | 외륜 결함 |
| W5 | A | Run 2 | 00:00.5~00:01.5 | 외륜 결함 |

W1과 W2는 거의 같은 raw signal을 공유합니다. W1이 train, W2가 test에 들어가면 성능이 비정상적으로 높아질 수 있습니다.

## 실무에서 어떻게 써야 하나?

1. UCR/UEA를 TSC 모델 학습용 benchmark로 사용합니다.
2. 새 모델을 만들면 1-NN ED, 1-NN DTW, catch22, tsfresh, MiniRocket 같은 baseline과 비교합니다.
3. UCR 결과를 설비 현장 성능으로 착각하지 않습니다.
4. 성공 사례뿐 아니라 실패 조건을 적습니다.
5. 실험 재현성을 위해 split 파일, preprocessing 기준, feature list, metric code를 남깁니다.

## 한계

1. UCR은 설비 진단 전용 archive가 아닙니다.
2. fixed split은 재현성에는 좋지만 일반화 검증에는 부족할 수 있습니다.
3. accuracy 중심 benchmark는 class imbalance에 약합니다.
4. UCR 데이터는 현장 데이터보다 clean한 편입니다.
5. 설비 데이터의 sliding window 문제를 UCR식으로 단순화하면 안 됩니다.

## 오늘 공부용 요약

1. UCR Time Series Archive는 TSC 알고리즘 비교를 위한 대표 benchmark입니다.
2. UCR은 dataset 모음일 뿐 아니라 실험을 공정하게 하는 방법을 알려줍니다.
3. fixed split은 재현성 장치지만 설비 진단에서는 unit/run/time split이 더 중요할 수 있습니다.
4. z-normalization은 모양 비교에는 좋지만 amplitude 정보를 지울 수 있습니다.
5. cherry-picking과 window random split은 성능을 크게 부풀립니다.
6. 설비 진단에서는 precision, recall, F1, confusion matrix, false alarm, missed detection이 필수입니다.

## 다음 논문 예고

다음은 **7편: Time Series Shapelets: A New Primitive for Data Mining**입니다.

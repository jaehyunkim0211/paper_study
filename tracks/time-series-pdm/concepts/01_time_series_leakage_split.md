# 시계열 데이터 누수와 Split 방식

## 결론부터

**시계열/설비 진단에서는 모델보다 데이터 분할이 먼저입니다.**

특히 sliding window를 만든 뒤 window 단위로 랜덤 train/test split을 하면, 같은 설비·같은 run·인접 시간 구간이 train과 test에 동시에 들어가 성능이 과대평가될 수 있습니다.

---

## 왜 위험한가?

시계열을 ML 모델에 넣으려면 보통 window로 자릅니다.

예를 들어 window 길이를 5분으로 두고 다음 1분 온도를 예측한다고 하겠습니다.

| window 입력 | 예측 target |
|---|---|
| 09:00~09:04 온도 | 09:05 온도 |
| 09:01~09:05 온도 | 09:06 온도 |
| 09:02~09:06 온도 | 09:07 온도 |

다음 두 window는 거의 같은 데이터를 공유합니다.

| window | 포함 시간 |
|---|---|
| A | 09:00~09:04 |
| B | 09:01~09:05 |

A가 train에 들어가고 B가 test에 들어가면, 모델은 사실상 test와 매우 비슷한 데이터를 이미 본 것입니다. 이것이 시계열 데이터 누수의 대표적인 형태입니다.

---

## 설비 진단에서 더 위험한 이유

설비 진단에서는 다음 상황이 흔합니다.

- 같은 베어링의 같은 고장 run에서 잘라낸 window가 train/test에 섞인다.
- 같은 모터의 인접 운전 구간이 train/test에 섞인다.
- 같은 설비의 같은 부하 조건 데이터가 양쪽에 들어간다.
- 모델은 고장 일반화가 아니라 특정 설비의 신호 fingerprint를 외운다.

이렇게 하면 accuracy가 99%가 나와도 실무 성능이라고 믿기 어렵습니다.

---

## 권장 split

| 상황 | 권장 split |
|---|---|
| 여러 설비가 있음 | 설비 unit 단위 split |
| 여러 run이 있음 | run 단위 split |
| 하나의 긴 run만 있음 | 시간 순서 기반 split |
| 여러 운전 조건이 있음 | 조건별 holdout도 별도 평가 |
| RUL 데이터 | engine/bearing unit 단위 split |

예를 들어 bearing fault diagnosis라면 다음이 더 현실적입니다.

| train | test |
|---|---|
| Bearing 1, Bearing 2, Bearing 3 | Bearing 4 |
| Run 1~7 | Run 8~10 |
| 0hp, 1hp 조건 | 2hp 조건 holdout |

반대로 아래는 위험합니다.

| train | test |
|---|---|
| 전체 window 중 랜덤 80% | 전체 window 중 랜덤 20% |

---

## STL과 전처리에서도 leakage가 생긴다

**STL을 train/test split 전에 전체 데이터에 적용하면 data leakage가 생길 수 있습니다.**

왜냐하면 decomposition은 주변 시점의 값을 사용해 trend와 seasonal을 추정하기 때문입니다. 특히 전체 시계열을 한 번에 STL로 분해한 뒤 그 결과 feature를 만들어 train/test를 나누면, test 구간의 정보가 train feature 생성 과정에 섞일 수 있습니다.

위험한 방식은 다음입니다.

| 단계 | 위험한 방식 |
|---|---|
| 1 | 전체 설비 데이터를 STL로 분해 |
| 2 | trend, seasonal, residual feature 생성 |
| 3 | window 단위로 랜덤 train/test split |
| 4 | accuracy 99% 보고 성능 좋다고 판단 |

더 안전한 방식은 다음입니다.

| 상황 | 권장 방식 |
|---|---|
| 예측 문제 | train 기간에서 STL 파라미터/패턴을 학습하고 미래 test에는 과거 정보만 사용 |
| 설비 분류 | 설비 unit 또는 run 단위로 split한 뒤, train 안에서만 전처리 기준 학습 |
| 이상탐지 | 정상 train 구간으로 baseline을 만들고, test 구간은 별도로 평가 |
| RUL | engine/bearing unit 단위 split 유지 |
| cross-validation | time-series split 또는 group split 사용 |

---

## 평가 지표도 같이 봐야 한다

accuracy 하나로 끝내면 안 됩니다.

| 지표 | 의미 |
|---|---|
| precision | 경보를 냈을 때 실제 이상일 확률 |
| recall | 실제 이상 중 얼마나 잡았는가 |
| F1 | precision과 recall의 균형 |
| confusion matrix | 정상/고장 판단의 전체 구조 |
| false alarm | 정상인데 경보를 낸 경우 |
| missed detection | 이상인데 놓친 경우 |
| lead time | 고장 전에 얼마나 빨리 감지했는가 |
| RUL error | 남은 수명 예측 오차 |

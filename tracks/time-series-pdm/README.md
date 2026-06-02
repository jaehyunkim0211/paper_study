# 시계열 데이터와 설비 진단 분류 / 예지보전 공부

이 저장소는 데이터사이언스/머신러닝/LLM 입문 논문 1~48편 이후의 새 주제인 **시계열 데이터**, **설비 진단 분류**, **예지보전**, **이상탐지**, **Remaining Useful Life(RUL)** 공부 내용을 정리하기 위한 Git 백업용 Markdown 구조입니다.

## 현재 진행 상황

| 구분 | 상태 |
|---|---|
| 로드맵 | 작성 완료 |
| Part 1. 시계열 분석 기본기 | 진행 중 |
| 1편. Box-Jenkins / ARIMA | 정리 완료 |
| 2편. STL Decomposition | 정리 완료 |
| 다음 논문 | 3편. Kalman Filter |

## 폴더 구조

```text
.
├── README.md
├── roadmap.md
├── papers/
│   ├── README.md
│   ├── 01_box_jenkins_arima.md
│   └── 02_stl_loess.md
└── concepts/
    ├── README.md
    ├── 01_time_series_leakage_split.md
    ├── 02_residual_based_anomaly_detection.md
    ├── 03_arima_basic_terms.md
    └── 04_stl_decomposition.md
```

## 공부 방식

논문 정리는 다음 형식을 유지합니다.

1. 제목은 `N편: 논문명`으로 시작합니다.
2. 반드시 `결론부터` 섹션으로 시작합니다.
3. 핵심 결론을 한 문장으로 먼저 말합니다.
4. `이 논문을 한 문장으로 요약하면`을 넣습니다.
5. `왜 이 논문이 필요했나?`를 설명합니다.
6. 핵심 개념을 5~10개 정도로 나눠 설명합니다.
7. 수식은 먼저 쉬운 비유와 직관을 설명한 뒤 보여줍니다.
8. 가능하면 작은 예시 데이터를 사용합니다.
9. `실무에서 어떻게 써야 하나?`를 반드시 넣습니다.
10. `한계`를 반드시 넣습니다.
11. 이전 논문들과 연결해서 이해할 수 있게 표로 연결합니다.
12. 마지막에는 `오늘 공부용 요약`을 넣습니다.
13. 다음 논문 제목을 예고합니다.

## 특히 반복해서 강조할 실무 원칙

시계열/설비 진단에서는 **모델보다 데이터 분할이 먼저**입니다.

특히 진동 데이터를 sliding window로 잘라놓고 window를 랜덤하게 train/test로 나누면, 같은 설비·같은 운전 run·인접 시간 구간이 train과 test에 동시에 들어갈 수 있습니다. 그러면 모델이 “고장 패턴”을 배운 것이 아니라 거의 같은 신호 조각을 외운 것처럼 되어 accuracy가 비정상적으로 높아질 수 있습니다.

실무에서는 accuracy 하나로 끝내면 안 됩니다. 설비 진단 분류에서는 **precision, recall, F1, confusion matrix, false alarm, missed detection**을 봐야 하고, 이상탐지에서는 **탐지 지연, threshold 민감도, 구간 단위 탐지 성능**을 봐야 하며, RUL에서는 **MAE, RMSE, NASA scoring function, early/late prediction penalty**를 봐야 합니다.

## 로드맵

전체 논문 로드맵은 [`roadmap.md`](./roadmap.md)에 정리되어 있습니다.

## 논문 정리

| 번호 | 파일 | 상태 |
|---:|---|---|
| 1 | [`papers/01_box_jenkins_arima.md`](./papers/01_box_jenkins_arima.md) | 완료 |
| 2 | [`papers/02_stl_loess.md`](./papers/02_stl_loess.md) | 완료 |

## 개념 정리

| 파일 | 내용 |
|---|---|
| [`concepts/01_time_series_leakage_split.md`](./concepts/01_time_series_leakage_split.md) | sliding window, train/test leakage, unit/run split |
| [`concepts/02_residual_based_anomaly_detection.md`](./concepts/02_residual_based_anomaly_detection.md) | ARIMA 잔차, STL remainder 기반 이상탐지 |
| [`concepts/03_arima_basic_terms.md`](./concepts/03_arima_basic_terms.md) | AR, MA, ARIMA, 정상성, 차분, ACF/PACF |
| [`concepts/04_stl_decomposition.md`](./concepts/04_stl_decomposition.md) | trend, seasonal, remainder, LOESS, robust STL |

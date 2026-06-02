# time-series-pdm

시계열 데이터, 설비 진단 분류, 이상탐지, 예지보전, Remaining Useful Life(RUL)를 공부하는 트랙입니다.

## 진행 상황

| 번호 | 논문 | 상태 |
|---:|---|---|
| 1 | The Box-Jenkins Approach to Time Series Analysis and Forecasting | 완료 |
| 2 | STL: A Seasonal-Trend Decomposition Procedure Based on Loess | 완료 |
| 3 | A New Approach to Linear Filtering and Prediction Problems | 완료 |
| 4 | Making Time-Series Classification More Accurate Using Learned Constraints | 완료 |
| 5 | The M4 Competition: 100,000 Time Series and 61 Forecasting Methods | 완료 |
| 6 | The UCR Time Series Archive / UCR Time Series Classification Archive | 완료 |
| 7 | Time Series Shapelets: A New Primitive for Data Mining | 완료 |
| 8 | Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests | 완료 |
| 9 | catch22: CAnonical Time-series CHaracteristics | 완료 |
| 10 | The BOSS is concerned with time series classification in the presence of noise | 완료 |
| 11 | ROCKET / MiniRocket | 완료 |
| 12 | HIVE-COTE 2.0 | 완료 |
| 13 | Time Series Classification from Scratch with Deep Neural Networks | 다음 |

## 폴더 구조

```text
tracks/time-series-pdm/
├── README.md
├── roadmap.md
├── papers/
│   ├── README.md
│   ├── 01_box_jenkins_arima.md
│   ├── 02_stl_loess.md
│   ├── 03_kalman_filter_state_space.md
│   ├── 04_dtw_learned_constraints.md
│   ├── 05_m4_competition.md
│   ├── 06_ucr_time_series_archive.md
│   ├── 07_time_series_shapelets.md
│   ├── 08_tsfresh.md
│   ├── 09_catch22.md
│   ├── 10_boss.md
│   ├── 11_rocket_minirocket.md
│   └── 12_hive_cote_2.md
└── concepts/
    ├── README.md
    └── ...
```

## 반복해서 지킬 평가 원칙

1. 모델보다 데이터 분할이 먼저입니다.
2. sliding window 이후 window 단위 random split은 매우 위험합니다.
3. 설비 진단 분류에서는 accuracy 외에 precision, recall, F1, confusion matrix, false alarm, missed detection을 봅니다.
4. 이상탐지에서는 detection delay, threshold sensitivity, event-level metric을 봅니다.
5. RUL에서는 MAE, RMSE, NASA scoring function, early/late prediction penalty를 봅니다.

# Time Series / Predictive Maintenance Track

시계열 데이터 분석, 설비 진단 분류, 이상탐지, 예지보전, Remaining Useful Life(RUL)를 공부하는 트랙입니다.

## 진행 상황

- Roadmap 작성 완료
- Part 1. 시계열 분석 기본기 완료: 01–05
- Part 2. 시계열 특징 추출과 전통 ML 완료: 06–12
- Part 3. 딥러닝 기반 시계열 모델 완료: 13–18
- Part 4. Transformer와 시계열 파운데이션 모델 진행 중: 19–23 완료
- Concepts 01–27 완료
- 다음 논문: 24. TimesFM / A Decoder-Only Foundation Model for Time-Series Forecasting

## 폴더 구조

```text
tracks/time-series-pdm/
├── README.md
├── roadmap.md
├── papers/
│   ├── README.md
│   ├── 01_box_jenkins_arima.md
│   ├── ...
│   └── 23_timegpt.md
└── concepts/
    ├── README.md
    ├── 01_time_series_leakage_split.md
    ├── ...
    └── 27_time_series_foundation_models.md
```

## 반복해서 강조할 실무 원칙

시계열/설비 진단에서는 모델보다 데이터 분할이 먼저입니다. Sliding window를 만든 뒤 window 단위로 랜덤 split하면 같은 설비, 같은 운전 run, 인접 window가 train/test에 섞일 수 있습니다. 설비 진단 분류에서는 accuracy만 보지 말고 precision, recall, F1, confusion matrix, false alarm, missed detection을 같이 봐야 합니다. 이상탐지에서는 detection delay, threshold sensitivity, event-level 성능을 봐야 합니다. RUL에서는 MAE, RMSE, NASA scoring function, early/late prediction penalty를 봐야 합니다.

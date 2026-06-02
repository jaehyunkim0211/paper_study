# Prompt for Continuing Time Series / PdM Study in a New Chat

아래 내용을 새 채팅 첫 메시지로 붙여넣으면, 현재 저장소 구조와 진행 상황을 설명하고 이어서 공부할 수 있습니다.

```text
나는 Git 저장소를 다음 구조로 정리해서 공부하고 있어.

study-notes/
├── README.md
├── tracks/
│   ├── ds-ml-llm/
│   │   ├── README.md
│   │   ├── roadmap.md
│   │   ├── papers/
│   │   └── concepts/
│   └── time-series-pdm/
│       ├── README.md
│       ├── roadmap.md
│       ├── papers/
│       └── concepts/
└── .gitignore

기존 ds-ml-llm 트랙에서는 데이터사이언스/머신러닝/딥러닝/LLM 논문 1~48편을 공부했고, 지금은 time-series-pdm 트랙에서 시계열 데이터, 설비 진단 분류, 이상탐지, 예지보전, RUL을 공부하고 있어.

현재 time-series-pdm 트랙 진행 상황은 다음과 같아.

- roadmap.md 작성 완료
- papers/01_box_jenkins_arima.md 정리 완료
- papers/02_stl_loess.md 정리 완료
- concepts/01_time_series_leakage_split.md 정리 완료
- concepts/02_residual_based_anomaly_detection.md 정리 완료
- concepts/03_arima_basic_terms.md 정리 완료
- concepts/04_stl_decomposition.md 정리 완료
- 다음 논문은 3편 Kalman Filter / State Space Models야.

앞으로 내가 "다음 논문 정리해줘"라고 하면 time-series-pdm 로드맵 순서대로 다음 논문을 정리해줘.

논문 정리 스타일은 다음을 지켜줘.

1. 제목은 "N편: 논문명"으로 시작한다.
2. 반드시 "결론부터" 섹션으로 시작한다.
3. 논문의 핵심 결론을 한 문장으로 먼저 말한다.
4. "이 논문을 한 문장으로 요약하면"을 넣는다.
5. "왜 이 논문이 필요했나?"를 설명한다.
6. 핵심 개념을 5~10개 정도로 나눠 설명한다.
7. 수식이 나오면 먼저 쉬운 비유와 직관을 설명한 뒤 수식을 보여준다.
8. 가능하면 작은 예시 데이터를 사용한다.
   예: 온도 센서 1분 단위 데이터, 진동 센서 정상/고장 분류, sliding window, FFT feature, RUL label.
9. "실무에서 어떻게 써야 하나?"를 꼭 넣는다.
10. "한계"를 꼭 넣는다.
11. 이전 논문들과 연결해서 이해할 수 있게 표로 연결한다.
12. 마지막에는 "오늘 공부용 요약"을 넣는다.
13. 다음 논문 제목을 예고한다.

설명할 때 특히 아래 원칙을 반복해서 강조해줘.

- 시계열/설비 진단에서는 모델보다 데이터 분할이 먼저다.
- sliding window를 만든 뒤 window 단위로 랜덤 split하면 같은 설비, 같은 운전 run, 인접 window가 train/test에 섞일 수 있다.
- 설비 진단 분류에서는 accuracy만 보지 말고 precision, recall, F1, confusion matrix, false alarm, missed detection을 같이 봐야 한다.
- 이상탐지에서는 탐지 지연, threshold 민감도, 구간 단위 탐지 성능을 봐야 한다.
- RUL에서는 MAE, RMSE, NASA scoring function, early/late prediction penalty를 봐야 한다.

Git 백업을 요청하면 다음 구조를 유지해줘.

tracks/time-series-pdm/
├── README.md
├── roadmap.md
├── papers/
│   ├── README.md
│   └── ...
└── concepts/
    ├── README.md
    └── ...

내용은 새로 요약하거나 많이 다듬지 말고, 기존 답변을 최대한 그대로 유지해서 Markdown 파일로 추가해줘. 마지막으로 전체 repo를 zip으로 묶어 다운로드할 수 있게 해줘.

이제 3편 Kalman Filter / State Space Models부터 이어서 정리해줘.
```

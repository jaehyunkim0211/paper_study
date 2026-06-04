# Study Notes

데이터사이언스, 머신러닝, LLM, 시계열 데이터, 설비 진단/예지보전을 공부하며 정리한 논문 및 개념 노트입니다.

이 저장소는 주제별 track 구조를 사용합니다. 루트는 전체 포털이고, 실제 논문 정리와 개념 정리는 각 track 아래에 있습니다.

## Repository Structure

```text
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
```

## Tracks

| Track | Description | Roadmap | Papers | Concepts |
|---|---|---|---|---|
| Data Science / ML / LLM | 데이터사이언스 기본기부터 현대 LLM reasoning, alignment, serving, 최신 LLM까지 정리 | [roadmap](tracks/ds-ml-llm/roadmap.md) | [papers](tracks/ds-ml-llm/papers/) | [concepts](tracks/ds-ml-llm/concepts/) |
| Time Series / Predictive Maintenance | 시계열 분석, 이상탐지, 설비 진단 분류, 예지보전, RUL 정리 | [roadmap](tracks/time-series-pdm/roadmap.md) | [papers](tracks/time-series-pdm/papers/) | [concepts](tracks/time-series-pdm/concepts/) |

## Current Progress

### Data Science / ML / LLM

- 01–10: 데이터사이언스 기본기 / 모델 평가 / 책임 있는 ML 완료
- 11–15: 딥러닝 기본기 완료
- 16–25: Transformer 이후 현대 AI 완료
- 26–30: 실무 ML 시스템 / 추천 / 데이터 중심 ML 완료
- 31–48: LLM reasoning / alignment / serving / 최신 LLM 완료

### Time Series / Predictive Maintenance

- Roadmap 작성 완료
- Part 1. 시계열 분석 기본기 완료: 01–05
  - 01. Box-Jenkins / ARIMA 완료
  - 02. STL / LOESS 완료
  - 03. Kalman Filter / State Space Models 완료
  - 04. Dynamic Time Warping / Learned Constraints 완료
  - 05. M4 Competition 완료
- Part 2. 시계열 특징 추출과 전통 ML 완료: 06–12
  - 06. UCR Time Series Archive 완료
  - 07. Time Series Shapelets 완료
  - 08. tsfresh 완료
  - 09. catch22 완료
  - 10. BOSS 완료
  - 11. ROCKET / MiniRocket 완료
  - 12. HIVE-COTE 2.0 완료
- Part 3. 딥러닝 기반 시계열 모델 완료: 13–18
  - 13. Time Series Classification from Scratch with Deep Neural Networks 완료
  - 14. InceptionTime 완료
  - 15. Temporal Convolutional Network, TCN 완료
  - 16. DeepAR 완료
  - 17. N-BEATS 완료
  - 18. TimesNet 완료
- Part 4. Transformer와 시계열 파운데이션 모델 진행 중: 19–23 완료
  - 19. Informer 완료
  - 20. Autoformer 완료
  - 21. FEDformer 완료
  - 22. PatchTST 완료
  - 23. TimeGPT-1 완료
- Concepts 01–27 완료
- 다음 논문: 24. TimesFM / A Decoder-Only Foundation Model for Time-Series Forecasting

## How to Use

각 track은 독립적인 `README.md`, `roadmap.md`, `papers/`, `concepts/`를 가집니다.

- 논문 정리: `tracks/<track-name>/papers/`
- 개념 정리: `tracks/<track-name>/concepts/`
- 학습 순서: `tracks/<track-name>/roadmap.md`

## Git Workflow

```bash
git add README.md tracks/time-series-pdm
git commit -m "update time-series predictive maintenance notes through TimeGPT"
```

새 track을 추가할 때는 `tracks/<new-track>/` 아래에 `README.md`, `roadmap.md`, `papers/`, `concepts/` 구조를 만들면 됩니다.

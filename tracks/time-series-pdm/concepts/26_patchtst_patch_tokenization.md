# PatchTST: patch tokenization

PatchTST는 시계열의 여러 time step을 하나의 patch token으로 묶어 Transformer에 넣는 모델입니다.

## 핵심

- Point token 대신 subseries patch token을 사용합니다.
- Token 수를 줄여 attention 비용을 낮춥니다.
- 더 긴 lookback history를 사용할 수 있습니다.
- Channel independence로 각 센서를 univariate series처럼 처리하고 weight를 공유합니다.
- Masked patch pretraining으로 label 없는 데이터 활용이 가능합니다.

## 설비 활용

온도, 전류 RMS, 압력, 진동 RMS, health index 같은 긴 저주파 센서 예측에 적합합니다. Raw vibration보다는 feature sequence에 우선 적용합니다.

## 주의

Patch가 너무 크면 작은 이상 신호가 희석될 수 있습니다. Channel independence는 전류-온도, 압력-유량 같은 센서 간 관계를 약하게 볼 수 있습니다.

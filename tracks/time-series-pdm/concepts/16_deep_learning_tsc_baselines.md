# 딥러닝 TSC baseline: FCN / ResNet

FCN과 ResNet은 raw time series를 사람이 만든 feature 없이 직접 학습하는 딥러닝 기반 시계열 분류 baseline입니다.

## 핵심

- 1D convolution filter는 작은 시계열 패턴 탐지기입니다.
- FCN은 Conv1D + BatchNorm + ReLU block과 Global Average Pooling을 사용합니다.
- ResNet은 residual shortcut으로 깊은 CNN 학습을 안정화합니다.
- CAM을 통해 어떤 시간 구간이 class 판단에 기여했는지 확인할 수 있습니다.

## 설비 진단 해석

베어링 진동에서는 짧은 충격 waveform, 모터 전류에서는 startup distortion, 압력에서는 valve cycle의 비정상 패턴을 CNN filter가 학습할 수 있습니다.

## 실무 주의

딥러닝은 window random split leakage를 매우 잘 외웁니다. Unit/run/condition split 없이 accuracy가 높으면 신뢰하기 어렵습니다. MiniRocket, catch22, tsfresh+XGBoost, BOSS, HIVE-COTE와 비교해야 합니다.

# InceptionTime과 multi-scale CNN

InceptionTime은 여러 길이의 1D convolution filter를 병렬로 적용해 시계열의 짧은 패턴과 긴 패턴을 동시에 보는 모델입니다.

## 핵심

- Inception module은 짧은/중간/긴 filter branch와 max-pooling branch를 병렬로 둡니다.
- Bottleneck layer는 1x1 convolution으로 채널 수를 줄여 계산량을 낮춥니다.
- Residual connection은 깊은 network 학습을 안정화합니다.
- Ensemble of 5는 초기값에 따른 성능 변동을 줄입니다.

## 설비 진단 해석

베어링 고장은 순간 충격과 반복 주기가 함께 중요합니다. InceptionTime은 짧은 filter로 충격을, 긴 filter로 반복 주기와 context를 볼 수 있습니다.

## 실무 주의

Multi-scale filter는 인접 window leakage도 여러 scale에서 잘 잡습니다. Sliding window random split을 피하고 unit/run split을 먼저 설계해야 합니다.

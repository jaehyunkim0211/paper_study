# Time-Series Shapelets

## 결론부터

Shapelet은 class를 가장 잘 구분하는 짧은 subsequence입니다.

## 핵심 직관

전체 진동 waveform을 다 볼 필요 없이, 고장 class에서 반복적으로 나타나는 짧은 충격 pattern만으로 정상/고장을 구분할 수 있습니다.

## SubsequenceDist

```math
SubsequenceDist(T, S) = min_{S' in T} Dist(S, S')
```

shapelet이 긴 시계열 안의 어느 위치에 가장 잘 맞는지 찾습니다.

## Information Gain

좋은 shapelet은 거리 threshold로 데이터를 나눴을 때 class entropy를 크게 줄입니다.

## 실무 주의

- shapelet은 반드시 train set에서만 찾아야 합니다.
- 전체 데이터에서 shapelet을 찾은 뒤 split하면 test label을 이용한 feature selection leakage입니다.
- 너무 짧은 shapelet은 noise spike를 고장 pattern으로 착각할 수 있습니다.

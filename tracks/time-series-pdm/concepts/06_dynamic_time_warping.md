# Dynamic Time Warping, DTW

## 결론부터

DTW는 두 시계열의 시간축이 조금 밀리거나 늘어나도 비슷한 패턴으로 비교할 수 있게 해주는 거리 계산 방법입니다.

## 왜 필요한가?

Euclidean distance는 같은 인덱스끼리만 비교합니다. 베어링 충격 peak가 한 window에서는 50번째 sample, 다른 window에서는 53번째 sample에 있으면 실제로는 같은 패턴인데 멀다고 볼 수 있습니다.

DTW는 시계열을 고무줄처럼 늘이거나 줄여서 가장 자연스럽게 맞춘 뒤 거리를 계산합니다.

## 수식

```math
gamma(i,j) = d(q_i,c_j) + min{gamma(i-1,j-1), gamma(i-1,j), gamma(i,j-1)}
```

현재 칸까지 오는 세 경로 중 가장 비용이 낮은 경로를 고르는 dynamic programming입니다.

## 제약

DTW를 무제한으로 허용하면 물리적으로 말이 안 되는 정렬이 생길 수 있습니다. 그래서 Sakoe-Chiba Band, slope constraint, R-K Band 같은 제약이 필요합니다.

## 실무 주의

- band width는 물리적 시간 변동 범위와 validation으로 정합니다.
- 너무 넓은 band는 정상과 고장도 억지로 비슷하게 만들 수 있습니다.
- DTW는 인접 window leakage에 특히 취약합니다.

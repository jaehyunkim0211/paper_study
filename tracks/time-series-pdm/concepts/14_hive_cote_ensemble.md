# HIVE-COTE 2.0

## 결론부터

HIVE-COTE 2.0은 shapelet, dictionary, interval, convolution 관점을 결합하는 heterogeneous TSC ensemble입니다.

## 구성 요소

| Component | 관점 |
|---|---|
| STC | shapelet 기반 짧은 부분 패턴 |
| TDE | dictionary 기반 symbolic word histogram |
| DrCIF | interval 기반 통계/주파수/catch22 feature |
| Arsenal | ROCKET classifier ensemble |

## CAWPE

각 component의 class probability를 성능 기반 가중 평균으로 결합합니다.

```math
P(c|x) = sum_m w_m P_m(c|x) / sum_m w_m
```

## 실무 위치

HC2는 정확하지만 무겁기 때문에 빠른 PoC용보다는 성능 상한선 benchmark로 보는 것이 좋습니다.

## 주의

HC2는 강력해서 잘못된 split의 leakage를 더 잘 이용합니다. unit/run/time/condition split이 먼저입니다.

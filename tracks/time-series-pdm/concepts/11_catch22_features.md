# catch22 / catch24

## 결론부터

catch22는 수천 개 feature 대신 빠르고 중복이 적은 22개 대표 시계열 feature를 사용하는 접근입니다.

## tsfresh와 차이

| tsfresh | catch22 |
|---|---|
| 많이 뽑고 통계적으로 고름 | 대표 feature 22개만 사용 |
| feature selection leakage 주의 | extraction 자체는 label을 보지 않음 |
| 탐색에 좋음 | 빠른 baseline에 좋음 |

## 설비 진단 주의

기본 catch22는 z-scored time series의 특성에 집중하므로 amplitude 정보가 약해질 수 있습니다. 온도 과열, 전류 과부하, 진동 RMS 증가처럼 절대값이 중요한 경우에는 catch24 또는 RMS/max/crest factor를 추가합니다.

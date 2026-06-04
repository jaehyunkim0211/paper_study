# TimesNet: period 기반 2D variation

TimesNet은 FFT로 주요 period를 찾고 1D 시계열을 period 기준 2D tensor로 접어 intraperiod와 interperiod variation을 동시에 모델링합니다.

## 핵심

- Intraperiod variation: 한 주기 내부 변화
- Interperiod variation: 주기 간 같은 phase의 변화
- FFT로 top-k period를 찾음
- 2D Inception block으로 2D tensor 처리
- Forecasting, imputation, classification, anomaly detection에 활용 가능

## 설비 활용

압력 cycle, 전류 RMS 주기, 온도 주야간 패턴, 진동 feature sequence처럼 주기가 있는 데이터에 적합합니다.

## 주의

FFT가 찾은 period가 실제 회전수, 제어 cycle, 결함 주기와 맞는지 검증해야 합니다. 전체 period 탐색 후 split하면 leakage가 될 수 있습니다.

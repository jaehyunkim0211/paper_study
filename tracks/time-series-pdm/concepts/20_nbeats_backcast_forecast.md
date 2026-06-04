# N-BEATS: Backcast와 Forecast

N-BEATS는 fully connected block이 과거를 설명하는 backcast와 미래를 예측하는 forecast를 동시에 내는 forecasting 모델입니다.

## 핵심

- Backcast: 입력 과거 구간을 설명하는 출력
- Forecast: 미래 예측 기여분
- Backward residual: 입력에서 backcast를 빼 다음 block으로 넘김
- Forecast accumulation: block별 forecast를 합쳐 최종 예측 생성
- Interpretable version: trend stack과 seasonality stack 사용

## 설비 활용

온도, 압력, 전류 RMS, 진동 RMS, health index 같은 저주파 센서의 point forecasting에 적합합니다.

## 주의

기본 N-BEATS는 univariate point forecast 모델입니다. 불확실성, 다변량 센서 관계, RUL 단조성은 별도 설계가 필요합니다.

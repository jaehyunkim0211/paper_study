# TCN: causal dilated convolution

TCN은 causal convolution과 dilated convolution을 사용해 미래를 보지 않고 긴 과거 history를 처리하는 sequence model입니다.

## 핵심

- Causal convolution은 현재와 과거만 사용합니다.
- Dilated convolution은 과거를 띄엄띄엄 보며 receptive field를 넓힙니다.
- Residual block은 깊은 TCN 학습을 안정화합니다.
- RNN/LSTM보다 병렬화가 쉽습니다.

## 설비 활용

- Forecasting: 과거 센서로 미래 온도/압력/진동 RMS 예측
- Anomaly detection: 예측 오차를 anomaly score로 사용
- RUL: 최근 cycle feature sequence로 RUL regression

## 주의

TCN이 causal이어도 전체 scaling, smoothing, window random split, RUL cycle random split은 leakage를 만들 수 있습니다.

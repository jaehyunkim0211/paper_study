# DeepAR와 probabilistic forecasting

DeepAR는 많은 관련 시계열을 하나의 autoregressive RNN으로 학습해 미래값의 분포를 예측하는 모델입니다.

## 핵심

- Global model: 여러 설비/센서 시계열을 함께 학습합니다.
- Probabilistic forecast: point forecast가 아니라 예측분포를 냅니다.
- Autoregressive: 이전 target 값을 다음 step 입력으로 사용합니다.
- Likelihood: Gaussian, negative binomial 등 데이터에 맞는 분포를 선택합니다.
- Sampling: 여러 미래 sample path로 quantile forecast를 계산합니다.

## 설비 활용

여러 모터의 온도, 전류 RMS, 진동 RMS, 압력 예측에 유용합니다. Prediction interval 밖의 실제값은 anomaly 후보가 될 수 있습니다.

## 주의

미래 실제 부하나 전류처럼 prediction 시점에 모르는 covariate를 넣으면 leakage입니다. Calibration과 coverage를 반드시 확인해야 합니다.

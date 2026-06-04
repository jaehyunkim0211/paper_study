# FEDformer: frequency-domain forecasting

FEDformer는 decomposition과 Fourier/Wavelet frequency-enhanced block을 결합한 장기 예측 모델입니다.

## 핵심

- Frequency domain에서 중요한 mode만 처리합니다.
- MOEDecomp는 여러 moving average filter를 섞어 trend를 뽑습니다.
- Fourier block은 global periodic structure에 적합합니다.
- Wavelet block은 local time-frequency 변화에 적합합니다.

## 설비 활용

온도, 압력, 전류 RMS, 진동 RMS, RMS/kurtosis/envelope feature sequence처럼 trend와 frequency 구조가 있는 데이터에 적합합니다.

## 주의

선택된 frequency mode가 실제 회전수, 결함 주파수, 전원 주파수, 제어 cycle과 맞는지 검증해야 합니다.

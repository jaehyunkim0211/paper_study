# ROCKET / MiniRocket

## 결론부터

ROCKET은 random convolution kernel을 많이 적용해 시계열을 feature vector로 바꾸고, MiniRocket은 이를 훨씬 빠르고 거의 deterministic하게 만든 강력한 TSC baseline입니다.

## 핵심 feature

- max: 어떤 kernel이 시계열 어딘가에서 얼마나 강하게 반응했는지
- PPV: convolution output이 threshold보다 큰 위치의 비율

```math
PPV = (1/n) sum 1(y_t > b)
```

## CNN과 차이

CNN은 kernel을 학습하지만 ROCKET/MiniRocket은 random 또는 fixed kernel을 사용합니다. transform 후에는 RidgeClassifier나 Logistic Regression 같은 선형 분류기를 씁니다.

## 실무 위치

MiniRocket은 CNN/Transformer 전에 반드시 비교해야 하는 강력한 baseline입니다.

## 주의

- transform과 scaler는 train에서만 fit합니다.
- window random split은 MiniRocket에서 특히 위험합니다.
- 물리 해석은 RMS, FFT, envelope feature보다 어렵습니다.

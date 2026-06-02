# BOSS / Dictionary-based TSC

## 결론부터

BOSS는 시계열을 짧은 window로 자르고 Fourier 기반 symbolic word로 바꾼 뒤, word histogram으로 분류하는 dictionary-based TSC 방법입니다.

## 핵심 흐름

```text
raw time series
→ sliding windows
→ DFT
→ SFA word
→ word histogram
→ 1-NN classification
```

## 핵심 개념

- SFA: Fourier coefficient를 symbol word로 변환합니다.
- MCB: Fourier coefficient를 symbol로 나누는 경계값을 train data에서 학습합니다.
- numerosity reduction: 같은 word가 연속 반복될 때 중복 count를 줄입니다.
- BOSS distance: word histogram 간 차이를 계산합니다.

## 실무 주의

- MCB breakpoints는 train에서만 학습합니다.
- window random split은 인접 window leakage를 만듭니다.
- BOSS는 순서보다 word 빈도를 보므로 phase shift에는 강하지만 순서 정보는 일부 잃습니다.

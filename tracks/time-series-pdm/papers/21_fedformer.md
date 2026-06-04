# 21편: FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting

## 결론부터

**긴 시계열 예측에서는 시간 영역의 모든 point를 직접 비교하기보다, trend와 seasonal을 분해하고 Fourier/Wavelet 같은 frequency domain에서 중요한 성분만 다루면 더 효율적이고 강력한 예측 모델을 만들 수 있습니다.**

## 이 논문을 한 문장으로 요약하면

FEDformer는 Autoformer의 decomposition 사고를 이어받되, self-attention과 cross-attention을 Fourier/Wavelet 기반 frequency-enhanced block으로 대체해 장기 예측의 계산량을 줄이고 global structure를 더 잘 잡으려는 Transformer 계열 모델입니다.

## 왜 이 논문이 필요했나?

많은 시계열은 frequency domain에서 몇 개의 중요한 성분으로 설명됩니다. 설비 데이터에서는 베어링 결함 주파수, 모터 전류 harmonic, 압력 제어 cycle, 온도 주야간 주기처럼 주파수 구조가 중요합니다. FEDformer는 모든 time point를 attention으로 비교하지 않고 중요한 frequency mode를 선택해 예측에 활용합니다.

## 핵심 개념

### 1. Frequency domain

시간 영역은 값이 시간에 따라 어떻게 변하는지 보는 관점이고, 주파수 영역은 어떤 반복 성분들이 섞여 있는지 보는 관점입니다.

### 2. Sparse frequency representation

모든 주파수가 중요한 것은 아닙니다. 장기 예측에서는 저주파 trend, 주기성, 주요 운전 cycle이 핵심일 수 있습니다.

### 3. Decomposition

FEDformer도 trend와 seasonal/detail component를 분리합니다. Autoformer보다 더 유연한 decomposition을 위해 MOEDecomp를 사용합니다.

### 4. MOEDecomp

여러 moving average filter를 expert로 두고, 데이터에 따라 가중합해 trend를 뽑습니다.


X_{trend}=Softmax(L(x))*F(x)


작은 filter는 짧은 부하 변화, 큰 filter는 장기 열화 trend를 잡을 수 있습니다.

### 5. FEB: Frequency Enhanced Block

시간 영역 시계열을 Fourier 또는 Wavelet domain으로 바꾸고, 중요한 mode만 선택해 처리한 뒤 inverse transform으로 돌아옵니다.

### 6. Fourier Enhanced Block


FEB_f(q)=F^{-1}(Padding(\tilde{Q}\odot R))


선택된 Fourier mode에 학습 가능한 kernel을 적용합니다.

### 7. Wavelet Enhanced Block

Wavelet은 global frequency뿐 아니라 언제 어떤 주파수가 나타나는지 보는 time-frequency 표현입니다. transient shock, startup 과도 현상, 베어링 초기 충격에 더 자연스러울 수 있습니다.

### 8. FEA: Frequency Enhanced Attention

Encoder-decoder cross-attention도 frequency domain에서 처리합니다.

## 작은 예시

압력 데이터가 4-step cycle을 갖는다면 Fourier domain에서 4-step 주기 성분이 강하게 나타납니다. FEDformer는 해당 mode를 선택해 장기 압력 profile 예측에 활용할 수 있습니다.

진동 RMS와 kurtosis feature sequence에서는 장기 trend와 충격성 증가가 frequency/time-frequency 구조로 나타날 수 있습니다.

## 실무에서 어떻게 써야 하나?

온도, 압력, 전류 RMS, 진동 RMS, health feature sequence처럼 trend와 frequency 구조가 있는 장기 예측에 유용합니다. Raw vibration waveform에는 직접 적용하기보다 envelope, RMS, kurtosis, FFT band energy 같은 feature sequence로 변환하는 것이 좋습니다.

Fourier와 Wavelet 버전을 모두 비교하고, 선택된 mode가 회전수, 베어링 결함 주파수, 전원 주파수, 제어 cycle과 맞는지 확인해야 합니다.

## 데이터 분할과 평가

Window random split, 전체 scaling, 같은 설비/run 혼합, 미래 실제 부하 입력, test로 mode 수 선택은 모두 leakage 위험입니다. Forecasting은 MAE/RMSE/MASE, anomaly detection은 false alarm/missed detection/delay, RUL은 NASA score와 early/late penalty를 봐야 합니다.

## 한계

Frequency sparsity 가정이 항상 맞지는 않습니다. 선택된 frequency mode가 물리적으로 중요한지 검증해야 합니다. Raw vibration classification의 최선 모델은 아닐 수 있습니다. Point forecast 중심이라 uncertainty가 부족합니다.

## 다음 논문 예고

다음은 **22편: PatchTST**입니다. 시계열을 point 단위가 아니라 patch token으로 묶어 Transformer에 넣는 모델입니다.

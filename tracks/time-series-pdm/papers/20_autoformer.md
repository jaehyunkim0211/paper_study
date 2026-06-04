# 20편: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

## 결론부터

**긴 시계열 예측에서는 Transformer의 point-wise attention만으로는 부족하므로, 시계열을 trend와 seasonal 성분으로 계속 분해하면서 같은 주기 위치의 sub-series끼리 연결하는 Auto-Correlation 구조를 쓰는 것이 더 적합합니다.**

## 이 논문을 한 문장으로 요약하면

Autoformer는 Transformer 안에 series decomposition block을 내장하고, self-attention 대신 주기 기반 Auto-Correlation mechanism을 사용해 장기 예측의 정확도와 효율을 개선하려는 모델입니다.

## 왜 이 논문이 필요했나?

Informer는 attention 계산량을 줄이는 데 집중했습니다. Autoformer는 더 근본적으로 “point-wise attention이 시계열에 가장 적합한가?”라고 묻습니다. 설비 센서는 valve cycle, 교대조, 회전 주기, 주야간처럼 반복 구조가 강합니다. 미래를 예측할 때 모든 point를 비교하기보다 같은 phase의 sub-series를 참고하는 것이 자연스럽습니다.

## 핵심 개념

### 1. Decomposition을 모델 내부로

STL처럼 trend와 seasonal을 나누는 사고를 전처리가 아니라 Transformer block 내부에 넣습니다.

### 2. Moving average decomposition


X_t = AvgPool(Padding(X))
X_s = X - X_t


Moving average로 trend-cyclical part를 뽑고, 원본에서 빼 seasonal part를 만듭니다.

### 3. Encoder와 decoder 역할

Encoder는 trend를 제거하고 seasonal pattern modeling에 집중합니다. Decoder는 trend part를 layer마다 누적하면서 prediction을 정제합니다.

### 4. Auto-Correlation

Self-attention이 point-wise 관계를 본다면, Auto-Correlation은 time-delay similarity와 period-based sub-series 관계를 봅니다.

### 5. Autocorrelation


R_{XX}(\tau)=\frac{1}{L}\sum_{t=1}^{L}X_t X_{t-\tau}


시계열을 τ만큼 지연했을 때 원래 시계열과 얼마나 닮았는지 봅니다.

### 6. Time delay aggregation

중요한 delay를 top-k로 고르고 value series를 roll해 같은 phase의 sub-series를 align한 뒤 가중합합니다.


AutoCorrelation(Q,K,V)=\sum_i Roll(V,\tau_i)\hat{R}(\tau_i)


### 7. FFT 기반 효율화

Autocorrelation은 FFT로 빠르게 계산할 수 있어 O(L log L) complexity를 목표로 합니다.

## 작은 예시

압력 cycle이 4분 주기라면 τ=4에서 autocorrelation이 강합니다. 미래 peak 압력을 예측할 때 과거 cycle의 같은 phase peak들을 time delay aggregation으로 참고하는 식입니다.

## 실무에서 어떻게 써야 하나?

온도, 압력, 전류 RMS, 진동 RMS처럼 trend와 cycle이 섞인 저주파 센서 장기 예측에 적합합니다. STL+ARIMA, seasonal naive, N-BEATS, TCN, Informer, TimesNet과 비교해야 합니다.

선택된 delay가 실제 회전 주기, valve cycle, batch cycle, 교대조, 주야간과 맞는지 물리적으로 확인해야 합니다.

## 데이터 분할과 평가

전체 scaling, 전체 decomposition 후 split, window random split, future covariate 사용은 leakage가 됩니다. 이상탐지에는 false alarm, missed detection, detection delay, event-level recall을 봐야 합니다. RUL에는 unit split과 early/late penalty가 필요합니다.

## 한계

주기성이 약한 데이터에서는 장점이 줄어들 수 있습니다. Moving average decomposition이 복잡한 설비 신호를 완벽히 나누지는 못합니다. 기본적으로 point forecasting 중심이라 uncertainty 보완이 필요합니다. Raw 고주파 vibration에는 직접 적용보다 health feature sequence 예측이 자연스럽습니다.

## 다음 논문 예고

다음은 **21편: FEDformer**입니다. Decomposition과 frequency-domain Transformer 구조를 결합합니다.

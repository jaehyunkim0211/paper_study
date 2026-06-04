# 18편: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis

## 결론부터

**복잡한 시계열을 1차원으로만 보지 말고 주기 기준으로 접어서 2차원 구조로 보면, 주기 내부 변화와 주기 간 변화를 동시에 모델링할 수 있습니다.**

## 이 논문을 한 문장으로 요약하면

TimesNet은 FFT로 시계열의 주요 period를 찾고, 1D 시계열을 period별 2D tensor로 reshape한 뒤 2D Inception block으로 forecasting, imputation, classification, anomaly detection을 함께 처리하려는 범용 시계열 backbone입니다.

## 왜 이 논문이 필요했나?

설비 시계열에는 여러 주기가 동시에 있습니다. 온도에는 주야간, 교대조, 부하 cycle이 있고, 압력에는 valve cycle과 batch cycle이 있으며, 진동에는 회전 주기와 결함 충격 반복이 있습니다. 1D CNN이나 TCN도 강력하지만 주기 구조를 명시적으로 드러내지는 않습니다.

TimesNet은 “주기가 있다면 그 주기 단위로 접어서 2D로 보자”는 모델입니다.

## 핵심 개념

### 1. Multi-periodicity

시계열에는 여러 period가 동시에 있습니다. TimesNet은 FFT로 top-k period를 찾고, 각각에 대해 2D tensor를 만듭니다.

### 2. Intraperiod variation

한 주기 내부의 변화입니다. 압력 cycle 안에서 phase 1→2→3→4로 압력이 오르내리는 패턴이 여기에 해당합니다.

### 3. Interperiod variation

같은 phase가 cycle 사이에서 어떻게 달라지는지 보는 것입니다. phase 3 peak 압력이 cycle마다 6.0→6.1→6.2로 상승하면 baseline 변화나 막힘 가능성을 의심할 수 있습니다.

### 4. 1D를 2D로 reshape

```text
1D: [5.0, 5.5, 6.0, 5.4, 5.1, 5.6, 6.1, 5.5]
period 4로 reshape:
[[5.0, 5.5, 6.0, 5.4],
 [5.1, 5.6, 6.1, 5.5]]
```

가로는 period 내부 phase, 세로는 period 반복 index입니다.

### 5. FFT로 period 찾기


A = Avg(Amp(FFT(X)))


Amplitude가 큰 frequency를 고르고 period를 계산합니다. 설비에서는 선택된 period가 회전수, 제어 cycle, 교대조, 결함 주기와 맞는지 확인해야 합니다.

### 6. TimesBlock

```text
입력 1D
→ FFT로 top-k period
→ period별 2D reshape
→ 2D Inception block
→ 1D로 되돌림
→ period별 결과 가중합
→ residual connection
```

### 7. Task-general backbone

TimesNet은 forecasting뿐 아니라 imputation, classification, anomaly detection도 다루려는 backbone입니다.

## 작은 예시

압력 cycle:

| cycle | phase1 | phase2 | phase3 | phase4 |
|---|---:|---:|---:|---:|
| 1 | 5.0 | 5.5 | 6.0 | 5.4 |
| 2 | 5.1 | 5.6 | 6.1 | 5.5 |
| 3 | 5.2 | 5.7 | 6.2 | 5.6 |

Intraperiod는 각 row 안의 모양, interperiod는 같은 column의 변화입니다. TimesNet은 이 2D structure를 2D convolution으로 처리합니다.

## 실무에서 어떻게 써야 하나?

주기적 운전 패턴이 강한 센서에 우선 적용합니다. 압력 cycle, 전류 RMS 주기 패턴, 온도 주야간 패턴, 진동 RMS trend에 적합합니다. Raw vibration waveform에는 직접 적용하기보다 RMS, kurtosis, envelope peak, FFT band energy 같은 feature sequence로 바꿔 적용하는 것이 실무적입니다.

FFT가 찾은 period는 반드시 물리적으로 검증해야 합니다. Noise나 sensor artifact를 period로 착각하면 모델이 잘못된 패턴을 학습할 수 있습니다.

## 평가와 split

Forecasting은 MAE, RMSE, MASE, sMAPE를 봅니다. Classification은 precision, recall, F1, confusion matrix를 봅니다. Anomaly detection은 false alarm, missed detection, detection delay, event-level recall을 봅니다. RUL은 MAE, RMSE, NASA scoring function, early/late penalty를 봅니다.

Window random split, 전체 scaling, 전체 period 탐색 후 split은 leakage가 될 수 있습니다.

## 한계

주기성이 약한 데이터에서는 장점이 줄어듭니다. FFT period가 물리 period와 다를 수 있습니다. Raw 고주파 vibration에는 직접 적용이 어려울 수 있습니다. Task-general이라고 모든 task에서 최선은 아닙니다. 해석성도 제한적입니다.

## 다음 논문 예고

다음은 **19편: Informer**입니다. 긴 시계열에서 Transformer attention 계산량을 줄이는 모델입니다.

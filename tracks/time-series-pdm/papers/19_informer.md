# 19편: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

## 결론부터

**긴 시계열 예측에 Transformer를 그대로 쓰면 계산량과 메모리 문제가 커지므로, attention을 희소화하고 encoder 길이를 줄이며 decoder가 긴 미래 구간을 한 번에 출력하도록 바꾸면 장기 예측을 더 효율적으로 할 수 있습니다.**

## 이 논문을 한 문장으로 요약하면

Informer는 long sequence time-series forecasting을 위해 vanilla Transformer의 self-attention을 ProbSparse attention으로 줄이고, self-attention distilling으로 encoder 길이를 압축하며, generative-style decoder로 긴 horizon을 한 번에 출력하는 efficient Transformer입니다.

## 왜 이 논문이 필요했나?

Transformer는 멀리 떨어진 시점끼리 직접 연결할 수 있지만, self-attention은 sequence length L에 대해 O(L²) 비용이 듭니다. 설비 데이터에서는 1주일 온도 history, 긴 압력 cycle, 장기 진동 RMS trend처럼 긴 입력이 필요합니다. Informer는 긴 history를 보면서도 계산량을 줄이는 방법을 제안합니다.

## 핵심 개념

### 1. LSTF

Long Sequence Time-Series Forecasting은 긴 과거를 보고 긴 미래 horizon을 예측하는 문제입니다. 예지보전에서는 과거 7일 센서로 다음 24시간 위험을 예측하는 식입니다.

### 2. Vanilla attention 비용


Attention(Q,K,V)=Softmax(QK^T/\sqrt{d})V


모든 query와 key를 비교하므로 attention matrix가 L×L입니다.

### 3. ProbSparse Attention

모든 query가 똑같이 중요하지 않습니다. 특정 key에 강하게 집중하는 active query만 full attention을 계산하고, lazy query는 간략하게 처리합니다.

### 4. Sparsity score

Attention이 얼마나 뾰족한지 근사해 active query를 고릅니다.


M(q_i,K)=max_j(q_i k_j^T/\sqrt{d}) - mean_j(q_i k_j^T/\sqrt{d})


값이 크면 특정 key에 강하게 몰리는 query입니다.

### 5. Self-attention distilling

Encoder layer 사이에서 sequence 길이를 줄입니다.

```text
길이 1000 → 500 → 250
```

계산량을 줄이고 dominant attention pattern을 유지하려는 목적입니다.

### 6. Generative-style decoder

미래를 step-by-step으로 예측하지 않고 긴 horizon을 한 번에 출력합니다.

```text
known label segment + future placeholder → future horizon 전체
```

### 7. seq_len, label_len, pred_len

`seq_len`은 encoder 입력 길이, `label_len`은 decoder가 참고할 최근 label 길이, `pred_len`은 예측 horizon입니다.

## 작은 예시

과거 2시간 온도/전류 RMS로 앞으로 30분 온도를 예측합니다.

| parameter | 예시 |
|---|---|
| seq_len | 120분 |
| label_len | 30분 |
| pred_len | 30분 |

오차가 특정 시점부터 커지면 부하 급변, 냉각 문제, 센서 drift, 과열 전조를 확인해야 합니다.

## 실무에서 어떻게 써야 하나?

온도, 전류 RMS, 압력, 진동 RMS 같은 긴 저주파 센서 예측에 먼저 검토합니다. Raw 고주파 vibration waveform에는 직접 적용하기보다 RMS, kurtosis, envelope peak, FFT band energy 같은 feature sequence가 더 실무적입니다.

Last value, seasonal naive, ARIMA, STL, TCN, N-BEATS, DeepAR, TimesNet과 비교해야 합니다.

## 데이터 분할

Window random split은 인접 history와 horizon이 겹쳐 누수됩니다. 과거 train, 미래 validation/test, unit holdout, run split, condition holdout을 목적에 맞게 설계해야 합니다. 미래 실제 부하나 전류를 covariate로 넣으면 leakage입니다.

## 한계

Point forecast 중심이라 uncertainty가 부족합니다. Sparse attention이 작은 초기 이상을 놓칠 수 있습니다. Attention 해석은 인과 해석이 아닙니다. Raw vibration classification에는 직접적인 최선이 아닐 수 있습니다.

## 다음 논문 예고

다음은 **20편: Autoformer**입니다. Decomposition과 Auto-Correlation을 Transformer 내부에 넣는 모델입니다.

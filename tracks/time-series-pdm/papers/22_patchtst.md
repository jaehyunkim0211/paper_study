# 22편: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers

## 결론부터

**긴 시계열을 Transformer에 한 시점씩 넣지 말고 여러 시점을 묶은 patch 단위 token으로 넣으면, 계산량은 줄이고 더 긴 history를 보면서 예측 성능도 높일 수 있습니다.**

## 이 논문을 한 문장으로 요약하면

PatchTST는 시계열을 작은 subseries patch로 잘라 Transformer token처럼 사용하고, 여러 센서 채널을 섞지 않고 channel-independent 방식으로 처리해 long-term forecasting 성능과 효율을 개선한 Transformer 기반 모델입니다.

## 왜 이 논문이 필요했나?

Transformer에 시계열을 넣을 때 한 시점 하나를 token처럼 쓰면 token 수가 너무 많고, 한 점만으로는 의미가 약합니다. 설비 데이터에서는 온도 한 점보다 “완만한 상승 patch”, 압력 한 점보다 “valve cycle patch”, 진동 한 점보다 “충격 waveform patch”가 더 의미 있는 단위입니다.

PatchTST는 입력 표현 자체를 point token에서 patch token으로 바꿉니다.

## 핵심 개념

### 1. Patching

길이 16 시계열을 patch length 4로 자르면 4개의 patch token이 됩니다.

```text
[70.0,70.1,70.2,70.4] → patch 1
[70.5,70.7,70.8,71.0] → patch 2
```

### 2. Patch는 짧은 시계열 문장

온도 patch는 상승/급상승, 압력 patch는 정상 cycle/비정상 peak, 진동 patch는 충격 pattern을 담을 수 있습니다.

### 3. Stride와 overlap

Patch length와 stride를 정해야 합니다. Overlap이 많으면 local continuity는 잘 보존하지만 계산량과 중복이 늘고, 너무 큰 patch는 작은 이상이 희석될 수 있습니다.

### 4. Channel independence

각 sensor channel을 univariate series처럼 독립적으로 patching하고 처리합니다. Embedding과 Transformer weight는 channel들이 공유합니다.

장점은 overfitting 감소와 temporal pattern 학습 안정성이고, 단점은 전류 상승 후 온도 상승 같은 sensor 간 관계를 직접 모델링하기 어렵다는 점입니다.

### 5. Patch 간 attention

Transformer는 point가 아니라 patch 간 관계를 봅니다. 미래 온도를 예측할 때 최근 급상승 patch가 중요한 token이 될 수 있습니다.

### 6. Attention 비용 감소

원본 길이 L, patch 수 N이면 attention 비용이 O(L²)에서 O(N²)로 줄어듭니다. 보통 N은 L보다 훨씬 작습니다.

### 7. Longer look-back

Token 수가 줄어 더 긴 과거 history를 볼 수 있습니다. 예지보전에서 장기 열화 trend를 보기 유리합니다.

### 8. Masked patch pretraining

일부 patch를 mask하고 복원하도록 학습할 수 있습니다. 설비에서는 label 없는 정상 데이터가 많으므로 self-supervised pretraining에 유용합니다.

## 작은 예시

온도 데이터:

| patch | 값 | 의미 |
|---|---|---|
| P1 | `[70.0,70.1,70.3,70.5]` | 완만한 상승 |
| P2 | `[70.8,71.0,71.3,71.6]` | 지속 상승 |
| P3 | `[72.0,72.5,73.0,73.8]` | 상승 속도 증가 |

PatchTST는 patch sequence를 보고 미래 온도를 예측합니다.

## 실무에서 어떻게 써야 하나?

온도, 전류 RMS, 압력, 진동 RMS, health index 같은 긴 저주파 센서 예측에 적합합니다. Raw vibration에는 직접 적용하기보다 RMS, kurtosis, envelope peak, FFT band energy 같은 feature sequence를 사용하는 것이 실무적입니다.

Patch length와 stride는 설비 물리 시간에 맞춰야 합니다. 압력은 valve cycle, 진동 RMS는 여러 회전/충격 반복, 온도는 수분~수십분 변화가 patch에 담기도록 정합니다.

Channel independence가 맞는지도 검증해야 합니다. 센서 간 관계가 중요하면 derived feature나 channel-mixing 모델과 비교합니다.

## 데이터 분할

Patch는 모델 내부 tokenization입니다. 긴 run을 window로 자른 뒤 random split하는 문제를 해결하지 못합니다. 전체 scaling, 같은 run 혼합, 미래 실제 부하 입력, test로 patch length 선택은 leakage입니다.

## 이상탐지와 RUL

Forecast error나 masked reconstruction error를 anomaly score로 쓸 수 있습니다. RUL에서는 health index patch sequence를 forecasting하고 threshold crossing으로 RUL 후보를 만들 수 있습니다. 반드시 unit split과 early/late penalty 평가가 필요합니다.

## 한계

Channel independence는 센서 간 물리 관계를 약하게 볼 수 있습니다. Patch가 너무 크면 작은 이상 신호가 희석됩니다. 기본 구조는 point forecast 중심이라 uncertainty 보완이 필요합니다. Raw 고주파 vibration classification에는 직접적인 최선이 아닐 수 있습니다.

## 다음 논문 예고

다음은 **23편: TimeGPT-1**입니다. 대규모 사전학습된 시계열 foundation model을 zero-shot/fine-tuning으로 활용하는 흐름입니다.

# 12편: HIVE-COTE 2.0: a new meta ensemble for time series classification

## 결론부터

**이 논문의 핵심 결론은, 시계열 분류에서는 하나의 표현 방식만 믿기보다 shapelet, dictionary, interval, convolution처럼 서로 다른 관점의 강한 모델들을 결합하면 더 안정적이고 높은 성능을 얻을 수 있다는 것입니다.**

## 이 논문을 한 문장으로 요약하면

**HIVE-COTE 2.0은 시계열을 여러 표현 공간에서 동시에 바라보는 heterogeneous meta-ensemble이며, STC, TDE, DrCIF, Arsenal이라는 네 가지 강한 구성 모델의 class probability를 가중 결합해 예측하는 고성능 TSC ensemble입니다.**

## 왜 이 논문이 필요했나?

Part 2에서 여러 시계열 분류 표현을 배웠습니다.

| 논문 | 시계열을 보는 관점 |
|---|---|
| Shapelet | class를 가르는 짧은 부분 패턴 |
| tsfresh | 많은 통계·신호 feature 자동 추출 |
| catch22 | 빠르고 대표성 있는 22개 feature |
| BOSS | symbolic word histogram |
| ROCKET/MiniRocket | random/fixed convolution feature |

설비 진단에서는 하나의 표현만으로 부족한 경우가 많습니다.

| 고장 신호의 모습 | 잘 맞는 접근 |
|---|---|
| 짧은 충격 패턴 | Shapelet |
| 반복되는 충격 word | BOSS/TDE |
| 특정 구간의 통계 변화 | Interval/DrCIF |
| 다양한 waveform 패턴 | ROCKET/Arsenal |
| RMS, kurtosis, FFT feature | tsfresh/catch22/domain feature |

HIVE-COTE 2.0은 여러 표현을 모두 쓰고 결합하자는 접근입니다.

## 핵심 개념 1. Heterogeneous ensemble

HIVE-COTE는 비슷한 모델을 여러 개 모은 것이 아니라, 서로 다른 표현 방식을 쓰는 모델들을 결합합니다.

| 전문가 | 보는 것 |
|---|---|
| 진동 전문가 | waveform 충격과 envelope |
| 전기 전문가 | current harmonic과 sideband |
| 공정 전문가 | 압력/유량 cycle |
| 통계 전문가 | trend, variance, residual |
| ML 전문가 | convolution feature와 classifier |

## 핵심 개념 2. HC2의 네 구성 요소

| 구성 요소 | 계열 | 직관 |
|---|---|---|
| STC | Shapelet | class를 가르는 짧은 부분 패턴 |
| TDE | Dictionary | 반복되는 symbolic word pattern |
| DrCIF | Interval | 특정 구간의 통계·주파수·catch22 feature |
| Arsenal | Convolution | 여러 ROCKET classifier의 ensemble |

## 핵심 개념 3. STC

STC는 shapelet 기반입니다. 고장 class를 잘 구분하는 짧은 부분 waveform을 찾고, 각 시계열을 shapelet까지의 거리 feature로 변환합니다.

| 원본 window | shapelet A까지 거리 | shapelet B까지 거리 | label |
|---|---:|---:|---|
| W1 | 5.2 | 4.8 | 정상 |
| W2 | 0.4 | 3.9 | 외륜 결함 |
| W3 | 3.8 | 0.3 | 내륜 결함 |

## 핵심 개념 4. TDE

TDE는 BOSS 계열 dictionary classifier입니다. 시계열을 SFA word histogram으로 바꾸고 반복되는 pattern word 분포로 분류합니다.

| 시계열 | 자주 나오는 symbolic word |
|---|---|
| 정상 진동 | `aaaa`, `aabb`, `abca` |
| 외륜 결함 | `dcba`, `dcbb`, `bcda` |
| 내륜 결함 | `ccbd`, `bdca`, `dcab` |

## 핵심 개념 5. DrCIF

DrCIF는 interval 기반 classifier입니다. 전체 시계열을 여러 구간으로 보고, 각 구간의 통계·주파수·catch22-style feature를 계산합니다.

| 센서 | DrCIF가 볼 수 있는 구간 feature |
|---|---|
| 온도 | 특정 구간의 평균, slope |
| 압력 | valve cycle 구간의 variance |
| 전류 | startup 중 특정 구간의 spectral feature |
| 진동 RMS | 고장 전 구간의 catch22-style feature |

## 핵심 개념 6. Arsenal

Arsenal은 ROCKET classifier들의 ensemble입니다. 다양한 random convolution feature로 raw waveform 또는 전처리된 vibration/envelope signal의 패턴을 잡습니다.

## 핵심 개념 7. CAWPE

HC2는 네 구성 모델의 class probability를 단순 다수결이 아니라 성능 기반 가중 평균으로 결합합니다.

\[
P(c \mid x) =
\frac{
\sum_{m=1}^{M} w_m P_m(c \mid x)
}{
\sum_{m=1}^{M} w_m
}
\]

\[
w_m = acc_m^{\alpha}
\]

더 믿을 만한 component의 확률을 더 크게 반영합니다.

## 핵심 개념 8. 왜 하나의 best model을 고르지 않나?

한 고장 class는 shapelet으로 잘 잡히고, 다른 class는 dictionary word, interval feature, convolution feature로 잘 잡힐 수 있습니다.

| class | 잘 잡는 표현 |
|---|---|
| 정상 | 안정적인 interval statistics |
| 외륜 결함 | 반복 충격 dictionary word |
| 내륜 결함 | phase-independent shapelet |
| 불평형 | frequency/interval feature |
| 복합 결함 | convolution feature |

Ensemble은 특정 표현의 약점을 줄입니다.

## 핵심 개념 9. 정확하지만 무겁다

HC2는 매우 강력하지만 계산량이 큽니다. 실무 배포 기본 모델이라기보다 성능 상한선 benchmark로 생각하는 것이 좋습니다.

| 목적 | HC2 사용 적합성 |
|---|---|
| 연구 benchmark 최고 성능 확인 | 적합 |
| 최종 accuracy ceiling 확인 | 적합 |
| 빠른 PoC | MiniRocket/catch22가 더 적합 |
| edge device 배포 | 보통 부적합 |
| 대규모 고주파 raw vibration | 계산량 주의 |

## 작은 예시: 구성 모델 결합

| 모델 | 정상 | 외륜 결함 | 내륜 결함 |
|---|---:|---:|---:|
| STC | 0.10 | 0.80 | 0.10 |
| TDE | 0.25 | 0.65 | 0.10 |
| DrCIF | 0.45 | 0.40 | 0.15 |
| Arsenal | 0.05 | 0.90 | 0.05 |

STC, TDE, Arsenal이 외륜 결함을 강하게 지지하고 DrCIF만 약간 애매하다면 최종 예측은 외륜 결함이 됩니다.

## 실무에서 어떻게 써야 하나?

1. HC2는 성능 상한선 benchmark로 씁니다.
2. 구성 요소별 예측을 해석에 활용합니다.
3. 설비 진단에서는 split을 더 엄격하게 해야 합니다.
4. 운영 지표를 반드시 같이 봅니다.
5. MiniRocket과 비용 대비 성능을 비교합니다.

| 위험한 방식 | 문제 |
|---|---|
| window random split | 인접 window 누수 |
| 같은 bearing run이 train/test에 섞임 | run fingerprint 학습 |
| 전체 데이터 scaling 후 split | test 분포 누수 |
| test set을 보며 threshold 조정 | 운영 성능 과대평가 |

## 이상탐지와 RUL 연결

HC2는 supervised classifier입니다. 정상/고장 또는 degradation stage classifier로 쓰고, class probability를 anomaly score나 RUL regressor input으로 활용할 수 있습니다. 이상탐지에서는 detection delay, event-level recall, false alarm per day를 봐야 합니다. RUL에서는 MAE, RMSE, NASA scoring function, early/late penalty를 봐야 합니다.

## 한계

1. 계산량이 큽니다.
2. 실시간 배포용 모델로는 부담스러울 수 있습니다.
3. 해석이 단일 모델보다 복잡합니다.
4. missing value와 unequal length는 사전 처리가 필요합니다.
5. split이 잘못되면 강력한 leakage 증폭기가 됩니다.
6. class imbalance를 해결해주지는 않습니다.

## 이전 논문들과 연결해서 이해하기

| 관점 | Shapelet | tsfresh | catch22 | BOSS | ROCKET | HIVE-COTE 2.0 |
|---|---|---|---|---|---|---|
| 표현 | subsequence | feature matrix | 22개 feature | word histogram | convolution feature | shapelet + dictionary + interval + convolution |
| 장점 | 고장 signature | 넓은 탐색 | 빠름 | 반복 pattern | 빠르고 강력 | 매우 높은 정확도 |
| 단점 | 탐색 비용 | leakage 위험 | 표현력 제한 | 순서 손실 | 해석 낮음 | 계산량 큼 |

## 오늘 공부용 요약

1. HIVE-COTE 2.0은 heterogeneous meta-ensemble입니다.
2. STC, TDE, DrCIF, Arsenal 네 구성 요소를 결합합니다.
3. CAWPE는 구성 모델의 class probability를 성능 기반으로 가중 결합합니다.
4. HC2는 정확하지만 계산량이 크므로 성능 상한선 benchmark로 유용합니다.
5. 시계열/설비 진단에서는 모델보다 데이터 분할이 먼저입니다.
6. sliding window random split은 HC2 같은 강한 모델에서 더 위험합니다.
7. Part 2의 결론은 하나의 표현만 믿지 말고 여러 표현을 공정한 split에서 비교하자는 것입니다.

## 다음 논문 예고

다음은 Part 3의 시작입니다. **13편: Time Series Classification from Scratch with Deep Neural Networks**입니다.

Part 2가 끝났습니다. 이 파트 내용을 Git에 올릴 수 있도록 정리하려면 `Git 백업` 요청을 하면 됩니다.

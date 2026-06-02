# 9편: catch22: CAnonical Time-series CHaracteristics

## 결론부터

**이 논문의 핵심 결론은, 시계열 feature를 수천 개씩 무작정 뽑지 않아도 “중복이 적고 계산이 빠르며 해석 가능한 22개 대표 feature”만으로도 강력한 시계열 분류 baseline을 만들 수 있다는 것입니다.**

## 이 논문을 한 문장으로 요약하면

**catch22는 hctsa의 수천 개 시계열 feature 중에서 분류 성능, 중복 최소화, 계산 효율, 해석 가능성을 기준으로 고른 22개의 대표 시계열 특징 세트입니다.**

## 왜 이 논문이 필요했나?

8편 tsfresh는 “많이 뽑고 통계적으로 고른다”는 전략이었습니다. 강력하지만 feature가 너무 많으면 문제가 생깁니다.

| 문제 | 설명 |
|---|---|
| 계산 비용 | 센서와 window가 많으면 feature 계산 시간이 커짐 |
| 중복 feature | 서로 비슷한 feature가 많아짐 |
| overfitting | train set에서만 우연히 잘 맞는 feature가 생김 |
| 해석 어려움 | 수백~수천 개 feature 중 무엇이 중요한지 보기 어려움 |
| 배포 부담 | edge device나 실시간 모니터링에서 계산이 부담됨 |

catch22는 “많이 뽑는 것” 대신 “대표 feature를 적게, 빠르게, 해석 가능하게 뽑자”는 접근입니다.

## 핵심 개념 1. 작은 대표 feature 세트

| 관점 | tsfresh | catch22 |
|---|---|---|
| 철학 | 많이 뽑고 통계적으로 고른다 | 적게, 대표적으로, 빠르게 뽑는다 |
| feature 수 | 수백~수천 개 가능 | 기본 22개 |
| target label 사용 | feature selection에서 사용 | 기본 feature extraction은 label 사용 안 함 |
| 장점 | 다양한 후보 탐색 | 빠름, 간단함, 중복 적음 |
| 위험 | feature selection leakage | 표현력이 부족할 수 있음 |

## 핵심 개념 2. hctsa에서 출발

hctsa는 수천 개 feature를 계산하는 거대한 feature library입니다. catch22는 이 큰 후보군에서 계산 가능성, 성능, 중복성, 해석 가능성을 고려해 22개 feature를 고른 것입니다.

## 핵심 개념 3. 중복이 적은가가 중요하다

RMS, variance, standard deviation, absolute energy처럼 서로 비슷한 정보를 담는 feature가 많으면 feature 수는 많아도 실제 정보량은 적습니다. catch22는 correlation과 clustering을 이용해 중복이 적은 대표 feature를 고르는 철학을 갖습니다.

## 핵심 개념 4. z-scored time series의 특성을 본다

catch22 feature는 대체로 z-scored time series의 통계적 속성과 time-ordering에 집중합니다. 따라서 절대 크기 자체는 약해질 수 있습니다.

| 상황 | 장점 | 위험 |
|---|---|---|
| 센서 gain 차이 | 설비별 scale 차이 완화 | 절대 진동 크기 증가를 놓칠 수 있음 |
| 모터 크기 차이 | 서로 다른 설비 비교 쉬움 | 과전류 절대값 정보 손실 |
| 온도 offset 차이 | baseline 완화 | 절대 과열 기준 손실 |

설비 진단에서는 catch22만 쓰기보다 mean과 standard deviation을 추가하는 catch24나 RMS, max, crest factor, FFT band energy를 함께 쓰는 것이 좋습니다.

## 핵심 개념 5. feature 그룹

| feature 그룹 | 쉬운 의미 | 설비 해석 |
|---|---|---|
| Distribution shape | 값 분포의 모양 | 진동 값이 한쪽으로 치우쳤는가 |
| Extreme event timing | 큰 outlier가 언제 나타나는가 | 충격 peak가 앞/뒤에 몰리는가 |
| Autocorrelation | 과거 값과 현재 값의 관계 | 반복 진동, 압력 cycle, 온도 관성 |
| Frequency summary | 저주파/중심 주파수 특성 | 주기성, 회전 관련 패턴 |
| Forecast error | 단순 예측이 잘 안 되는 정도 | 불규칙성, 이상 변동 |
| Incremental differences | 변화량 패턴 | 급격한 상승/하강, 충격성 |
| Symbolic pattern | 값을 기호화했을 때의 패턴 | 단순 반복인지 복잡한지 |

## 핵심 개념 6. 예시 feature 직관

### ACF timescale

현재 값이 과거 값과 얼마나 오래 닮아 있는지 봅니다. 온도는 ACF가 오래 유지될 수 있고, 충격성 진동은 빠르게 변할 수 있습니다.

### Outlier timing

큰 양/음 outlier가 window 어디에 나타나는지 봅니다. 단, window 정렬 방식에 민감합니다.

### Low-frequency power / centroid frequency

에너지가 저주파 쪽에 몰려 있는지, 중심 주파수가 어디인지 봅니다. 베어링 결함 주파수를 직접 계산하는 것은 아니므로 envelope/FFT domain feature를 별도로 추가하는 것이 좋습니다.

### Forecast error

단순 rolling mean으로 예측하기 어려운 정도를 봅니다. 불규칙한 충격이 많으면 커질 수 있습니다.

## 작은 예시 데이터

| window | label | sequence |
|---|---|---|
| W1 | 정상 | `[0.10, 0.11, 0.09, 0.10, 0.12, 0.10]` |
| W2 | 정상 | `[0.08, 0.10, 0.11, 0.09, 0.10, 0.09]` |
| W3 | 고장 | `[0.10, 0.20, 1.50, -1.20, 0.30, 1.40]` |
| W4 | 고장 | `[0.20, 1.60, -1.30, 0.40, 1.50, -1.10]` |

| feature 성격 | 정상 | 고장 | 해석 |
|---|---|---|---|
| outlier timing | 거의 없음 | 큰 outlier 존재 | 충격성 |
| high fluctuation | 낮음 | 높음 | 급격한 변화 |
| forecast error | 낮음 | 높음 | 예측 어려움 |
| entropy | 낮음 | 높음 | 복잡도 증가 |

## 실무에서 어떻게 써야 하나?

1. tsfresh보다 먼저 빠른 baseline으로 돌려봅니다.
2. 설비 진단에서는 catch24와 domain feature를 같이 고려합니다.
3. multivariate sensor에는 센서별로 계산하고 조합합니다.
4. split은 여전히 모델보다 먼저입니다.
5. SHAP이나 class별 분포로 feature importance를 확인합니다.

| 위험한 방식 | 문제 |
|---|---|
| 긴 진동 run을 window로 자르고 random split | 인접 window 누수 |
| 전체 데이터 기준 z-normalization | test 분포 정보 누수 |
| test 성능 보고 catch24 여부 결정 | model selection leakage |
| 같은 설비/run이 train/test에 섞임 | fingerprint 학습 |

## 이상탐지와 RUL 연결

정상 window의 catch22 feature 분포를 학습하고, 새 window가 이 분포에서 멀어지면 anomaly score로 쓸 수 있습니다. RUL에서는 cycle별 catch22/catch24 feature를 만들어 regressor에 넣을 수 있습니다. RUL 평가는 MAE, RMSE, NASA scoring function, early/late penalty를 봐야 합니다.

## 한계

1. 22개 feature만으로 모든 시계열을 설명할 수는 없습니다.
2. 기본 catch22는 raw amplitude 정보를 약하게 다룹니다.
3. 센서 간 관계를 자동으로 충분히 반영하지 않습니다.
4. window 정렬에 민감한 feature가 있을 수 있습니다.
5. 데이터 누수를 막아주지 않습니다.
6. class imbalance 해결책은 아닙니다.

## 이전 논문들과 연결해서 이해하기

| 관점 | Shapelet | tsfresh | catch22 |
|---|---|---|---|
| 핵심 질문 | class를 가르는 부분 패턴은? | 많은 feature 중 중요한 것은? | 적고 빠른 대표 feature만으로 충분한가? |
| feature 수 | 선택된 subsequence | 수백~수천 개 가능 | 22개 |
| 장점 | 국소 패턴 발견 | 넓은 탐색 | 빠르고 중복 적음 |

## 오늘 공부용 요약

1. catch22는 22개의 대표 시계열 feature 세트입니다.
2. 성능, 중복 최소화, 계산 효율, 해석 가능성을 함께 고려했습니다.
3. 기본 catch22는 z-scored time series 특성에 집중하므로 amplitude 정보가 약해질 수 있습니다.
4. 설비 진단에서는 catch24나 RMS, FFT, envelope feature를 함께 쓰는 것이 좋습니다.
5. window random split과 전체 scaling은 여전히 위험합니다.

## 다음 논문 예고

다음은 **10편: The BOSS is concerned with time series classification in the presence of noise**입니다.

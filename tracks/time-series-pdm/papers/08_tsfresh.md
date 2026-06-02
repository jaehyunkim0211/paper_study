# 8편: Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh)

## 결론부터

**이 논문의 핵심 결론은, 시계열을 딥러닝에 바로 넣기 전에 평균, 분산, 자기상관, FFT, entropy, peak, energy 같은 특징을 대량으로 자동 추출하고, 통계적 검정으로 target과 관련 있는 feature만 골라내면 강력하고 해석 가능한 전통 ML 파이프라인을 만들 수 있다는 것입니다.**

## 이 논문을 한 문장으로 요약하면

**tsfresh는 시계열 feature engineering을 자동화하기 위해 여러 시계열 특성 추출 함수를 대량으로 적용한 뒤, classification/regression target과 통계적으로 관련 있는 feature를 가설검정 기반으로 선택하는 Python 패키지입니다.**

## 왜 이 논문이 필요했나?

설비 고장은 여러 방식으로 나타납니다.

| 고장 신호 | 필요한 feature 예시 |
|---|---|
| 베어링 충격 | kurtosis, crest factor, envelope peak |
| 회전체 불평형 | FFT 1X 성분, 진동 RMS |
| 모터 전기적 이상 | 전류 harmonic, sideband energy |
| 냉각 성능 저하 | 온도 trend slope |
| 압력 불안정 | peak count, variance, autocorrelation |
| RUL 열화 | 최근 feature의 증가율, 누적 손상 지표 |

feature engineering을 사람이 하나씩 하면 시간이 오래 걸립니다. tsfresh는 가능한 많은 후보 feature를 자동으로 만들고, target과 관련 있는 것만 통계적으로 걸러내자는 접근입니다.

## 핵심 개념 1. Feature extraction

시계열 하나를 여러 개의 숫자 feature로 바꿉니다.

```text
vibration = [0.1, 0.2, 0.1, 1.5, 3.0, 1.4, 0.2, 0.1]
```

| feature | 값 | 의미 |
|---|---:|---|
| mean | 0.825 | 평균 진동 수준 |
| max | 3.0 | 최대 충격 크기 |
| std | 1.05 | 변동성 |
| energy | 12.72 | 전체 에너지 |
| peak_count | 1 | 큰 peak 수 |
| kurtosis | 높음 | 충격성 |

이제 시계열 분류 문제를 tabular ML 문제처럼 바꿀 수 있습니다.

## 핵심 개념 2. 많이 뽑고 통계적으로 고른다

| 단계 | 설비 예시 |
|---|---|
| raw data | 진동, 전류, 온도, 압력 센서 |
| windowing | 1초 또는 10초 단위 window 생성 |
| feature extraction | RMS, FFT, entropy, autocorrelation 등 대량 추출 |
| feature selection | 정상/고장 label과 관련 있는 feature만 선택 |
| model training | RandomForest, XGBoost, SVM |
| evaluation | precision, recall, F1, confusion matrix |

## 핵심 개념 3. FRESH algorithm

FRESH는 `FeatuRe Extraction based on Scalable Hypothesis tests`입니다.

| 단계 | 하는 일 |
|---|---|
| Feature extraction | raw time series를 feature matrix로 변환 |
| Feature significance testing | 각 feature가 target과 관련 있는지 p-value 계산 |
| Multiple test procedure | feature가 많을 때 false discovery를 통제 |

## 핵심 개념 4. p-value

p-value는 “현재 데이터에서 이 feature가 target과 통계적으로 관련 있어 보이는가?”를 판단하는 데 쓰입니다.

| p-value | 해석 |
|---:|---|
| 작음 | 이 feature가 label과 관련 있어 보임 |
| 큼 | 관련 있다고 보기 어려움 |

p-value가 작다고 해서 고장의 원인이라는 뜻은 아닙니다. 운전 조건 차이, 센서 위치 차이, 부하 차이를 반영한 것일 수도 있습니다.

## 핵심 개념 5. Multiple testing

feature를 794개처럼 많이 뽑으면 일부 feature는 우연히 target과 관련 있어 보일 수 있습니다. 그래서 multiple testing correction이 필요합니다.

동전 1개를 10번 던져 앞면이 9번 나오면 이상하지만, 동전 1,000개를 각각 10번 던지면 우연히 앞면이 9번 나오는 동전이 생길 수 있습니다.

## 핵심 개념 6. Feature 종류

| feature 계열 | 예시 | 설비 해석 |
|---|---|---|
| 기본 통계 | mean, min, max, variance | 센서 수준과 변동성 |
| 에너지 | abs_energy, RMS 유사 feature | 진동/전류 에너지 증가 |
| 변화량 | absolute_sum_of_changes | 값이 얼마나 급격히 변하는가 |
| peak 계열 | number of peaks, max | 충격성 이벤트 |
| 자기상관 | autocorrelation | 주기성, 반복 패턴 |
| trend | linear trend | 열화 증가/감소 |
| entropy | approximate entropy | 불규칙성, 복잡도 |
| 주파수 | FFT aggregated | 주파수 분포 요약 |
| wavelet | CWT coefficients | 시간-주파수 국소 변화 |

## 핵심 개념 7. Long format

| column | 의미 | 예시 |
|---|---|---|
| id | 하나의 sample/window/entity | `motor_A_run3_window_005` |
| time | window 내부 시간 순서 | 0, 1, 2, 3 |
| kind | 센서 종류 | vibration, current, temperature |
| value | 센서값 | 0.13, 10.4, 72.1 |

결과는 각 id가 한 row인 feature matrix가 됩니다.

## 핵심 개념 8. tsfresh는 모델이 아니라 feature engineering 도구다

```text
raw sensor data
→ train/test split
→ windowing
→ tsfresh feature extraction
→ train-only feature selection
→ scaler/imputer
→ RandomForest/XGBoost/LogisticRegression/SVM
→ evaluation
```

## 핵심 개념 9. Feature selection은 반드시 train 안에서만 한다

`select_features`는 label을 보고 feature를 고릅니다. 전체 데이터에서 feature selection을 하면 test label을 사용한 leakage가 생깁니다.

안전한 방식:

| 단계 | 안전한 방식 |
|---|---|
| 1 | 설비 unit/run/time 기준 split |
| 2 | train set에서만 feature extraction과 feature selection |
| 3 | 선택된 feature 목록을 validation/test에 그대로 적용 |
| 4 | scaler/imputer도 train에서 fit |
| 5 | test는 마지막 한 번만 평가 |

## 작은 예시 데이터

| id | mean | max | std | abs_energy | label |
|---|---:|---:|---:|---:|---|
| W1 | 0.105 | 0.12 | 0.013 | 0.0446 | 정상 |
| W2 | 0.120 | 0.14 | 0.018 | 0.0586 | 정상 |
| W3 | 1.500 | 3.00 | 1.029 | 13.22 | 고장 |
| W4 | 1.475 | 2.80 | 0.920 | 11.73 | 고장 |

`max`, `std`, `abs_energy`는 정상과 고장을 잘 나눌 수 있습니다. 하지만 실제 현장에서는 부하 조건 차이를 학습한 것인지 반드시 확인해야 합니다.

## 실무에서 어떻게 써야 하나?

1. 딥러닝 전에 강력한 feature baseline으로 씁니다.
2. feature를 물리 의미로 그룹화합니다.
3. feature selection은 train-only로 수행합니다.
4. windowing과 id 정의를 신중히 합니다.
5. feature importance를 현장 검증과 연결합니다.

## 한계

1. feature가 많아서 overfitting 위험이 큽니다.
2. 자동 feature가 물리적으로 의미 있다는 보장은 없습니다.
3. 고주파 raw vibration에는 계산 비용이 클 수 있습니다.
4. streaming에 바로 쓰기 어렵습니다.
5. 센서 간 동시 관계를 자동으로 충분히 만들지는 않습니다.
6. 불균형 데이터에서는 p-value와 metric을 더 조심해야 합니다.

## 이전 논문들과 연결해서 이해하기

| 관점 | UCR Archive | Shapelet | tsfresh |
|---|---|---|---|
| 핵심 질문 | TSC 모델을 어떻게 공정하게 비교할 것인가? | class를 가르는 짧은 부분 패턴은? | 유용한 feature를 자동으로 뽑고 고를 수 있는가? |
| 출력 | benchmark 성능 | shapelet + threshold | feature matrix |
| leakage 위험 | split misuse | test로 shapelet 선택 | test label로 feature selection |

## 오늘 공부용 요약

1. tsfresh는 시계열 feature extraction과 selection을 자동화합니다.
2. FRESH는 feature extraction, significance testing, multiple testing correction으로 구성됩니다.
3. p-value가 작다는 것은 target과 관련 있어 보인다는 뜻이지 원인이라는 뜻은 아닙니다.
4. feature selection은 label을 사용하므로 반드시 train set에서만 해야 합니다.
5. sliding window random split은 같은 설비, 같은 run, 인접 window가 섞일 수 있어 위험합니다.
6. accuracy만 보지 말고 precision, recall, F1, confusion matrix, false alarm, missed detection을 봐야 합니다.

## 다음 논문 예고

다음은 **9편: catch22: CAnonical Time-series CHaracteristics**입니다.

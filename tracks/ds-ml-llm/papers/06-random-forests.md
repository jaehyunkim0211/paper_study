# Random Forests — Leo Breiman

## 결론부터

이 논문의 결론은 이겁니다.

**하나의 의사결정나무는 불안정하지만, 서로 조금씩 다른 의사결정나무를 많이 만들고 그 예측을 평균내면 훨씬 강하고 안정적인 모델이 된다.**  
이 방법이 바로 **Random Forest**입니다.

조금 더 정확히 말하면:

> Random Forest는 **bootstrap sampling**으로 서로 다른 학습 데이터를 만들고, 각 나무의 split마다 **일부 변수만 무작위로 후보로 선택**해서 여러 개의 decision tree를 만든 뒤, 분류에서는 다수결, 회귀에서는 평균으로 최종 예측을 만드는 앙상블 모델입니다.

---

## 이 논문을 한 문장으로 요약하면

**좋은 Random Forest는 “개별 나무는 꽤 잘 맞추되, 서로는 최대한 다르게 틀리게 만드는 모델”이다.**

Random Forest가 잘 작동하는 이유는 단순히 “나무를 많이 만들기 때문”이 아닙니다.

중요한 조건은 두 가지입니다.

| 조건 | 의미 |
|---|---|
| **각 tree가 어느 정도 강해야 한다** | 완전히 엉터리 나무를 많이 모아도 좋은 모델이 되지 않음 |
| **tree들끼리 너무 비슷하면 안 된다** | 다 똑같이 틀리면 평균내도 소용없음 |

즉, Random Forest의 핵심 철학은 이것입니다.

> **강하지만 서로 다른 모델들을 많이 만든다.**

---

## 왜 하나의 Decision Tree만으로는 부족한가?

Decision tree는 이런 식으로 판단합니다.

```text
고객의 가입기간이 3개월 미만인가?
 ├─ 예: 월요금이 80,000원 이상인가?
 │    ├─ 예: 이탈 가능성 높음
 │    └─ 아니오: 이탈 가능성 중간
 └─ 아니오: 최근 불만 접수가 있었는가?
      ├─ 예: 이탈 가능성 중간
      └─ 아니오: 이탈 가능성 낮음
```

장점은 분명합니다.

| 장점 | 설명 |
|---|---|
| 해석이 쉽다 | if-then 규칙처럼 볼 수 있음 |
| 비선형 관계를 잡을 수 있다 | 선형회귀보다 유연함 |
| 변수 스케일링이 덜 중요하다 | 표준화하지 않아도 잘 작동하는 경우가 많음 |
| 범주형/수치형 변수 모두 다룰 수 있다 | 전처리 부담이 비교적 낮음 |

하지만 큰 단점이 있습니다.

**decision tree는 데이터가 조금만 바뀌어도 구조가 크게 바뀔 수 있습니다.**

예를 들어 훈련 데이터에서 샘플 몇 개만 바뀌어도 첫 번째 split이 달라지고, 그러면 나무 전체 구조가 달라질 수 있습니다.

이걸 통계적으로 말하면:

> decision tree는 **variance가 큰 모델**입니다.

즉, 훈련 데이터에 너무 민감합니다.

---

## Random Forest의 핵심 아이디어

Random Forest는 이 문제를 이렇게 해결합니다.

> 불안정한 나무 하나에 의존하지 말고,  
> 불안정한 나무를 많이 만들고 평균내자.

분류 문제에서는 다수결을 합니다.

| Tree | 예측 |
|---|---|
| Tree 1 | 이탈 |
| Tree 2 | 유지 |
| Tree 3 | 이탈 |
| Tree 4 | 이탈 |
| Tree 5 | 유지 |

```text
이탈 3표, 유지 2표 → 최종 예측: 이탈
```

회귀 문제에서는 평균을 냅니다.

| Tree | 예측 집값 |
|---|---:|
| Tree 1 | 5.1억 |
| Tree 2 | 5.4억 |
| Tree 3 | 5.0억 |
| Tree 4 | 5.3억 |
| Tree 5 | 5.2억 |

```text
최종 예측 = 평균 = 5.2억
```

이렇게 하면 개별 나무의 흔들림이 줄어듭니다.

---

## Random Forest가 “random”인 이유

Random Forest에는 중요한 무작위성이 두 가지 있습니다.

### 1. 데이터 샘플을 무작위로 뽑는다: Bootstrap sampling

각 나무는 전체 데이터를 그대로 쓰지 않습니다.

전체 데이터에서 **복원추출**로 bootstrap sample을 만듭니다.

예를 들어 원래 데이터가 이렇다고 해봅시다.

```text
A, B, C, D, E
```

한 나무를 만들 때는 복원추출로 이런 샘플을 만들 수 있습니다.

```text
A, A, C, E, E
```

다른 나무는 이렇게 받을 수 있습니다.

```text
B, C, C, D, E
```

즉, 각 나무가 보는 데이터가 조금씩 다릅니다.

이 방식은 **bagging**, 즉 bootstrap aggregating의 핵심입니다.

### 2. split할 때 변수 후보를 무작위로 고른다

Random Forest는 단순한 bagging tree와 다릅니다.

Bagging은 여러 개의 나무를 bootstrap sample로 만들고 평균냅니다. Random Forest는 여기에 한 가지 무작위성을 더 넣습니다.

> 각 node에서 split을 고를 때 전체 변수 중 일부 변수만 후보로 본다.

예를 들어 변수가 10개 있다고 해봅시다.

```text
나이, 소득, 지역, 가입기간, 월요금, 사용량, 불만횟수, 결제방식, 할인여부, 상담횟수
```

일반 decision tree라면 모든 변수를 보고 가장 좋은 split을 고릅니다.

하지만 Random Forest는 각 node에서 무작위로 일부 변수만 봅니다.

```text
이번 node에서는: 가입기간, 월요금, 상담횟수만 후보
다음 node에서는: 나이, 사용량, 할인여부만 후보
```

강력한 변수가 하나 있으면, 모든 나무가 그 변수로 비슷한 split을 만들 수 있습니다. 그러면 나무들이 서로 너무 비슷해집니다.

Random Forest는 일부 변수만 보게 해서 나무들이 서로 다른 구조를 갖도록 만듭니다.

---

## Random Forest 알고리즘을 단계별로 보면

분류 문제 기준으로 Random Forest는 대략 이렇게 작동합니다.

```text
1. 만들 tree 개수 B를 정한다. 예: 500개

2. 각 tree마다:
   2-1. 원본 train data에서 bootstrap sample을 뽑는다.
   2-2. decision tree를 만든다.
   2-3. 각 node에서 전체 변수 중 m개만 무작위로 고른다.
   2-4. 그 m개 변수 중 가장 좋은 split을 선택한다.
   2-5. 나무를 충분히 깊게 키운다.

3. 새 데이터가 들어오면:
   3-1. 모든 tree가 각각 예측한다.
   3-2. 분류는 majority vote를 한다.
   3-3. 회귀는 평균을 낸다.
```

논문의 핵심은 2-3번입니다.

단순히 나무를 많이 만드는 것이 아니라, **나무들 사이의 상관을 낮추기 위해 split 후보 변수까지 무작위화**합니다.

---

## 핵심 개념 1: Strength

**Strength는 개별 tree 하나하나가 얼마나 잘 맞추는지를 의미합니다.**

Random Forest가 잘 작동하려면 각 나무가 완전히 약하면 안 됩니다.

예를 들어 이진분류에서 각 나무의 정확도가 50% 근처라면, 사실상 동전 던지기와 비슷합니다.

이런 나무를 1,000개 모아도 좋은 모델이 되기 어렵습니다.

| 개별 tree 수준 | forest 결과 |
|---|---|
| 각 tree가 너무 약함 | 많이 모아도 성능이 낮음 |
| 각 tree가 어느 정도 강함 | 평균내면 강한 모델이 됨 |

즉, Random Forest는 “아무 모델이나 많이 모으면 된다”가 아닙니다.

**각 tree가 기본적으로 의미 있는 패턴을 잡아야 합니다.**

---

## 핵심 개념 2: Correlation

**Correlation은 tree들이 서로 얼마나 비슷하게 예측하거나 비슷하게 틀리는지를 의미합니다.**

tree들이 서로 너무 비슷하면 forest의 장점이 줄어듭니다.

예를 들어 시험 문제를 푸는 학생 100명이 있다고 합시다.

그런데 100명이 모두 같은 답안지를 보고 베꼈다면, 100명의 다수결은 사실상 한 명의 답과 같습니다.

반대로 각자 다른 방식으로 공부했고, 어느 정도 실력이 있다면 다수결이 훨씬 안정적일 수 있습니다.

| 상황 | 결과 |
|---|---|
| tree들이 서로 비슷함 | 평균내도 성능 개선이 제한적 |
| tree들이 서로 다름 | 평균내면 오류가 상쇄됨 |

---

## Random Forest의 핵심 균형

Random Forest의 핵심은 이 균형입니다.

```text
개별 tree는 강하게 만들고,
tree들끼리는 덜 비슷하게 만든다.
```

이를 다시 말하면:

| 목표 | 방법 |
|---|---|
| tree를 강하게 만들기 | 충분히 깊은 decision tree 사용 |
| tree들을 다르게 만들기 | bootstrap sampling, random feature selection |
| 최종 예측을 안정화하기 | 다수결 또는 평균 |

이게 Random Forest가 잘 되는 이유입니다.

---

## 쉬운 예시: 고객 이탈 예측

고객 이탈 예측 문제를 생각해봅시다.

변수는 다음과 같습니다.

| 변수 |
|---|
| 가입기간 |
| 월요금 |
| 최근 사용량 |
| 고객센터 문의 횟수 |
| 결제 실패 여부 |
| 할인 여부 |
| 요금제 종류 |
| 최근 로그인 횟수 |

하나의 decision tree는 이렇게 학습할 수 있습니다.

```text
가입기간 < 2개월?
 ├─ 예: 최근 로그인 횟수 < 3회?
 │    ├─ 예: 이탈
 │    └─ 아니오: 유지
 └─ 아니오: 고객센터 문의 횟수 > 2회?
      ├─ 예: 이탈
      └─ 아니오: 유지
```

다른 bootstrap sample을 받은 tree는 이렇게 학습할 수 있습니다.

```text
결제 실패 여부 = 예?
 ├─ 예: 이탈
 └─ 아니오: 월요금 > 80,000원?
      ├─ 예: 이탈
      └─ 아니오: 유지
```

또 다른 tree는 이런 구조가 될 수 있습니다.

```text
최근 사용량 < 1GB?
 ├─ 예: 이탈
 └─ 아니오: 할인 여부 = 예?
      ├─ 예: 유지
      └─ 아니오: 중간 위험
```

이렇게 각 tree가 조금씩 다른 관점으로 고객을 봅니다.

최종적으로 수백 개 tree의 투표를 모으면, 단일 tree보다 훨씬 안정적인 예측이 됩니다.

---

## 핵심 개념 3: Out-of-Bag Error, OOB Error

**OOB error는 Random Forest가 내부적으로 만들 수 있는 검증 성능 추정치입니다.**

Random Forest에서는 각 tree를 만들 때 bootstrap sample을 씁니다.

복원추출을 하기 때문에 어떤 샘플은 특정 tree의 학습에 포함되지 않습니다.

예를 들어 원래 데이터가:

```text
A, B, C, D, E
```

어떤 tree의 bootstrap sample이:

```text
A, A, C, E, E
```

라면, 이 tree는 B와 D를 학습에 쓰지 않았습니다.

이때 B와 D는 이 tree 입장에서 **out-of-bag sample**입니다.

OOB error는 이런 느낌입니다.

```text
각 tree는 자기가 못 본 샘플에 대해서만 시험을 본다.
그리고 모든 tree의 시험 결과를 모아 전체 성능을 추정한다.
```

장점은 따로 validation set을 크게 떼어내지 않아도 된다는 것입니다.

| 장점 | 설명 |
|---|---|
| 데이터 효율적 | 작은 데이터에서도 train data를 많이 활용 가능 |
| 빠른 성능 추정 | 별도 cross-validation 없이 대략적인 일반화 성능 확인 |
| tree 수 선택에 도움 | OOB error가 안정되는 지점을 볼 수 있음 |

하지만 OOB error가 항상 최종 test set을 완전히 대체하는 것은 아닙니다. 실무에서는 가능하면 여전히 독립 test set을 두고 최종 성능을 확인하는 것이 좋습니다.

---

## 핵심 개념 4: Variable Importance

**Random Forest는 어떤 변수가 예측에 중요했는지 측정하는 방법도 제공합니다.**

대표적으로 두 가지가 있습니다.

| 방법 | 설명 |
|---|---|
| **Impurity-based importance** | split할 때 불순도를 얼마나 줄였는지 누적 |
| **Permutation importance** | 특정 변수를 섞었을 때 성능이 얼마나 떨어지는지 측정 |

### Impurity-based importance

Decision tree는 split을 할 때 Gini impurity, entropy, MSE 같은 기준을 줄이는 방향으로 나눕니다.

어떤 변수가 여러 tree에서 좋은 split을 많이 만들었다면, 그 변수의 중요도가 높게 계산됩니다.

### Permutation importance

Permutation importance는 더 직관적입니다.

절차는 이렇습니다.

```text
1. 학습된 Random Forest의 기본 성능을 측정한다.
2. 어떤 변수 하나의 값을 무작위로 섞는다.
3. 다시 성능을 측정한다.
4. 성능이 많이 떨어지면 그 변수는 중요하다고 본다.
```

예를 들어 `가입기간`을 섞었더니 AUC가 0.87에서 0.76으로 크게 떨어졌다면, 가입기간은 중요한 변수일 가능성이 큽니다.

---

## Variable Importance를 해석할 때 주의할 점

중요도는 유용하지만 조심해야 합니다.

| 주의점 | 설명 |
|---|---|
| 중요도는 인과관계가 아니다 | 중요하다고 해서 원인이라는 뜻은 아님 |
| 상관된 변수들이 있으면 중요도가 나뉠 수 있다 | 비슷한 변수끼리 importance를 나눠 가질 수 있음 |
| impurity importance는 편향될 수 있다 | 연속형 변수나 cardinality가 높은 변수에 유리할 수 있음 |
| 중요도는 데이터와 모델에 종속적이다 | 다른 데이터, 다른 split에서는 달라질 수 있음 |

---

## Random Forest와 Bagging의 차이

| 방법 | 데이터 샘플링 | split 후보 변수 | 핵심 효과 |
|---|---|---|---|
| **Decision Tree** | 전체 데이터 사용 | 전체 변수 사용 | 단순하지만 불안정 |
| **Bagging Trees** | bootstrap sample 사용 | 전체 변수 사용 | variance 감소 |
| **Random Forest** | bootstrap sample 사용 | 일부 변수만 무작위 사용 | variance 감소 + tree 간 correlation 감소 |

즉, Random Forest는 bagging tree에 **feature randomness**를 추가한 방법이라고 볼 수 있습니다.

---

## Random Forest와 Boosting의 차이

| 구분 | Random Forest | Boosting |
|---|---|---|
| 학습 방식 | 여러 tree를 독립적으로 학습 | 앞선 tree의 오류를 다음 tree가 보완 |
| 병렬화 | 비교적 쉬움 | 순차 학습이라 제한적 |
| 주된 목적 | variance 감소 | bias와 variance를 함께 줄임 |
| 튜닝 난이도 | 비교적 낮음 | 보통 더 민감함 |
| 과적합 위험 | 상대적으로 낮은 편 | 튜닝에 따라 커질 수 있음 |
| 대표 모델 | Random Forest | Gradient Boosting, XGBoost, LightGBM |

쉽게 말하면:

> Random Forest는 여러 나무를 독립적으로 많이 만들고 평균낸다.  
> Boosting은 나무를 순서대로 만들면서 이전 실수를 고친다.

---

## Random Forest의 장점

1. **성능이 안정적이다** — 하나의 decision tree보다 과적합에 덜 취약합니다.
2. **튜닝을 많이 안 해도 잘 된다** — 기본 설정으로도 꽤 괜찮은 성능을 내는 경우가 많습니다.
3. **비선형 관계와 변수 상호작용을 잘 잡는다** — 사람이 interaction을 직접 넣지 않아도 어느 정도 학습합니다.
4. **변수 스케일링에 덜 민감하다** — 거리 기반 모델이 아니기 때문에 표준화 부담이 상대적으로 낮습니다.
5. **OOB error를 활용할 수 있다** — bootstrap sampling 덕분에 내부 검증 성능을 추정할 수 있습니다.

---

## Random Forest의 한계

1. **하나의 tree보다 해석이 어렵다** — 수백 개 tree의 조합이므로 단일 tree처럼 직관적이지 않습니다.
2. **매우 희소한 고차원 데이터에서는 항상 최선이 아니다** — 텍스트 데이터처럼 feature가 수만 개이고 대부분 0인 경우 linear model이나 neural model이 더 적합할 수 있습니다.
3. **외삽 extrapolation에 약하다** — 훈련 데이터 범위를 벗어난 값을 예측하는 데 약할 수 있습니다.
4. **feature importance 해석이 조심스럽다** — 중요도는 인과관계가 아니고 상관된 변수에 의해 왜곡될 수 있습니다.

---

## 실무에서 Random Forest를 어떻게 써야 하나?

### 1. 강력한 baseline으로 쓴다

Random Forest는 tabular data에서 첫 모델로 매우 좋습니다.

```text
Logistic Regression → Random Forest → XGBoost/LightGBM
```

| 모델 | 역할 |
|---|---|
| Logistic Regression | 단순하고 해석 가능한 baseline |
| Random Forest | 비선형 baseline |
| XGBoost / LightGBM | 고성능 boosting baseline |

Random Forest가 logistic regression보다 훨씬 좋다면, 데이터에 비선형성이나 interaction이 많다는 신호일 수 있습니다.

### 2. `n_estimators`는 충분히 크게 잡는다

tree 수가 적으면 forest가 불안정할 수 있습니다.

보통은 100, 300, 500, 1000개 정도를 실험합니다.

### 3. `max_features`가 중요하다

`max_features`는 각 split에서 몇 개의 변수를 후보로 볼지 결정합니다.

| `max_features` | 효과 |
|---|---|
| 큼 | 개별 tree 강함, tree 간 correlation 증가 가능 |
| 작음 | tree 다양성 증가, 개별 tree 약화 가능 |

즉, `max_features`는 strength와 correlation 사이의 균형을 조절하는 핵심 하이퍼파라미터입니다.

### 4. 과적합이 보이면 tree를 덜 깊게 만든다

| 파라미터 | 조정 방향 |
|---|---|
| `max_depth` | 줄이면 tree가 단순해짐 |
| `min_samples_leaf` | 키우면 leaf가 더 안정적 |
| `min_samples_split` | 키우면 split이 덜 발생 |
| `max_features` | 줄이면 tree 간 다양성 증가 |

### 5. 시간 순서 데이터에서는 split을 조심한다

고객 이탈, 매출 예측, 수요 예측처럼 시간 순서가 있는 문제에서는 random split을 하면 leakage가 생길 수 있습니다.

```text
train: 2022~2024년
validation: 2025년 상반기
test: 2025년 하반기
```

---

## p-value와 Random Forest의 관계

Random Forest는 보통 회귀분석처럼 각 변수에 p-value를 주지 않습니다.

즉, Random Forest에서 이런 식으로 말하지 않습니다.

```text
가입기간의 p-value는 0.03이므로 유의하다.
```

대신 이런 식으로 봅니다.

```text
가입기간을 permutation했더니 OOB AUC가 크게 떨어졌다.
따라서 가입기간은 예측에 중요한 변수로 보인다.
```

차이는 큽니다.

| 회귀분석 p-value | Random Forest importance |
|---|---|
| 계수가 0이라는 귀무가설 검정 | 변수가 예측 성능에 기여하는 정도 |
| 통계적 유의성 중심 | 예측 기여도 중심 |
| 해석/추론에 적합 | 예측 모델 해석에 보조적으로 사용 |
| 선형모형 가정에 의존 | tree ensemble 구조에 의존 |

따라서 Random Forest의 feature importance를 보고 “이 변수는 인과적으로 유의하다”고 말하면 안 됩니다.

---

## 이 논문이 데이터사이언스 입문에서 중요한 이유

Random Forest는 데이터사이언스 입문자에게 매우 중요한 모델입니다.

1. **앙상블의 핵심 철학을 배운다** — 여러 모델을 어떻게 조합하면 하나의 모델보다 더 좋아지는가를 이해할 수 있습니다.
2. **Bias-Variance Tradeoff를 체감할 수 있다** — 깊은 decision tree 하나는 낮은 bias, 높은 variance이고, Random Forest는 variance를 줄입니다.
3. **실무 tabular data에서 아직도 강력하다** — 고객 이탈, 신용평가, 사기 탐지, 매출 예측 등에서 좋은 baseline입니다.

---

## 이전 논문들과 연결해서 이해하기

| 논문 | Random Forest와의 연결 |
|---|---|
| **Tidy Data** | Random Forest에 넣을 수 있도록 관측치와 변수를 정리해야 함 |
| **Two Cultures** | Random Forest는 algorithmic modeling culture의 대표 모델 |
| **Model Evaluation** | OOB error, cross-validation, test set 평가가 중요 |
| **Datasheets** | 학습 데이터의 출처와 한계를 알아야 모델을 믿을 수 있음 |
| **Model Cards** | Random Forest 모델도 사용 범위, 집단별 성능, 한계를 문서화해야 함 |

특히 Breiman의 **Two Cultures**와 이번 **Random Forests**는 아주 강하게 연결됩니다.

---

## 입문자가 꼭 기억해야 할 문장

Random Forest를 공부할 때는 이 문장을 기억하면 됩니다.

**Random Forest는 여러 개의 깊은 decision tree를 만들되, 각 tree가 보는 데이터와 변수를 무작위로 다르게 해서 서로 덜 비슷한 tree들을 만들고, 이들의 예측을 평균내어 안정적인 예측을 만드는 모델이다.**

더 짧게 말하면:

> **강한 나무를 많이 만들고, 서로 다르게 만들고, 평균낸다.**

---

## 오늘 공부용 요약

**논문명:** Random Forests  
**저자:** Leo Breiman  
**핵심 결론:** 하나의 decision tree는 불안정하지만, bootstrap sampling과 random feature selection으로 서로 다른 tree를 많이 만들고 그 예측을 결합하면 강력하고 안정적인 모델을 만들 수 있다.  
**왜 중요함:** Random Forest는 현대 앙상블 학습의 대표 모델이고, tabular data에서 강력한 baseline이며, “개별 모델의 strength와 모델 간 correlation의 균형”이라는 앙상블의 핵심 원리를 보여준다.  
**입문자가 배울 점:** 좋은 앙상블은 단순히 모델을 많이 모으는 것이 아니라, 각 모델이 충분히 강하면서도 서로 다르게 틀리도록 설계해야 한다.  
**가장 중요한 문장:** Random Forest의 성능은 개별 tree의 강함과 tree들 사이의 낮은 상관관계 사이의 균형에서 나온다.


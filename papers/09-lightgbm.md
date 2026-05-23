# LightGBM: A Highly Efficient Gradient Boosting Decision Tree — Ke et al.

## 결론부터

이 논문의 결론은 이겁니다.

**LightGBM은 GBDT, 즉 Gradient Boosting Decision Tree를 훨씬 빠르고 메모리 효율적으로 학습시키기 위해 만든 모델이다.**  
핵심 아이디어는 두 가지입니다.

| 핵심 기법 | 쉽게 말하면 |
|---|---|
| **GOSS** | 아직 잘 못 맞추는 중요한 데이터는 많이 남기고, 이미 잘 맞추는 데이터는 일부만 샘플링한다 |
| **EFB** | 동시에 0이 아닌 경우가 거의 없는 sparse feature들을 하나로 묶어 feature 수를 줄인다 |

LightGBM 논문은 기존 GBDT가 큰 데이터와 많은 feature를 다룰 때 split 후보를 찾는 비용이 커진다는 문제를 지적하고, 이를 해결하기 위해 **Gradient-based One-Side Sampling, GOSS**와 **Exclusive Feature Bundling, EFB**를 제안합니다.

---

## 이 논문을 한 문장으로 요약하면

**LightGBM은 “모든 데이터와 모든 feature를 다 보지 않아도, 중요한 정보는 유지하면서 GBDT를 빠르게 학습할 수 있다”는 논문입니다.**

이전 논문인 XGBoost를 기억하면 흐름이 쉽습니다.

```text
Gradient Boosting
→ XGBoost: 정규화와 시스템 최적화로 강력한 GBDT 구현
→ LightGBM: 더 빠르고 가볍게 학습하기 위한 GBDT 구현
```

XGBoost가 “정규화된 objective와 scalable system”에 초점이 있었다면, LightGBM은 특히 다음 질문에 답합니다.

> 데이터가 너무 많을 때, 모든 row를 다 봐야 할까?  
> feature가 너무 많을 때, 모든 feature를 따로따로 봐야 할까?

LightGBM의 답은:

> 아니다.  
> 중요한 row를 더 많이 보고, 같이 나타나지 않는 feature는 묶자.

입니다.

---

## 왜 LightGBM이 필요했나?

GBDT는 성능이 좋지만, 큰 데이터에서는 느려집니다.

GBDT가 tree를 만들 때 가장 비싼 작업은 보통 **best split point**를 찾는 일입니다.

예를 들어 feature가 1,000개 있고 데이터가 1,000만 행이라면, tree의 node를 나눌 때마다 이런 질문을 반복해야 합니다.

```text
feature 1은 어디서 자를까?
feature 2는 어디서 자를까?
...
feature 1000은 어디서 자를까?
```

기존 GBDT 구현은 각 feature마다 많은 데이터 instance를 훑으면서 split gain을 계산해야 했고, 그래서 계산량이 데이터 수와 feature 수에 모두 비례했습니다.

이 병목을 해결하려면 데이터 instance 수 또는 feature 수를 줄여야 합니다.

---

## 핵심 개념 1: Histogram-based Algorithm

**LightGBM은 연속형 feature 값을 그대로 전부 비교하지 않고, bin이라는 구간으로 묶어서 split을 빠르게 찾습니다.**

예를 들어 나이라는 feature가 있다고 합시다.

원래 값은 다양합니다.

```text
23, 24, 25, 31, 42, 53, 64, ...
```

이걸 LightGBM은 여러 bin으로 나눕니다.

```text
20대 bin
30대 bin
40대 bin
50대 bin
60대 bin
```

그다음 split 후보를 모든 실제 값에서 찾는 대신, bin 기준으로 찾습니다.

```text
20대 이하 vs 초과
30대 이하 vs 초과
40대 이하 vs 초과
...
```

이렇게 하면 계산량과 메모리 사용량이 줄어듭니다.

---

## 핵심 개념 2: GOSS — Gradient-based One-Side Sampling

**GOSS는 gradient가 큰 데이터는 모두 남기고, gradient가 작은 데이터는 일부만 샘플링하는 방법입니다.**

여기서 gradient가 크다는 말은 쉽게 말해:

> 현재 모델이 아직 잘 못 맞추고 있는 데이터다.

라는 뜻입니다.

반대로 gradient가 작다는 말은:

> 현재 모델이 이미 꽤 잘 맞추고 있는 데이터다.

라는 뜻입니다.

GBDT에서 다음 tree는 현재 모델이 부족한 부분을 고쳐야 합니다. 그러므로 현재 모델이 크게 틀리는 데이터는 split gain 계산에서 더 중요한 역할을 합니다.

---

## GOSS를 아주 쉽게 예시로 보면

고객 이탈 예측 모델을 학습한다고 해봅시다.

현재 모델이 10명의 고객을 예측했습니다.

| 고객 | 실제 이탈 여부 | 현재 예측 | 모델 상태 |
|---|---:|---:|---|
| A | 이탈 | 이탈 확률 0.90 | 잘 맞춤 |
| B | 유지 | 이탈 확률 0.05 | 잘 맞춤 |
| C | 이탈 | 이탈 확률 0.20 | 크게 틀림 |
| D | 유지 | 이탈 확률 0.80 | 크게 틀림 |
| E | 이탈 | 이탈 확률 0.55 | 애매함 |

GBDT에서 다음 tree는 현재 모델이 부족한 부분을 고쳐야 합니다.

그렇다면 모든 데이터를 똑같이 중요하게 볼 필요가 있을까요?

LightGBM은 이렇게 봅니다.

```text
크게 틀린 데이터 C, D는 중요하다.
이미 잘 맞춘 A, B는 상대적으로 덜 중요하다.
```

그래서 GOSS는:

```text
gradient 큰 데이터: 모두 사용
gradient 작은 데이터: 일부만 랜덤 샘플링
```

합니다.

다만 작은 gradient 데이터를 버리기만 하면 전체 데이터 분포가 왜곡될 수 있습니다. 그래서 GOSS는 작은 gradient 데이터 중 샘플링된 데이터에 가중치를 조정해, 정보량 추정의 편향을 줄입니다.

---

## GOSS와 일반 랜덤 샘플링의 차이

일반 random sampling은 데이터가 중요한지 아닌지 보지 않고 뽑습니다.

```text
전체 데이터 중 30%를 랜덤으로 뽑자.
```

GOSS는 다릅니다.

```text
중요한 데이터, 즉 gradient가 큰 데이터는 반드시 남기자.
덜 중요한 데이터, 즉 gradient가 작은 데이터는 일부만 뽑자.
```

비교하면 이렇습니다.

| 방식 | 샘플링 기준 | 장점 | 단점 |
|---|---|---|---|
| 일반 랜덤 샘플링 | 모든 데이터 동일 취급 | 단순함 | 중요한 어려운 데이터를 버릴 수 있음 |
| GOSS | gradient 큰 데이터 우선 보존 | 적은 데이터로도 split gain을 잘 추정 | gradient 기준 샘플링과 보정이 필요 |

---

## 핵심 개념 3: EFB — Exclusive Feature Bundling

**EFB는 동시에 0이 아닌 경우가 거의 없는 feature들을 하나의 feature bundle로 묶어서 feature 수를 줄이는 방법입니다.**

예를 들어 one-hot encoding을 생각해봅시다.

`지역`이라는 범주형 변수가 있습니다.

```text
서울, 부산, 대구, 광주
```

one-hot encoding을 하면 이렇게 됩니다.

| 고객 | 지역_서울 | 지역_부산 | 지역_대구 | 지역_광주 |
|---|---:|---:|---:|---:|
| A | 1 | 0 | 0 | 0 |
| B | 0 | 1 | 0 | 0 |
| C | 0 | 0 | 1 | 0 |
| D | 0 | 0 | 0 | 1 |

여기서 중요한 점은:

```text
지역_서울과 지역_부산은 동시에 1이 될 수 없다.
지역_대구와 지역_광주도 동시에 1이 될 수 없다.
```

이런 feature들을 **mutually exclusive features**, 즉 상호 배타적인 feature라고 합니다.

LightGBM은 이런 feature들을 하나로 묶습니다.

---

## EFB를 쉬운 비유로 보면

여러 개의 신호등이 있다고 합시다.

그런데 규칙상 이 신호등들은 동시에 켜지지 않습니다.

```text
신호등 A가 켜지면 B는 꺼짐
신호등 B가 켜지면 A는 꺼짐
```

그러면 굳이 두 개의 별도 공간을 쓸 필요가 없습니다.

하나의 공간에 이렇게 저장할 수 있습니다.

```text
0 = 아무것도 없음
1 = A 켜짐
2 = B 켜짐
3 = C 켜짐
```

EFB도 비슷합니다.

서로 충돌하지 않는 sparse feature들을 하나의 bundle로 묶어서 feature 수를 줄입니다.

feature 수가 줄면 histogram을 만드는 비용도 줄어듭니다.

---

## GOSS와 EFB의 역할 차이

LightGBM의 핵심은 GOSS와 EFB를 같이 쓰는 것입니다.

둘은 줄이는 대상이 다릅니다.

| 기법 | 줄이는 것 | 핵심 질문 |
|---|---|---|
| **GOSS** | 데이터 행 row 수 | 모든 데이터 instance를 다 봐야 할까? |
| **EFB** | feature 열 column 수 | 모든 sparse feature를 따로 봐야 할까? |

즉:

```text
GOSS = row 방향 효율화
EFB = column 방향 효율화
```

GBDT의 계산량은 대략 데이터 수와 feature 수 양쪽에 영향을 받습니다.

그래서 LightGBM은 두 방향을 모두 줄입니다.

```text
데이터 instance를 줄인다 → GOSS
feature 수를 줄인다 → EFB
```

---

## 핵심 개념 4: Leaf-wise Tree Growth

**LightGBM은 tree를 level-wise가 아니라 leaf-wise로 키웁니다.**

대부분의 tree 학습은 level-wise로 자랍니다.

```text
깊이 1의 모든 node를 나눔
→ 깊이 2의 모든 node를 나눔
→ 깊이 3의 모든 node를 나눔
```

이렇게 하면 tree가 비교적 균형 잡힌 모양으로 자랍니다.

반면 LightGBM은 leaf-wise, 또는 best-first 방식으로 자랍니다.

```text
현재 leaf 중 loss를 가장 많이 줄일 수 있는 leaf 하나를 고름
→ 그 leaf만 나눔
→ 다시 가장 이득이 큰 leaf를 고름
→ 반복
```

### Level-wise

```text
모든 가지를 균형 있게 키움
```

예:

```text
        root
       /    \
      A      B
     / \    / \
    C   D  E   F
```

장점은 안정적입니다. 단점은 별로 필요 없는 가지도 같이 키울 수 있습니다.

### Leaf-wise

```text
가장 많이 개선될 가지를 집중적으로 키움
```

예:

```text
        root
       /    \
      A      B
     /
    C
   /
  D
```

장점은 같은 leaf 수로 더 큰 loss 감소를 만들 수 있습니다. 단점은 특정 방향으로 너무 깊게 자라면 과적합될 수 있습니다.

그래서 LightGBM에서는 `num_leaves`, `max_depth`, `min_data_in_leaf` 같은 파라미터가 중요합니다.

---

## LightGBM과 XGBoost의 차이

둘 다 GBDT 계열이고, 둘 다 강력한 tabular model입니다. 하지만 강조점이 다릅니다.

| 구분 | XGBoost | LightGBM |
|---|---|---|
| 핵심 방향 | 정규화된 GBDT와 scalable system | 더 빠르고 메모리 효율적인 GBDT |
| split 탐색 | exact/approx/histogram 지원 | histogram 기반 중심 |
| 데이터 샘플링 | row subsampling 가능 | GOSS로 gradient 기반 샘플링 |
| feature 효율화 | sparse-aware, column block 등 | EFB로 exclusive feature bundling |
| tree growth | 보통 level-wise 중심으로 이해 | leaf-wise, best-first |
| 강점 | 안정적이고 범용적인 GBDT | 대규모·고차원 sparse feature에서 빠름 |
| 주의점 | 튜닝 필요 | leaf-wise overfitting 주의 |

---

## 실무 예시: 광고 클릭 예측

LightGBM이 특히 잘 맞는 전형적인 상황은 광고 클릭 예측 같은 문제입니다.

광고 클릭 데이터는 보통 이런 특징을 가집니다.

| 특징 | 설명 |
|---|---|
| 데이터 행이 많다 | 수백만~수억 로그 |
| feature가 많다 | 유저, 광고, 기기, 지역, 시간, 카테고리 |
| sparse하다 | one-hot encoding 이후 대부분 0 |
| 빠른 학습이 중요하다 | 실험을 자주 돌려야 함 |
| tabular 구조다 | GBDT 계열이 강함 |

예를 들어 one-hot feature가 많으면 EFB가 유리합니다.

```text
광고카테고리_A
광고카테고리_B
광고카테고리_C
...
```

한 광고는 한 카테고리에만 속하므로 여러 one-hot feature는 동시에 1이 되지 않습니다. 이런 feature를 묶으면 feature 수가 줄어듭니다.

또한 데이터가 수천만 건이면 GOSS가 유리합니다.

```text
이미 잘 맞추는 쉬운 로그는 일부만 본다.
아직 틀리는 어려운 로그는 더 많이 본다.
```

---

## 핵심 개념 5: LightGBM은 “가벼운 모델”이라는 뜻이 아니다

이름 때문에 오해하기 쉽습니다.

**LightGBM의 Light는 모델이 단순하다는 뜻이 아니라, 학습이 빠르고 효율적이라는 뜻에 가깝습니다.**

LightGBM도 충분히 복잡한 모델을 만들 수 있습니다. 특히 leaf-wise 방식 때문에, 제약을 잘 걸지 않으면 작은 데이터에서는 과적합될 수 있습니다.

---

## LightGBM의 장점

1. **빠른 학습 속도** — 대규모 데이터에서 강합니다.
2. **낮은 메모리 사용량** — histogram 기반 알고리즘은 연속값을 discrete bin으로 저장합니다.
3. **sparse feature에 강하다** — EFB는 one-hot feature와 sparse feature를 효율적으로 다룹니다.
4. **대용량 tabular data에 적합하다** — 병렬·분산·GPU 학습 지원과 결합하면 큰 데이터 처리에 강합니다.

---

## LightGBM의 한계와 주의점

### 1. 작은 데이터에서는 과적합에 주의해야 한다

LightGBM은 leaf-wise 방식으로 loss를 가장 많이 줄이는 leaf를 계속 키웁니다. 이 방식은 강력하지만, 데이터가 작거나 노이즈가 많으면 특정 branch가 너무 깊어져 train data에 과적합될 수 있습니다.

### 2. 튜닝 파라미터가 중요하다

| 파라미터 | 의미 | 실무 감각 |
|---|---|---|
| `num_leaves` | leaf 개수 | 너무 크면 과적합 위험 |
| `max_depth` | 최대 깊이 | leaf-wise tree의 깊이 제한 |
| `min_data_in_leaf` | leaf에 필요한 최소 데이터 수 | 키우면 과적합 감소 |
| `learning_rate` | 각 tree의 반영 정도 | 낮추면 tree 수를 늘려야 함 |
| `n_estimators` | tree 개수 | early stopping과 함께 사용 |
| `feature_fraction` | feature subsampling | 과적합 감소, 속도 향상 |
| `bagging_fraction` | row subsampling | 데이터 샘플링 |
| `lambda_l1`, `lambda_l2` | 정규화 | leaf score 제약 |

### 3. 해석 가능성이 단순하지 않다

LightGBM은 수백~수천 개 tree를 쌓는 모델입니다.

단일 decision tree처럼:

```text
이 조건 때문에 예측했다.
```

라고 쉽게 설명하기 어렵습니다.

그래서 실제로는 feature importance, permutation importance, PDP, SHAP 등을 함께 씁니다.

---

## GOSS를 다시 직관적으로 정리

GOSS는 이런 생각입니다.

```text
모델이 이미 잘 맞추는 쉬운 데이터는 조금 덜 봐도 된다.
모델이 아직 못 맞추는 어려운 데이터는 꼭 봐야 한다.
```

수업에서 시험공부를 한다고 생각해봅시다.

이미 다 맞히는 쉬운 문제를 계속 반복해서 보는 것보다, 자꾸 틀리는 어려운 문제를 더 많이 보는 게 효율적입니다.

| 데이터 상태 | gradient | GOSS 처리 |
|---|---:|---|
| 모델이 크게 틀림 | 큼 | 반드시 유지 |
| 모델이 잘 맞춤 | 작음 | 일부만 샘플링 |
| 분포 보정 필요 | — | 작은 gradient 샘플에 weight 조정 |

---

## EFB를 다시 직관적으로 정리

EFB는 이런 생각입니다.

```text
동시에 켜질 일이 없는 feature들은 한 칸에 같이 저장해도 된다.
```

one-hot encoding을 생각하면 쉽습니다.

| 원래 feature들 | 동시에 1 가능? |
|---|---|
| 지역_서울, 지역_부산 | 불가능 |
| 요일_월, 요일_화 | 불가능 |
| 상품카테고리_A, 상품카테고리_B | 대개 불가능 |

이런 feature를 묶으면 feature 수가 줄어듭니다.

feature 수가 줄면 tree split 탐색과 histogram building이 빨라집니다.

---

## LightGBM을 언제 쓰면 좋은가?

| 상황 | 이유 |
|---|---|
| 데이터가 크다 | GOSS와 histogram 기반 학습으로 빠름 |
| feature가 많다 | EFB로 feature 수를 줄일 수 있음 |
| sparse feature가 많다 | one-hot, text-derived feature 등에 강함 |
| tabular data다 | GBDT 계열이 강한 영역 |
| 빠른 실험 반복이 필요하다 | 학습 속도가 빠름 |
| ranking 문제를 다룬다 | LambdaRank와 NDCG metric 지원 |

---

## 실무에서 LightGBM을 처음 쓸 때의 추천 흐름

### 1. 먼저 평가 설계를 제대로 한다

LightGBM은 강력해서 leakage가 있으면 말도 안 되게 좋은 성능이 나올 수 있습니다.

| 데이터 유형 | split 방식 |
|---|---|
| 일반 tabular data | train / validation / test |
| 시간 순서 데이터 | 과거 train, 이후 validation, 최신 test |
| 고객 단위 데이터 | 같은 고객이 train/test에 동시에 들어가지 않게 주의 |
| 그룹 데이터 | group split 고려 |

### 2. 작은 learning rate + early stopping을 쓴다

```text
learning_rate = 0.03 ~ 0.1
n_estimators는 크게
validation set으로 early stopping
```

### 3. 과적합이면 leaf를 줄인다

| 문제 | 조정 방향 |
|---|---|
| train 성능은 좋고 validation 성능이 나쁨 | `num_leaves` 줄이기 |
| leaf가 너무 세부적으로 쪼개짐 | `min_data_in_leaf` 키우기 |
| tree가 너무 깊음 | `max_depth` 제한 |
| feature가 너무 많음 | `feature_fraction` 낮추기 |
| 데이터 row가 많고 noisy함 | `bagging_fraction` 사용 |
| leaf score가 과격함 | L1/L2 regularization 강화 |

---

## XGBoost, LightGBM, CatBoost를 어떻게 구분하면 좋나?

| 모델 | 한 줄 감각 |
|---|---|
| **XGBoost** | 안정적이고 정규화가 잘 설계된 고성능 GBDT |
| **LightGBM** | 빠르고 메모리 효율적인 대규모 GBDT |
| **CatBoost** | 범주형 feature와 target leakage 제어에 강한 GBDT |

LightGBM은 특히 데이터가 크고 sparse feature가 많을 때 좋은 선택지가 됩니다.

---

## 이전 논문들과 연결해서 이해하기

| 이전 논문 | LightGBM과의 연결 |
|---|---|
| **Tidy Data** | LightGBM에 넣을 tabular feature table을 명확히 구성해야 함 |
| **Two Cultures** | LightGBM은 예측 성능 중심의 algorithmic modeling culture 모델 |
| **Model Evaluation** | 빠르게 튜닝할 수 있지만 validation/test 오염을 막아야 함 |
| **Datasheets** | 데이터 수집·라벨 정의·편향이 모델에 그대로 반영됨 |
| **Model Cards** | LightGBM 모델도 사용 범위와 집단별 성능을 문서화해야 함 |
| **Random Forests** | 둘 다 tree ensemble이지만 RF는 독립 평균, LightGBM은 순차 boosting |
| **Gradient Boosting** | LightGBM은 GBDT의 효율화 버전 |
| **XGBoost** | 둘 다 GBDT 시스템이지만 LightGBM은 GOSS/EFB와 leaf-wise 성장으로 속도에 강점 |

---

## 이 논문이 주는 진짜 메시지

LightGBM의 핵심은 단순히 “XGBoost보다 빠르다”가 아닙니다.

더 중요한 메시지는 이것입니다.

**큰 데이터에서 좋은 모델을 만들려면, 알고리즘이 모든 데이터를 똑같이 다루지 않아도 된다. 중요한 데이터와 feature를 더 효율적으로 선택하고 압축하면, 정확도를 크게 잃지 않으면서 학습 비용을 줄일 수 있다.**

GOSS는 데이터 instance의 중요도를 gradient로 봅니다.

EFB는 feature의 동시 발생 구조를 봅니다.

즉 LightGBM은 데이터의 구조를 활용해서 빠르게 학습합니다.

```text
GOSS: 어떤 row가 중요한가?
EFB: 어떤 feature를 같이 묶을 수 있는가?
```

이 두 질문이 LightGBM의 핵심입니다.

---

## 오늘 공부용 요약

**논문명:** LightGBM: A Highly Efficient Gradient Boosting Decision Tree  
**저자:** Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu  
**핵심 결론:** LightGBM은 GOSS와 EFB를 이용해 GBDT 학습에서 데이터 수와 feature 수의 부담을 줄이고, 빠른 학습 속도와 낮은 메모리 사용량을 달성한다.  
**왜 중요함:** 대규모 tabular data와 sparse feature가 많은 실무 문제에서 GBDT를 빠르게 학습할 수 있게 만든 대표 모델이다.  
**입문자가 배울 점:** 모든 데이터와 모든 feature를 항상 똑같이 볼 필요는 없다. gradient가 큰 어려운 데이터와 mutually exclusive한 sparse feature 구조를 활용하면 학습을 크게 효율화할 수 있다.  
**가장 중요한 문장:** LightGBM은 GBDT에서 row 방향은 GOSS로 줄이고, column 방향은 EFB로 줄여 빠르고 메모리 효율적인 boosting을 만든다.


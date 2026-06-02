# XGBoost: A Scalable Tree Boosting System — Chen & Guestrin

## 결론부터

이 논문의 결론은 이겁니다.

**XGBoost는 Gradient Boosting Tree를 더 빠르고, 더 안정적이고, 더 큰 데이터에 쓸 수 있게 만든 시스템이다.**  
단순히 “성능 좋은 부스팅 모델”이 아니라, **정규화 regularization, 2차 미분 정보, 결측치·희소 데이터 처리, 빠른 split 탐색, 병렬화·캐시 최적화·out-of-core 학습**까지 묶은 실전형 tree boosting 시스템입니다.

쉽게 말하면:

> **Gradient Boosting이 “이전 모델의 실수를 다음 나무가 고친다”는 아이디어라면, XGBoost는 그 아이디어를 실무 대용량 데이터에서 빠르고 덜 과적합되게 구현한 버전입니다.**

---

## 이 논문을 한 문장으로 요약하면

**XGBoost는 Gradient Boosting Tree에 정규화와 고성능 시스템 설계를 결합해, tabular data에서 강력한 예측 성능과 확장성을 동시에 달성한 모델이다.**

이전 논문인 Friedman의 Gradient Boosting은 이 질문에 답했습니다.

> “현재 모델의 손실을 줄이는 방향으로 다음 나무를 어떻게 추가할 것인가?”

XGBoost는 여기에 질문을 더합니다.

> “그걸 어떻게 더 빠르게 할 것인가?”  
> “어떻게 과적합을 줄일 것인가?”  
> “결측치와 sparse feature를 어떻게 처리할 것인가?”  
> “대용량 데이터에서도 어떻게 학습할 것인가?”

즉, XGBoost는 **알고리즘 + 시스템 최적화 + 실무 튜닝 장치**가 합쳐진 논문입니다.

---

## 먼저 Gradient Boosting과 XGBoost의 차이

둘 다 tree를 순차적으로 쌓습니다.

```text
모델 0
→ tree 1 추가
→ tree 2 추가
→ tree 3 추가
→ ...
```

하지만 XGBoost는 일반 Gradient Boosting보다 몇 가지를 더 명시적으로 넣습니다.

| 구분 | Gradient Boosting | XGBoost |
|---|---|---|
| 기본 아이디어 | 손실을 줄이는 방향으로 tree를 순차 추가 | 동일 |
| 학습 방향 | 주로 1차 gradient 중심 설명 | 1차 gradient + 2차 gradient 사용 |
| 과적합 제어 | learning rate, tree depth 등 | objective에 정규화 항을 명시적으로 포함 |
| split 평가 | impurity 또는 loss reduction | 정규화된 gain 공식 사용 |
| 결측치 처리 | 별도 전처리 필요할 수 있음 | split마다 default direction 학습 |
| sparse data | 일반 구현은 비효율적일 수 있음 | sparsity-aware algorithm |
| 대용량 처리 | 구현마다 다름 | column block, cache-aware, out-of-core, distributed 설계 |

한마디로:

> **Gradient Boosting은 방법론이고, XGBoost는 그 방법론을 매우 강력하게 구현한 시스템입니다.**

---

## 핵심 개념 1: 정규화된 목적함수

**XGBoost는 “예측을 잘하는 것”뿐 아니라 “모델이 너무 복잡해지지 않는 것”까지 목적함수에 넣습니다.**

일반적으로 머신러닝 모델은 loss를 줄이려고 합니다.

```text
목적 = 예측 오차 줄이기
```

XGBoost는 여기에 모델 복잡도 penalty를 추가합니다.

```text
목적 = 예측 오차 + 모델 복잡도 penalty
```

논문에서는 목적함수를 다음 형태로 둡니다.

```text
Objective = loss + regularization
```

정규화 항은 대략 이렇게 생겼습니다.

```text
Ω(f) = γ × leaf 개수 + 1/2 × λ × leaf score 크기²
```

여기서:

| 기호 | 의미 |
|---|---|
| `γ` gamma | leaf를 하나 더 만들 때 드는 비용 |
| `T` | tree의 leaf 개수 |
| `λ` lambda | leaf score가 너무 커지는 것을 막는 L2 정규화 |
| `w` | leaf score |

쉽게 말하면 XGBoost는 이렇게 생각합니다.

> “split을 더 해서 train loss가 조금 줄어든다고 무조건 좋은 것은 아니다.  
> leaf를 더 만드는 비용까지 고려했을 때 이득이 있어야 split하자.”

이게 XGBoost의 중요한 차이입니다.

---

## 핵심 개념 2: leaf마다 점수를 가진 tree

XGBoost의 tree는 단순히 class label만 주는 tree가 아닙니다.

각 leaf가 **score**를 가집니다.

예를 들어 고객 이탈 예측에서 어떤 tree가 이렇게 생겼다고 합시다.

```text
최근 사용량 < 1GB?
 ├─ 예: +0.8
 └─ 아니오: -0.2
```

여기서 `+0.8`, `-0.2`가 leaf score입니다.

최종 예측은 여러 tree의 leaf score를 모두 더해서 만듭니다.

```text
최종 예측 = tree1 점수 + tree2 점수 + tree3 점수 + ...
```

즉, XGBoost는 이런 구조입니다.

```text
입력 x
→ tree 1에서 leaf score 얻음
→ tree 2에서 leaf score 얻음
→ tree 3에서 leaf score 얻음
→ 전부 더함
→ 최종 예측
```

---

## 핵심 개념 3: 1차 gradient와 2차 gradient를 같이 쓴다

**XGBoost는 loss를 줄일 때 gradient뿐 아니라 Hessian, 즉 2차 미분 정보도 사용합니다.**

| 개념 | 쉬운 의미 |
|---|---|
| **Gradient** | loss를 줄이려면 예측을 어느 방향으로 움직여야 하는가 |
| **Hessian** | loss 곡면이 얼마나 휘어져 있는가, 즉 조심해서 움직여야 하는 정도 |

XGBoost에서는 각 데이터마다 두 값을 계산합니다.

```text
g_i = 1차 gradient
h_i = 2차 gradient
```

쉽게 비유하면 이렇습니다.

```text
gradient = 어느 방향으로 내려가야 하는지
hessian = 그 길이 얼마나 급하고 휘어져 있는지
```

Gradient만 쓰면 “방향”만 봅니다. Hessian까지 쓰면 “방향 + 곡률”을 같이 봅니다.

그래서 XGBoost는 split과 leaf score를 계산할 때 더 정교한 정보를 씁니다.

---

## 핵심 개념 4: split을 할지 말지 gain으로 판단한다

Decision Tree를 만들 때 가장 중요한 질문은 이것입니다.

> “이 node를 어느 feature의 어느 값으로 나눌까?”

XGBoost는 split 후보를 평가할 때 단순히 Gini impurity 같은 것만 보지 않습니다.

각 split이 objective를 얼마나 줄이는지 계산합니다.

쉽게 말하면:

```text
split gain = 왼쪽 leaf에서 얻는 이득
           + 오른쪽 leaf에서 얻는 이득
           - split 전 leaf의 이득
           - 새 leaf를 만드는 비용 γ
```

즉, split을 해서 loss가 줄어도, 그 이득이 `γ`보다 작으면 split하지 않는 것이 낫다고 봅니다.

실무적으로는 매우 중요합니다.

```text
γ가 작다 → split을 쉽게 허용 → 복잡한 tree
γ가 크다 → split을 엄격하게 허용 → 단순한 tree
```

그래서 `gamma`는 XGBoost에서 tree 복잡도를 조절하는 중요한 하이퍼파라미터입니다.

---

## 핵심 개념 5: shrinkage, 즉 learning rate

XGBoost도 Gradient Boosting처럼 한 번에 크게 고치지 않습니다.

새 tree가 계산한 보정을 그대로 더하지 않고, `eta` 또는 `learning_rate`를 곱해서 조금만 반영합니다.

```text
새 모델 = 이전 모델 + η × 새 tree
```

예를 들어 새 tree가 `+1.0`을 제안했는데 `eta = 0.1`이면 실제로는 `+0.1`만 반영합니다.

쉽게 말하면:

> **XGBoost는 오답노트를 한 번에 고치지 않고, 조금씩 조심스럽게 고칩니다.**

| learning_rate | tree 개수 | 특징 |
|---:|---:|---|
| 큼 | 적게 필요 | 빠르지만 과적합 위험 |
| 작음 | 많이 필요 | 느리지만 일반화가 좋아질 수 있음 |

---

## 핵심 개념 6: column subsampling

XGBoost는 Random Forest처럼 feature를 일부만 보는 기법도 씁니다.

예를 들어 feature가 100개라면, 매 tree 또는 매 level 또는 매 split에서 일부 feature만 사용하게 할 수 있습니다.

이렇게 하면 두 가지 장점이 있습니다.

| 장점 | 설명 |
|---|---|
| 과적합 감소 | 모든 tree가 비슷한 feature만 쓰는 것을 줄임 |
| 계산 속도 향상 | 고려할 feature 수가 줄어듦 |

이 부분은 Random Forest와 연결됩니다.

```text
Random Forest: feature randomness로 tree 간 상관을 줄임
XGBoost: column subsampling으로 과적합을 줄이고 계산을 빠르게 함
```

---

## 핵심 개념 7: 결측치와 sparse data를 똑똑하게 처리한다

**XGBoost는 결측치를 단순히 평균이나 0으로 채우지 않고, split마다 결측치가 어느 방향으로 가야 좋은지 학습합니다.**

예를 들어 어떤 feature가 `최근 로그인 횟수`라고 합시다.

어떤 고객은 이 값이 없습니다.

일반적인 방법은:

```text
결측치를 평균으로 채운다
결측치를 0으로 채운다
별도 전처리를 한다
```

입니다.

XGBoost는 다르게 접근합니다.

어떤 split에서 값이 missing이면:

```text
missing 값을 왼쪽으로 보낼까?
missing 값을 오른쪽으로 보낼까?
```

둘 다 시도해보고, gain이 더 좋은 방향을 **default direction**으로 학습합니다.

이건 실무에서 매우 큽니다.

특히 one-hot encoding을 하면 데이터가 대부분 0인 sparse matrix가 됩니다. XGBoost는 이런 sparse structure를 계산상 효율적으로 활용합니다.

---

## 핵심 개념 8: weighted quantile sketch

**XGBoost는 대용량 데이터에서 모든 split 후보를 전부 보지 않고, 좋은 후보 split 지점을 효율적으로 고릅니다.**

정확한 greedy split은 모든 feature의 모든 가능한 split을 검사합니다.

작은 데이터에서는 가능합니다.

하지만 데이터가 수천만, 수억 건이면 너무 비쌉니다.

그래서 approximate algorithm이 필요합니다.

XGBoost는 feature 값의 분위수 quantile를 이용해 후보 split 지점을 고릅니다. 그런데 XGBoost에서는 각 샘플이 같은 무게가 아닙니다. 앞서 본 `h_i`, 즉 2차 gradient가 weight처럼 작동합니다.

쉽게 말하면:

> “모든 split을 다 보지는 말자.  
> 하지만 중요한 데이터의 무게를 반영해서 좋은 후보 split을 고르자.”

이게 weighted quantile sketch입니다.

---

## XGBoost가 빠른 이유

XGBoost의 중요한 기여는 모델 수식만이 아닙니다.

시스템 설계가 핵심입니다.

| 기술 | 쉬운 설명 |
|---|---|
| **Column block** | feature column을 정렬된 block으로 저장해 split 탐색을 빠르게 함 |
| **Parallel split search** | feature별 split 계산을 병렬화 |
| **Cache-aware design** | CPU cache에 잘 맞게 데이터 접근 패턴 최적화 |
| **Out-of-core computation** | 메모리에 다 안 올라가는 데이터도 디스크를 활용해 학습 |
| **Data compression / sharding** | 큰 데이터를 효율적으로 저장·분산 처리 |
| **Sparsity-aware algorithm** | 결측치·0이 많은 데이터를 효율적으로 처리 |

이 논문은 “좋은 알고리즘”만으로는 부족하고, **좋은 시스템 구현**이 성능을 크게 좌우한다는 걸 보여줍니다.

---

## 쉬운 예시: 고객 이탈 예측에서 XGBoost가 하는 일

고객 이탈 예측 데이터를 생각해봅시다.

| 고객 | 가입기간 | 월요금 | 최근 사용량 | 불만 횟수 | 이탈 여부 |
|---|---:|---:|---:|---:|---:|
| A | 24개월 | 55,000원 | 15GB | 0 | 0 |
| B | 3개월 | 90,000원 | 2GB | 2 | 1 |
| C | 12개월 | 70,000원 | 결측 | 1 | 0 |

XGBoost는 처음에 단순한 예측을 합니다.

```text
초기 예측: 전체 평균 이탈률
```

그다음 반복합니다.

```text
1. 현재 예측이 얼마나 틀렸는지 loss를 본다.
2. 각 데이터의 gradient와 hessian을 계산한다.
3. 어떤 split이 loss를 가장 많이 줄이는지 계산한다.
4. 단, split을 추가하는 비용 γ를 고려한다.
5. leaf score가 너무 커지지 않도록 λ로 정규화한다.
6. 결측치는 왼쪽/오른쪽 중 더 좋은 default direction으로 보낸다.
7. 새 tree를 learning_rate만큼만 반영한다.
8. validation 성능을 보며 반복한다.
```

즉, XGBoost는 단순히 “오차를 맞추는 tree”를 만드는 게 아니라, 매번 이렇게 묻습니다.

> “이 split은 정규화까지 고려해도 진짜 이득인가?”  
> “이 leaf score는 너무 과격하지 않은가?”  
> “missing value는 어느 방향으로 보내야 loss가 줄어드는가?”  
> “전체 데이터를 빠르게 처리하려면 어떤 후보 split만 봐야 하는가?”

---

## XGBoost와 Random Forest, Gradient Boosting 비교

| 구분 | Random Forest | Gradient Boosting | XGBoost |
|---|---|---|---|
| tree 학습 방식 | 독립적으로 여러 tree 학습 | 순차적으로 tree 학습 | 순차적으로 tree 학습 |
| 최종 결합 | 평균 또는 다수결 | 누적합 | 누적합 |
| 각 tree의 역할 | 전체 문제를 각각 예측 | 이전 모델의 부족한 부분 보정 | 정규화된 objective를 줄이는 방향으로 보정 |
| 과적합 제어 | bagging, feature randomness | learning rate, depth | learning rate, depth, γ, λ, α, subsampling |
| 결측치 처리 | 구현마다 다름 | 구현마다 다름 | default direction 학습 |
| sparse data 처리 | 보통 별도 고려 필요 | 보통 별도 고려 필요 | sparsity-aware |
| 대용량 처리 | 비교적 쉬운 병렬화 | 순차 구조라 어려움 | split 탐색·시스템 최적화로 확장성 강화 |
| 튜닝 난이도 | 비교적 낮음 | 높음 | 높지만 성능 강함 |

짧게 말하면:

```text
Random Forest = 여러 tree를 독립적으로 만들고 평균
Gradient Boosting = tree를 순서대로 쌓으며 오차 보정
XGBoost = Gradient Boosting을 정규화하고 대규모로 빠르게 구현
```

---

## 실무에서 중요한 하이퍼파라미터

| 파라미터 | 의미 | 커지면 대체로 |
|---|---|---|
| `n_estimators` / `num_boost_round` | tree 개수 | 복잡해짐, 과적합 가능 |
| `eta` / `learning_rate` | 각 tree 반영 정도 | 빠르게 학습, 과적합 위험 |
| `max_depth` | tree 최대 깊이 | 복잡한 interaction 학습, 과적합 위험 |
| `min_child_weight` | child node에 필요한 최소 hessian 합 | 더 보수적 |
| `gamma` | split에 필요한 최소 loss reduction | split이 줄어듦 |
| `subsample` | row sampling 비율 | randomness 증가, 과적합 감소 가능 |
| `colsample_bytree` | tree마다 feature sampling 비율 | feature randomness 증가 |
| `lambda` / `reg_lambda` | L2 정규화 | leaf score가 더 작아짐 |
| `alpha` / `reg_alpha` | L1 정규화 | feature/leaf 효과가 더 sparse해질 수 있음 |
| `scale_pos_weight` | class imbalance 조정 | 양성 클래스 가중치 증가 |

---

## XGBoost를 처음 쓸 때의 실전 감각

### 1. 먼저 validation set을 제대로 나눈다

XGBoost는 강력하지만 과적합도 가능합니다.

그래서 제일 먼저 해야 할 일은 모델 튜닝이 아니라 평가 설계입니다.

```text
train / validation / test
```

시간 순서가 있는 데이터라면 random split을 하면 안 됩니다.

```text
train: 과거
validation: 그다음 기간
test: 가장 최근 기간
```

### 2. early stopping을 쓴다

XGBoost는 tree를 계속 추가하면 train loss는 계속 줄어들 수 있습니다. 하지만 validation 성능은 어느 순간부터 나빠질 수 있습니다.

```text
train 성능 계속 좋아짐
validation 성능 멈춤 또는 나빠짐
→ 과적합 시작
```

### 3. learning rate와 tree 개수를 같이 조절한다

```text
learning_rate 낮춤
→ tree 개수 늘림
```

| 설정 | 느낌 |
|---|---|
| `learning_rate=0.3`, tree 100개 | 빠르지만 거칠 수 있음 |
| `learning_rate=0.1`, tree 500개 | 일반적인 시작점 |
| `learning_rate=0.03`, tree 2000개 | 느리지만 세밀함 |

### 4. 과적합이면 복잡도를 줄인다

| 조치 | 효과 |
|---|---|
| `max_depth` 줄이기 | tree 단순화 |
| `min_child_weight` 키우기 | 작은 leaf 방지 |
| `gamma` 키우기 | 쓸모없는 split 방지 |
| `subsample` 낮추기 | row randomness |
| `colsample_bytree` 낮추기 | feature randomness |
| `lambda`, `alpha` 키우기 | 정규화 강화 |
| `learning_rate` 낮추고 early stopping | 더 부드러운 학습 |

---

## 이 논문의 진짜 메시지

이 논문은 “XGBoost가 성능이 좋다”라는 단순한 이야기가 아닙니다.

더 중요한 메시지는 이것입니다.

**좋은 머신러닝 시스템은 모델 수식, 최적화 알고리즘, 데이터 구조, 하드웨어 친화적 구현이 함께 설계되어야 한다.**

Friedman의 Gradient Boosting이 수학적 아이디어를 줬다면, XGBoost는 그 아이디어를 실제 데이터사이언스 현장에서 쓸 수 있도록 엔지니어링했습니다.

---

## XGBoost의 한계도 알아야 한다

| 한계 | 설명 |
|---|---|
| 튜닝에 민감함 | learning rate, depth, regularization에 따라 성능 차이가 큼 |
| 해석이 단순하지 않음 | 단일 tree보다 설명이 어렵고 SHAP/PDP 같은 도구가 필요 |
| 인과관계를 말하지 않음 | feature importance가 높아도 원인이라는 뜻은 아님 |
| 외삽에 약함 | tree 기반 모델은 train 범위 밖 값을 예측하는 데 약할 수 있음 |
| 데이터 leakage에 취약 | 강한 모델이라 leakage가 있으면 비현실적으로 높은 성능을 냄 |
| 확률 calibration이 필요할 수 있음 | 분류 확률을 그대로 의사결정 확률로 쓰기 전 calibration 확인 필요 |

특히 중요한 것은 이것입니다.

> XGBoost의 성능이 좋다고 해서 데이터 품질, 평가 설계, leakage 점검, model card 작성이 생략되는 것은 아닙니다.

---

## 이전 논문들과 연결해서 이해하기

| 이전 논문 | XGBoost와의 연결 |
|---|---|
| **Tidy Data** | XGBoost에 넣을 feature table을 올바르게 구성해야 함 |
| **Two Cultures** | XGBoost는 algorithmic modeling culture의 대표 모델 |
| **Model Evaluation** | validation, test set, early stopping, leakage 방지가 매우 중요 |
| **Datasheets** | 데이터 수집·라벨 정의·편향이 XGBoost 모델에 그대로 반영됨 |
| **Model Cards** | XGBoost 모델도 사용 범위, 집단별 성능, 한계를 문서화해야 함 |
| **Random Forests** | 둘 다 tree ensemble이지만 RF는 독립 평균, XGBoost는 순차 보정 |
| **Gradient Boosting** | XGBoost는 Gradient Boosting의 정규화·확장성 강화 버전 |

---

## 입문자가 꼭 기억해야 할 문장

**XGBoost는 Gradient Boosting Tree에 정규화, 2차 gradient 정보, 결측치·희소 데이터 처리, 빠른 split 탐색, 병렬·대용량 시스템 최적화를 결합한 모델이다.**

더 짧게 말하면:

> **XGBoost = 실무형 고성능 Gradient Boosting Tree**

---

## 오늘 공부용 요약

**논문명:** XGBoost: A Scalable Tree Boosting System  
**저자:** Tianqi Chen, Carlos Guestrin  
**핵심 결론:** Gradient Boosting Tree를 정규화하고, 1차·2차 gradient 정보를 이용하며, sparse data와 missing value를 효율적으로 처리하고, 대용량 데이터에서 빠르게 학습하도록 시스템 최적화한 모델이 XGBoost다.  
**왜 중요함:** tabular data에서 강력한 성능을 내는 대표 모델이며, XGBoost 이후 LightGBM, CatBoost 같은 고성능 GBDT 계열을 이해하는 기반이 된다.  
**입문자가 배울 점:** 좋은 모델은 알고리즘만으로 만들어지지 않는다. 손실함수, 정규화, split 탐색, 데이터 구조, 시스템 최적화, 평가 설계가 함께 중요하다.  
**가장 중요한 문장:** XGBoost는 “이전 모델의 실수를 고치는 tree”를 단순히 반복해서 쌓는 것이 아니라, 정규화된 objective를 가장 효율적으로 줄이는 tree를 빠르게 찾아 누적하는 시스템이다.


# Greedy Function Approximation: A Gradient Boosting Machine — Jerome H. Friedman

## 결론부터

이 논문의 결론은 이겁니다.

**모델을 한 번에 완성하려고 하지 말고, 현재 모델이 틀린 부분을 다음 모델이 조금씩 고치게 만들면 강력한 예측 모델을 만들 수 있다.**  
이 방식이 바로 **Gradient Boosting**입니다.

조금 더 정확히 말하면:

> Gradient Boosting은 예측 모델을 하나의 함수 `F(x)`로 보고, 손실함수 loss를 줄이는 방향으로 작은 모델들을 순차적으로 더해 가는 방법입니다.

Friedman은 이 논문에서 함수 추정을 “파라미터 공간에서의 최적화”가 아니라 **함수 공간에서의 최적화**로 바라봅니다. 그리고 stagewise additive model, steepest descent, boosting을 연결해 일반적인 gradient boosting 프레임워크를 제시합니다.

---

## 이 논문을 한 문장으로 요약하면

**Gradient Boosting은 “이전 모델의 오차를 다음 모델이 학습한다”는 아이디어를 손실함수의 gradient descent로 일반화한 방법입니다.**

Random Forest와 비교하면 차이가 선명합니다.

| 모델 | 핵심 아이디어 |
|---|---|
| **Random Forest** | 여러 나무를 독립적으로 만들고 평균낸다 |
| **Gradient Boosting** | 나무를 순서대로 만들며 이전 모델의 실수를 고친다 |

즉, Random Forest는 **병렬적 앙상블**이고, Gradient Boosting은 **순차적 앙상블**입니다.

---

## 왜 이 논문이 중요한가?

이 논문은 지금의 XGBoost, LightGBM, CatBoost 같은 boosting 계열 모델을 이해하기 위한 핵심 기반입니다.

데이터사이언스 실무에서 tabular data를 다룰 때 자주 쓰는 고성능 모델들이 대부분 이 흐름 위에 있습니다.

| 모델 | 계열 |
|---|---|
| Gradient Boosting Machine, GBM | Friedman식 gradient boosting |
| XGBoost | regularized gradient boosting |
| LightGBM | 효율화된 gradient boosting tree |
| CatBoost | 범주형 변수 처리에 강한 gradient boosting |
| sklearn GradientBoosting | 고전적 GBM 구현 |

이 논문을 이해하면 “XGBoost를 그냥 쓰는 사람”에서 “boosting이 왜 작동하는지 아는 사람”으로 넘어갈 수 있습니다.

---

## 핵심 개념 1: Boosting은 모델을 더해 가는 방식이다

Gradient Boosting의 모델은 이런 모양입니다.

```text
최종 모델 = 초기 모델 + 작은 모델 1 + 작은 모델 2 + 작은 모델 3 + ...
```

수식으로 쓰면:

```text
F_M(x) = F_0(x) + h_1(x) + h_2(x) + ... + h_M(x)
```

여기서:

| 기호 | 의미 |
|---|---|
| `F_0(x)` | 처음 시작하는 단순한 예측값 |
| `h_m(x)` | m번째로 추가되는 작은 모델 |
| `F_M(x)` | M번 업데이트 후 최종 모델 |

이걸 **additive model**, 즉 더해서 만드는 모델이라고 합니다.

---

## 쉬운 예시: 집값 예측

집값 예측 문제를 생각해봅시다.

처음에는 아무것도 모르니까 전체 평균 집값으로 예측합니다.

```text
F_0(x) = 5억
```

그런데 실제값과 비교해보니 이런 오차가 생깁니다.

| 집 | 실제 집값 | 현재 예측 | 오차 |
|---|---:|---:|---:|
| A | 7억 | 5억 | +2억 |
| B | 4억 | 5억 | -1억 |
| C | 6억 | 5억 | +1억 |
| D | 3억 | 5억 | -2억 |

이제 첫 번째 작은 나무 `h_1(x)`는 집값 자체를 예측하는 것이 아니라, **현재 모델의 오차**를 예측합니다.

예를 들어 이런 규칙을 배울 수 있습니다.

```text
면적이 크고 강남이면 +2억 보정
면적이 작고 구축이면 -1.5억 보정
```

그러면 모델을 업데이트합니다.

```text
F_1(x) = F_0(x) + h_1(x)
```

이제 다시 오차를 계산합니다. 그리고 두 번째 나무 `h_2(x)`는 남은 오차를 또 학습합니다.

```text
F_2(x) = F_1(x) + h_2(x)
```

이 과정을 반복합니다.

즉, Gradient Boosting은 이런 식입니다.

```text
처음에는 대충 예측
→ 어디서 틀렸는지 확인
→ 그 틀린 부분을 작은 모델이 보정
→ 다시 틀린 부분 확인
→ 또 보정
→ 반복
```

---

## 핵심 개념 2: “Residual을 학습한다”는 말의 정확한 의미

입문 단계에서는 Gradient Boosting을 이렇게 설명해도 됩니다.

> 다음 나무가 이전 모델의 residual을 학습한다.

여기서 residual은 보통:

```text
residual = 실제값 - 예측값
```

입니다.

예를 들어 실제 집값이 7억이고 현재 예측이 5억이면:

```text
residual = 7억 - 5억 = +2억
```

다음 나무는 이 `+2억`을 맞추려고 합니다.

하지만 이 논문의 핵심은 한 단계 더 일반적입니다.

Friedman은 단순한 residual이 아니라 **pseudo-residual**, 즉 손실함수의 **negative gradient**를 학습하게 만듭니다.

---

## residual과 pseudo-residual의 차이

### squared error에서는 residual이 맞다

손실함수가 squared error라면:

```text
L(y, F) = 1/2 × (y - F)^2
```

이때 negative gradient는:

```text
y - F
```

즉, residual과 같습니다.

그래서 회귀 문제에서 평균제곱오차를 쓰는 Gradient Boosting은 “오차를 학습한다”고 이해해도 됩니다.

### 다른 loss에서는 residual이 아니라 pseudo-residual이다

하지만 손실함수가 바뀌면 달라집니다.

예를 들어 분류 문제에서는 squared error보다 logistic loss를 많이 씁니다. 이때 다음 모델은 단순한 `y - 예측값`이 아니라, logistic loss를 줄이는 방향의 gradient를 학습합니다.

그래서 더 정확한 표현은 이것입니다.

> Gradient Boosting은 이전 모델의 residual을 학습한다기보다, 현재 손실함수를 가장 빨리 줄이는 방향인 **negative gradient**를 학습한다.

이게 논문의 핵심입니다.

---

## 핵심 개념 3: 왜 “Gradient” Boosting인가?

Gradient descent는 원래 파라미터를 조금씩 업데이트하는 방식입니다.

예를 들어 선형회귀라면:

```text
w_new = w_old - learning_rate × gradient
```

여기서 `w`는 파라미터입니다.

그런데 Friedman은 이것을 함수 자체로 확장합니다.

```text
F_new(x) = F_old(x) + 작은 모델
```

즉, 파라미터 `w`를 직접 움직이는 것이 아니라, 예측 함수 `F(x)`를 손실이 줄어드는 방향으로 움직입니다.

이걸 **function space에서의 gradient descent**라고 볼 수 있습니다.

| 일반 gradient descent | Gradient Boosting |
|---|---|
| 파라미터를 업데이트 | 예측 함수 `F(x)`를 업데이트 |
| `w ← w - η∇L` | `F ← F + ηh` |
| gradient를 직접 계산 | negative gradient를 작은 모델이 근사 |
| 하나의 모델 내부 최적화 | 여러 모델을 순차적으로 추가 |

---

## 핵심 개념 4: 왜 “Greedy”인가?

논문 제목에 **Greedy Function Approximation**이라고 되어 있습니다.

여기서 greedy는 알고리즘 용어입니다.

**Greedy algorithm**은 매 단계에서 지금 당장 가장 좋아 보이는 선택을 하는 방식입니다.

Gradient Boosting은 모든 나무를 한꺼번에 최적화하지 않습니다.

```text
나무 1, 나무 2, ..., 나무 100을 동시에 최적으로 찾자
```

이렇게 하지 않습니다.

대신 이렇게 합니다.

```text
현재 모델이 있다.
지금 손실을 가장 줄일 수 있는 다음 작은 모델 하나를 찾자.
그걸 더하자.
다음 단계로 가자.
```

이것이 greedy입니다.

장점은 계산이 가능하고 실용적이라는 것입니다. 단점은 매 단계의 최선 선택이 전체적으로 완벽한 최적해를 보장하지는 않는다는 것입니다.

---

## 핵심 개념 5: Base learner는 보통 작은 tree다

Gradient Boosting에서 매번 추가되는 작은 모델을 **base learner** 또는 **weak learner**라고 부릅니다.

논문에서는 특히 regression tree를 base learner로 쓰는 경우를 중요하게 다룹니다.

예를 들어 작은 tree는 이런 식입니다.

```text
가입기간 < 3개월?
 ├─ 예: +0.8 보정
 └─ 아니오: 최근 사용량 < 1GB?
      ├─ 예: +0.5 보정
      └─ 아니오: -0.2 보정
```

이 tree 하나는 약합니다. 하지만 이런 tree를 수백 개 순서대로 더하면 복잡한 비선형 관계를 학습할 수 있습니다.

---

## Gradient Boosting 알고리즘을 단계별로 보면

가장 기본적인 흐름은 다음과 같습니다.

```text
1. 초기 모델 F_0(x)를 만든다.
   예: 회귀에서는 y의 평균값

2. m = 1부터 M까지 반복한다.

3. 현재 모델 F_{m-1}(x)의 손실함수 gradient를 계산한다.

4. negative gradient, 즉 pseudo-residual을 만든다.

5. 작은 tree h_m(x)를 pseudo-residual에 맞춘다.

6. 그 tree를 얼마나 더할지 결정한다.

7. 기존 모델에 조금 더한다.
   F_m(x) = F_{m-1}(x) + learning_rate × h_m(x)

8. 최종적으로 F_M(x)를 사용한다.
```

---

## 핵심 개념 6: Loss function을 바꾸면 문제 유형이 바뀐다

Gradient Boosting의 강력한 점은 특정 문제에 묶여 있지 않다는 것입니다.

손실함수를 어떻게 정의하느냐에 따라 회귀, 분류, robust regression 등을 다룰 수 있습니다.

| 문제 | 손실함수 예시 | 의미 |
|---|---|---|
| 일반 회귀 | squared error | 큰 오차를 크게 벌줌 |
| 이상치에 강한 회귀 | absolute error | 큰 이상치의 영향을 덜 받음 |
| 적당히 robust한 회귀 | Huber loss | squared error와 absolute error의 절충 |
| 이진분류 | logistic loss | 확률적 분류 |
| 다중분류 | multiclass logistic likelihood | 여러 클래스 분류 |

이게 중요합니다.

Random Forest는 보통 impurity나 MSE 같은 tree 기준에 의존하지만, Gradient Boosting은 **“어떤 loss를 줄일 것인가?”**를 중심에 둡니다.

---

## 쉬운 예시: 고객 이탈 분류

고객 이탈 예측 문제를 생각해봅시다.

목표는:

```text
이 고객이 이탈할 확률을 예측하자.
```

처음에는 모든 고객에게 평균 이탈률을 줍니다.

```text
F_0(x): 전체 평균 이탈률 기반 예측
```

그런데 어떤 고객들은 실제로 이탈했는데 모델이 낮은 확률을 줬습니다.

```text
실제 이탈 = 1
예측 확률 = 0.2
```

이런 고객들에 대해서는 모델이 다음 단계에서 예측을 올려야 합니다.

반대로 어떤 고객들은 이탈하지 않았는데 모델이 높은 확률을 줬습니다.

```text
실제 이탈 = 0
예측 확률 = 0.8
```

이런 고객들에 대해서는 다음 단계에서 예측을 내려야 합니다.

Gradient Boosting은 logistic loss의 negative gradient를 계산해서:

```text
어떤 고객의 점수를 올려야 하는가?
어떤 고객의 점수를 내려야 하는가?
얼마나 조정해야 하는가?
```

를 학습합니다.

---

## 핵심 개념 7: Learning rate, 즉 shrinkage

Gradient Boosting에서 매우 중요한 하이퍼파라미터가 **learning rate**입니다.

업데이트를 그대로 더하면:

```text
F_m(x) = F_{m-1}(x) + h_m(x)
```

learning rate를 쓰면:

```text
F_m(x) = F_{m-1}(x) + ν × h_m(x)
```

여기서 `ν`는 보통 0과 1 사이의 작은 값입니다.

예를 들어:

```text
ν = 0.1
```

이면 새 tree의 보정을 10%만 반영합니다.

이렇게 하면 한 번에 크게 고치지 않고 조금씩 조심스럽게 고칩니다.

| learning rate | 특징 |
|---|---|
| 큼 | 빠르게 학습하지만 과적합 위험 증가 |
| 작음 | 천천히 학습하지만 일반화 성능이 좋아질 수 있음 |
| 너무 작음 | tree 수가 많이 필요하고 계산량 증가 |

---

## Learning rate와 tree 개수의 관계

Gradient Boosting에서는 보통 이런 trade-off가 있습니다.

```text
learning_rate 작게
→ 한 번에 조금씩 학습
→ tree를 더 많이 필요로 함
→ 보통 더 안정적일 수 있음
```

예:

| learning_rate | n_estimators | 특징 |
|---:|---:|---|
| 0.3 | 100 | 빠르지만 과적합 위험 |
| 0.1 | 500 | 실무에서 자주 쓰는 균형 |
| 0.03 | 2000 | 느리지만 더 세밀한 학습 가능 |

이것은 XGBoost나 LightGBM에서도 그대로 이어지는 핵심 감각입니다.

---

## 핵심 개념 8: Tree 크기는 interaction을 조절한다

Gradient Boosting Tree에서 각 tree의 깊이나 terminal node 수는 매우 중요합니다.

작은 tree는 단순한 패턴만 잡습니다.

```text
가입기간이 짧으면 이탈 위험 증가
```

깊은 tree는 변수 간 상호작용을 잡습니다.

```text
가입기간이 짧고,
월요금이 높고,
최근 사용량이 줄었으면
이탈 위험 증가
```

쉽게 말하면:

| tree 크기 | 의미 |
|---|---|
| stump, 깊이 1 | 변수 하나의 단순 효과 |
| 얕은 tree | 낮은 차수 interaction |
| 깊은 tree | 복잡한 interaction |
| 너무 깊은 tree | 과적합 위험 증가 |

Gradient Boosting에서는 보통 **얕은 tree를 많이 쌓는 방식**이 강력합니다.

---

## Random Forest와 Gradient Boosting의 핵심 차이

| 구분 | Random Forest | Gradient Boosting |
|---|---|---|
| 학습 방식 | 여러 tree를 독립적으로 학습 | tree를 순서대로 학습 |
| 각 tree의 역할 | 전체 문제를 각각 예측 | 이전 모델의 부족한 부분 보정 |
| 결합 방식 | 평균 또는 다수결 | 누적합 |
| 주된 효과 | variance 감소 | loss를 직접 줄이는 방향으로 성능 개선 |
| tree 크기 | 보통 깊은 tree | 보통 얕은 tree |
| 병렬화 | 쉬움 | 순차적이라 상대적으로 어려움 |
| 튜닝 민감도 | 비교적 낮음 | 상대적으로 높음 |
| 과적합 | 상대적으로 안정적 | learning rate, tree 수, depth에 민감 |

짧게 정리하면:

> Random Forest는 **여러 전문가에게 독립적으로 물어보고 평균내는 방식**입니다.  
> Gradient Boosting은 **한 사람이 계속 틀린 문제를 오답노트로 고쳐 가는 방식**입니다.

---

## 해석 도구: 변수 중요도와 Partial Dependence Plot

Gradient Boosting은 단일 decision tree보다 해석이 어렵습니다.

하지만 TreeBoost 모델을 해석하기 위한 도구도 있습니다.

| 도구 | 의미 |
|---|---|
| **Relative variable importance** | 어떤 변수가 예측 함수 변화에 많이 기여했는가 |
| **Partial dependence plot, PDP** | 특정 변수가 바뀔 때 모델 예측이 어떻게 변하는가 |

예를 들어 고객 이탈 모델에서 PDP를 보면 이런 것을 확인할 수 있습니다.

```text
가입기간이 3개월 미만일 때 이탈 위험이 급격히 높아진다.
월요금이 8만 원을 넘으면 이탈 위험이 증가한다.
최근 사용량이 0에 가까워지면 이탈 위험이 상승한다.
```

다만 PDP도 인과관계를 보여주는 것은 아닙니다. 모델이 학습한 예측 패턴을 보여주는 도구입니다.

---

## 이 논문이 말하는 진짜 메시지

표면적으로는 새로운 boosting 알고리즘 논문처럼 보이지만, 더 깊은 메시지는 이것입니다.

**머신러닝 모델 학습은 “손실함수를 줄이는 함수 최적화 문제”로 볼 수 있다.**

기존에는 boosting이 약한 분류기를 여러 개 합치는 특별한 기법처럼 보였습니다. Friedman은 boosting을 더 일반적인 최적화 프레임워크로 정리했습니다.

즉:

```text
어떤 loss를 줄이고 싶은가?
그 loss의 gradient는 무엇인가?
그 gradient를 어떤 base learner로 근사할 것인가?
얼마나 작게 업데이트할 것인가?
몇 번 반복할 것인가?
```

이 질문들이 Gradient Boosting의 핵심입니다.

---

## 입문자가 꼭 잡아야 할 용어

| 용어 | 의미 |
|---|---|
| Loss function | 모델이 얼마나 틀렸는지를 측정하는 함수 |
| Residual | 실제값과 예측값의 차이, `y - F(x)` |
| Pseudo-residual | 일반 loss에서의 negative gradient |
| Base learner | 매 단계에서 추가되는 작은 모델 |
| Stagewise additive modeling | 작은 구성요소를 하나씩 더해 가는 방식 |
| Learning rate | 각 단계의 보정을 얼마나 작게 반영할지 정하는 값 |

---

## Gradient Boosting의 장점

1. **예측 성능이 강하다** — tabular data에서 매우 강력합니다.
2. **다양한 loss를 쓸 수 있다** — squared error뿐 아니라 absolute error, Huber loss, logistic loss 등 다양한 목적함수를 직접 최적화할 수 있습니다.
3. **상대적으로 해석 도구를 붙일 수 있다** — 변수 중요도, PDP, SHAP 같은 도구와 잘 결합됩니다.
4. **약한 모델을 많이 쌓아 강한 모델을 만든다** — 단순한 tree를 수백 개 더해 유연한 함수 근사가 가능합니다.

---

## Gradient Boosting의 한계

1. **튜닝에 민감하다** — `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf` 등이 중요합니다.
2. **과적합될 수 있다** — tree를 너무 많이 쌓거나 learning rate가 너무 크거나 tree가 너무 깊으면 위험합니다.
3. **학습이 순차적이다** — 이전 tree의 residual을 알아야 다음 tree를 학습할 수 있으므로 완전히 병렬화하기 어렵습니다.
4. **이상치에 민감할 수 있다** — squared error loss를 쓰면 큰 이상치가 강하게 영향을 줍니다.

---

## 실무적으로 어떻게 이해하면 좋나?

1. 먼저 단순 모델과 Random Forest를 baseline으로 둡니다.

```text
Linear / Logistic Regression
→ Random Forest
→ Gradient Boosting / XGBoost / LightGBM
```

2. learning rate와 tree 개수를 같이 봅니다.

```text
learning_rate ↓
n_estimators ↑
```

3. tree depth는 interaction 복잡도를 조절합니다.

얕은 tree는 단순한 패턴, 깊은 tree는 복잡한 패턴을 잡습니다.

4. validation set을 반드시 둡니다.

```text
train loss 계속 감소
validation loss 어느 순간 증가
→ 과적합 시작
```

그래서 early stopping이 중요합니다.

---

## 이전 논문들과 연결해서 이해하기

| 이전 논문 | 이번 논문과의 연결 |
|---|---|
| **Tidy Data** | Gradient Boosting에 넣을 tabular feature를 정리해야 함 |
| **Two Cultures** | Gradient Boosting은 algorithmic modeling culture의 대표 모델 |
| **Model Evaluation** | boosting은 validation, early stopping, test set 보호가 중요 |
| **Datasheets** | 학습 데이터의 편향과 라벨 정의가 모델에 그대로 반영됨 |
| **Model Cards** | boosting 모델도 사용 범위, 집단별 성능, 한계를 문서화해야 함 |
| **Random Forests** | 둘 다 tree ensemble이지만, RF는 독립 평균, GBM은 순차 보정 |

---

## 이 논문을 공부할 때 가장 중요한 질문

이 논문을 읽으면서 계속 물어야 할 질문은 하나입니다.

**“지금 다음 tree가 무엇을 학습하고 있는가?”**

정답은:

> 현재 모델의 손실을 줄이는 방향, 즉 negative gradient를 학습하고 있다.

squared error에서는 그게 residual입니다. classification이나 robust loss에서는 pseudo-residual입니다.

---

## 오늘 공부용 요약

**논문명:** Greedy Function Approximation: A Gradient Boosting Machine  
**저자:** Jerome H. Friedman  
**핵심 결론:** 예측 모델을 함수 `F(x)`로 보고, 손실함수를 줄이는 방향의 작은 모델들을 순차적으로 더하면 강력한 예측 모델을 만들 수 있다.  
**왜 중요함:** boosting을 단순한 앙상블 기법이 아니라, 함수 공간에서의 gradient descent로 일반화했다. 현대 GBM, XGBoost, LightGBM, CatBoost를 이해하는 핵심 기반이다.  
**입문자가 배울 점:** Gradient Boosting은 “이전 모델의 residual을 다음 모델이 학습한다”로 시작해 이해할 수 있지만, 더 정확히는 “현재 손실함수의 negative gradient를 다음 모델이 학습한다”는 방법이다.  
**가장 중요한 문장:** Gradient Boosting은 현재 모델이 틀린 방향을 gradient로 계산하고, 작은 tree를 더해 그 방향으로 예측 함수를 조금씩 고쳐 가는 알고리즘이다.


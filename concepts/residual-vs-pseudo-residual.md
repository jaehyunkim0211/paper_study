# Residual vs Pseudo-residual, Gradient Descent vs Gradient Boosting

## 결론부터

**residual은 “얼마나 틀렸는가”이고, pseudo-residual은 “손실함수를 줄이려면 예측값을 어느 방향으로 얼마나 움직여야 하는가”입니다.**

MSE 회귀에서는 둘이 같아서 헷갈립니다.  
하지만 loss function이 바뀌면 둘은 달라집니다.

---

## 1. Residual은 그냥 오차다

가장 단순한 회귀 문제를 생각해봅시다.

```text
실제값 y = 10
현재 예측값 ŷ = 7
```

그러면 residual은:

```text
residual = 실제값 - 예측값
         = 10 - 7
         = 3
```

의미는 간단합니다.

> 현재 예측이 실제보다 3만큼 낮다.  
> 다음 모델은 예측을 3만큼 올리는 방향으로 도와주면 좋다.

반대로:

```text
실제값 y = 10
현재 예측값 ŷ = 13
```

이면:

```text
residual = 10 - 13 = -3
```

의미는:

> 현재 예측이 실제보다 3만큼 높다.  
> 다음 모델은 예측을 3만큼 낮추는 방향으로 도와주면 좋다.

즉, residual은 아주 직관적으로:

```text
정답 - 현재 예측
```

입니다.

---

## 2. 그런데 왜 pseudo-residual이 필요한가?

residual은 회귀 문제, 특히 **MSE**, 즉 평균제곱오차에서는 잘 맞습니다.

MSE loss는 이런 형태입니다.

```text
loss = (실제값 - 예측값)^2
```

이때는 예측값을 고치는 가장 좋은 방향이 residual과 같습니다.

```text
실제값 10, 예측값 7
→ residual = +3
→ 예측값을 올려야 함
```

그래서 MSE에서는:

```text
pseudo-residual = residual
```

입니다.

하지만 모든 문제가 MSE를 쓰는 것은 아닙니다.

분류에서는 logistic loss를 쓰고, 이상치에 강한 회귀에서는 absolute error나 Huber loss를 쓸 수 있습니다. 이때는 단순한 `실제값 - 예측값`이 최적의 수정 방향이 아닐 수 있습니다.

그래서 Gradient Boosting은 더 일반적으로 이렇게 묻습니다.

> “현재 loss를 줄이려면, 각 데이터의 예측값을 어느 방향으로 움직여야 하지?”

그 답이 바로 **pseudo-residual**입니다.

---

## 3. Pseudo-residual은 “loss를 줄이는 방향”이다

수식으로는 이렇게 씁니다.

```text
pseudo-residual = - loss의 gradient
```

조금 더 정확히는:

```text
r_i = - ∂L(y_i, F(x_i)) / ∂F(x_i)
```

여기서:

| 기호 | 의미 |
|---|---|
| `y_i` | i번째 데이터의 실제값 |
| `F(x_i)` | 현재 모델의 예측값 |
| `L` | loss function |
| `r_i` | i번째 데이터의 pseudo-residual |

쉽게 말하면:

> pseudo-residual은 “현재 예측값을 어느 쪽으로 움직이면 loss가 줄어드는가?”를 알려주는 값입니다.

그래서 pseudo-residual은 진짜 residual처럼 생겼지만, 실제로는 loss function에서 나온 값입니다. 그래서 이름에 **pseudo**, 즉 “가짜/유사”가 붙습니다.

---

## 4. 예시 1: MSE에서는 residual과 pseudo-residual이 같다

```text
실제값 y = 10
현재 예측값 F(x) = 7
```

residual:

```text
10 - 7 = 3
```

MSE loss 기준 pseudo-residual도:

```text
3
```

따라서 다음 tree는 `+3`을 예측하도록 학습합니다.

```text
현재 예측 7
다음 tree가 +3 보정
최종적으로 10에 가까워짐
```

이 경우에는 둘이 같습니다.

| 개념 | 값 |
|---|---:|
| residual | 3 |
| pseudo-residual | 3 |

---

## 5. 예시 2: Absolute Error에서는 둘이 다르다

이번에는 loss를 MSE가 아니라 absolute error로 바꿔봅시다.

```text
loss = |실제값 - 예측값|
```

다시 같은 예시입니다.

```text
실제값 y = 10
현재 예측값 F(x) = 7
```

residual은 여전히:

```text
10 - 7 = 3
```

그런데 absolute error의 pseudo-residual은 대략:

```text
+1
```

입니다.

왜냐하면 absolute error는 “얼마나 크게 틀렸는지”보다 “어느 방향으로 틀렸는지”가 더 중요하게 작동합니다.

```text
예측이 낮다 → 올려라 → +1
예측이 높다 → 내려라 → -1
```

예를 들어:

| 실제값 | 예측값 | residual | absolute loss 기준 pseudo-residual |
|---:|---:|---:|---:|
| 10 | 7 | +3 | +1 |
| 10 | 2 | +8 | +1 |
| 10 | 13 | -3 | -1 |
| 10 | 100 | -90 | -1 |

여기서 차이가 보입니다.

residual은 오차의 크기를 그대로 반영합니다.

```text
+3, +8, -3, -90
```

하지만 absolute error의 pseudo-residual은 방향만 강하게 반영합니다.

```text
+1, +1, -1, -1
```

이런 방식은 이상치에 덜 민감합니다. 예측이 90만큼 틀렸다고 해서 다음 모델이 그 한 점에 지나치게 끌려가지 않기 때문입니다.

---

## 6. 예시 3: Huber Loss에서는 큰 residual을 적당히 잘라낸다

Huber loss는 MSE와 absolute error의 절충입니다.

작은 오차에는 MSE처럼 행동하고, 큰 오차에는 absolute error처럼 행동합니다.

예를 들어 residual이 작으면:

```text
residual = 3
pseudo-residual ≈ 3
```

하지만 residual이 너무 크면:

```text
residual = 100
pseudo-residual ≈ 적당한 값으로 제한됨
```

즉, Huber loss는 이렇게 말합니다.

> 적당히 틀린 데이터는 열심히 고치자.  
> 그런데 너무 심하게 튀는 이상치 하나에 모델 전체가 끌려가지는 말자.

그래서 robust regression에 유용합니다.

---

## 7. 예시 4: 분류에서는 residual 개념이 더 애매하다

이진분류를 생각해봅시다.

```text
y = 1  # 실제로 이탈함
현재 예측 확률 p = 0.2
```

이 모델은 실제 이탈 고객에게 이탈 확률을 낮게 줬습니다.  
그러면 다음 모델은 이 고객의 점수를 올려야 합니다.

logistic loss 기준 pseudo-residual은 보통 이런 형태입니다.

```text
pseudo-residual = y - p
```

그러면:

```text
1 - 0.2 = 0.8
```

의미:

> 이 고객의 이탈 점수를 꽤 많이 올려라.

다른 예시를 봅시다.

```text
y = 1
p = 0.9
```

이미 잘 맞추고 있습니다.

```text
pseudo-residual = 1 - 0.9 = 0.1
```

의미:

> 이미 잘 맞추고 있으니 조금만 올려라.

반대로:

```text
y = 0
p = 0.8
```

실제로는 이탈하지 않았는데 이탈 확률을 높게 줬습니다.

```text
pseudo-residual = 0 - 0.8 = -0.8
```

의미:

> 이 고객의 이탈 점수를 꽤 많이 내려라.

정리하면:

| 실제값 y | 예측 확률 p | pseudo-residual `y - p` | 의미 |
|---:|---:|---:|---|
| 1 | 0.2 | +0.8 | 이탈 점수를 많이 올려라 |
| 1 | 0.9 | +0.1 | 조금만 올려라 |
| 0 | 0.8 | -0.8 | 이탈 점수를 많이 내려라 |
| 0 | 0.1 | -0.1 | 조금만 내려라 |

분류에서는 단순히:

```text
실제 class - 예측 class
```

를 쓰면 정보가 너무 거칠어집니다.

예를 들어 둘 다 오답이라고 해도:

```text
실제 1, 예측 확률 0.49
실제 1, 예측 확률 0.01
```

은 심각도가 다릅니다.

pseudo-residual은 이런 “얼마나 자신 있게 틀렸는가”까지 반영합니다.

---

## 8. residual vs pseudo-residual 한 번에 정리

| 구분 | residual | pseudo-residual |
|---|---|---|
| 뜻 | 실제값 - 예측값 | loss를 줄이는 방향 |
| 기준 | 실제값과 예측값의 차이 | loss function의 gradient |
| 주로 쓰이는 설명 | 회귀, 특히 MSE | 일반적인 Gradient Boosting |
| MSE에서 | pseudo-residual과 같음 | residual과 같음 |
| logistic loss에서 | 단순 residual로 보기 어려움 | `y - p` 형태로 나타남 |
| absolute error에서 | 오차 크기 그대로 | 방향 중심, 보통 `+1` 또는 `-1` |
| 핵심 질문 | “얼마나 틀렸나?” | “loss를 줄이려면 어느 쪽으로 움직이나?” |

가장 쉽게 말하면:

```text
residual = 정답과 예측의 차이
pseudo-residual = 다음 모델이 따라가야 할 수정 방향
```

---

## 9. Gradient라는 단어는 Gradient Descent와 같은 의미인가?

네, **같은 핵심 의미**입니다.

둘 다에서 gradient는:

> loss가 어느 방향으로 얼마나 빠르게 증가하는지를 나타내는 기울기

입니다.

그래서 loss를 줄이려면 gradient의 반대 방향으로 움직입니다.

```text
negative gradient = loss가 줄어드는 방향
```

이게 공통점입니다.

---

## 10. Gradient Descent에서의 gradient

Gradient Descent는 보통 모델의 **파라미터**를 업데이트합니다.

예를 들어 선형회귀라면:

```text
ŷ = w₁x₁ + w₂x₂ + b
```

여기서 파라미터는:

```text
w₁, w₂, b
```

입니다.

Gradient Descent는 이렇게 합니다.

```text
현재 파라미터가 있다.
loss를 계산한다.
각 파라미터를 어느 방향으로 바꾸면 loss가 줄어드는지 gradient를 구한다.
그 반대 방향으로 파라미터를 조금 움직인다.
```

수식으로는:

```text
w_new = w_old - learning_rate × gradient
```

즉, Gradient Descent에서는:

```text
파라미터를 직접 움직인다.
```

---

## 11. Gradient Boosting에서의 gradient

Gradient Boosting은 조금 다릅니다.

Gradient Boosting은 파라미터 `w`를 직접 조금씩 움직이는 방식이라기보다, **예측 함수 `F(x)` 자체를 조금씩 고치는 방식**입니다.

현재 모델이 있습니다.

```text
F_0(x)
```

그 모델이 각 데이터에 대해 예측을 합니다.

```text
F_0(x₁), F_0(x₂), F_0(x₃), ...
```

이제 각 데이터마다 묻습니다.

> 이 데이터의 예측값을 어느 방향으로 움직이면 loss가 줄어들까?

그 답이 pseudo-residual입니다.

그다음 작은 tree를 학습합니다.

```text
x → pseudo-residual
```

즉, 다음 tree는 원래 정답 `y`를 직접 맞추는 게 아니라, **현재 모델이 고쳐야 할 방향**을 맞춥니다.

그리고 기존 모델에 더합니다.

```text
F_1(x) = F_0(x) + learning_rate × tree_1(x)
```

다시 pseudo-residual을 계산하고, 또 다음 tree를 더합니다.

```text
F_2(x) = F_1(x) + learning_rate × tree_2(x)
```

이 과정을 반복합니다.

즉, Gradient Boosting에서는:

```text
예측 함수 자체를 negative gradient 방향으로 조금씩 고친다.
```

---

## 12. 둘의 차이를 비유로 설명하면

### Gradient Descent

산을 내려간다고 생각해봅시다.

현재 위치가 있습니다.

```text
현재 위치 = 모델 파라미터
```

기울기를 봅니다.

```text
이쪽으로 가면 올라간다.
저쪽으로 가면 내려간다.
```

그래서 내려가는 방향으로 직접 한 걸음 움직입니다.

```text
파라미터를 직접 수정
```

### Gradient Boosting

Gradient Boosting은 조금 다릅니다.

현재 모델이 여러 데이터에서 틀리고 있습니다.

각 데이터마다 메모를 남깁니다.

```text
1번 데이터: 예측을 올려야 함
2번 데이터: 예측을 조금 내려야 함
3번 데이터: 예측을 많이 올려야 함
4번 데이터: 거의 그대로 둬도 됨
```

이 메모가 pseudo-residual입니다.

그다음 작은 decision tree에게 말합니다.

> “입력 x를 보고, 이 메모를 예측해봐.”

tree가 그 메모를 어느 정도 배웁니다.

그 tree를 기존 모델에 더합니다.

```text
기존 모델 + 수정용 tree
```

즉, Gradient Boosting은 직접 파라미터 하나하나를 움직이기보다:

> “현재 모델이 고쳐야 할 방향을 새 모델로 학습해서 더하는 방식”

입니다.

---

## 13. 핵심 차이 표

| 구분 | Gradient Descent | Gradient Boosting |
|---|---|---|
| gradient의 의미 | loss를 줄이기 위한 기울기 | loss를 줄이기 위한 기울기 |
| gradient 개념은 같은가? | 예 | 예 |
| 무엇을 업데이트하나? | 파라미터 `w` | 예측 함수 `F(x)` |
| 업데이트 방식 | 파라미터를 직접 조금 이동 | 작은 모델을 추가 |
| 예시 | 선형회귀의 weight 조정 | decision tree를 하나씩 추가 |
| 수식 느낌 | `w ← w - η∇L` | `F ← F + ηh` |
| 핵심 | 파라미터 공간에서 내려감 | 함수 공간에서 내려감 |

정확히 말하면:

> Gradient Boosting은 Gradient Descent의 아이디어를 **함수 공간**에 적용한 것이다.

---

## 14. 가장 중요한 직관

Gradient Boosting을 이해할 때는 이렇게 생각하면 됩니다.

```text
현재 모델이 있다.
각 데이터마다 "어느 방향으로 예측을 고쳐야 loss가 줄어드는지" 계산한다.
그 방향이 pseudo-residual이다.
다음 tree는 그 pseudo-residual을 맞추도록 학습한다.
그 tree를 기존 모델에 조금 더한다.
반복한다.
```

MSE에서는 이 pseudo-residual이 그냥 residual입니다.

```text
실제값 - 예측값
```

그래서 입문 설명에서:

> Gradient Boosting은 이전 모델의 residual을 학습한다

라고 말합니다.

하지만 더 정확히는:

> Gradient Boosting은 현재 loss의 negative gradient, 즉 pseudo-residual을 학습한다

입니다.

---

## 15. 한 문장으로 최종 정리

**Residual은 “정답과 예측의 차이”이고, pseudo-residual은 “선택한 loss function을 줄이기 위해 현재 예측값이 움직여야 할 방향”입니다. MSE에서는 둘이 같지만, logistic loss, absolute error, Huber loss처럼 다른 loss를 쓰면 pseudo-residual은 residual과 달라집니다. Gradient Boosting의 Gradient는 Gradient Descent의 Gradient와 같은 의미로, loss가 줄어드는 방향을 찾는 데 쓰입니다. 차이는 Gradient Descent는 파라미터를 직접 움직이고, Gradient Boosting은 그 방향을 작은 모델이 학습하게 해서 예측 함수를 조금씩 고친다는 점입니다.**


# The Matrix Calculus You Need For Deep Learning — Terence Parr & Jeremy Howard

## 결론부터

이 글의 결론은 이겁니다.

**딥러닝의 학습은 결국 “loss를 줄이기 위해 weight와 bias를 어느 방향으로 얼마나 바꿀 것인가?”를 계산하는 일이고, 그 계산의 언어가 matrix calculus다.**

쉽게 말하면:

> 딥러닝 모델은 수많은 숫자, 즉 weight와 bias로 이루어져 있다.  
> 학습은 loss를 줄이는 방향으로 그 숫자들을 조금씩 바꾸는 과정이다.  
> 그 “바꿀 방향”을 계산하려면 벡터와 행렬에 대한 미분을 알아야 한다.

이 글은 Terence Parr와 Jeremy Howard가 쓴 튜토리얼 성격의 arXiv 글입니다. 저자들은 이 글의 목적을 “deep neural network의 training을 이해하는 데 필요한 matrix calculus를 설명하는 것”이라고 밝히고, 미적분 1 수준 이상의 수학을 거의 가정하지 않는다고 설명합니다. 또한 실무에서 딥러닝을 시작하기 전에 반드시 알아야 하는 내용이라기보다는, 신경망의 기본을 이미 알고 있고 그 내부 수학을 더 깊게 이해하고 싶은 사람을 위한 글이라고 말합니다.

---

## 이 글을 한 문장으로 요약하면

**Matrix calculus는 backpropagation이 각 weight에 대해 “loss를 줄이는 방향”을 어떻게 계산하는지 이해하기 위한 도구다.**

우리가 앞에서 본 딥러닝 학습 흐름은 이렇습니다.

```text
입력 x
→ 모델 f(x)
→ 예측값 ŷ
→ loss 계산
→ gradient 계산
→ weight 업데이트
```

여기서 핵심은 이 부분입니다.

```text
gradient 계산
```

딥러닝 프레임워크는 이걸 자동으로 해줍니다.

```python
loss.backward()
optimizer.step()
```

하지만 내부적으로는 다음을 계산하고 있습니다.

```text
∂Loss / ∂W
∂Loss / ∂b
```

즉:

> weight W를 조금 바꾸면 loss가 어떻게 변하는가?  
> bias b를 조금 바꾸면 loss가 어떻게 변하는가?

를 계산합니다.

이걸 이해하려면 scalar calculus만으로는 부족합니다.  
왜냐하면 딥러닝의 weight는 보통 숫자 하나가 아니라 **벡터, 행렬, 텐서**이기 때문입니다.

---

## 왜 matrix calculus가 필요한가?

간단한 선형회귀라면 weight가 하나일 수 있습니다.

```text
ŷ = wx + b
```

이때는 scalar 미분만으로도 충분합니다.

```text
∂Loss / ∂w
```

하지만 딥러닝에서는 보통 이렇게 됩니다.

```text
X: 입력 행렬
W: weight 행렬
b: bias 벡터
Z = XW + b
A = activation(Z)
Loss = L(A, y)
```

이제 우리가 구해야 하는 것은 숫자 하나에 대한 미분이 아닙니다.

```text
∂Loss / ∂W
∂Loss / ∂b
∂Loss / ∂X
```

여기서 `W`는 행렬입니다.

예를 들어:

```text
W.shape = (784, 128)
```

이라면 `∂Loss / ∂W`도 같은 shape를 가져야 합니다.

```text
∂Loss / ∂W.shape = (784, 128)
```

왜냐하면 `W`의 각 원소마다 loss에 대한 영향이 다르기 때문입니다.

즉, matrix calculus의 실용적 목적은 이것입니다.

> 각 parameter와 같은 shape의 gradient를 계산해서, optimizer가 그 parameter를 업데이트할 수 있게 하는 것.

---

## 핵심 개념 1: Scalar derivative

가장 기본은 scalar derivative입니다.

```text
y = x²
dy/dx = 2x
```

의미는:

> x가 조금 변할 때 y가 얼마나 변하는가?

입니다.

예를 들어:

```text
x = 3
dy/dx = 6
```

이면, x를 아주 조금 증가시키면 y는 그보다 약 6배 빠르게 증가합니다.

딥러닝에서도 결국 이 생각이 확장됩니다.

```text
weight 하나가 조금 변하면 loss가 얼마나 변하는가?
```

다만 weight가 수백만 개이므로, 이걸 한꺼번에 계산하려면 vector와 matrix 미분이 필요합니다.

---

## 핵심 개념 2: Partial derivative

변수가 여러 개일 때는 partial derivative, 즉 편미분을 씁니다.

예를 들어:

```text
f(x, y) = 3x²y
```

가 있다고 합시다.

x에 대해 미분할 때는 y를 상수처럼 봅니다.

```text
∂f/∂x = 6xy
```

y에 대해 미분할 때는 x를 상수처럼 봅니다.

```text
∂f/∂y = 3x²
```

딥러닝에서는 weight가 많습니다.

```text
w1, w2, w3, ..., wn
```

loss는 이 모든 weight의 함수입니다.

```text
Loss = L(w1, w2, ..., wn)
```

그래서 각 weight에 대해 편미분을 구합니다.

```text
∂L/∂w1
∂L/∂w2
∂L/∂w3
...
```

이 편미분들을 모은 것이 gradient입니다.

---

## 핵심 개념 3: Gradient

### 결론부터

**Gradient는 scalar output을 vector input에 대해 미분한 결과입니다.**

예를 들어 loss는 scalar입니다.

```text
Loss = 하나의 숫자
```

weight vector가 있다고 합시다.

```text
w = [w1, w2, w3]
```

그러면 loss의 gradient는:

```text
∇w L = [∂L/∂w1, ∂L/∂w2, ∂L/∂w3]
```

입니다.

의미는:

> 각 weight를 조금 바꾸면 loss가 얼마나 변하는가?

입니다.

Gradient descent는 이 gradient의 반대 방향으로 움직입니다.

```text
w_new = w_old - learning_rate × gradient
```

왜 반대 방향일까요?

gradient는 loss가 가장 빠르게 증가하는 방향입니다.

그러므로 loss를 줄이려면:

```text
negative gradient 방향
```

으로 가야 합니다.

---

## 핵심 개념 4: Jacobian

### 결론부터

**Jacobian은 vector output을 vector input에 대해 미분한 결과입니다.**

예를 들어 입력 vector가 있습니다.

```text
x = [x1, x2]
```

출력도 vector입니다.

```text
y = [y1, y2, y3]
```

그러면 Jacobian은 모든 조합의 편미분을 모은 행렬입니다.

```text
J =
[ ∂y1/∂x1   ∂y1/∂x2
  ∂y2/∂x1   ∂y2/∂x2
  ∂y3/∂x1   ∂y3/∂x2 ]
```

딥러닝에서 Jacobian이 중요한 이유는, layer의 출력도 vector이고 입력도 vector인 경우가 많기 때문입니다.

```text
z = Wx + b
a = ReLU(z)
```

여기서 `z`도 vector, `a`도 vector입니다.

그러면:

```text
∂a / ∂z
```

는 Jacobian입니다.

---

## 핵심 개념 5: Shape가 제일 중요하다

Matrix calculus에서 초보자가 가장 많이 헷갈리는 것은 수식 자체보다 **shape**입니다.

딥러닝에서는 항상 이렇게 확인해야 합니다.

```text
이 값은 scalar인가?
vector인가?
matrix인가?
tensor인가?
shape는 무엇인가?
```

예를 들어 mini-batch notation을 이렇게 잡아봅시다.

```text
X.shape = (N, D)
W.shape = (D, H)
b.shape = (H,)
Z = XW + b
Z.shape = (N, H)
```

여기서:

| 기호 | 의미 | shape |
|---|---|---|
| `N` | batch size | |
| `D` | input feature 수 | |
| `H` | hidden unit 수 | |
| `X` | 입력 batch | `(N, D)` |
| `W` | weight matrix | `(D, H)` |
| `b` | bias | `(H,)` |
| `Z` | pre-activation | `(N, H)` |

forward는 이렇게 됩니다.

```text
Z = XW + b
```

backward에서는 다음 gradient들이 필요합니다.

```text
∂L/∂W.shape = (D, H)
∂L/∂b.shape = (H,)
∂L/∂X.shape = (N, D)
```

즉, gradient는 업데이트할 대상과 같은 shape를 갖습니다.

```text
W를 업데이트하려면 W와 같은 shape의 gradient가 필요하다.
b를 업데이트하려면 b와 같은 shape의 gradient가 필요하다.
```

이 감각이 중요합니다.

---

## 핵심 개념 6: Chain Rule

### 결론부터

**Backpropagation은 chain rule을 neural network에 적용한 것이다.**

단순한 합성함수를 봅시다.

```text
y = f(g(x))
```

이때 미분은:

```text
dy/dx = dy/dg × dg/dx
```

입니다.

딥러닝도 똑같습니다.

예를 들어 한 뉴런이 있습니다.

```text
z = wᵀx + b
a = σ(z)
L = loss(a, y)
```

loss `L`은 `a`에 의존하고, `a`는 `z`에 의존하고, `z`는 `w`에 의존합니다.

그러면:

```text
∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w
```

이게 backpropagation의 핵심입니다.

즉:

```text
loss에서 시작해서
출력층 → 이전층 → 더 이전층
순서로 chain rule을 적용한다.
```

---

## 쉬운 예시: 뉴런 하나의 gradient

가장 단순한 뉴런 하나를 생각해봅시다.

```text
z = wᵀx + b
a = σ(z)
L = loss(a, y)
```

여기서:

| 기호 | 의미 |
|---|---|
| `x` | 입력 vector |
| `w` | weight vector |
| `b` | bias scalar |
| `z` | activation function 전 값 |
| `σ` | sigmoid, ReLU 같은 activation function |
| `a` | activation output |
| `L` | loss |

우리는 `w`와 `b`를 업데이트해야 합니다.

그러려면:

```text
∂L/∂w
∂L/∂b
```

가 필요합니다.

chain rule로 보면:

```text
∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w
```

그런데:

```text
z = wᵀx + b
```

이므로:

```text
∂z/∂w = x
∂z/∂b = 1
```

따라서:

```text
∂L/∂w = δ x
∂L/∂b = δ
```

여기서:

```text
δ = ∂L/∂z
```

입니다.

딥러닝 backprop에서 자주 나오는 delta가 바로 이런 의미입니다.

> 현재 뉴런의 pre-activation `z`가 loss에 얼마나 영향을 주는가?

---

## 더 구체적인 숫자 예시

입력이 두 개라고 합시다.

```text
x = [2, 3]
w = [0.5, -1]
b = 1
```

먼저 forward를 계산합니다.

```text
z = wᵀx + b
  = 0.5×2 + (-1)×3 + 1
  = 1 - 3 + 1
  = -1
```

activation은 일단 단순화를 위해 identity라고 합시다.

```text
a = z = -1
```

정답이:

```text
y = 2
```

이고 loss를 squared error로 둡니다.

```text
L = 1/2 × (a - y)²
```

그러면:

```text
a - y = -1 - 2 = -3
```

따라서:

```text
∂L/∂a = a - y = -3
```

activation이 identity라서:

```text
∂a/∂z = 1
```

그러면:

```text
δ = ∂L/∂z = -3 × 1 = -3
```

이제 weight gradient는:

```text
∂L/∂w = δx
       = -3 × [2, 3]
       = [-6, -9]
```

bias gradient는:

```text
∂L/∂b = δ = -3
```

이제 gradient descent는:

```text
w_new = w - η ∂L/∂w
b_new = b - η ∂L/∂b
```

로 업데이트합니다.

예를 들어 learning rate `η = 0.1`이면:

```text
w_new = [0.5, -1] - 0.1 × [-6, -9]
      = [0.5, -1] + [0.6, 0.9]
      = [1.1, -0.1]

b_new = 1 - 0.1 × (-3)
      = 1.3
```

이게 딥러닝 학습의 가장 작은 단위입니다.

---

## 핵심 개념 7: Matrix form으로 보면 훨씬 효율적이다

딥러닝에서는 뉴런 하나씩 계산하지 않습니다.

mini-batch 전체와 layer 전체를 한꺼번에 계산합니다.

예를 들어:

```text
X.shape = (N, D)
W.shape = (D, H)
b.shape = (H,)
Z = XW + b
Z.shape = (N, H)
```

여기서 `N`은 batch size, `D`는 input dimension, `H`는 hidden unit 수입니다.

forward:

```text
Z = XW + b
A = σ(Z)
L = loss(A, Y)
```

backward에서 위쪽 gradient가 들어왔다고 합시다.

```text
G = ∂L/∂Z
G.shape = (N, H)
```

그러면 중요한 gradient는 다음과 같습니다.

```text
∂L/∂W = XᵀG
```

shape를 확인해봅시다.

```text
Xᵀ.shape = (D, N)
G.shape  = (N, H)

XᵀG.shape = (D, H)
```

`W.shape`와 같습니다.

```text
W.shape = (D, H)
```

bias gradient는 batch 방향으로 합칩니다.

```text
∂L/∂b = sum over batch of G
```

shape는:

```text
(H,)
```

입력으로 전달할 gradient는:

```text
∂L/∂X = GWᵀ
```

shape는:

```text
G.shape  = (N, H)
Wᵀ.shape = (H, D)

GWᵀ.shape = (N, D)
```

`X.shape`와 같습니다.

이 세 식은 딥러닝에서 정말 중요합니다.

```text
dW = Xᵀ dZ
db = sum(dZ, axis=batch)
dX = dZ Wᵀ
```

이걸 이해하면 linear layer의 backprop이 보입니다.

---

## 핵심 개념 8: Element-wise activation의 derivative

Activation function은 보통 element-wise로 적용됩니다.

예를 들어 ReLU:

```text
a = ReLU(z)
```

`z`가 vector이면:

```text
a_i = ReLU(z_i)
```

각 원소가 자기 자신에만 의존합니다.

```text
a1은 z1에만 의존
a2는 z2에만 의존
a3은 z3에만 의존
```

그래서 Jacobian은 대각행렬이 됩니다.

예를 들어:

```text
a = σ(z)
```

이면:

```text
∂a/∂z =
[ σ'(z1)    0       0
   0      σ'(z2)    0
   0        0     σ'(z3) ]
```

이게 의미하는 것은 간단합니다.

> element-wise activation에서는 각 원소의 gradient도 원소별로 계산된다.

실무에서는 대각행렬을 실제로 만들지 않고, 그냥 element-wise multiplication을 합니다.

```text
dZ = dA * σ'(Z)
```

예를 들어 ReLU라면:

```text
ReLU'(z) = 1 if z > 0 else 0
```

그래서:

```text
dZ = dA * (Z > 0)
```

입니다.

이것이 PyTorch나 TensorFlow 내부에서 자동미분으로 처리되는 계산입니다.

---

## 핵심 개념 9: Bias gradient는 왜 sum인가?

이 부분은 자주 헷갈립니다.

forward에서 bias는 batch의 모든 row에 더해집니다.

```text
Z = XW + b
```

예를 들어:

```text
XW.shape = (N, H)
b.shape  = (H,)
```

`b`는 각 sample에 broadcast되어 더해집니다.

```text
Z[0, :] = XW[0, :] + b
Z[1, :] = XW[1, :] + b
...
Z[N-1, :] = XW[N-1, :] + b
```

즉, 같은 `b`가 batch의 모든 sample에 사용됩니다.

그러므로 loss에 대한 `b`의 gradient는 각 sample에서 온 gradient를 모두 더해야 합니다.

```text
db = sum(dZ over batch)
```

예를 들어 batch size가 3이고 hidden unit이 2개라면:

```text
dZ =
[ [1, 2],
  [3, 4],
  [5, 6] ]
```

그러면:

```text
db = [1+3+5, 2+4+6]
   = [9, 12]
```

이게 bias gradient입니다.

---

## 핵심 개념 10: Backpropagation은 “local gradient × upstream gradient”다

딥러닝 backprop을 가장 쉽게 말하면 이겁니다.

```text
내가 받은 gradient × 내가 이 연산에서 만든 local gradient
```

예를 들어 어떤 layer가 있습니다.

```text
Z = XW + b
```

위쪽에서 gradient가 내려옵니다.

```text
dZ = ∂L/∂Z
```

이 layer는 세 가지 방향으로 gradient를 계산해야 합니다.

```text
dW = ∂L/∂W
db = ∂L/∂b
dX = ∂L/∂X
```

즉, 이 layer가 하는 일은:

```text
upstream gradient를 받아서
자기 parameter gradient를 만들고
이전 layer로 보낼 gradient도 만든다.
```

입니다.

이것이 chain rule의 반복입니다.

```text
output layer
→ hidden layer
→ previous hidden layer
→ input layer
```

순서로 계속 적용합니다.

---

## 이 글을 이해하면 PyTorch가 다르게 보인다

PyTorch에서 학습 코드는 보통 이렇게 생겼습니다.

```python
optimizer.zero_grad()

pred = model(x)
loss = loss_fn(pred, y)

loss.backward()
optimizer.step()
```

이때:

```python
loss.backward()
```

가 하는 일은 전체 계산 그래프에 대해 chain rule을 적용해서 각 parameter의 gradient를 계산하는 것입니다.

그러면 각 parameter에는 이런 값이 생깁니다.

```python
param.grad
```

예를 들어 어떤 linear layer가 있다면:

```python
layer.weight.grad
layer.bias.grad
```

가 계산됩니다.

그다음:

```python
optimizer.step()
```

은 이 gradient를 이용해 parameter를 업데이트합니다.

```text
W ← W - learning_rate × dW
b ← b - learning_rate × db
```

즉, matrix calculus를 알면 `loss.backward()`가 단순한 마법이 아니라는 걸 이해할 수 있습니다.

---

## 앞에서 공부한 논문들과의 연결

### Deep Learning 논문과 연결

Deep Learning 리뷰에서는 neural network가 여러 layer를 통해 representation을 학습한다고 했습니다.

이번 글은 그 representation이 어떻게 학습되는지를 수학적으로 설명합니다.

```text
좋은 representation을 만들려면 weight를 업데이트해야 한다.
weight를 업데이트하려면 gradient가 필요하다.
gradient를 계산하려면 matrix calculus가 필요하다.
```

### Dropout과 연결

Dropout은 학습 중 일부 activation을 랜덤하게 0으로 만듭니다.

그러면 forward 계산이 달라지고, backward 계산도 그 mask를 반영합니다.

예를 들어 dropout mask가:

```text
mask = [1, 0, 1]
```

이면, 꺼진 뉴런에는 gradient도 흐르지 않습니다.

```text
dH = dH * mask
```

즉, dropout도 matrix calculus와 chain rule 위에서 작동합니다.

### BatchNorm과 연결

BatchNorm은 batch 평균과 분산으로 activation을 정규화합니다.

```text
x_hat = (x - μ) / sqrt(σ² + ε)
y = γx_hat + β
```

이 식도 학습하려면 gradient가 필요합니다.

```text
∂L/∂γ
∂L/∂β
∂L/∂x
```

BatchNorm의 backward는 일반 linear layer보다 복잡하지만, 원리는 같습니다.

```text
chain rule
```

입니다.

### ResNet과 연결

ResNet block은:

```text
y = F(x) + x
```

입니다.

backward를 보면 gradient가 두 경로로 흐릅니다.

```text
F(x)를 통과하는 경로
shortcut x를 통과하는 경로
```

matrix calculus 관점에서 보면, addition의 gradient는 각 입력으로 그대로 분배됩니다.

즉:

```text
y = a + b
```

이면:

```text
∂L/∂a += ∂L/∂y
∂L/∂b += ∂L/∂y
```

ResNet의 shortcut gradient도 이런 chain rule의 결과로 이해할 수 있습니다.

---

## 이 글에서 입문자가 꼭 가져가야 할 것

이 글을 읽을 때 모든 수식을 완벽하게 외울 필요는 없습니다.

입문자가 가져가야 할 핵심은 다음 5개입니다.

### 1. Gradient는 parameter와 같은 shape다

```text
W.shape = (D, H)
dW.shape = (D, H)
```

이것이 가장 중요합니다.

### 2. Backpropagation은 chain rule이다

```text
전체 미분 = 뒤쪽 gradient × 현재 연산의 local derivative
```

### 3. Linear layer의 핵심 backward 식

```text
Z = XW + b

dW = Xᵀ dZ
db = sum(dZ, axis=batch)
dX = dZ Wᵀ
```

이 세 식은 꼭 기억하면 좋습니다.

### 4. Element-wise activation은 element-wise로 미분된다

```text
A = ReLU(Z)
dZ = dA * (Z > 0)
```

### 5. 자동미분은 수학을 없앤 것이 아니라 자동화한 것이다

PyTorch가 gradient를 계산해주지만, 내부에서는 chain rule과 matrix calculus가 작동합니다.

---

## 이 글이 어려운 이유

이 글이 어려운 이유는 미분 자체보다 표기 때문입니다.

딥러닝 수식에서는 다음이 계속 섞입니다.

```text
scalar
vector
matrix
tensor
```

또 표기 convention도 자료마다 다릅니다.

어떤 자료는 vector를 column으로 보고, 어떤 자료는 row로 봅니다.  
어떤 자료는 gradient를 row vector로 쓰고, 어떤 자료는 column vector로 씁니다.

그래서 수식을 볼 때는 항상 이렇게 확인해야 합니다.

```text
이 값의 shape가 무엇인가?
곱셈이 가능한 shape인가?
결과 gradient가 parameter와 같은 shape인가?
```

shape가 맞으면 대부분 이해가 맞습니다.

---

## 쉬운 예시: Linear layer backward 한 번 더 정리

이 예시는 꼭 기억하면 좋습니다.

```text
X.shape = (N, D)
W.shape = (D, H)
b.shape = (H,)
Z = XW + b
Z.shape = (N, H)
```

위쪽에서:

```text
dZ.shape = (N, H)
```

가 내려옵니다.

그러면:

```text
dW = Xᵀ dZ
```

shape:

```text
(D, N) × (N, H) = (D, H)
```

```text
db = sum(dZ, axis=0)
```

shape:

```text
(H,)
```

```text
dX = dZ Wᵀ
```

shape:

```text
(N, H) × (H, D) = (N, D)
```

이 세 개가 linear layer backward의 핵심입니다.

---

## 이 글의 한계

이 글은 딥러닝에 필요한 matrix calculus를 매우 친절하게 설명하지만, 모든 딥러닝 수학을 다루는 것은 아닙니다.

예를 들어 다음 주제는 별도 공부가 필요합니다.

| 주제 | 왜 별도인가 |
|---|---|
| 확률론 | cross-entropy, likelihood, Bayesian model 이해 |
| 최적화 이론 | SGD, Adam, momentum, learning rate schedule |
| 고급 선형대수 | eigenvalue, SVD, conditioning |
| 정보이론 | entropy, KL divergence |
| 딥러닝 일반화 이론 | 왜 큰 모델이 일반화되는가 |
| 자동미분 구현 | computational graph, reverse-mode autodiff |

하지만 이 글은 다음 질문에는 아주 좋은 출발점입니다.

> backpropagation에서 gradient가 어떻게 계산되는가?

---

## 이 논문이 딥러닝 기본기 파트의 마지막에 온 이유

지금까지 딥러닝 기본기 파트에서 본 흐름은 이랬습니다.

| 논문 | 핵심 |
|---|---|
| Deep Learning | 딥러닝은 representation learning이다 |
| Dropout | overfitting을 줄이는 regularization |
| BatchNorm | activation 분포를 안정화해 학습을 빠르게 함 |
| ResNet | residual connection으로 깊은 모델을 학습 가능하게 함 |
| Matrix Calculus | 이 모든 학습의 수학적 기반인 gradient 계산 |

즉, 이번 글은 모델 구조를 제안하는 논문이라기보다는, 앞에서 배운 딥러닝 기법들이 내부적으로 어떻게 학습되는지 이해하기 위한 수학적 기초입니다.

---

## 오늘 공부용 요약

**논문명:** The Matrix Calculus You Need For Deep Learning  
**저자:** Terence Parr, Jeremy Howard

**핵심 결론:** 딥러닝 학습을 이해하려면 scalar calculus를 넘어 vector와 matrix에 대한 미분을 이해해야 한다. 특히 gradient, Jacobian, chain rule을 알면 backpropagation이 weight와 bias에 대한 gradient를 어떻게 계산하는지 이해할 수 있다.

**왜 중요함:** PyTorch나 TensorFlow는 자동미분으로 gradient를 계산해주지만, 그 내부는 chain rule과 matrix calculus로 이루어져 있다. 이 글은 `loss.backward()`가 어떤 수학적 원리로 작동하는지 이해하게 해준다.

**입문자가 배울 점:** 모든 수식을 외우기보다 shape를 추적하는 습관이 중요하다. gradient는 업데이트할 parameter와 같은 shape를 가져야 하며, backpropagation은 local derivative와 upstream gradient를 chain rule로 곱해가는 과정이다.

**가장 중요한 문장:** 딥러닝 학습은 loss를 줄이기 위해 각 parameter와 같은 shape의 gradient를 계산하고, 그 gradient의 반대 방향으로 parameter를 업데이트하는 과정이다.

---

## 딥러닝 기본기 파트 마무리

이제 **딥러닝 기본기 파트**가 끝났습니다.

```text
11. Deep Learning
12. Dropout
13. Batch Normalization
14. ResNet
15. Matrix Calculus
```

다음 파트는 **Transformer 이후 현대 AI**로 넘어가면 됩니다. 다음 논문은 **Attention Is All You Need**입니다.

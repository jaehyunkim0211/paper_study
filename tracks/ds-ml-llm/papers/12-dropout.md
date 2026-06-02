# Dropout: A Simple Way to Prevent Neural Networks from Overfitting — Srivastava et al.

## 결론부터

이 논문의 결론은 이겁니다.

**딥러닝 모델이 훈련 데이터에 과하게 맞춰지는 것을 막으려면, 학습 중에 일부 뉴런을 무작위로 꺼버리면 된다.**  
이 방법이 바로 **Dropout**입니다.

조금 더 쉽게 말하면:

> 학습할 때마다 매번 다른 일부 뉴런을 꺼서,  
> 특정 뉴런들이 서로에게만 지나치게 의존하지 못하게 만든다.

논문은 dropout의 핵심 아이디어를 “training 중에 neural network의 unit과 그 연결을 무작위로 drop한다”고 설명합니다. 이 방식은 unit들이 서로 과도하게 co-adapt하지 못하게 만들고, training 중에는 지수적으로 많은 “thinned network”를 샘플링한 것과 비슷한 효과를 낸다고 설명합니다.

---

## 이 논문을 한 문장으로 요약하면

**Dropout은 학습할 때마다 다른 작은 신경망을 훈련시키는 것처럼 만들어, 여러 모델을 앙상블한 효과를 저렴하게 얻는 regularization 기법이다.**

즉, Dropout은 단순히 “뉴런을 랜덤하게 끈다”가 아닙니다.

진짜 의도는 이것입니다.

```text
하나의 큰 모델이 훈련 데이터에 과적합되는 것을 막고,
여러 작은 모델을 평균낸 것 같은 효과를 얻는다.
```

논문은 큰 neural network 여러 개를 따로 학습해서 평균내는 것은 계산 비용이 너무 크지만, dropout은 파라미터를 공유하는 지수적으로 많은 sub-network를 효율적으로 조합하는 방법을 제공한다고 설명합니다.

---

## 왜 Dropout이 필요한가?

직전 논문에서 딥러닝의 장점은 **표현을 스스로 학습한다**는 것이라고 했습니다.

하지만 딥러닝 모델은 보통 parameter가 많습니다.

```text
weight 수가 많다
→ 복잡한 패턴을 배울 수 있다
→ 하지만 훈련 데이터를 외울 수도 있다
```

이게 overfitting입니다.

예를 들어 훈련 데이터에서는 정확도가 99%인데, test 데이터에서는 70%라면 모델이 일반화하지 못한 것입니다.

| 상황 | 의미 |
|---|---|
| train 성능 좋음, test 성능 좋음 | 좋은 일반화 |
| train 성능 좋음, test 성능 나쁨 | overfitting |
| train 성능 나쁨, test 성능 나쁨 | underfitting |

Dropout 논문은 deep neural network가 많은 parameter를 가진 강력한 시스템이지만 overfitting이 심각한 문제라고 설명합니다. 특히 큰 network 여러 개를 따로 학습하고 test time에 평균내는 방식은 계산 비용 때문에 실용적이지 않다고 지적합니다.

---

## 핵심 개념 1: Dropout은 학습 중 일부 뉴런을 꺼버린다

### 결론부터

**Dropout은 training 중에 hidden unit 또는 input unit을 일정 확률로 0으로 만든다.**

예를 들어 어떤 hidden layer에 뉴런이 6개 있다고 합시다.

```text
h = [h1, h2, h3, h4, h5, h6]
```

Dropout을 적용하면 어떤 training step에서는 이렇게 될 수 있습니다.

```text
h = [h1, 0, h3, 0, h5, h6]
```

다음 training step에서는 또 다르게 꺼집니다.

```text
h = [0, h2, h3, h4, 0, h6]
```

즉, 매번 다른 network가 됩니다.

논문은 dropout이 unit을 network에서 임시로 제거하는 것을 의미하며, 제거된 unit은 들어오는 연결과 나가는 연결까지 함께 제거된다고 설명합니다. 논문 그림도 일반 neural net과 dropout 적용 후 일부 unit이 제거된 “thinned net”을 비교해 보여줍니다.

---

## 핵심 개념 2: Dropout은 Bernoulli mask를 씌우는 것

수식으로는 어렵지 않습니다.

hidden activation이 있다고 합시다.

```text
h = [2.0, 1.5, 0.7, 3.0]
```

각 뉴런을 살릴지 죽일지 random mask를 뽑습니다.

```text
mask = [1, 0, 1, 0]
```

그러면 dropout 적용 후 activation은:

```text
h_dropout = h * mask
          = [2.0, 0, 0.7, 0]
```

즉, mask가 0인 뉴런은 꺼집니다.

논문은 dropout이 각 layer에서 독립적인 Bernoulli random variable로 이루어진 vector를 샘플링하고, 이 vector를 layer activation에 element-wise product로 곱하는 방식으로 feed-forward 연산을 바꾼다고 설명합니다.

---

## 중요한 용어: keep probability와 dropout probability

여기서 헷갈리기 쉬운 점이 있습니다.

논문에서는 보통 `p`를 **unit을 유지할 확률**, 즉 **keep probability**로 설명합니다.

```text
p = 0.5
→ 뉴런을 50% 확률로 살림
→ 뉴런을 50% 확률로 끔
```

그런데 PyTorch 같은 현대 프레임워크에서는 `Dropout(p=0.5)`의 `p`가 **dropout probability**, 즉 **0으로 만들 확률**입니다.

```text
PyTorch Dropout(p=0.5)
→ 입력 요소를 50% 확률로 0으로 만듦
```

정리하면:

| 문맥 | `p`의 의미 |
|---|---|
| 원 논문 설명 | 살릴 확률, keep probability |
| PyTorch `nn.Dropout(p=...)` | 끌 확률, dropout probability |

이 둘을 섞으면 혼란이 생깁니다.

---

## 쉬운 예시: 시험 공부 팀 프로젝트

Dropout의 직관을 비유로 이해해봅시다.

어떤 팀이 시험 문제를 같이 풉니다. 팀원 5명이 있습니다.

```text
A, B, C, D, E
```

항상 5명이 같이 문제를 풀면 이런 일이 생길 수 있습니다.

```text
A는 계산만 한다.
B는 공식만 외운다.
C는 B가 틀린 부분만 보완한다.
D는 C가 있으면 안심하고 대충 한다.
E는 특정 유형만 맡는다.
```

즉, 서로에게 너무 의존합니다.

이 상태에서는 연습 문제는 잘 풀 수 있지만, 새로운 문제가 나오면 약할 수 있습니다.

Dropout은 훈련할 때마다 일부 팀원을 랜덤하게 빠지게 합니다.

```text
1번째 훈련: A, C, E만 참여
2번째 훈련: B, C, D만 참여
3번째 훈련: A, B, D, E만 참여
```

그러면 각 팀원은 이렇게 배워야 합니다.

> “내가 항상 특정 팀원에게 의존할 수는 없구나.  
> 어떤 조합에서도 쓸모 있는 능력을 가져야겠구나.”

Dropout도 같은 원리입니다.

각 뉴런은 특정 다른 뉴런이 항상 존재한다고 가정할 수 없습니다.  
그래서 더 독립적이고 일반화 가능한 feature를 배우게 됩니다.

논문은 표준 backpropagation 학습에서는 unit들이 서로의 실수를 보정하는 방향으로 복잡한 co-adaptation을 만들 수 있고, 이런 co-adaptation이 unseen data로 일반화되지 않아 overfitting을 유발할 수 있다고 설명합니다. Dropout은 다른 hidden unit의 존재를 불확실하게 만들어 이런 co-adaptation을 줄인다고 설명합니다.

---

## 핵심 개념 3: Co-adaptation을 막는다

### 결론부터

**Dropout의 핵심 효과는 뉴런들이 서로에게 과도하게 의존하지 못하게 하는 것입니다.**

co-adaptation은 쉽게 말해:

> 어떤 뉴런이 다른 특정 뉴런들이 항상 같이 있을 때만 쓸모 있게 되는 현상

입니다.

예를 들어 어떤 뉴런 A가 이런 식으로 학습됐다고 합시다.

```text
B와 C가 같이 켜질 때만 A가 의미 있음
```

이런 패턴은 훈련 데이터에서는 잘 맞을 수 있지만, 새로운 데이터에서는 취약할 수 있습니다.

Dropout은 B나 C가 훈련 중 랜덤하게 사라지도록 만듭니다.

그러면 A는 이렇게 배워야 합니다.

```text
B와 C가 없어도 어느 정도 쓸모 있는 feature를 만들어야 함
```

이게 dropout의 핵심 철학입니다.

논문의 결론도 표준 backpropagation은 training data에서는 작동하지만 unseen data로 일반화되지 않는 brittle co-adaptation을 만들 수 있고, dropout은 hidden unit의 존재를 불확실하게 만들어 이런 co-adaptation을 깨뜨린다고 설명합니다.

---

## 핵심 개념 4: Dropout은 앙상블처럼 작동한다

Dropout은 매 training step마다 다른 sub-network를 만듭니다.

예를 들어 뉴런이 4개인 layer가 있다고 합시다.

각 뉴런은 켜지거나 꺼질 수 있습니다.

```text
h1 on/off
h2 on/off
h3 on/off
h4 on/off
```

가능한 조합은:

```text
2^4 = 16개
```

뉴런이 100개면 가능한 sub-network 수는 어마어마하게 많습니다.

Dropout은 training 중 이 sub-network들을 계속 샘플링합니다.

```text
step 1: network A
step 2: network B
step 3: network C
...
```

하지만 이 sub-network들은 완전히 별개의 모델이 아닙니다.  
원래 큰 network의 weight를 공유합니다.

그래서 dropout은 이런 효과를 냅니다.

```text
많은 작은 network를 따로 학습한 것 같은 효과
+ weight는 공유하므로 계산은 훨씬 저렴
```

논문은 dropout이 training 중 지수적으로 많은 thinned network를 샘플링하고, test time에는 작은 weight를 가진 하나의 unthinned network로 이들의 평균 예측 효과를 근사할 수 있다고 설명합니다.

---

## 핵심 개념 5: Training 때와 Test 때가 다르다

Dropout은 training 때만 무작위로 뉴런을 끕니다.

```text
training mode: 일부 뉴런을 랜덤하게 끔
test / inference mode: 모든 뉴런을 사용
```

왜 test 때는 dropout을 끄냐면, test 때는 안정적인 예측이 필요하기 때문입니다.

하지만 한 가지 문제가 있습니다.

training 때는 예를 들어 절반의 뉴런만 살아 있습니다.

```text
training: 평균적으로 50% 뉴런만 사용
```

test 때는 모든 뉴런을 사용합니다.

```text
test: 100% 뉴런 사용
```

그러면 test 때 activation 크기가 너무 커질 수 있습니다.

그래서 scaling이 필요합니다.

원 논문은 test time에 모든 unit을 사용하는 하나의 network를 쓰되, dropout network들의 평균 효과를 근사하기 위해 weight scaling을 적용한다고 설명합니다.

---

## 요즘 구현: Inverted Dropout

현대 딥러닝 프레임워크에서는 보통 **inverted dropout**을 씁니다.

이 방식은 training 때 살아남은 activation을 미리 키워둡니다.

예를 들어 dropout probability가 0.5라면, 살아남은 activation을 `1 / 0.5 = 2`배 합니다.

```text
training: 살아남은 뉴런 값을 2배
test: 아무것도 하지 않음
```

이렇게 하면 test 때 별도 scaling을 하지 않아도 됩니다.

정리하면:

| 방식 | Training | Test |
|---|---|---|
| 원 논문식 설명 | dropout 적용 | weight scaling |
| 현대 inverted dropout | dropout 적용 + surviving activation scaling | 그대로 통과 |

실무에서 PyTorch를 쓰면 `model.train()`일 때 dropout이 작동하고, `model.eval()`일 때 dropout이 꺼집니다.

---

## 숫자로 보는 Dropout 예시

dropout probability가 0.5라고 합시다.

어떤 layer activation이 있습니다.

```text
h = [4, 2, 6, 8]
```

training 때 dropout mask가 이렇게 나왔다고 합시다.

```text
mask = [1, 0, 1, 0]
```

그럼 단순 dropout은:

```text
h_dropout = [4, 0, 6, 0]
```

inverted dropout에서는 살아남은 값을 `1 / (1 - dropout_probability)`만큼 키웁니다.

dropout probability가 0.5면 keep probability는 0.5입니다.

```text
scaling factor = 1 / 0.5 = 2
```

그래서:

```text
h_dropout = [8, 0, 12, 0]
```

이렇게 training 때 평균 activation 규모를 test 때와 맞춰둡니다.

test 때는:

```text
h = [4, 2, 6, 8]
```

그대로 사용합니다.

---

## Dropout은 Regularization이다

Dropout은 regularization 기법입니다.

regularization은 모델이 훈련 데이터에 너무 딱 맞지 않도록 제약을 주는 방법입니다.

| Regularization | 핵심 아이디어 |
|---|---|
| L2 weight decay | weight가 너무 커지지 않게 함 |
| L1 regularization | 일부 weight를 0에 가깝게 만듦 |
| Early stopping | validation 성능이 나빠지기 전에 멈춤 |
| Data augmentation | 데이터를 변형해서 일반화 향상 |
| Dropout | 일부 뉴런을 랜덤하게 꺼서 co-adaptation 방지 |

논문도 L2, L1, sparsity, max-norm 같은 기존 regularization 방법들과 dropout을 비교하고, dropout을 neural network를 regularize하는 또 다른 방법으로 설명합니다.

---

## Dropout이 왜 과적합을 줄이나?

핵심 이유는 세 가지입니다.

### 1. 특정 뉴런 조합에 의존하지 못하게 한다

Dropout은 매번 다른 뉴런을 끄므로, 어떤 뉴런도 특정 동료 뉴런이 항상 있을 것이라고 믿을 수 없습니다.

그래서 각 뉴런은 더 robust한 feature를 배워야 합니다.

### 2. 여러 sub-network를 평균낸 효과가 있다

Dropout은 training 중 많은 thinned network를 샘플링합니다.

test 때는 그 효과를 하나의 network로 근사합니다.

이것은 앙상블과 비슷합니다.

### 3. Noise injection 효과가 있다

Dropout은 activation에 random noise를 주는 것과 비슷합니다.

훈련이 매번 조금씩 흔들리므로, 모델이 훈련 데이터의 세부 패턴을 그대로 외우기 어렵습니다.

---

## 쉬운 예시: 손글씨 숫자 분류

MNIST 손글씨 숫자 분류를 생각해봅시다.

모델은 숫자 이미지를 보고 0~9 중 하나를 맞춰야 합니다.

Dropout 없이 학습하면 어떤 뉴런들이 특정 stroke나 pixel pattern 조합에 지나치게 의존할 수 있습니다.

예를 들어:

```text
왼쪽 위 pixel 패턴 + 가운데 선 + 아래쪽 curve가 같이 있을 때만 8로 판단
```

이런 조합이 train set에서는 잘 맞을 수 있지만, 새로운 손글씨에서는 다르게 나타날 수 있습니다.

Dropout을 쓰면 훈련 중 일부 feature detector가 랜덤하게 꺼집니다.

그러면 모델은 다양한 조합에서도 숫자를 알아볼 수 있도록 학습해야 합니다.

---

## Dropout 확률은 어떻게 정하나?

입문 단계에서는 이렇게 기억하면 됩니다.

```text
hidden layer dropout: 0.5 근처에서 자주 시작
input layer dropout: 보통 더 약하게
```

다만 여기서 다시 용어를 구분해야 합니다.

원 논문에서 `p=0.5`는 **keep probability**입니다.

즉:

```text
논문식 p = 0.5
→ 50% 살림
→ 50% 끔
```

PyTorch에서 `nn.Dropout(p=0.5)`는 **dropout probability**입니다.

즉:

```text
PyTorch p = 0.5
→ 50% 끔
```

논문은 가장 단순한 경우 각 unit이 fixed probability `p`로 유지된다고 설명하고, 이 `p`는 validation set으로 고르거나 단순히 0.5로 둘 수 있으며, input unit의 optimal retention probability는 보통 0.5보다 1에 가깝다고 설명합니다.

---

## Dropout과 Random Forest의 연결

예전에 Random Forest를 공부했죠.

Random Forest는 여러 tree를 다르게 만들고 평균냅니다.

Dropout도 비슷한 면이 있습니다.

| Random Forest | Dropout |
|---|---|
| tree를 여러 개 만든다 | sub-network를 여러 개 샘플링한다 |
| bootstrap sample과 feature randomness를 쓴다 | neuron을 랜덤하게 끈다 |
| 여러 tree 예측을 평균낸다 | 여러 thinned network의 평균 효과를 근사한다 |
| 과적합을 줄인다 | 과적합을 줄인다 |

하지만 차이도 있습니다.

| 구분 | Random Forest | Dropout |
|---|---|---|
| 모델 종류 | tree ensemble | neural network regularization |
| 여러 모델의 관계 | 각 tree가 독립적으로 학습 | sub-network들이 weight를 공유 |
| 적용 시점 | ensemble 자체가 모델 | 학습 중 regularization 기법 |
| test time | 모든 tree 평균 | 하나의 full network 사용 |

즉, Dropout은 neural network 안에서 일어나는 **효율적인 앙상블 근사**라고 볼 수 있습니다.

---

## Dropout과 Gradient Descent의 연결

Dropout은 loss function 자체를 바꾸는 것처럼 보이지 않을 수 있지만, 학습 과정에 강한 noise를 넣습니다.

일반 neural network 학습은 이렇게 진행됩니다.

```text
forward pass
→ loss 계산
→ backpropagation
→ gradient descent로 weight 업데이트
```

Dropout을 쓰면 forward pass 전에 random mask가 들어갑니다.

```text
random mask 생성
→ 일부 neuron 제거
→ forward pass
→ loss 계산
→ backpropagation
→ weight 업데이트
```

중요한 점은 매 step마다 mask가 다릅니다.

그래서 같은 데이터라도 매번 조금 다른 network로 학습합니다.

---

## Dropout의 장점

### 1. 구현이 단순하다

Dropout은 아이디어가 매우 간단합니다.

```text
일부 activation을 랜덤하게 0으로 만든다.
```

하지만 효과는 강력합니다.

### 2. 과적합을 줄인다

Dropout은 큰 neural network가 training data를 외우는 것을 줄이는 데 도움이 됩니다.

특히 데이터가 제한적이고 모델이 큰 경우 유용합니다.

### 3. 앙상블 효과를 저렴하게 얻는다

여러 neural network를 따로 학습해서 ensemble하는 것은 비쌉니다.

Dropout은 하나의 network 안에서 여러 sub-network를 샘플링하고, test 때 full network로 그 평균 효과를 근사합니다.

### 4. 다른 regularization과 함께 쓸 수 있다

Dropout은 L2 weight decay, max-norm, data augmentation, early stopping 등과 함께 사용할 수 있습니다.

---

## Dropout의 한계

### 1. 학습 시간이 늘어날 수 있다

Dropout은 매 step마다 noisy한 sub-network를 학습하므로 수렴이 느려질 수 있습니다.

### 2. 항상 도움이 되는 것은 아니다

모델이 이미 underfitting 중이라면 dropout은 오히려 학습을 더 어렵게 만들 수 있습니다.

예를 들어 train 성능도 낮고 validation 성능도 낮은 상황에서 dropout을 강하게 넣으면 모델이 더 못 배울 수 있습니다.

### 3. 위치와 확률을 잘 정해야 한다

Dropout을 어디에 넣을지, 얼마나 강하게 넣을지는 중요합니다.

예를 들어 CNN에서는 convolution layer보다 fully connected layer에 dropout을 많이 쓰는 경우가 많았습니다. 현대 architecture에서는 BatchNorm, residual connection, data augmentation, weight decay 등과 함께 고려해야 합니다.

### 4. Inference 때는 보통 꺼야 한다

Dropout은 training mode에서 regularization으로 쓰입니다.

일반 inference에서는 보통 꺼야 합니다.

예외적으로 uncertainty estimation을 위해 test time에도 dropout을 켜는 Monte Carlo Dropout 같은 방법이 있지만, 이는 원래 표준 inference 사용법과는 별도 주제입니다.

---

## Dropout을 실무에서 어떻게 이해하면 좋나?

### 1. Overfitting이 있을 때 고려한다

이런 상황이면 dropout을 고려할 수 있습니다.

```text
train loss는 계속 내려감
validation loss는 어느 순간부터 올라감
```

또는:

```text
train accuracy는 높음
validation accuracy는 낮음
```

이때 dropout은 regularization 후보입니다.

### 2. 너무 강하게 넣으면 underfitting될 수 있다

dropout probability가 너무 높으면 모델이 학습할 수 있는 정보가 너무 많이 사라집니다.

예를 들어 hidden unit의 80%를 꺼버리면 학습이 매우 어려워질 수 있습니다.

일반적으로는 작은 값부터 실험하는 것이 좋습니다.

```text
0.1, 0.2, 0.3, 0.5
```

### 3. `model.train()`과 `model.eval()`을 구분해야 한다

PyTorch에서는 매우 중요합니다.

```python
model.train()  # dropout 켜짐
model.eval()   # dropout 꺼짐
```

validation이나 test 평가 때 `model.eval()`을 하지 않으면, dropout이 계속 켜져 있어서 예측이 랜덤하게 흔들릴 수 있습니다.

---

## Dropout과 Batch Normalization의 관계

다음 논문에서 볼 **Batch Normalization**과도 연결됩니다.

Dropout은 주로 regularization입니다.

```text
과적합을 줄이기 위해 일부 뉴런을 랜덤하게 끔
```

Batch Normalization은 주로 학습 안정화와 속도 개선에 초점이 있습니다.

```text
layer 입력 분포를 안정화해서 학습을 쉽게 만듦
```

물론 BatchNorm도 regularization 효과가 어느 정도 있을 수 있습니다. 그래서 현대 모델에서는 dropout을 예전만큼 강하게 쓰지 않는 경우도 있지만, Transformer나 fully connected network에서는 여전히 dropout이 많이 쓰입니다.

---

## Dropout과 Data Augmentation의 차이

두 방법 모두 overfitting을 줄이는 데 쓰입니다.

하지만 작동 위치가 다릅니다.

| 방법 | 어디에 noise를 넣나? |
|---|---|
| Data augmentation | 입력 데이터에 변형을 줌 |
| Dropout | network 내부 activation에 noise를 줌 |

예를 들어 이미지 분류에서 data augmentation은 이미지를 회전하거나 자르거나 색을 바꿉니다.

```text
고양이 이미지
→ 조금 회전
→ crop
→ 색상 변화
```

Dropout은 이미지가 아니라 neural network 내부 뉴런 일부를 끕니다.

```text
hidden unit 일부 제거
```

둘은 함께 쓸 수 있습니다.

---

## Dropout과 L2 Regularization의 차이

L2 regularization은 weight 크기를 작게 유지합니다.

```text
큰 weight에 penalty를 준다.
```

Dropout은 뉴런의 존재 자체를 랜덤하게 불확실하게 만듭니다.

```text
일부 unit을 임시로 제거한다.
```

| 구분 | L2 | Dropout |
|---|---|---|
| 제약 대상 | weight 크기 | unit activation |
| 방식 | penalty 추가 | random masking |
| 효과 | weight가 과도하게 커지는 것 방지 | co-adaptation 방지 |
| 직관 | 모델을 부드럽게 만듦 | 여러 sub-network를 학습 |

---

## 이전 논문들과 연결해서 이해하기

| 이전 논문 | Dropout과의 연결 |
|---|---|
| **Deep Learning** | 여러 층의 neural network는 표현력이 크지만 overfitting 위험이 있음 |
| **Model Evaluation** | dropout 효과는 train 성능이 아니라 validation/test 일반화 성능으로 판단해야 함 |
| **Random Forest** | dropout도 여러 sub-model을 평균내는 앙상블 직관을 가짐 |
| **Gradient Descent** | dropout은 training 중 random mask를 넣은 상태에서 gradient를 계산하게 만듦 |
| **SHAP** | dropout을 쓴 neural network도 해석이 필요하지만, dropout 자체는 설명 도구가 아니라 regularization 기법 |

특히 Random Forest와의 연결을 기억하면 좋습니다.

```text
Random Forest = 여러 tree를 다르게 만들어 평균
Dropout = 여러 sub-network를 랜덤하게 만들어 평균 효과를 근사
```

---

## 이 논문이 딥러닝 역사에서 중요한 이유

Dropout은 딥러닝이 본격적으로 커지던 시기에 매우 중요한 regularization 기법이었습니다.

큰 neural network는 강력하지만 과적합 위험이 컸고, dropout은 그 문제를 아주 간단한 아이디어로 완화했습니다.

이 논문의 가치는 단순히 성능 개선만이 아닙니다.

더 중요한 관점은 이것입니다.

> 큰 모델 하나를 그대로 믿지 말고,  
> 학습 중에 구조적 불확실성을 주어 더 robust한 표현을 배우게 하자.

이 생각은 이후 stochastic depth, drop path, attention dropout, feature dropout 등 다양한 변형으로 이어졌습니다.

---

## 입문자가 꼭 기억해야 할 문장

**Dropout은 학습 중 일부 뉴런을 무작위로 꺼서, 뉴런들이 서로에게 과도하게 의존하지 못하게 만들고, 여러 sub-network를 앙상블한 것 같은 효과를 내는 regularization 기법이다.**

더 짧게 말하면:

> **Dropout = 학습 중 뉴런을 랜덤하게 꺼서 overfitting을 줄이는 방법**

---

## 오늘 공부용 요약

**논문명:** Dropout: A Simple Way to Prevent Neural Networks from Overfitting  
**저자:** Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov  
**출판:** *Journal of Machine Learning Research*, 2014년 15권 56호, 1929–1958쪽.

**핵심 결론:** Dropout은 training 중 neural network의 일부 unit과 연결을 무작위로 제거해 overfitting을 줄이는 regularization 기법이다.

**왜 중요함:** 큰 neural network는 강력하지만 overfitting되기 쉽다. Dropout은 unit 간 co-adaptation을 막고, 여러 thinned network를 평균낸 것 같은 효과를 효율적으로 제공한다.

**입문자가 배울 점:** Dropout은 모델 구조를 영구적으로 잘라내는 pruning이 아니라, 학습 중에만 무작위로 일부 뉴런을 끄는 stochastic regularization이다.

**가장 중요한 문장:** Dropout은 특정 뉴런 조합에 의존하는 약한 표현을 막고, 어떤 뉴런이 사라져도 작동하는 robust한 표현을 배우게 만든다.

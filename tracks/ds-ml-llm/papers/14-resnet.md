# Deep Residual Learning for Image Recognition — ResNet

## 결론부터

이 논문의 결론은 이겁니다.

**딥러닝 모델을 더 깊게 만들수록 성능이 좋아질 것 같지만, 단순히 layer를 계속 쌓으면 오히려 학습이 어려워지고 성능이 나빠질 수 있다. ResNet은 이 문제를 해결하기 위해 “출력 전체”를 직접 학습하지 않고, 입력에서 얼마나 바뀌어야 하는지에 해당하는 “잔차 residual”만 학습하게 만든다.**

한 줄로 말하면:

> **ResNet은 layer가 직접 `H(x)`를 배우게 하지 않고, `H(x) - x`를 배우게 해서 매우 깊은 네트워크를 학습 가능하게 만든 구조다.**

논문은 깊은 네트워크가 학습하기 어렵다는 문제를 다루며, 기존처럼 layer가 원하는 mapping 전체를 직접 학습하게 하지 않고, layer 입력을 기준으로 한 residual function을 학습하게 하는 residual learning framework를 제안합니다. 저자들은 이 방식으로 매우 깊은 residual network가 더 쉽게 최적화되고, depth 증가로부터 성능 이득을 얻을 수 있음을 실험적으로 보였습니다.

---

## 이 논문을 한 문장으로 요약하면

**ResNet은 “새로운 것을 전부 배워라”가 아니라 “기존 입력에서 필요한 수정량만 배워라”라고 neural network에게 문제를 다시 던진 논문입니다.**

기존 plain network는 이런 걸 배웁니다.

```text
입력 x
→ 여러 layer
→ 원하는 출력 H(x)
```

ResNet은 이렇게 바꿉니다.

```text
입력 x
→ 여러 layer가 F(x)를 학습
→ 출력은 F(x) + x
```

여기서:

| 기호 | 의미 |
|---|---|
| `x` | residual block의 입력 |
| `H(x)` | 원래 배우고 싶었던 mapping |
| `F(x)` | residual mapping, 즉 `H(x) - x` |
| `F(x) + x` | residual block의 최종 출력 |

논문은 원하는 mapping을 `H(x)`라고 할 때, stacked nonlinear layer가 직접 `H(x)`를 학습하게 하는 대신 `F(x) := H(x) - x`를 학습하게 하고, 원래 mapping을 `F(x) + x`로 재구성한다고 설명합니다.

---

## 왜 ResNet이 필요했나?

딥러닝에서는 보통 이런 직관이 있습니다.

```text
layer가 많아지면 더 복잡한 표현을 배울 수 있다.
그러면 성능도 좋아질 것이다.
```

어느 정도는 맞습니다. Deep Learning 리뷰 논문에서도 봤듯이, 낮은 layer는 edge 같은 단순 feature를 배우고, 높은 layer는 object part나 object 같은 더 추상적인 feature를 배울 수 있습니다.

그런데 문제가 생깁니다.

> 깊게 쌓기만 하면 성능이 계속 좋아지는가?

ResNet 논문의 답은 **아니다**입니다.

저자들은 CIFAR-10에서 20-layer plain network와 56-layer plain network를 비교했는데, 더 깊은 56-layer plain network가 training error와 test error 모두 더 높게 나오는 현상을 보였습니다. 논문은 이를 **degradation problem**이라고 부르고, 이 현상이 단순한 overfitting이 아니라고 설명합니다. 왜냐하면 test error만 나빠지는 것이 아니라 training error 자체도 높아졌기 때문입니다.

---

## 핵심 개념 1: Degradation Problem

### 결론부터

**Degradation problem은 네트워크를 더 깊게 만들었는데, 오히려 training error까지 나빠지는 현상입니다.**

이건 overfitting과 다릅니다.

| 현상 | train error | test error | 의미 |
|---|---:|---:|---|
| Overfitting | 낮음 | 높음 | 훈련 데이터는 잘 외웠지만 일반화가 안 됨 |
| Degradation | 높음 | 높음 | 훈련 데이터조차 잘 못 맞춤 |
| Underfitting | 높음 | 높음 | 모델이 충분히 학습하지 못함 |

ResNet 논문에서 중요한 점은 “더 깊은 모델은 적어도 얕은 모델만큼은 할 수 있어야 하지 않나?”라는 문제의식입니다.

예를 들어 20-layer 모델이 잘 학습됐다고 해봅시다. 여기에 layer를 더 붙여서 56-layer 모델을 만들 수 있습니다. 이론적으로는 새로 추가한 layer들이 그냥 입력을 그대로 통과시키는 **identity mapping**만 하면, 56-layer 모델은 최소한 20-layer 모델만큼은 성능을 내야 합니다.

그런데 실제 plain deep network에서는 그렇게 되지 않았습니다. 논문은 더 깊은 모델에는 얕은 모델의 solution을 포함하는 constructed solution이 존재하지만, 실제 solver가 그 좋은 solution을 찾지 못한다고 설명합니다.

---

## 핵심 개념 2: Residual Learning

### 결론부터

**Residual learning은 원래 함수 `H(x)`를 직접 배우는 대신, 입력에서 얼마나 수정해야 하는지 `F(x) = H(x) - x`를 배우는 방식입니다.**

기존 방식:

```text
H(x)를 직접 배워라
```

ResNet 방식:

```text
F(x) = H(x) - x를 배워라
마지막 출력은 F(x) + x로 만들자
```

이걸 아주 쉽게 비유하면 이렇습니다.

### 기존 방식

선생님이 학생에게 말합니다.

```text
정답 전체를 처음부터 만들어라.
```

### ResNet 방식

선생님이 학생에게 말합니다.

```text
일단 기존 답안을 그대로 두고,
고쳐야 할 부분만 표시해라.
```

보통 후자가 더 쉽습니다.

이미 입력 `x`가 꽤 괜찮은 표현이라면, 네트워크가 해야 할 일은 완전히 새로운 `H(x)`를 만드는 것이 아니라, `x`에서 부족한 부분만 보정하는 것입니다.

---

## 핵심 개념 3: Skip Connection / Shortcut Connection

### 결론부터

**Skip connection은 입력 `x`를 몇 개 layer 뒤로 그대로 건너뛰어 보내고, layer가 계산한 `F(x)`와 더하는 연결입니다.**

Residual block의 구조는 이렇게 생겼습니다.

```text
        x
        │
        ├──────────────┐
        │              │
        ▼              │
   weight layer        │
        ▼              │
      ReLU             │
        ▼              │
   weight layer        │
        ▼              │
      F(x)             │
        │              │
        └──── + x ◀────┘
              │
              ▼
           F(x) + x
```

즉, block의 출력은 다음과 같습니다.

```text
y = F(x) + x
```

논문은 residual building block을 `y = F(x, {Wi}) + x`로 정의하고, shortcut connection이 몇 개 layer를 건너뛰어 입력을 그대로 전달한 뒤 stacked layer의 출력과 element-wise addition을 수행한다고 설명합니다. 또한 identity shortcut은 추가 parameter나 추가 계산 복잡도를 만들지 않는다고 설명합니다.

---

## 왜 `F(x) + x`가 좋은가?

가장 중요한 직관은 이것입니다.

```text
필요하면 block이 x를 많이 바꿀 수 있다.
필요 없으면 F(x)를 0에 가깝게 만들어 x를 거의 그대로 통과시킬 수 있다.
```

즉, residual block은 두 가지 모두 가능합니다.

| 상황 | block이 할 일 |
|---|---|
| 입력 표현을 바꿔야 함 | `F(x)`가 의미 있는 보정값을 학습 |
| 입력 표현을 그대로 둬도 됨 | `F(x) ≈ 0`이 되도록 학습 |
| 차원이 다름 | projection shortcut 등으로 shape를 맞춤 |

이게 plain network보다 쉬운 이유입니다.

plain network에서 identity mapping을 배우려면 여러 nonlinear layer가 전체적으로 `H(x) = x`가 되도록 정확히 맞춰야 합니다.

ResNet에서는 identity가 필요하면:

```text
F(x) = 0
```

만 배우면 됩니다.

논문도 identity mapping이 최적이라면, nonlinear layer stack이 직접 identity mapping을 맞추는 것보다 residual을 0으로 만드는 편이 더 쉬울 수 있다고 설명합니다.

---

## 쉬운 예시: 이미지 feature map 보정

CNN 중간 feature map을 생각해봅시다.

어떤 block의 입력 `x`가 이미 꽤 좋은 이미지 표현입니다.

예를 들어 `x`에는 이런 정보가 들어 있습니다.

```text
edge
texture
object part
```

plain block은 이 입력을 받아서 완전히 새로운 표현 `H(x)`를 만들어야 합니다.

```text
x → H(x)
```

ResNet block은 이렇게 생각합니다.

```text
x는 이미 쓸 만하다.
나는 x에서 부족한 부분만 보정하겠다.
```

그래서 block은 `F(x)`를 학습합니다.

```text
F(x) = 추가해야 할 정보 또는 수정량
```

최종 출력은:

```text
x + F(x)
```

입니다.

즉, ResNet block은 feature map을 완전히 다시 만드는 장치라기보다, **feature map을 단계적으로 보정하는 장치**처럼 볼 수 있습니다.

---

## 핵심 개념 4: Element-wise Addition

### 결론부터

**ResNet에서는 skip으로 넘어온 `x`와 residual branch에서 나온 `F(x)`를 같은 위치끼리 더합니다.**

CNN feature map 기준으로 생각하면, `x`와 `F(x)`는 shape가 같아야 합니다.

예를 들어:

```text
x:    (N, C, H, W)
F(x): (N, C, H, W)
```

그러면 같은 위치끼리 더합니다.

```text
y[n, c, h, w] = F(x)[n, c, h, w] + x[n, c, h, w]
```

논문은 convolutional layer에도 같은 notation을 적용할 수 있으며, `F(x, {Wi})`가 여러 convolution layer를 나타낼 수 있고, element-wise addition은 두 feature map에서 channel별로 수행된다고 설명합니다.

---

## Shape가 다르면 어떻게 하나?

문제가 하나 있습니다.

어떤 residual block에서는 feature map 크기나 channel 수가 바뀝니다.

예를 들어:

```text
x:    (N, 64, 56, 56)
F(x): (N, 128, 28, 28)
```

이러면 그냥 더할 수 없습니다.

shape가 다르기 때문입니다.

ResNet 논문에서는 이런 경우 shortcut 쪽도 shape를 맞춰야 한다고 설명합니다. 입력과 출력 차원이 같으면 identity shortcut을 바로 쓰고, 차원이 증가하는 경우에는 zero-padding을 쓰거나, projection shortcut을 사용해 차원을 맞추는 옵션을 제시합니다. projection shortcut은 보통 `1×1 convolution`으로 구현됩니다.

정리하면:

| 상황 | shortcut 방식 |
|---|---|
| shape 같음 | `x`를 그대로 더함 |
| channel 또는 spatial size 다름 | zero-padding 또는 `1×1 conv` projection |
| downsampling 필요 | shortcut에도 stride 적용 |

---

## 핵심 개념 5: Plain Network vs Residual Network

두 구조의 차이를 간단히 비교하면 이렇습니다.

| 구분 | Plain Network | ResNet |
|---|---|---|
| block이 배우는 것 | `H(x)` | `F(x) = H(x) - x` |
| 출력 | `H(x)` | `F(x) + x` |
| 입력 전달 | layer를 순서대로 통과 | 일부 입력이 skip connection으로 직접 전달 |
| identity mapping | 여러 layer가 직접 배워야 함 | `F(x) = 0`이면 됨 |
| 깊어질 때 | degradation problem 발생 가능 | 더 깊은 모델 학습이 쉬워짐 |
| 추가 parameter | 없음 | identity shortcut은 추가 parameter 없음 |

논문은 18-layer와 34-layer plain network를 먼저 비교하고, 34-layer plain network가 18-layer plain network보다 validation error가 높다는 degradation problem을 관찰했습니다. 반면 shortcut을 추가한 residual network에서는 34-layer ResNet이 18-layer ResNet보다 좋아지고, training error도 크게 낮아져 depth 증가로 성능 이득을 얻었습니다.

---

## 핵심 개념 6: Basic Block과 Bottleneck Block

ResNet에는 대표적인 block 구조가 있습니다.

### 1. Basic Block

얕은 ResNet, 예를 들어 ResNet-18, ResNet-34에서 많이 떠올리는 구조입니다.

```text
3×3 conv
→ BN
→ ReLU
→ 3×3 conv
→ BN
→ + shortcut
→ ReLU
```

논문에서 18-layer와 34-layer 구조는 주로 `3×3 conv` 두 개를 쌓은 residual block을 사용합니다. Table 1과 Figure 5에서 18/34-layer는 두 개의 `3×3` convolution으로 구성된 block을 사용하고, 더 깊은 50/101/152-layer는 bottleneck block을 사용합니다.

### 2. Bottleneck Block

더 깊은 ResNet, 예를 들어 ResNet-50, ResNet-101, ResNet-152에서 사용됩니다.

```text
1×1 conv  → channel 줄임
3×3 conv  → 핵심 convolution
1×1 conv  → channel 다시 늘림
```

왜 이렇게 할까요?

깊은 모델에서는 계산량이 커집니다.  
그래서 `1×1 conv`로 channel 수를 줄인 다음, 작은 channel에서 `3×3 conv`를 수행하고, 마지막 `1×1 conv`로 channel을 다시 복원합니다.

논문은 ResNet-50/101/152를 위해 residual function `F`를 2-layer가 아니라 3-layer stack으로 바꾸는 bottleneck design을 사용했으며, 그 세 layer가 `1×1`, `3×3`, `1×1` convolution이라고 설명합니다.

---

## BatchNorm과 ResNet의 연결

직전에 BatchNorm을 봤기 때문에 이 부분이 중요합니다.

ResNet 논문에서는 convolution 뒤, activation function 앞에 BatchNorm을 적용했습니다.

구조는 대략 이렇습니다.

```text
Conv
→ BatchNorm
→ ReLU
```

논문은 ImageNet 구현에서 각 convolution 직후, activation 전에 BatchNorm을 사용했고, weight initialization, SGD, mini-batch size 256, weight decay, momentum 등을 사용했다고 설명합니다. 또 BatchNorm을 사용한 관행을 따르며 dropout은 사용하지 않았다고 밝힙니다.

이 흐름이 아주 중요합니다.

| 논문 | 해결하려는 문제 |
|---|---|
| Dropout | overfitting 완화 |
| BatchNorm | 학습 안정화와 빠른 수렴 |
| ResNet | 매우 깊은 network의 optimization/degradation 문제 완화 |

즉, ResNet은 BatchNorm 위에 쌓인 다음 단계의 아이디어입니다.

BatchNorm으로 깊은 네트워크의 학습이 어느 정도 안정화됐지만, 단순히 layer를 깊게 쌓는 plain network에는 여전히 degradation problem이 있었습니다. ResNet은 여기에 shortcut connection과 residual learning을 추가해 훨씬 깊은 네트워크를 학습 가능하게 만들었습니다.

---

## ResNet이 gradient 흐름에 주는 직관

ResNet을 공부할 때 자주 나오는 설명이 있습니다.

> skip connection이 gradient가 뒤로 잘 흐를 수 있는 길을 만들어준다.

이 설명은 직관적으로 유용합니다.

일반적인 깊은 네트워크에서는 gradient가 많은 layer를 거치며 약해지거나 학습이 어려워질 수 있습니다.

ResNet에서는 shortcut이 있으므로 forward 방향에서는 정보가 더 직접적으로 흐르고, backward 방향에서도 gradient가 더 직접적인 경로를 가질 수 있습니다.

다만 원 논문이 강조한 핵심은 단순히 vanishing gradient 하나가 아닙니다. 논문은 plain network의 degradation이 vanishing gradient 때문에 발생했을 가능성은 낮다고 보고, BatchNorm을 사용해 forward signal의 variance와 backward gradient norm이 건강하게 유지됨을 확인했다고 설명합니다. 저자들은 이 문제를 optimization difficulty로 봅니다.

그래서 입문자 입장에서는 이렇게 이해하는 게 좋습니다.

```text
ResNet은 gradient 흐름에도 도움을 주지만,
논문의 핵심 문제의식은 “깊은 plain network가 좋은 solution을 찾기 어렵다”는 optimization 문제다.
```

---

## 쉬운 비유: 전체 답을 새로 쓰기 vs 수정사항만 쓰기

보고서 초안이 있다고 합시다.

기존 방식은 매번 이렇게 하는 것입니다.

```text
초안을 보고 완전히 새 보고서를 작성하라.
```

ResNet 방식은 이렇게 하는 것입니다.

```text
초안을 그대로 두고,
고쳐야 할 부분만 빨간펜으로 표시하라.
```

후자가 훨씬 쉽습니다.

왜냐하면 초안이 이미 쓸 만하면, 수정할 부분만 작게 쓰면 되기 때문입니다.

ResNet의 residual branch `F(x)`는 이 빨간펜 수정사항에 해당합니다.

```text
최종 보고서 = 기존 초안 + 수정사항
H(x) = x + F(x)
```

---

## ResNet의 실험 결과가 보여준 것

이 논문은 단순히 구조를 제안한 것에서 끝나지 않고, 매우 강한 실험 결과를 보였습니다.

ImageNet에서 ResNet은 최대 152-layer까지 평가됐고, 이 152-layer residual network는 VGG보다 훨씬 깊지만 낮은 complexity를 가졌습니다. ResNet ensemble은 ImageNet test set에서 3.57% top-5 error를 달성했고, ILSVRC 2015 classification task에서 1위를 차지했습니다. 논문은 CIFAR-10에서도 100-layer와 1000-layer 규모의 실험을 제시했습니다.

ImageNet validation 결과에서도 depth가 증가할수록 ResNet-50, ResNet-101, ResNet-152가 좋은 성능을 보였습니다.

---

## 이 논문이 딥러닝 역사에서 중요한 이유

ResNet은 단순한 CNN architecture 하나가 아닙니다.

더 큰 의미는 이것입니다.

**매우 깊은 네트워크를 학습 가능하게 만든 일반적인 설계 원칙을 제시했다.**

이후 많은 모델들이 skip connection 또는 residual connection을 사용합니다.

| 분야 | residual connection의 영향 |
|---|---|
| CNN | ResNet, DenseNet, EfficientNet 계열 |
| Object Detection | Faster R-CNN, Mask R-CNN backbone 등 |
| Segmentation | U-Net 변형, DeepLab 등 |
| Transformer | self-attention block과 MLP block 주변의 residual connection |
| LLM | Transformer layer의 residual stream |

특히 Transformer와 LLM을 공부할 때도 residual connection은 핵심입니다.

Transformer block도 대략 이렇게 되어 있습니다.

```text
x + Attention(x)
x + MLP(x)
```

즉, ResNet의 사고방식은 컴퓨터비전 CNN을 넘어 현대 딥러닝 architecture의 기본 문법이 되었습니다.

---

## ResNet과 이전 논문들의 연결

### Deep Learning 논문과의 연결

Deep Learning 리뷰는 layer를 깊게 쌓으면 더 추상적인 representation을 학습할 수 있다고 설명했습니다.

ResNet은 그다음 질문에 답합니다.

```text
그럼 layer를 매우 깊게 쌓으려면 어떻게 해야 하는가?
```

ResNet의 답은:

```text
residual connection을 넣어라.
```

입니다.

### Dropout과의 연결

Dropout은 overfitting을 줄이기 위한 regularization입니다.

ResNet은 주로 overfitting보다 **optimization difficulty**를 다룹니다.

| 논문 | 주요 문제 |
|---|---|
| Dropout | 큰 모델이 training data에 과적합됨 |
| ResNet | 깊은 plain network가 training data조차 잘 못 맞춤 |

즉, 둘은 해결하는 문제가 다릅니다.

### BatchNorm과의 연결

BatchNorm은 layer activation 분포를 안정화해 학습을 빠르게 합니다.

ResNet 논문도 BatchNorm을 사용합니다.

하지만 BatchNorm만으로는 깊은 plain network의 degradation problem을 완전히 해결하지 못했고, ResNet은 residual connection으로 그 문제를 직접 겨냥했습니다. 논문은 plain network에 BN을 사용했음에도 34-layer plain network가 18-layer plain network보다 더 높은 training error를 보였다고 설명합니다.

---

## ResNet을 오해하면 안 되는 부분

### 오해 1. ResNet은 단순히 vanishing gradient만 해결한 논문이다

일부는 맞지만, 불완전합니다.

ResNet이 gradient 흐름에 도움을 주는 것은 맞습니다. 하지만 논문이 강조한 degradation problem은 단순 vanishing gradient 문제가 아니었습니다. 저자들은 BN을 사용한 plain network에서 forward/backward signal이 완전히 사라지는 것은 아니라고 보고, 더 깊은 plain network가 optimization상 좋은 solution을 찾기 어렵다고 해석했습니다.

더 정확한 표현은:

```text
ResNet은 매우 깊은 네트워크의 optimization을 쉽게 만든다.
```

입니다.

### 오해 2. ResNet은 parameter를 많이 늘려서 성능이 좋아진 것이다

identity shortcut 자체는 parameter를 추가하지 않습니다.

논문은 identity shortcut connection이 추가 parameter나 계산 복잡도를 만들지 않기 때문에, plain network와 residual network를 동일한 parameter 수, depth, width, computational cost 조건에서 비교할 수 있다고 설명합니다.

즉, ResNet의 핵심 성능 향상은 단순히 parameter 수를 늘린 데서 나온 것이 아니라, **문제를 residual form으로 재구성한 데서 나온 것**입니다.

### 오해 3. Residual block은 항상 입력을 그대로 보존한다

아닙니다.

입력 `x`가 shortcut으로 더해지긴 하지만, residual branch `F(x)`가 큰 값을 만들면 출력은 충분히 달라질 수 있습니다.

```text
y = x + F(x)
```

여기서 `F(x)`가 작으면 거의 identity에 가깝고, `F(x)`가 크면 입력을 많이 수정합니다.

즉, ResNet은 “아무것도 안 바꾸는 모델”이 아니라, **필요한 만큼만 바꾸기 쉬운 모델**입니다.

---

## 실무에서 ResNet을 어떻게 이해하면 좋나?

### 1. 이미지 모델의 기본 backbone

ResNet은 오랫동안 이미지 분류, 객체 탐지, 세그멘테이션에서 기본 backbone으로 쓰였습니다.

예를 들어:

```text
ResNet-50 backbone
→ detection head
→ object detection
```

또는:

```text
ResNet encoder
→ segmentation decoder
→ semantic segmentation
```

같은 구조입니다.

### 2. ResNet-18, 34, 50, 101, 152의 차이

숫자는 대략 layer 수를 의미합니다.

| 모델 | 특징 |
|---|---|
| ResNet-18 | 작고 빠름, basic block |
| ResNet-34 | basic block, 더 깊음 |
| ResNet-50 | bottleneck block, 실무에서 매우 자주 사용 |
| ResNet-101 | 더 깊고 강력하지만 비용 증가 |
| ResNet-152 | 매우 깊음, 논문에서 ImageNet 실험의 핵심 |

논문 Table 1은 ImageNet용 18/34/50/101/152-layer ResNet 구조를 비교하고, 50-layer 이상에서는 `1×1-3×3-1×1` bottleneck block을 사용합니다.

### 3. Residual connection은 CNN에만 쓰이는 것이 아니다

이게 정말 중요합니다.

ResNet을 배운 이유는 단지 이미지 분류 모델 하나를 알기 위해서가 아닙니다.

앞으로 Transformer를 볼 때 이런 구조가 계속 나옵니다.

```text
x = x + self_attention(x)
x = x + feed_forward(x)
```

즉, residual connection은 현대 딥러닝의 기본 구성요소입니다.

---

## 이 논문의 핵심 직관을 다시 정리하면

ResNet은 이런 문제의식에서 시작합니다.

```text
깊은 모델이 더 표현력이 크다면,
왜 layer를 더 쌓았을 때 training error가 더 나빠질까?
```

그리고 이렇게 답합니다.

```text
깊은 plain network는 좋은 mapping, 특히 identity mapping을 찾기 어렵다.
그러니 layer가 전체 mapping H(x)를 직접 배우게 하지 말고,
입력 x에서 얼마나 바뀌어야 하는지 residual F(x)를 배우게 하자.
```

그래서 최종 구조는:

```text
output = x + F(x)
```

입니다.

이 한 줄이 ResNet의 본질입니다.

---

## 오늘 공부용 요약

**논문명:** Deep Residual Learning for Image Recognition  
**저자:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

**핵심 결론:** 매우 깊은 plain network는 training error까지 나빠지는 degradation problem을 겪을 수 있다. ResNet은 layer가 원하는 mapping `H(x)`를 직접 학습하는 대신 residual mapping `F(x) = H(x) - x`를 학습하게 하고, shortcut connection으로 `F(x) + x`를 만들어 이 문제를 완화한다.

**왜 중요함:** ResNet은 매우 깊은 neural network를 실제로 학습 가능하게 만든 대표 구조이고, 이후 CNN뿐 아니라 Transformer와 LLM architecture에도 영향을 준 residual connection의 핵심 원리를 보여준다.

**입문자가 배울 점:** 깊은 모델의 문제는 단순히 overfitting만이 아니다. training error 자체가 나빠지는 optimization 문제가 생길 수 있고, residual learning은 이 문제를 “전체 mapping 학습”에서 “수정량 학습”으로 바꿔 해결한다.

**가장 중요한 문장:** ResNet은 `H(x)`를 직접 배우는 대신 `F(x) = H(x) - x`를 배우고, 최종 출력은 `F(x) + x`로 만든다.

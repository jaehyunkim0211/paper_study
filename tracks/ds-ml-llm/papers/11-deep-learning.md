# Deep Learning — Yann LeCun, Yoshua Bengio, Geoffrey Hinton

## 결론부터

이 논문의 결론은 이겁니다.

**딥러닝의 핵심은 사람이 직접 feature를 설계하지 않고, 모델이 데이터로부터 여러 층의 표현 representation을 스스로 학습한다는 것이다.**  
낮은 층은 단순한 패턴을 배우고, 높은 층은 더 추상적인 개념을 배웁니다.

이미지라면 대략 이런 식입니다.

```text
raw pixels
→ edge
→ texture / motif
→ object part
→ object
```

텍스트라면 이런 식입니다.

```text
word
→ word vector
→ phrase meaning
→ sentence representation
→ translation / classification / answer
```

LeCun, Bengio, Hinton의 2015년 *Nature* 리뷰 논문은 딥러닝을 “여러 processing layer로 이루어진 계산 모델이 여러 수준의 추상화로 데이터 표현을 학습하는 방법”으로 설명합니다. 이 논문은 딥러닝이 음성 인식, 이미지 인식, 객체 탐지, 약물 발견, 유전체학 등 여러 영역에서 성능을 크게 끌어올렸다고 정리합니다.

---

## 이 논문을 한 문장으로 요약하면

**딥러닝은 feature engineering을 사람이 하는 방식에서, representation learning을 모델이 하는 방식으로 넘어가게 만든 기술이다.**

이전의 전통적 머신러닝에서는 보통 사람이 먼저 feature를 설계했습니다.

예를 들어 이미지 분류라면:

```text
이미지
→ 사람이 edge, texture, color histogram, shape feature 설계
→ classifier가 분류
```

딥러닝은 이 과정을 바꿉니다.

```text
이미지
→ neural network가 필요한 feature를 층별로 학습
→ classifier까지 함께 학습
```

논문은 기존 머신러닝이 raw natural data를 직접 처리하는 데 한계가 있었고, 오랫동안 도메인 전문가가 raw data를 적절한 feature vector로 바꾸는 feature extractor를 설계해야 했다고 설명합니다. 반면 representation learning은 raw data를 입력받아 detection이나 classification에 필요한 표현을 자동으로 발견하게 해주는 방법이라고 정리합니다.

---

## 왜 이 논문이 중요한가?

이 논문은 새로운 단일 알고리즘을 제안한 논문이라기보다는, **딥러닝이 왜 중요한지, 어떤 원리로 작동하는지, 왜 2010년대에 폭발적으로 성공했는지 정리한 대표 리뷰 논문**입니다.

지금까지 우리가 본 논문 흐름과 연결하면 위치가 분명합니다.

| 이전 논문 | 핵심 질문 |
|---|---|
| Tidy Data | 데이터를 분석하기 좋은 구조로 만들었는가? |
| Two Cultures | 설명인가, 예측인가? |
| Model Evaluation | 모델을 공정하게 평가했는가? |
| Random Forest / XGBoost / LightGBM | tabular data에서 강한 예측 모델은 어떻게 만드는가? |
| SHAP | 복잡한 모델의 예측을 어떻게 설명하는가? |
| **Deep Learning** | raw data에서 표현 자체를 어떻게 학습하는가? |

트리 기반 모델들은 주로 tabular data에서 강했습니다.  
딥러닝은 이미지, 음성, 텍스트, 영상처럼 **raw signal**에 가까운 데이터에서 특히 강력해졌습니다.

---

## 핵심 개념 1: Representation Learning

### 결론부터

**Representation learning은 모델이 문제를 풀기 좋은 표현을 스스로 배우는 것이다.**

예를 들어 고양이 이미지를 분류한다고 해봅시다.

전통적인 방식에서는 사람이 이런 feature를 만들 수 있습니다.

```text
귀 모양
눈 위치
털 색깔
윤곽선
수염 패턴
```

하지만 이런 feature를 사람이 모든 상황에 맞게 설계하기는 어렵습니다.

왜냐하면 고양이는 사진마다 다르게 보이기 때문입니다.

| 변화 | 예시 |
|---|---|
| 위치 | 화면 왼쪽, 오른쪽, 중앙 |
| 크기 | 가까이 찍힘, 멀리 찍힘 |
| 자세 | 앉음, 누움, 뛰어감 |
| 조명 | 밝음, 어두움 |
| 배경 | 집, 거리, 숲 |
| 가려짐 | 일부가 가려짐 |

딥러닝은 이런 feature를 사람이 일일이 만들지 않고, 데이터에서 자동으로 배웁니다.

논문은 딥러닝을 여러 수준의 representation을 가진 representation learning이라고 설명합니다. 각 layer는 이전 layer의 표현을 조금 더 추상적인 표현으로 바꾸며, 충분히 많은 변환을 쌓으면 복잡한 함수를 학습할 수 있다고 설명합니다. 이미지 예시에서는 첫 번째 layer가 edge를, 두 번째 layer가 edge 조합인 motif를, 그 다음 layer가 object part와 object 조합을 학습한다고 설명합니다.

---

## 핵심 개념 2: 여러 층이 추상화를 만든다

딥러닝에서 “deep”은 단순히 layer가 많다는 뜻만이 아닙니다.

중요한 건 layer가 쌓이면서 표현이 점점 추상화된다는 점입니다.

이미지 예시로 보면:

| 층 | 배우는 것 |
|---|---|
| 입력층 | 픽셀값 |
| 낮은 hidden layer | edge, 방향, 색 대비 |
| 중간 hidden layer | texture, motif, 작은 형태 |
| 높은 hidden layer | 눈, 귀, 바퀴, 문, 얼굴 같은 object part |
| 마지막 layer | 고양이, 자동차, 사람 같은 class |

이게 딥러닝의 핵심 직관입니다.

> 낮은 층은 단순한 시각 패턴을 배우고, 높은 층은 의미 있는 개념을 배운다.

물론 실제 neural network가 항상 인간이 이름 붙이기 쉬운 feature만 배우는 것은 아닙니다. 하지만 중요한 점은 **표현이 층을 지나며 점점 더 분류나 예측에 유용한 형태로 변환된다**는 것입니다.

---

## 핵심 개념 3: Backpropagation

### 결론부터

**Backpropagation은 neural network가 자신의 weight를 어떻게 바꿔야 loss가 줄어드는지 계산하는 방법입니다.**

딥러닝 모델에는 수많은 weight가 있습니다.

예를 들어 어떤 neural network가 이미지를 보고 “고양이”라고 예측해야 하는데 “개”라고 틀렸다고 합시다.

그러면 모델은 질문해야 합니다.

```text
어떤 weight를 얼마나 바꾸면 다음에는 덜 틀릴까?
```

Backpropagation은 이 질문에 답합니다.

논문은 딥러닝이 backpropagation algorithm을 이용해 각 layer의 representation을 계산하는 내부 parameter를 어떻게 바꿔야 하는지 알려준다고 설명합니다. 또한 학습 과정에서는 objective function이 출력과 정답 사이의 error를 측정하고, weight라는 조절 가능한 parameter를 error가 줄어드는 방향으로 수정한다고 설명합니다.

---

## Backpropagation을 쉽게 이해하기

학생이 시험을 봤다고 해봅시다.

```text
예상 답: 70점
실제 정답 기준: 90점
오차: 20점
```

학생은 틀린 원인을 거꾸로 추적합니다.

```text
마지막 계산에서 뭘 잘못했나?
그 이전 단계에서는 뭘 잘못했나?
처음 입력을 어떻게 잘못 해석했나?
```

Neural network도 비슷합니다.

```text
출력층에서 error 계산
→ 이전 layer의 weight가 error에 얼마나 기여했는지 계산
→ 그 이전 layer도 계산
→ 입력 쪽까지 거꾸로 전파
```

그래서 이름이 **back-propagation**, 즉 error gradient를 뒤로 전파하는 방법입니다.

논문의 Figure 1은 multilayer neural network와 backpropagation을 설명합니다. 그림은 forward pass에서 layer별 출력을 계산하고, output과 정답의 차이에서 시작한 error derivative를 chain rule로 아래 layer까지 전달해 weight gradient를 계산하는 구조를 보여줍니다.

---

## 핵심 개념 4: Gradient Descent와 SGD

Backpropagation은 gradient를 계산하는 방법입니다.  
그 gradient를 사용해 실제 weight를 바꾸는 대표 방법이 **gradient descent**입니다.

논문은 각 weight에 대해 error가 증가하거나 감소하는 방향을 나타내는 gradient vector를 계산하고, weight vector를 gradient의 반대 방향으로 조정한다고 설명합니다. 평균 objective function을 고차원 weight 공간의 울퉁불퉁한 지형으로 볼 수 있고, negative gradient는 error가 낮은 지점으로 내려가는 가장 가파른 방향을 가리킨다고 설명합니다.

수식 느낌은 이렇습니다.

```text
새 weight = 기존 weight - learning_rate × gradient
```

이전 Gradient Boosting에서 gradient를 배웠죠.

거기서도 핵심은 같았습니다.

```text
gradient = loss가 커지는 방향
negative gradient = loss가 줄어드는 방향
```

딥러닝에서도 같습니다.

다만 딥러닝에서는 보통 neural network의 수많은 weight를 gradient 방향으로 업데이트합니다.

실무에서는 전체 데이터를 한 번에 쓰기보다 mini-batch 단위로 gradient를 계산하는 **SGD, stochastic gradient descent**를 많이 씁니다. 논문도 SGD를 몇 개의 example을 보여주고 output과 error를 계산한 뒤, 그 작은 sample에 대한 평균 gradient로 weight를 조정하는 절차라고 설명합니다.

---

## 핵심 개념 5: Non-linearity

### 결론부터

**Neural network가 강력한 이유는 layer 사이에 non-linear function이 들어가기 때문입니다.**

만약 모든 layer가 단순한 선형 변환이면, layer를 아무리 많이 쌓아도 결국 하나의 선형 변환과 같습니다.

```text
linear + linear + linear = still linear
```

그래서 neural network에는 ReLU 같은 non-linear activation function이 필요합니다.

대표적인 ReLU는 이렇게 생겼습니다.

```text
ReLU(z) = max(0, z)
```

즉:

| z | ReLU(z) |
|---:|---:|
| -3 | 0 |
| -1 | 0 |
| 0 | 0 |
| 2 | 2 |
| 5 | 5 |

논문은 현재 가장 널리 쓰이는 non-linear function으로 ReLU를 설명하고, ReLU가 과거의 tanh나 sigmoid 같은 매끄러운 비선형 함수보다 많은 layer를 가진 network에서 보통 훨씬 빠르게 학습된다고 설명합니다. 또한 hidden layer가 input을 비선형적으로 왜곡해 마지막 layer에서 category가 linearly separable해지도록 돕는다고 설명합니다.

쉽게 말하면:

> Non-linearity가 있어야 neural network가 복잡한 경계와 추상적 패턴을 배울 수 있다.

---

## 쉬운 예시: 고양이와 강아지 분류

딥러닝이 하는 일을 아주 간단히 생각해봅시다.

입력은 이미지입니다.

```text
224 × 224 × 3 픽셀
```

처음에는 숫자 배열일 뿐입니다.

```text
[123, 95, 88, 130, ...]
```

사람은 이걸 보고 고양이인지 강아지인지 알 수 있지만, 모델은 처음에는 모릅니다.

딥러닝 모델은 학습 과정에서 다음을 배웁니다.

```text
픽셀 조합 → edge
edge 조합 → 귀, 눈, 코, 털 패턴
부품 조합 → 얼굴
얼굴/몸통 조합 → 고양이 또는 강아지
```

즉, 딥러닝은 raw pixel에서 class label로 바로 가는 게 아니라, 중간에 여러 수준의 representation을 만들어 갑니다.

이것이 논문의 가장 중요한 메시지입니다.

---

## 핵심 개념 6: CNN, Convolutional Neural Network

### 결론부터

**CNN은 이미지처럼 공간 구조가 있는 데이터를 잘 처리하도록 설계된 neural network입니다.**

이미지는 아무 숫자 배열이 아닙니다.  
주변 픽셀끼리 의미가 있습니다.

예를 들어 고양이 눈을 찾으려면 전체 이미지를 한 번에 보는 것보다, 작은 지역 patch를 보는 것이 자연스럽습니다.

CNN은 이 성질을 이용합니다.

논문은 ConvNet이 여러 array 형태의 데이터를 처리하도록 설계되었다고 설명합니다. 예를 들어 컬러 이미지는 세 개의 2D pixel intensity array로 볼 수 있고, 1D signal, 2D image나 spectrogram, 3D video나 volumetric image도 이런 구조를 가집니다. 논문은 ConvNet의 네 가지 핵심 아이디어로 local connections, shared weights, pooling, many layers를 제시합니다.

---

## CNN의 핵심 아이디어

### 1. Local connection

이미지 전체를 한 번에 연결하지 않고, 작은 영역을 봅니다.

```text
작은 patch를 보고 edge나 texture를 감지
```

### 2. Shared weights

같은 filter를 이미지 여러 위치에 반복 적용합니다.

예를 들어 “세로 edge 감지 filter”는 이미지 왼쪽에서도, 오른쪽에서도, 위쪽에서도 유용합니다.

```text
같은 detector를 여러 위치에서 사용
```

### 3. Pooling

작은 위치 변화에 덜 민감해지게 정보를 요약합니다.

예를 들어 고양이 귀가 정확히 1픽셀 오른쪽으로 이동해도 여전히 고양이입니다.

```text
작은 위치 변화에 robust하게 만들기
```

### 4. Many layers

낮은 layer는 edge를, 높은 layer는 object part와 object를 학습합니다.

---

## CNN이 왜 혁명적이었나?

논문은 2012년 ImageNet competition에서 deep convolutional network가 약 100만 장의 이미지와 1,000개 class를 가진 데이터셋에 적용되어, 기존 최고 경쟁 방법들의 error rate를 거의 절반으로 줄이는 결과를 냈다고 설명합니다. 이 성공에는 GPU, ReLU, dropout, data augmentation 등이 함께 기여했고, 이후 ConvNet이 computer vision의 대부분 recognition과 detection task에서 지배적인 접근이 되었다고 정리합니다.

이 부분이 딥러닝 역사에서 매우 중요합니다.

2012년 ImageNet 성공은 많은 연구자와 기업이 딥러닝을 본격적으로 받아들이게 만든 계기였습니다.

---

## 핵심 개념 7: Distributed Representation

### 결론부터

**Distributed representation은 하나의 개념을 하나의 feature로 표현하지 않고, 여러 feature의 조합으로 표현하는 방식입니다.**

예를 들어 단어를 생각해봅시다.

전통적인 one-hot encoding은 이런 식입니다.

```text
cat = [0, 0, 1, 0, 0, ...]
dog = [0, 0, 0, 1, 0, ...]
car = [0, 1, 0, 0, 0, ...]
```

문제는 `cat`과 `dog`가 의미적으로 가깝다는 사실이 표현에 잘 드러나지 않는다는 것입니다.

딥러닝의 word vector는 다릅니다.

```text
cat = [0.2, -0.7, 1.1, 0.4, ...]
dog = [0.3, -0.6, 1.0, 0.5, ...]
car = [-0.8, 0.1, -0.4, 1.2, ...]
```

이제 의미적으로 비슷한 단어들이 vector space에서 가까워질 수 있습니다.

논문은 neural language model이 각 단어를 real-valued feature vector와 연결하고, 의미적으로 관련된 단어들이 vector space에서 서로 가깝게 위치한다고 설명합니다. 또한 이런 semantic feature들은 입력에 명시적으로 존재한 것이 아니라 학습 과정에서 발견된 것이라고 설명합니다.

---

## 핵심 개념 8: RNN, Recurrent Neural Network

### 결론부터

**RNN은 순서가 있는 데이터를 처리하기 위한 neural network입니다.**

이미지는 공간 구조가 중요합니다.  
텍스트와 음성은 순서가 중요합니다.

예를 들어 문장을 봅시다.

```text
나는 어제 영화를 봤다
```

단어 하나하나도 중요하지만, 순서가 바뀌면 의미도 달라집니다.

RNN은 sequence를 한 요소씩 읽으면서 hidden state에 과거 정보를 저장합니다.

```text
단어 1 읽음 → hidden state 업데이트
단어 2 읽음 → hidden state 업데이트
단어 3 읽음 → hidden state 업데이트
...
```

논문은 speech나 language처럼 sequential input을 다루는 task에서는 RNN을 쓰는 것이 더 나은 경우가 많다고 설명합니다. RNN은 sequence를 한 요소씩 처리하며, hidden unit의 state vector가 과거 요소들의 정보를 암묵적으로 담는다고 설명합니다.

---

## RNN의 어려움: vanishing / exploding gradients

RNN은 강력하지만 훈련이 어렵습니다.

왜냐하면 긴 sequence에서는 gradient가 time step을 따라 계속 전달되는데, 이 과정에서 gradient가 너무 작아지거나 너무 커질 수 있기 때문입니다.

| 문제 | 의미 |
|---|---|
| Vanishing gradient | 먼 과거 정보의 영향이 거의 사라짐 |
| Exploding gradient | gradient가 너무 커져 학습이 불안정해짐 |

논문은 RNN이 강력한 dynamic system이지만, backpropagated gradient가 time step마다 커지거나 작아져 긴 시간에 걸쳐 explode 또는 vanish하는 문제가 있다고 설명합니다.

이 문제를 완화하기 위해 LSTM 같은 구조가 등장했습니다.

논문은 LSTM이 입력을 오래 기억하도록 설계된 특수 hidden unit을 사용하며, 이후 conventional RNN보다 더 효과적임이 입증되었고 speech recognition과 machine translation에서 사용되었다고 설명합니다.

---

## CNN과 RNN을 비교하면

| 구분 | CNN | RNN |
|---|---|---|
| 주로 쓰는 데이터 | 이미지, 영상, spectrogram | 텍스트, 음성, 시계열 |
| 핵심 구조 | convolution, pooling | recurrent state |
| 잘 잡는 패턴 | 공간적 지역 패턴 | 순서와 시간 의존성 |
| 예시 | 객체 인식, 이미지 분류 | 언어 모델, 번역, 음성 인식 |
| 직관 | 작은 filter가 이미지 곳곳을 훑음 | 문장을 순서대로 읽으며 상태를 업데이트 |

현대에는 Transformer가 NLP와 vision에서도 많이 쓰이지만, 이 논문이 나온 2015년 당시에는 CNN과 RNN이 딥러닝의 대표 구조였습니다.

---

## 이 논문이 말하는 딥러닝 성공의 이유

이 논문은 딥러닝이 갑자기 “아이디어가 새로워서”만 성공한 것은 아니라고 봅니다.

여러 조건이 함께 맞았습니다.

| 요인 | 설명 |
|---|---|
| 큰 데이터 | 대규모 labeled data가 생김 |
| GPU | 대규모 행렬 연산을 빠르게 수행 |
| Backpropagation | 깊은 network의 weight를 효율적으로 학습 |
| ReLU | 깊은 network 학습을 더 쉽게 만듦 |
| Dropout | overfitting을 줄이는 regularization |
| CNN/RNN 구조 | 이미지와 sequence에 맞는 inductive bias |
| End-to-end training | feature extractor와 classifier를 함께 학습 |

논문은 초기 unsupervised pre-training이 딥러닝 부활에 중요한 역할을 했지만, 이후 큰 데이터와 발전한 학습 방법 덕분에 purely supervised learning의 성공이 커졌다고 설명합니다. 또한 unsupervised learning이 장기적으로 다시 중요해질 것이라고 전망합니다.

---

## 이 논문의 관점에서 딥러닝과 기존 ML의 차이

가장 중요한 차이는 feature를 누가 만드느냐입니다.

| 구분 | 전통적 ML | 딥러닝 |
|---|---|---|
| 입력 | 사람이 설계한 feature | raw data 또는 덜 가공된 data |
| feature engineering | 매우 중요 | 모델이 상당 부분 자동 학습 |
| 모델 구조 | 보통 shallow model | 여러 layer |
| 강한 영역 | tabular data, 작은 데이터 | 이미지, 음성, 텍스트, 대규모 데이터 |
| 예시 | Random Forest, XGBoost, SVM | CNN, RNN, Transformer |
| 장점 | 해석과 튜닝이 비교적 쉬움 | 표현 학습이 강력함 |
| 단점 | raw signal 처리에 한계 | 데이터와 계산량이 많이 필요할 수 있음 |

단, “딥러닝이 항상 전통 ML보다 좋다”는 뜻은 아닙니다.

많은 비즈니스 tabular data에서는 여전히 XGBoost, LightGBM 같은 모델이 매우 강합니다.  
반대로 이미지, 음성, 텍스트처럼 raw data의 구조가 복잡한 영역에서는 딥러닝이 강력합니다.

---

## 쉬운 실무 예시: 이미지 불량 검출

제조 공장에서 제품 이미지를 보고 불량 여부를 판단한다고 해봅시다.

### 전통적 ML 방식

사람이 feature를 설계합니다.

```text
스크래치 길이
색상 변화
가장자리 패턴
특정 영역 밝기
texture 지표
```

그리고 Random Forest나 SVM을 학습합니다.

문제는 불량 유형이 다양하고 조명, 각도, 배경이 바뀌면 feature 설계가 어려워진다는 점입니다.

### 딥러닝 방식

CNN에 이미지를 넣습니다.

```text
제품 이미지 → CNN → 불량 확률
```

CNN은 낮은 layer에서 edge와 texture를 배우고, 높은 layer에서 불량 패턴을 배웁니다.

사람이 feature를 전부 설계하지 않아도 됩니다.

이게 딥러닝의 강점입니다.

---

## 쉬운 실무 예시: 고객 이탈 예측과의 비교

반대로 고객 이탈 예측은 보통 tabular data입니다.

| 고객 | 가입기간 | 월요금 | 사용량 | 불만 횟수 | 이탈 여부 |
|---|---:|---:|---:|---:|---|
| A | 3개월 | 80,000원 | 2GB | 2 | 예 |

이런 데이터에서는 XGBoost나 LightGBM이 여전히 강력합니다.

딥러닝도 쓸 수 있지만, 데이터 크기와 feature 구조에 따라 반드시 더 좋지는 않습니다.

즉, 모델 선택 감각은 이렇게 잡으면 됩니다.

| 데이터 유형 | 첫 후보 |
|---|---|
| 깔끔한 tabular data | Logistic Regression, Random Forest, XGBoost, LightGBM |
| 이미지 | CNN, Vision Transformer |
| 텍스트 | RNN historically, 현재는 Transformer |
| 음성 | CNN/RNN/Transformer 계열 |
| 대규모 multimodal data | Deep learning 계열 |

---

## 이 논문의 한계: 2015년 리뷰라는 점

이 논문은 매우 중요한 고전이지만, 2015년에 나온 리뷰입니다.

그래서 현재 관점에서는 빠진 것이 많습니다.

| 빠진 현대 주제 | 이유 |
|---|---|
| Transformer | 2017년 이후 핵심 구조 |
| BERT, GPT | 논문 이후 등장 |
| Diffusion model | 이미지 생성 분야의 이후 핵심 흐름 |
| Foundation model | 대규모 사전학습 모델 패러다임 |
| RLHF | LLM alignment에서 중요 |
| RAG | LLM 응용에서 중요 |
| LoRA/QLoRA | 현대 fine-tuning 실무에서 중요 |

하지만 이 논문은 여전히 중요합니다.

왜냐하면 Transformer나 LLM을 공부하기 전에 알아야 하는 기본 개념이 대부분 여기 들어 있기 때문입니다.

```text
representation learning
backpropagation
gradient descent
hidden layer
non-linearity
CNN
RNN
distributed representation
end-to-end learning
```

즉, 이 논문은 최신 모델 논문이 아니라 **딥러닝 사고방식의 기초 문서**로 읽어야 합니다.

---

## 이전 논문들과 연결해서 이해하기

### Two Cultures와의 연결

Breiman의 *Two Cultures*는 algorithmic modeling culture를 강조했습니다.

딥러닝은 그 흐름의 대표적인 사례입니다.

```text
정확한 데이터 생성 확률모형을 사람이 세우기보다,
큰 데이터에서 복잡한 input-output mapping을 학습한다.
```

다만 딥러닝은 단순히 예측 모델만이 아니라, representation 자체를 학습한다는 점에서 더 강력합니다.

### Model Evaluation과의 연결

딥러닝은 parameter가 많고 overfitting 위험도 큽니다.

그래서 train/validation/test split이 매우 중요합니다.

```text
train: weight 학습
validation: architecture, learning rate, regularization 선택
test: 최종 일반화 성능 평가
```

특히 딥러닝에서는 validation loss를 보며 early stopping을 하거나, learning rate scheduling을 합니다.

이전 Raschka 논문의 평가 설계 원칙이 그대로 적용됩니다.

### XGBoost/LightGBM과의 연결

트리 부스팅과 딥러닝은 둘 다 강력한 예측 모델이지만, 강한 영역이 다릅니다.

| 모델 계열 | 강한 데이터 |
|---|---|
| XGBoost / LightGBM | structured tabular data |
| Deep Learning | unstructured raw data: image, audio, text, video |

따라서 실무에서는 “딥러닝이 최신이니 무조건 딥러닝”이 아니라, 데이터 형태와 문제를 보고 선택해야 합니다.

### SHAP과의 연결

딥러닝 모델도 해석이 필요합니다.

다만 트리 모델에 비해 딥러닝 해석은 더 어렵습니다.

SHAP, saliency map, integrated gradients, attention visualization 같은 방법들이 쓰이지만, 해석은 항상 조심해야 합니다.

SHAP에서 말했듯이:

```text
설명은 모델의 예측 논리를 설명하는 것이지,
현실의 인과관계를 자동으로 증명하는 것이 아니다.
```

---

## 입문자가 꼭 잡아야 할 개념 7개

### 1. Representation

모델이 데이터를 이해하기 위해 내부적으로 만드는 표현입니다.

```text
raw data → useful representation
```

### 2. Layer

입력을 다른 표현으로 바꾸는 단계입니다.

```text
layer 1 → layer 2 → layer 3
```

### 3. Weight

모델이 학습하는 parameter입니다.

```text
weight가 바뀌면 input-output function이 바뀜
```

### 4. Loss / Objective Function

예측이 얼마나 틀렸는지 측정하는 함수입니다.

```text
loss가 낮을수록 모델이 더 잘 맞춤
```

### 5. Backpropagation

loss를 줄이기 위해 각 weight를 어떻게 바꿔야 하는지 계산하는 방법입니다.

### 6. Gradient Descent / SGD

계산한 gradient를 이용해 실제 weight를 업데이트하는 방법입니다.

### 7. Non-linearity

여러 layer를 쌓았을 때 복잡한 패턴을 배울 수 있게 해주는 요소입니다.

---

## 이 논문을 공부할 때 가장 중요한 질문

이 논문을 읽을 때 계속 물어야 할 질문은 이것입니다.

**“이 모델은 어떤 representation을 배우고 있는가?”**

딥러닝은 단순히 layer가 많은 모델이 아닙니다.

딥러닝은:

```text
데이터를 문제 해결에 유용한 표현으로 바꾸는 계층적 representation learning 시스템
```

입니다.

이미지 모델을 볼 때도:

```text
어떤 layer가 edge를 배우는가?
어떤 layer가 object part를 배우는가?
어떤 layer가 class 정보를 담는가?
```

텍스트 모델을 볼 때도:

```text
단어가 어떤 vector로 표현되는가?
문장이 어떤 representation으로 압축되는가?
의미적으로 비슷한 표현이 가까워지는가?
```

이 질문을 해야 합니다.

---

## 이 논문을 오해하면 안 되는 부분

### 오해 1. 딥러닝은 feature engineering이 전혀 필요 없다

완전히 맞는 말은 아닙니다.

딥러닝은 feature learning을 많이 자동화했지만, 여전히 다음은 중요합니다.

| 여전히 중요한 것 |
|---|
| 데이터 수집 |
| 라벨 품질 |
| input normalization |
| augmentation |
| architecture 선택 |
| loss function 선택 |
| evaluation 설계 |
| leakage 방지 |
| 배포 후 monitoring |

즉, 딥러닝은 feature engineering의 일부를 모델 안으로 흡수했지만, 데이터사이언티스트의 일이 사라진 것은 아닙니다.

### 오해 2. layer가 많으면 무조건 좋다

layer가 많아지면 표현력은 커질 수 있지만, 학습이 어려워지고 overfitting 위험도 생깁니다.

그래서 이후 Dropout, Batch Normalization, ResNet 같은 논문들이 중요해집니다.

이 논문 다음에 우리가 볼 논문들이 바로 그 문제를 다룹니다.

### 오해 3. 딥러닝은 인간 뇌를 그대로 모방한다

딥러닝은 생물학적 신경망에서 영감을 받았지만, 실제 뇌를 그대로 복제한 것은 아닙니다.

실무적으로는 딥러닝을 **gradient 기반 함수 근사 모델**로 이해하는 것이 더 정확합니다.

---

## 오늘 공부용 요약

**논문명:** Deep Learning  
**저자:** Yann LeCun, Yoshua Bengio, Geoffrey Hinton  
**출판:** *Nature*, 2015년, 521권, 436–444쪽.

**핵심 결론:** 딥러닝은 여러 processing layer를 통해 raw data로부터 여러 수준의 representation을 자동으로 학습하는 방법이다. 낮은 layer는 단순한 패턴을, 높은 layer는 추상적 개념을 학습한다.

**왜 중요함:** 사람이 feature를 직접 설계하던 전통적 머신러닝에서, 모델이 feature와 classifier를 함께 학습하는 representation learning 패러다임으로 넘어가게 만든 핵심 흐름을 설명한다.

**입문자가 배울 점:** 딥러닝의 본질은 “layer가 많은 모델”이 아니라, “데이터를 점점 더 유용한 표현으로 바꾸는 계층적 representation learning”이다.

**가장 중요한 문장:** 딥러닝은 raw data를 입력받아, 여러 층의 비선형 변환을 통해 예측에 필요한 표현을 스스로 학습한다.

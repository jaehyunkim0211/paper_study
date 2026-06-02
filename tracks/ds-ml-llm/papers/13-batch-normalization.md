# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift — Ioffe & Szegedy

## 결론부터

이 논문의 결론은 이겁니다.

**딥러닝 학습을 더 빠르고 안정적으로 만들려면, 각 layer에 들어가는 값의 분포를 mini-batch 단위로 정규화하면 된다.**  
이 방법이 바로 **Batch Normalization**, 줄여서 **BatchNorm** 또는 **BN**입니다.

쉽게 말하면:

> layer 중간중간에서 값들이 너무 커지거나, 너무 작아지거나, 분포가 계속 흔들리는 것을 막기 위해  
> mini-batch마다 평균은 0, 분산은 1에 가깝게 맞춰준다.

논문은 deep neural network 학습이 어려운 이유 중 하나로, 앞 layer의 parameter가 바뀌면서 뒤 layer가 받는 입력 분포도 계속 바뀌는 현상을 지적하고, 이를 **internal covariate shift**라고 부릅니다. 저자들은 mini-batch 단위로 layer 입력을 normalization하는 BatchNorm을 제안했고, 이 방법이 더 높은 learning rate 사용, initialization 민감도 감소, regularization 효과를 줄 수 있다고 설명합니다.

---

## 이 논문을 한 문장으로 요약하면

**BatchNorm은 neural network의 중간 activation을 mini-batch 기준으로 표준화한 뒤, 다시 학습 가능한 scale과 shift를 적용해서 학습을 안정화하는 기법입니다.**

형태는 이렇게 볼 수 있습니다.

```text
입력 x
→ mini-batch 평균과 분산 계산
→ 평균 0, 분산 1로 정규화
→ γ, β로 다시 scale/shift
→ 다음 layer로 전달
```

여기서 `γ`와 `β`는 모델이 학습하는 parameter입니다.

즉, BatchNorm은 단순히 값을 강제로 평균 0, 분산 1로 고정하는 것이 아닙니다.  
먼저 정규화한 다음, 모델이 필요하면 다시 적절한 크기와 위치로 바꿀 수 있게 해줍니다.

---

## 왜 BatchNorm이 필요했나?

딥러닝에서는 layer가 여러 개 쌓입니다.

```text
input
→ layer 1
→ layer 2
→ layer 3
→ ...
→ output
```

학습 중에는 각 layer의 weight가 계속 바뀝니다.

그런데 앞 layer의 weight가 바뀌면, 뒤 layer가 받는 입력 분포도 바뀝니다.

예를 들어 layer 3 입장에서는 어제는 이런 값들이 들어왔습니다.

```text
평균 0.2, 분산 1.1
```

그런데 weight update 이후에는 이런 값들이 들어올 수 있습니다.

```text
평균 5.0, 분산 20.0
```

그러면 layer 3는 계속 바뀌는 입력 분포에 적응해야 합니다.  
이런 상황에서는 학습이 불안정해지고, learning rate를 크게 쓰기 어렵고, initialization에도 민감해질 수 있습니다.

저자들은 이런 문제를 **internal covariate shift**라는 관점으로 설명합니다. 원 논문은 각 layer의 input distribution이 training 중 계속 바뀌면 더 낮은 learning rate와 신중한 parameter initialization이 필요해지고, saturating nonlinearity를 가진 모델 학습이 어려워진다고 설명합니다.

---

## 핵심 개념 1: Mini-batch 단위로 평균과 분산을 맞춘다

### 결론부터

**BatchNorm은 한 mini-batch 안에서 특정 activation의 평균과 분산을 계산한 뒤, 그 값을 기준으로 정규화합니다.**

수식은 이렇습니다.

```text
μ_B = mini-batch 평균
σ_B² = mini-batch 분산

x_hat = (x - μ_B) / sqrt(σ_B² + ε)
```

여기서:

| 기호 | 의미 |
|---|---|
| `x` | BatchNorm에 들어온 값 |
| `μ_B` | 현재 mini-batch의 평균 |
| `σ_B²` | 현재 mini-batch의 분산 |
| `ε` | 0으로 나누는 것을 막기 위한 작은 값 |
| `x_hat` | 정규화된 값 |

결과적으로 `x_hat`은 mini-batch 안에서 대략 평균 0, 분산 1이 됩니다.

---

## 숫자로 보는 BatchNorm 예시

어떤 layer의 activation 값이 mini-batch에서 다음과 같다고 합시다.

```text
x = [2, 4, 6, 8]
```

### 1. 평균 계산

```text
평균 μ = (2 + 4 + 6 + 8) / 4 = 5
```

### 2. 분산 계산

각 값에서 평균을 뺍니다.

```text
2 - 5 = -3
4 - 5 = -1
6 - 5 = 1
8 - 5 = 3
```

제곱 평균을 구합니다.

```text
분산 = (9 + 1 + 1 + 9) / 4 = 5
```

표준편차는:

```text
sqrt(5) ≈ 2.236
```

### 3. 정규화

```text
x_hat = (x - 5) / 2.236
```

결과는 대략:

```text
[-1.34, -0.45, 0.45, 1.34]
```

이제 값들이 평균 0, 분산 1에 가까워졌습니다.

---

## 핵심 개념 2: γ와 β가 필요하다

### 결론부터

**정규화만 하면 표현력이 제한될 수 있으므로, BatchNorm은 정규화 후 다시 학습 가능한 scale과 shift를 적용합니다.**

정규화 후 값은 `x_hat`입니다.

BatchNorm의 최종 출력은 보통 이렇게 씁니다.

```text
y = γ × x_hat + β
```

여기서:

| 기호 | 의미 |
|---|---|
| `γ` gamma | scale, 값을 얼마나 키우거나 줄일지 |
| `β` beta | shift, 값을 얼마나 이동할지 |

왜 이게 필요할까요?

만약 모든 layer의 activation을 항상 평균 0, 분산 1로만 고정하면, 모델이 필요한 표현을 만들기 어려울 수 있습니다.

예를 들어 어떤 layer는 평균이 3이고 분산이 5인 표현이 더 유용할 수도 있습니다.

그래서 BatchNorm은 이렇게 합니다.

```text
일단 안정적으로 정규화한다.
하지만 필요하면 γ와 β를 학습해서 다시 원하는 분포로 바꿀 수 있게 한다.
```

즉, BatchNorm은 표현력을 없애는 것이 아니라, **학습이 쉬운 좌표계로 바꾼 뒤 필요한 scale과 shift를 다시 배우게 하는 방법**입니다.

---

## 핵심 개념 3: Training 때와 Inference 때가 다르다

이 부분은 실무에서 정말 중요합니다.

BatchNorm은 training 때 mini-batch의 평균과 분산을 사용합니다.

```text
training mode:
현재 mini-batch의 mean/variance 사용
```

하지만 inference, 즉 evaluation이나 test 때는 mini-batch 통계를 그대로 쓰면 불안정합니다.

예를 들어 test에서 샘플 1개만 들어오면 batch 평균과 분산을 제대로 계산하기 어렵습니다.

그래서 training 중에 running mean과 running variance를 계속 추적해둡니다.

```text
inference mode:
training 중 누적한 running mean/running variance 사용
```

PyTorch에서는 보통 이렇게 구분합니다.

```python
model.train()  # BatchNorm이 mini-batch 통계를 사용하고 running stats를 업데이트함
model.eval()   # BatchNorm이 running mean/variance를 사용함
```

Dropout 때도 비슷한 이야기를 했죠.

| 모듈 | `model.train()` | `model.eval()` |
|---|---|---|
| Dropout | 뉴런 일부를 랜덤하게 끔 | Dropout 꺼짐 |
| BatchNorm | batch 통계 사용, running stats 업데이트 | running stats 사용 |

---

## 쉬운 예시: 시험 점수 정규화

BatchNorm을 시험 점수로 비유해봅시다.

어떤 반의 시험 점수가 있습니다.

```text
[20, 30, 40, 50]
```

다른 반은 점수가 이렇습니다.

```text
[70, 80, 90, 100]
```

원점수만 보면 두 반이 너무 다릅니다.

그런데 각 반 안에서 평균과 표준편차를 기준으로 표준화하면:

```text
이 학생은 자기 반에서 평균보다 얼마나 높은가?
이 학생은 자기 반에서 평균보다 얼마나 낮은가?
```

를 볼 수 있습니다.

BatchNorm도 비슷합니다.

각 mini-batch 안에서 activation을 표준화해서, 다음 layer가 너무 들쭉날쭉한 값이 아니라 안정적인 범위의 값을 받도록 합니다.

단, neural network에서는 단순 표준화로 끝내지 않고, `γ`와 `β`를 통해 모델이 필요한 scale과 shift를 다시 학습합니다.

---

## 핵심 개념 4: BatchNorm은 더 큰 learning rate를 가능하게 한다

딥러닝 학습에서 learning rate가 너무 크면 loss가 튀거나 발산할 수 있습니다.

```text
learning rate 너무 큼
→ weight update가 과격함
→ activation 분포가 크게 흔들림
→ 학습 불안정
```

BatchNorm은 layer 입력의 분포를 안정화하기 때문에 더 큰 learning rate를 쓸 수 있게 도와줍니다.

원 논문은 BatchNorm이 더 높은 learning rate를 사용할 수 있게 하고, initialization에 덜 민감하게 만들며, 경우에 따라 Dropout 필요성을 줄일 수 있다고 주장합니다. ImageNet 실험에서는 batch-normalized network가 같은 accuracy를 훨씬 적은 training step으로 달성했다고 보고합니다.

쉽게 말하면:

> BatchNorm은 학습 중 값의 스케일을 안정화해서, gradient descent가 더 과감하게 움직일 수 있게 해준다.

---

## 핵심 개념 5: BatchNorm은 regularization 효과도 있다

BatchNorm은 원래 학습 안정화 기법으로 제안됐지만, regularization 효과도 있습니다.

왜냐하면 training 때 mini-batch마다 평균과 분산이 조금씩 다르기 때문입니다.

즉, 같은 샘플이라도 어떤 mini-batch에 들어가느냐에 따라 normalized value가 약간 달라집니다.

```text
mini-batch A에 있을 때의 정규화 값
mini-batch B에 있을 때의 정규화 값
```

이 차이가 학습에 noise처럼 작용할 수 있습니다.

그래서 모델이 특정 훈련 데이터에 너무 딱 맞는 것을 어느 정도 막아줍니다.

원 논문도 BatchNorm이 regularizer처럼 작용할 수 있고, 어떤 경우에는 Dropout의 필요를 제거할 수도 있다고 설명합니다.

---

## Dropout과 BatchNorm의 차이

직전 논문이 Dropout이었기 때문에 비교하면 좋습니다.

| 구분 | Dropout | BatchNorm |
|---|---|---|
| 핵심 목적 | overfitting 방지 | 학습 안정화와 가속 |
| 방식 | 뉴런 일부를 랜덤하게 끔 | activation을 batch 기준으로 정규화 |
| training 때 | random mask 적용 | batch mean/variance 사용 |
| inference 때 | 꺼짐 | running mean/variance 사용 |
| regularization 효과 | 강함 | 부수적으로 있음 |
| 주의점 | 너무 강하면 underfitting | batch size가 너무 작으면 불안정 |

짧게 말하면:

```text
Dropout = 일부 뉴런을 랜덤하게 꺼서 robust하게 학습
BatchNorm = 중간 activation 분포를 안정화해서 빠르게 학습
```

둘 다 overfitting 완화에 도움이 될 수 있지만, 주된 철학은 다릅니다.

---

## BatchNorm은 어디에 넣는가?

전통적으로는 linear 또는 convolution 연산 뒤, activation function 앞에 두는 경우가 많습니다.

```text
Linear / Conv
→ BatchNorm
→ ReLU
```

예:

```text
Conv2d
→ BatchNorm2d
→ ReLU
```

다만 모델 구조나 프레임워크에 따라 순서는 달라질 수 있습니다.

일반적인 CNN에서는 다음과 같은 패턴이 흔합니다.

```text
Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU
```

BatchNorm은 activation이 다음 layer로 들어가기 전에 scale을 안정화하는 역할을 합니다.

---

## 핵심 개념 6: Internal Covariate Shift는 현재도 해석에 주의가 필요하다

원 논문은 BatchNorm의 동기를 **internal covariate shift 감소**로 설명했습니다.

하지만 이후 연구에서는 “BatchNorm이 정말 internal covariate shift를 줄여서 잘 되는가?”에 대해 논쟁이 있었습니다.

대표적으로 Santurkar et al.의 2018년 논문은 BatchNorm의 성공이 layer input distribution의 안정화보다는, optimization landscape를 더 smooth하게 만들어 gradient가 더 안정적이고 예측 가능하게 되는 효과와 관련이 크다고 주장합니다.

입문자 관점에서는 이렇게 정리하면 좋습니다.

```text
원 논문 설명:
BatchNorm은 internal covariate shift를 줄인다.

현대적 보강:
BatchNorm은 학습을 안정화하고 optimization을 쉽게 만든다.
그 이유를 internal covariate shift 하나로만 설명하기는 어렵다.
```

즉, BatchNorm을 실무적으로 이해할 때는:

> “각 layer의 값 스케일을 안정화해서 optimization을 쉽게 만든다”

정도로 이해하는 것이 안전합니다.

---

## BatchNorm의 한계 1: Batch size가 너무 작으면 불안정하다

BatchNorm은 batch의 평균과 분산을 사용합니다.

그런데 batch size가 너무 작으면 평균과 분산 추정이 부정확해집니다.

예를 들어 batch size가 2라면:

```text
두 샘플만 보고 평균과 분산을 추정
```

이건 불안정할 수 있습니다.

특히 object detection, segmentation, video model처럼 GPU memory 때문에 batch size를 크게 잡기 어려운 작업에서는 BatchNorm이 약해질 수 있습니다.

---

## BatchNorm의 한계 2: Training과 inference의 통계 차이

BatchNorm은 training 때 batch statistics를 사용하고, inference 때 running statistics를 사용합니다.

이 둘이 잘 맞으면 문제 없습니다.

하지만 training data와 inference data의 분포가 다르거나, running statistics가 충분히 안정적으로 추정되지 않았으면 문제가 생길 수 있습니다.

예를 들어:

```text
training: 밝은 조명 이미지 위주
inference: 어두운 조명 이미지 많음
```

이면 running mean/variance가 실제 inference 분포를 잘 대표하지 못할 수 있습니다.

그래서 BatchNorm을 사용하는 모델은 `train/eval` 모드 전환, batch size, domain shift에 주의해야 합니다.

---

## BatchNorm의 한계 3: RNN이나 sequence model에서는 항상 자연스럽지 않다

BatchNorm은 CNN에서 특히 강력하게 쓰였습니다.

하지만 RNN처럼 time step이 있고 sequence 길이가 다양한 모델에서는 BatchNorm 적용이 더 복잡할 수 있습니다.

이후 Transformer 계열에서는 BatchNorm보다 **Layer Normalization**이 더 널리 쓰입니다.

왜냐하면 LayerNorm은 batch dimension에 의존하지 않고, 각 sample 내부 feature 차원에서 normalization하기 때문입니다.

이건 나중에 Transformer 논문을 볼 때 다시 중요해집니다.

---

## BatchNorm과 LayerNorm을 아주 간단히 비교하면

아직 LayerNorm 논문을 자세히 보진 않았지만, 미리 감각만 잡으면 좋습니다.

| 구분 | BatchNorm | LayerNorm |
|---|---|---|
| 평균/분산 계산 기준 | batch 방향 | sample 내부 feature 방향 |
| batch size 영향 | 큼 | 작음 |
| CNN에서 | 매우 많이 사용 | 상대적으로 덜 전통적 |
| Transformer에서 | 거의 표준은 아님 | 핵심적으로 사용 |
| inference 통계 | running mean/var 사용 | 보통 현재 sample 기준 |

지금은 이렇게만 기억하면 됩니다.

```text
CNN 계열에서는 BatchNorm이 강력했다.
Transformer 계열에서는 LayerNorm이 중요해졌다.
```

---

## BatchNorm을 실무에서 쓸 때 주의할 점

### 1. Validation과 test 때 `eval()`을 해야 한다

PyTorch 기준으로:

```python
model.train()
# training

model.eval()
# validation / test
```

를 정확히 구분해야 합니다.

Dropout과 BatchNorm 모두 이 모드에 따라 동작이 달라집니다.

### 2. Batch size가 너무 작으면 성능이 흔들릴 수 있다

batch size가 작으면 BatchNorm 통계가 불안정합니다.

이럴 때는 다음을 고려할 수 있습니다.

| 상황 | 대안 |
|---|---|
| batch size가 매우 작음 | GroupNorm, LayerNorm |
| multi-GPU에서 GPU별 batch가 작음 | SyncBatchNorm |
| fine-tuning 데이터가 작음 | BatchNorm freeze 고려 |
| inference domain이 다름 | running stats 재추정 또는 calibration 고려 |

### 3. BatchNorm과 bias는 중복될 수 있다

`Linear`나 `Conv` 뒤에 바로 BatchNorm이 붙는 경우, 앞 layer의 bias는 종종 생략할 수 있습니다.

왜냐하면 BatchNorm에 이미 shift parameter `β`가 있기 때문입니다.

예:

```text
Conv(bias=False) → BatchNorm → ReLU
```

이런 패턴을 많이 봅니다.

### 4. BatchNorm은 data leakage와는 별개로 train/test 모드를 조심해야 한다

BatchNorm은 training data의 running statistics를 inference에 사용합니다.

일반적인 사용에서는 문제가 아닙니다.

하지만 test data를 이용해 BatchNorm statistics를 업데이트해버리면 평가가 오염될 수 있습니다.

즉:

```text
test 단계에서 model.train() 상태로 예측
→ BatchNorm running stats가 test data로 업데이트될 수 있음
→ 평가 오염
```

그래서 test 전에 반드시 `model.eval()`을 해야 합니다.

---

## BatchNorm이 학습을 빠르게 만든다는 말의 의미

BatchNorm을 쓰면 보통 다음이 가능해집니다.

| 효과 | 의미 |
|---|---|
| 더 큰 learning rate | 학습을 더 빠르게 진행 가능 |
| initialization 민감도 감소 | 초기 weight 선택에 덜 예민 |
| gradient 흐름 안정화 | 깊은 network 학습이 쉬워짐 |
| saturating activation 문제 완화 | sigmoid/tanh 계열에서 특히 중요했던 문제 |
| regularization 효과 | batch 통계 noise로 overfitting 완화 가능 |

원 논문은 BatchNorm이 훨씬 적은 training step으로 같은 accuracy를 달성했고, ImageNet classification에서 좋은 성능을 보였다고 보고합니다.

---

## Dropout 이후 BatchNorm이 왜 자연스러운 다음 논문인가?

Dropout은 이렇게 말했습니다.

```text
모델이 훈련 데이터에 과적합되지 않도록 일부 뉴런을 랜덤하게 끄자.
```

BatchNorm은 이렇게 말합니다.

```text
깊은 network가 학습 중 불안정해지지 않도록 중간 activation 분포를 안정화하자.
```

딥러닝 기본기에서 두 논문은 서로 보완적입니다.

| 문제 | 대표 기법 |
|---|---|
| overfitting | Dropout |
| 학습 불안정, 느린 수렴 | BatchNorm |
| gradient 소실/폭주와 깊은 모델 학습 | BatchNorm, Residual connection |
| 너무 깊은 모델의 degradation | ResNet |

그래서 다음 논문인 **ResNet**으로 가면, “깊은 모델을 어떻게 더 깊게 만들 수 있는가?”라는 질문으로 자연스럽게 이어집니다.

---

## 이전 논문들과 연결해서 이해하기

| 이전 논문 | BatchNorm과의 연결 |
|---|---|
| **Deep Learning** | 깊은 network는 representation을 학습하지만 학습이 어렵다 |
| **Dropout** | Dropout은 regularization, BatchNorm은 학습 안정화 중심 |
| **Gradient Descent** | BatchNorm은 gradient-based optimization을 더 안정적으로 만든다 |
| **Model Evaluation** | BatchNorm 모델도 train/validation/test 모드와 분리를 엄격히 해야 한다 |
| **SHAP** | BatchNorm은 설명 도구가 아니라 학습 안정화 기법이며, 해석은 별도 도구가 필요하다 |

---

## 입문자가 꼭 기억해야 할 문장

**BatchNorm은 mini-batch의 평균과 분산으로 중간 activation을 정규화하고, 학습 가능한 γ와 β로 다시 조정해서 deep network의 학습을 빠르고 안정적으로 만드는 방법이다.**

더 짧게 말하면:

> **BatchNorm = layer 중간값을 batch 기준으로 정규화해서 학습을 안정화하는 기법**

---

## 오늘 공부용 요약

**논문명:** Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift  
**저자:** Sergey Ioffe, Christian Szegedy  
**출판:** ICML 2015.

**핵심 결론:** BatchNorm은 각 layer의 입력 또는 activation을 mini-batch 단위로 정규화하고, 학습 가능한 scale과 shift를 적용해 deep neural network의 학습을 빠르고 안정적으로 만든다.

**왜 중요함:** 딥러닝 모델이 깊어질수록 학습이 불안정하고 initialization과 learning rate에 민감해지는데, BatchNorm은 이를 크게 완화해 더 깊고 빠른 학습을 가능하게 했다.

**입문자가 배울 점:** BatchNorm은 단순한 전처리가 아니라 network 안에 들어가는 layer다. training 때는 batch 통계를 쓰고, inference 때는 running 통계를 쓴다는 점이 매우 중요하다.

**가장 중요한 주의점:** BatchNorm의 원 논문은 internal covariate shift 감소를 주요 동기로 제시했지만, 현대적으로는 optimization을 더 smooth하고 안정적으로 만드는 효과까지 함께 이해하는 것이 좋다.

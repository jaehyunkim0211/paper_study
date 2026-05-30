# Training Compute-Optimal Large Language Models — Chinchilla

## 결론부터

이 논문의 결론은 이겁니다.

**LLM을 잘 만들려면 모델 크기만 무작정 키우면 안 된다. 정해진 compute budget 안에서는 모델 파라미터 수와 학습 token 수를 균형 있게 키워야 한다.**

쉽게 말하면:

> GPT-3 이후 사람들은 “모델을 더 크게 만들자”에 집중했다. 그런데 Chinchilla 논문은 “모델은 너무 큰데, 학습 데이터 token 수가 부족했다”고 지적했다.

이 논문의 핵심 주장은 다음 한 줄입니다.

```text
compute-optimal training에서는 모델 크기와 학습 token 수를 대략 같이 키워야 한다.
```

논문은 400개 이상의 모델을 학습해 분석했고, compute-optimal한 학습에서는 모델 크기를 두 배로 키우면 학습 token 수도 두 배로 키우는 것이 좋다고 결론내립니다. 그리고 이 가설을 검증하기 위해 70B parameter 모델인 **Chinchilla**를 1.4T token으로 학습했고, 같은 compute budget을 쓴 280B parameter **Gopher**보다 더 좋은 성능을 냈습니다.

---

## 이 논문을 한 문장으로 요약하면

**Chinchilla는 “더 큰 모델”보다 “같은 compute로 더 적절한 크기의 모델을 더 많은 token으로 학습하는 것”이 낫다는 것을 보여준 논문입니다.**

직전 논문인 Kaplan의 Scaling Laws와 비교하면 차이가 선명합니다.

| 논문 | 핵심 메시지 |
|---|---|
| **Scaling Laws, Kaplan et al.** | 모델 크기, 데이터, compute가 커지면 loss가 예측 가능하게 줄어든다 |
| **Chinchilla, Hoffmann et al.** | 고정 compute에서는 모델만 키우지 말고, 모델 크기와 학습 token 수를 균형 있게 키워야 한다 |

즉, Scaling Laws가 “scale이 중요하다”고 말했다면, Chinchilla는 이렇게 보강합니다.

> scale을 키우되, **모델 크기와 데이터 token 수의 비율**을 잘 맞춰야 한다.

---

## 왜 이 논문이 필요했나?

GPT-3 이후 대형 언어 모델 경쟁은 주로 파라미터 수를 키우는 방향으로 갔습니다.

예를 들어 논문이 비교한 모델들을 보면 다음과 같습니다.

| 모델 | 파라미터 수 | 학습 token 수 |
|---|---:|---:|
| GPT-3 | 175B | 300B |
| Jurassic-1 | 178B | 300B |
| Gopher | 280B | 300B |
| MT-NLG | 530B | 270B |
| Chinchilla | 70B | 1.4T |

여기서 재미있는 점은, GPT-3, Jurassic-1, Gopher, MT-NLG는 파라미터 수는 크게 다르지만 학습 token 수는 대부분 약 300B 근처였다는 것입니다. 반면 Chinchilla는 70B로 훨씬 작지만, 1.4T token으로 훨씬 더 오래 학습됐습니다.

이 논문은 이 흐름을 보고 이렇게 말합니다.

> 기존 대형 모델들은 모델 크기에 비해 너무 적은 token으로 학습됐다. 즉, **undertrained**였다.

---

## 핵심 개념 1: Compute-optimal이란?

### 결론부터

**Compute-optimal은 “정해진 계산량 안에서 loss를 가장 낮게 만드는 모델 크기와 학습 token 수의 조합”을 찾는 것입니다.**

예를 들어 compute budget이 정해져 있다고 합시다.

```text
총 학습 예산 = 100
```

이 예산을 어떻게 쓸 수 있을까요?

### 선택 A: 큰 모델, 적은 데이터

```text
모델 크기: 매우 큼
학습 token: 적음
```

### 선택 B: 중간 모델, 많은 데이터

```text
모델 크기: 적당히 큼
학습 token: 많음
```

### 선택 C: 작은 모델, 아주 많은 데이터

```text
모델 크기: 작음
학습 token: 아주 많음
```

Chinchilla 논문은 이 중 어떤 선택이 주어진 compute에서 가장 낮은 loss를 만드는지 찾습니다.

---

## 핵심 개념 2: 모델 크기와 token 수는 trade-off 관계다

같은 compute budget이 있을 때, 모델을 크게 만들면 한 token을 학습하는 비용이 커집니다.

```text
큰 모델
→ token 하나 처리하는 비용 큼
→ 같은 compute로 볼 수 있는 token 수 줄어듦
```

반대로 모델을 작게 만들면 한 token을 처리하는 비용이 줄어듭니다.

```text
작은 모델
→ token 하나 처리하는 비용 작음
→ 같은 compute로 더 많은 token을 볼 수 있음
```

따라서 고정 compute에서는 이런 trade-off가 생깁니다.

```text
모델을 크게 할 것인가?
더 많은 token을 학습할 것인가?
```

---

## 핵심 개념 3: Chinchilla의 핵심 결론 — 둘을 같이 키워라

### 결론부터

**모델 크기와 학습 token 수를 비슷한 비율로 키워야 한다는 것이 Chinchilla의 핵심 결론입니다.**

Kaplan의 이전 Scaling Laws에서는 compute가 늘어날 때 모델 크기를 더 빠르게 키우는 쪽의 결론이 나왔습니다.

Chinchilla는 다른 결과를 냈습니다.

```text
Kaplan식 직관:
compute가 늘면 모델 크기를 훨씬 더 많이 키워라.

Chinchilla식 직관:
compute가 늘면 모델 크기와 학습 token 수를 비슷하게 키워라.
```

이 차이가 매우 중요합니다.

---

## 아주 쉬운 비유: 학생과 문제집

모델을 학생, token을 문제집이라고 생각해봅시다.

### 기존 대형 모델 방식

```text
학생은 엄청 똑똑하게 만든다.
그런데 문제집은 별로 많이 안 풀린다.
```

즉:

```text
큰 모델 + 적은 token
```

이 경우 학생의 잠재력은 크지만, 연습량이 부족합니다.

### Chinchilla 방식

```text
학생을 적당히 똑똑하게 만든다.
대신 문제집을 훨씬 많이 풀린다.
```

즉:

```text
중간 크기 모델 + 많은 token
```

이 방식이 같은 학습 예산에서는 더 성적이 좋을 수 있다는 것입니다.

---

## 핵심 개념 4: Undertrained model

### 결론부터

**Undertrained model은 모델 크기에 비해 학습 token 수가 부족한 모델입니다.**

예를 들어 500B parameter 모델이 있다고 합시다.

이 모델은 엄청난 용량을 가지고 있습니다.

그런데 300B token만 학습했다고 합시다.

Chinchilla 관점에서는 이렇게 볼 수 있습니다.

```text
모델은 너무 큰데,
학습 데이터 token 수가 부족하다.
```

이 경우 같은 compute budget이라면 차라리 이렇게 하는 게 나을 수 있습니다.

```text
500B 모델 × 300B token
```

보다

```text
100B 모델 × 훨씬 많은 token
```

이 낫다는 것입니다.

---

## 핵심 개념 5: Chinchilla vs Gopher

이 논문에서 가장 유명한 비교가 **Chinchilla vs Gopher**입니다.

| 모델 | 파라미터 수 | 학습 token 수 | compute |
|---|---:|---:|---|
| Gopher | 280B | 300B | Chinchilla와 같음 |
| Chinchilla | 70B | 1.4T | Gopher와 같음 |

즉, Chinchilla는 Gopher보다:

```text
파라미터 수는 4배 작고
학습 token 수는 약 4배 이상 많다
```

그런데 compute는 같게 맞췄습니다.

결과는 Chinchilla 쪽이 더 좋았습니다.

Chinchilla는 MMLU, BIG-bench, LAMBADA, RACE, commonsense benchmark 등에서 Gopher와 GPT-3 계열 모델을 대체로 앞섰습니다.

---

## 왜 작은 Chinchilla가 큰 Gopher보다 좋았을까?

핵심은 이겁니다.

```text
Gopher는 너무 큰 모델을 너무 적은 token으로 학습했다.
Chinchilla는 더 작은 모델을 훨씬 더 많은 token으로 학습했다.
```

이 차이는 “시험을 보기 전 연습 문제를 얼마나 풀었는가”와 비슷합니다.

Gopher:

```text
두뇌 용량은 매우 큼
그런데 연습 문제를 상대적으로 적게 풂
```

Chinchilla:

```text
두뇌 용량은 Gopher보다 작음
하지만 연습 문제를 훨씬 많이 풂
```

같은 총 공부 예산에서는 후자가 더 성적이 좋을 수 있습니다.

---

## 핵심 개념 6: 20 tokens per parameter 감각

Chinchilla를 보면 중요한 감각이 생깁니다.

```text
70B parameters
1.4T training tokens
```

계산하면:

```text
1.4T / 70B = 20
```

즉, 대략 **parameter 하나당 20 training tokens** 정도입니다.

이 숫자는 이후 LLM 개발에서 “Chinchilla-optimal ratio”라고 자주 불립니다.

단, 이걸 절대 법칙처럼 외우면 안 됩니다.

실제 최적 비율은 다음에 따라 달라질 수 있습니다.

```text
데이터 품질
모델 구조
optimizer
learning rate schedule
repeated token 여부
target task
inference 비용 고려 여부
```

그래도 입문적으로는 이렇게 기억하면 좋습니다.

> GPT-3식 대형 모델들은 parameter 수에 비해 token이 부족했다. Chinchilla는 parameter당 훨씬 많은 token을 먹여야 한다는 감각을 줬다.

---

## 핵심 개념 7: inference 비용도 중요하다

Chinchilla는 Gopher보다 작습니다.

```text
Gopher: 280B
Chinchilla: 70B
```

모델이 작으면 좋은 점이 있습니다.

```text
추론할 때 메모리가 덜 필요함
추론 비용이 낮음
fine-tuning 비용도 낮음
배포가 더 쉬움
```

이 부분은 매우 중요합니다.

LLM에서는 training compute만 중요한 것이 아닙니다.

모델을 만든 뒤 실제로 수많은 사용자가 inference를 합니다.

```text
training cost: 한 번 큰 비용
inference cost: 계속 반복되는 비용
```

그래서 같은 성능이라면 작은 모델이 훨씬 유리합니다. Chinchilla는 심지어 더 작은데 성능도 더 좋았습니다.

---

## Scaling Laws와 Chinchilla를 비교해서 다시 이해하기

## Kaplan Scaling Laws

Kaplan 논문은 큰 메시지를 줬습니다.

```text
언어 모델 loss는 scale이 커질수록 power law 형태로 줄어든다.
```

그리고 compute-efficient training에서는 큰 모델을 수렴 전까지 학습하는 쪽이 유리하다고 봤습니다.

## Chinchilla

Chinchilla는 이렇게 수정합니다.

```text
모델을 크게만 만들면 안 된다.
학습 token 수도 비슷하게 키워야 한다.
```

즉, 두 논문을 함께 이해하면 이렇게 됩니다.

```text
1. Scale은 중요하다.
2. 하지만 모델 크기만 scale하면 안 된다.
3. 데이터 token 수까지 같이 scale해야 한다.
```

---

## 이 논문에서 사용한 방법

논문은 크게 세 가지 접근으로 compute-optimal frontier를 추정했습니다.

## 1. 모델 크기 고정, 학습 token 수 변화

여러 모델 크기를 고정해두고, 각 모델을 다양한 training horizon으로 학습했습니다.

```text
70M 모델을 짧게/중간/길게 학습
1B 모델을 짧게/중간/길게 학습
10B 모델을 짧게/중간/길게 학습
```

## 2. IsoFLOP 분석

같은 FLOPs budget 안에서 모델 크기와 token 수 조합을 바꾸고, 어느 조합이 가장 낮은 loss를 내는지 봤습니다.

```text
같은 compute 안에서
작은 모델 오래 학습 vs 큰 모델 짧게 학습
```

## 3. Parametric loss function fitting

최종 loss를 모델 크기 `N`과 token 수 `D`의 함수로 직접 모델링했습니다.

세 접근 모두 결론이 비슷했습니다.

```text
모델 크기와 token 수를 대략 같은 비율로 키워라.
```

---

## 이 논문이 주는 실무적 교훈

## 1. 모델 크기만 자랑하면 안 된다

LLM 발표에서 parameter 수만 보면 부족합니다.

꼭 같이 봐야 합니다.

```text
parameter 수
training token 수
compute budget
data quality
inference cost
```

## 2. 데이터 수집과 품질이 더 중요해졌다

Chinchilla는 더 많은 token의 중요성을 강조했습니다.

하지만 token을 무작정 많이 모으면 안 됩니다.

정확히는:

```text
모델 크기에 맞는 충분한 양의 고품질 token을 확보하라
```

입니다.

## 3. 작은 모델도 잘 학습하면 강하다

70B Chinchilla가 280B Gopher보다 성능이 좋았다는 사실은 매우 중요합니다.

```text
무조건 큰 모델이 이기는 것이 아니다.
작더라도 충분한 token으로 compute-optimal하게 학습한 모델이 더 나을 수 있다.
```

---

## Chinchilla의 한계도 알아야 한다

이 논문도 모든 것을 끝낸 것은 아닙니다.

## 1. 큰 scale 검증은 제한적이었다

비용 때문에 Chinchilla와 Gopher처럼 대규모로 비교 가능한 학습 run은 많지 않았습니다.

## 2. Power-law 가정의 한계

efficient computational frontier가 power-law 관계로 설명된다고 가정하지만, high compute budget에서 약간의 curvature가 관찰될 수 있습니다.

## 3. 데이터 품질과 반복 학습 문제

training runs는 주로 데이터 1 epoch 미만으로 학습됐고, multiple epoch regime은 future work로 남겨뒀습니다.

즉, Chinchilla 법칙은 매우 강력한 실용 지침이지만, 모든 조건에서 불변의 자연법칙은 아닙니다.

---

## GPT-3, Scaling Laws, Chinchilla 흐름 정리

| 논문 | 핵심 질문 | 답 |
|---|---|---|
| **GPT-3** | 모델을 크게 만들면 few-shot 능력이 생기는가? | 175B 모델이 prompt만으로 다양한 task를 수행함 |
| **Scaling Laws** | 모델을 키우면 성능이 어떻게 변하는가? | loss가 scale에 따라 예측 가능하게 감소함 |
| **Chinchilla** | 고정 compute에서 모델 크기와 token 수를 어떻게 배분할까? | 모델 크기와 token 수를 균형 있게 키워야 함 |

한 줄로 연결하면:

```text
GPT-3: 큰 모델은 강하다.
Scaling Laws: scale과 loss 사이에는 법칙이 있다.
Chinchilla: 큰 모델만으로는 부족하고, 충분한 token 학습이 필요하다.
```

---

## 입문자가 꼭 기억해야 할 문장

**Chinchilla는 같은 compute budget이라면 모델을 무작정 크게 만들기보다, 더 작은 모델을 더 많은 token으로 학습하는 것이 더 좋은 성능과 더 낮은 inference 비용을 줄 수 있다는 것을 보여준 논문입니다.**

더 짧게 말하면:

> **Chinchilla = “모델 크기와 학습 token 수의 균형이 중요하다”는 논문**

---

## 오늘 공부용 요약

**논문명:** Training Compute-Optimal Large Language Models  
**저자:** Jordan Hoffmann et al.  
**핵심 결론:** 고정된 compute budget에서 LLM을 최적으로 학습하려면 모델 파라미터 수와 학습 token 수를 대략 같은 비율로 키워야 한다. 기존 대형 모델들은 파라미터 수에 비해 학습 token 수가 부족해 undertrained 상태였을 가능성이 크다.

**왜 중요함:** GPT-3 이후 “더 큰 모델” 경쟁에 대해, Chinchilla는 “더 큰 모델”보다 “compute-optimal하게 학습된 모델”이 중요하다는 관점을 제시했습니다.

**입문자가 배울 점:** LLM 성능은 parameter 수만으로 결정되지 않습니다. parameter 수, 학습 token 수, compute budget, data quality, inference cost를 함께 봐야 합니다.

**가장 중요한 문장:** 같은 compute라면, 너무 큰 모델을 적게 학습시키는 것보다, 더 작은 모델을 더 많은 token으로 충분히 학습시키는 것이 더 좋을 수 있다.

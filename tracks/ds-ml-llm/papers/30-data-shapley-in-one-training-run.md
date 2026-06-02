# Data Shapley in One Training Run

## 결론부터

이 논문의 결론은 이겁니다.

**기존 Data Shapley는 각 데이터의 가치를 계산하려면 모델을 여러 번 다시 학습해야 해서 너무 비쌌다. 이 논문은 한 번의 학습 과정 안에서 각 데이터가 모델 성능에 얼마나 기여했는지를 추적하는 In-Run Data Shapley를 제안한다.**

한 줄로 말하면:

> **In-Run Data Shapley = 모델을 여러 번 재학습하지 않고, 한 번의 training run 중에 데이터 기여도를 계산하는 방법**

직전 논문인 **Data Shapley**는 각 학습 데이터의 기여도를 공정하게 계산하는 원칙을 제시했습니다. 하지만 문제는 계산 비용이었습니다.

```text
기존 Data Shapley:
여러 데이터 subset을 만든다
→ 각 subset으로 모델을 다시 학습한다
→ 성능 변화를 본다
→ 각 데이터의 평균 기여도를 계산한다
```

이 방식은 작은 데이터셋에서는 가능할 수 있지만, GPT, BERT, LLaMA 같은 대규모 모델에서는 사실상 불가능합니다.

이 논문은 이 문제를 이렇게 바꿉니다.

```text
모델을 반복해서 새로 학습하지 말자.
어차피 모델은 SGD 같은 반복 업데이트로 학습된다.
각 training step에서 어떤 데이터가 validation 성능을 어느 방향으로 움직였는지 추적하자.
그 step별 기여도를 누적하자.
```

---

## 이 논문을 한 문장으로 요약하면

**Data Shapley in One Training Run은 “학습 데이터를 여러 subset으로 나눠 모델을 계속 재학습하지 말고, 실제 모델이 학습되는 한 번의 과정에서 데이터별 기여도를 회계 장부처럼 기록하자”는 논문입니다.**

기존 retraining-based Data Shapley는 가능한 데이터 subset마다 모델을 다시 학습해야 해서 비효율적이고 불안정할 수 있지만, In-Run Data Shapley는 한 번의 학습 과정에서 데이터 가치 점수를 추적합니다.

---

## 왜 이 논문이 필요한가?

직전 Data Shapley 논문은 이 질문에 답했습니다.

```text
각 학습 데이터가 모델 성능에 얼마나 기여했는가?
```

하지만 현실적으로 문제가 있었습니다.

```text
데이터가 100개면 subset이 2^100개
각 subset마다 모델 학습 필요
대형 모델에서는 불가능
```

그래서 기존 Data Shapley는 원칙적으로는 멋지지만, 실제 foundation model pretraining 같은 규모에서는 적용하기 어려웠습니다.

이 논문은 그 한계를 직접 겨냥합니다.

---

## 핵심 차이: 기존 Data Shapley vs In-Run Data Shapley

| 구분 | 기존 Retraining-based Data Shapley | In-Run Data Shapley |
|---|---|---|
| 핵심 방식 | 여러 subset으로 모델을 반복 재학습 | 한 번의 학습 과정에서 step별 기여 추적 |
| 계산 비용 | 매우 큼 | 훨씬 작음 |
| 평가 대상 | learning algorithm의 평균적 기여 | 특정 training run의 특정 target model에 대한 기여 |
| 대형 모델 적용 | 사실상 어려움 | foundation model pretraining에도 적용 가능성을 제시 |
| 직관 | “이 데이터가 여러 subset에서 평균적으로 얼마나 도움 되나?” | “이 데이터가 실제 이 모델 학습 중 언제 얼마나 도움 됐나?” |

---

## 핵심 개념 1: 특정 모델에 대한 데이터 기여도

**In-Run Data Shapley는 “일반적인 학습 알고리즘”이 아니라, 실제로 내가 학습한 그 모델에 대한 데이터 기여도를 계산합니다.**

이 차이는 생각보다 중요합니다.

기존 Data Shapley는 이렇게 묻습니다.

```text
이 학습 알고리즘이 여러 번 실행될 때,
이 데이터는 평균적으로 얼마나 기여하는가?
```

하지만 실제 실무자는 보통 이렇게 묻습니다.

```text
이번에 내가 학습해서 배포하려는 이 모델에,
이 데이터가 얼마나 기여했는가?
```

예를 들어 같은 데이터셋과 같은 코드로 모델을 학습해도 random seed, batch order, 초기화, 학습 중 stochasticity 때문에 서로 다른 모델이 나올 수 있습니다.

기존 retraining-based Data Shapley는 이런 특정 run의 차이를 잘 반영하지 못합니다.

In-Run Data Shapley는 실제 학습 과정에서 어떤 데이터가 어떤 step에 들어갔고, 그때 validation loss에 어떤 영향을 줬는지 추적합니다.

---

## 핵심 개념 2: training step별 Shapley value

**In-Run Data Shapley는 전체 학습을 한 번에 보지 않고, 각 gradient update step에서 데이터의 기여를 계산한 뒤 누적합니다.**

일반적인 SGD 학습은 이렇게 진행됩니다.

```text
step 1: batch 1로 gradient 계산 → weight update
step 2: batch 2로 gradient 계산 → weight update
step 3: batch 3로 gradient 계산 → weight update
...
```

In-Run Data Shapley는 각 step마다 묻습니다.

```text
이번 batch에 들어온 데이터들 중,
각 데이터가 validation 성능 변화에 얼마나 기여했는가?
```

그리고 이 값을 계속 누적합니다.

```text
데이터 i의 최종 가치
= step 1에서의 기여
+ step 2에서의 기여
+ step 3에서의 기여
+ ...
```

---

## 쉬운 비유: 데이터 기여도 회계 장부

모델 학습을 회사 프로젝트라고 생각해봅시다.

기존 Data Shapley는 이렇게 합니다.

```text
팀원 조합을 여러 번 바꿔가며 프로젝트를 다시 해본다.
A, B만 있을 때 성과
A, C만 있을 때 성과
B, C만 있을 때 성과
A, B, C가 있을 때 성과
...
```

이건 공정하지만 너무 비쌉니다.

In-Run Data Shapley는 다릅니다.

```text
실제 프로젝트가 진행되는 동안,
각 회의와 각 작업 단계에서 누가 성과에 얼마나 기여했는지 기록한다.
```

즉, 여러 번 프로젝트를 다시 하지 않고, **한 번의 실제 프로젝트 진행 로그를 보고 기여도를 계산**하는 방식입니다.

이게 “one training run”의 의미입니다.

---

## 핵심 개념 3: Taylor approximation

**In-Run Data Shapley는 각 training step의 성능 변화가 작다고 보고, Taylor expansion으로 성능 변화를 근사합니다.**

모델은 step마다 조금씩 바뀝니다.

```text
θ_t → θ_{t+1}
```

학습률이 너무 크지 않다면, 한 step에서 모델 성능 변화는 비교적 작습니다.

그러면 validation loss 변화를 Taylor expansion으로 근사할 수 있습니다.

아주 쉽게 말하면:

```text
weight가 조금 움직였을 때 validation loss가 얼마나 바뀌는가?
≈ gradient 정보로 근사할 수 있다.
```

---

## 핵심 개념 4: 1차 In-Run Data Shapley

**1차 In-Run Data Shapley는 training data gradient와 validation data gradient가 얼마나 같은 방향을 보는지를 계산합니다.**

직관은 이렇습니다.

어떤 학습 데이터 `x_train`이 있습니다.

이 데이터로 gradient update를 하면 모델 weight가 어떤 방향으로 움직입니다.

또 validation data `x_val`이 있습니다.

validation loss를 줄이려면 모델 weight가 어떤 방향으로 움직여야 하는지도 gradient로 알 수 있습니다.

이때 두 gradient 방향이 잘 맞으면:

```text
training data가 validation 성능 개선에 도움
```

반대로 방향이 반대면:

```text
training data가 validation 성능을 해칠 수 있음
```

즉, 1차 In-Run Data Shapley는 대략 이런 질문을 합니다.

```text
이 학습 데이터가 만든 gradient 방향은
validation loss를 줄이는 방향과 얼마나 잘 맞는가?
```

---

## gradient dot product를 쉽게 이해하기

두 화살표를 생각하면 됩니다.

```text
학습 데이터가 모델을 움직이려는 방향: → 
validation loss가 줄어드는 방향: →
```

두 방향이 같으면 dot product가 양수입니다.

```text
도움이 됨
```

두 방향이 직각이면 dot product가 0에 가깝습니다.

```text
별 영향 없음
```

두 방향이 반대면 dot product가 음수입니다.

```text
해로울 수 있음
```

그래서 1차 In-Run Data Shapley는 학습 데이터와 validation 데이터의 gradient 방향 정렬을 기반으로 데이터 가치를 추정합니다.

---

## 핵심 개념 5: 2차 In-Run Data Shapley

**2차 In-Run Data Shapley는 1차 gradient alignment에 더해, 데이터들 사이의 상호작용과 loss curvature까지 고려합니다.**

1차 방식은 각 training point가 validation loss에 미치는 직접적인 방향성을 봅니다.

하지만 데이터는 서로 상호작용합니다.

예를 들어 같은 데이터가 여러 번 중복되어 있다고 합시다.

```text
A, A-prime, A-double-prime, A-triple-prime
```

A 하나의 기여도는 중복 데이터들과 나눠 가져야 합니다.

2차 In-Run Data Shapley는 이런 상호작용을 더 잘 반영하려고 Hessian 정보를 사용합니다.

쉽게 말하면:

```text
1차: 이 데이터가 validation 방향과 맞는가?
2차: 이 데이터가 다른 데이터와 중복되거나 상호작용하는가?
```

입니다.

---

## 핵심 개념 6: Ghost dot-product

**Ghost dot-product는 per-sample gradient를 실제로 전부 저장하지 않고도 gradient dot product를 계산하는 효율화 기법입니다.**

문제는 gradient dot product 계산이 비싸다는 것입니다.

각 training sample마다 gradient vector를 계산하고 저장한다고 생각해봅시다.

LLM의 parameter가 수억~수십억 개라면, sample별 gradient vector는 엄청나게 큽니다.

```text
sample 1 gradient: 수억 차원
sample 2 gradient: 수억 차원
...
```

이걸 전부 만들고 dot product를 계산하면 불가능에 가깝습니다.

Ghost dot-product는 이걸 피합니다.

```text
개별 gradient vector를 직접 만들지 않고,
backpropagation 중 생기는 activation과 output gradient를 이용해
dot product만 효율적으로 계산한다.
```

---

## 핵심 개념 7: 계산 비용이 얼마나 줄어드나?

**1차 In-Run Data Shapley는 최적화된 구현에서 regular training과 거의 비슷한 속도로 계산될 수 있고, 2차 방식은 추가 backward pass 때문에 더 느리지만 여전히 naive 구현보다 훨씬 빠릅니다.**

핵심 결과는:

```text
1차 In-Run Data Shapley:
ghost dot-product 사용 시 regular training과 runtime이 거의 비슷

2차 In-Run Data Shapley:
extra backpropagation 때문에 regular training보다 느림
하지만 naive implementation보다 훨씬 빠름
```

입니다.

---

## 핵심 개념 8: 기존 Data Shapley와의 개념적 차이

**기존 Data Shapley는 “이 데이터가 learning algorithm에 평균적으로 얼마나 중요한가?”를 보고, In-Run Data Shapley는 “이 데이터가 이번 training run에서 실제 target model에 얼마나 기여했는가?”를 봅니다.**

같은 데이터셋으로 모델을 3번 학습합니다.

```text
Run 1: seed 1, batch order A
Run 2: seed 2, batch order B
Run 3: seed 3, batch order C
```

각 run에서 모델은 조금 다를 수 있습니다.

기존 retraining-based Data Shapley는 이런 stochasticity를 평균낸 learning algorithm 차원의 기여도에 가깝습니다.

In-Run Data Shapley는 특정 run 하나를 봅니다.

```text
Run 2에서 실제로 학습된 이 모델에 대해,
각 데이터는 얼마나 기여했나?
```

---

## 언제 기존 Data Shapley가 더 낫나?

이 논문이 기존 방식을 완전히 버리라고 말하는 것은 아닙니다.

기존 retraining-based Data Shapley가 더 적합한 경우도 있습니다.

```text
특정 training run보다 learning algorithm 전체에 대한 평균적 데이터 가치가 궁금할 때
iterative training algorithm이 아닌 경우
training loop 내부를 수정하기 어려울 때
clean한 Monte Carlo 구현이 더 필요한 경우
```

---

## 활용 1: Foundation model pretraining 데이터 가치 분석

**In-Run Data Shapley는 foundation model pretraining 단계에서 데이터 attribution을 가능하게 하려는 시도입니다.**

기존 Data Shapley는 작은 데이터셋과 작은 모델에서나 가능했습니다.

하지만 이 논문은 GPT-2와 Pile 데이터셋을 이용해 pretraining data contribution을 분석하는 case study를 수행했습니다.

LLM은 웹 규모 데이터로 학습됩니다.

```text
웹 문서
코드
논문
책
Q&A
위키
뉴스
포럼
```

이제 중요한 질문이 생깁니다.

```text
어떤 corpus가 모델 성능에 기여했는가?
어떤 데이터가 특정 benchmark 성능에 기여했는가?
어떤 데이터가 오히려 학습을 방해했는가?
어떤 데이터가 저작권 논의에서 contribution을 가진다고 볼 수 있는가?
어떤 데이터 source를 제거하거나 더 모아야 하는가?
```

기존 Data Shapley는 LLM 규모에서 사실상 어려웠습니다.

In-Run Data Shapley는 이 질문을 대규모 pretraining에 더 가깝게 가져오려는 시도입니다.

---

## 활용 2: 기여는 memorization과 다르다

**어떤 training data가 생성 결과와 똑같이 복사되지 않아도, 그 데이터는 여전히 모델 성능에 크게 기여할 수 있습니다.**

이건 특히 저작권 논의와 연결됩니다.

많은 논의에서 “모델이 훈련 데이터를 그대로 외워서 출력했는가?”가 중요하게 다뤄집니다.

하지만 이 논문은 contribution이 꼭 memorization이나 verbatim copying과 같지는 않다고 봅니다.

예를 들어 training corpus가 있습니다.

```text
원문: 어떤 역사적 사건에 대한 Wikipedia 문서
```

validation corpus는 GPT-4로 paraphrase된 문서입니다.

```text
내용 주제는 비슷하지만 문장 표현은 완전히 다름
```

BM25 같은 lexical similarity 기준으로는 두 문서가 많이 달라 보일 수 있습니다.

하지만 In-Run Data Shapley는 원 training corpus가 paraphrased validation corpus에도 높은 기여를 한다고 평가할 수 있습니다.

주의할 점은, 이것이 곧 법적 결론은 아니라는 것입니다.

정확한 해석은:

> 이 논문은 “데이터 기여”가 단순한 문장 복사 여부보다 넓은 개념일 수 있음을 보여준다.

입니다.

---

## 활용 3: 데이터 기여도는 학습 단계에 따라 달라진다

**어떤 데이터가 중요한지는 학습 초반과 후반에 달라질 수 있습니다.**

기존 Data Shapley는 보통 최종 결과에 대한 하나의 가치 점수를 줍니다.

하지만 In-Run Data Shapley는 training step별로 기여를 추적합니다.

그래서 이런 질문을 할 수 있습니다.

```text
학습 초반에는 어떤 데이터가 중요했나?
중반에는?
후반에는?
특정 validation task에 중요한 데이터는 언제부터 중요해졌나?
```

쉽게 말하면:

```text
초반: 일반 언어 패턴을 배우는 데이터가 중요
후반: 특정 도메인이나 task 관련 데이터가 더 중요
```

처럼 볼 수 있습니다.

이건 pretraining data curation에 매우 중요한 시각입니다.

---

## 활용 4: 저품질 데이터 찾기와 data curation

**In-Run Data Shapley는 pretraining corpus 안에서 학습에 해로운 데이터를 찾는 데 사용할 수 있습니다.**

절차는 대략 이렇습니다.

```text
1. GPT-2를 Pile subset으로 학습
2. 학습 중 In-Run Data Shapley 계산
3. 음수 value를 가진 corpus를 제거
4. cleaned subset으로 다시 학습
5. test loss와 convergence 비교
```

결과적으로 음수 Data Shapley 값을 가진 corpus를 제거했을 때, 모델이 같은 test loss에 더 빨리 도달했습니다.

이건 데이터 중심 ML 관점에서 매우 중요한 메시지입니다.

> 대규모 공개 데이터셋도 깨끗하다고 가정하면 안 된다.  
> 어떤 데이터가 실제 학습에 해로운지 추적할 수 있어야 한다.

---

## Data Shapley와 In-Run Data Shapley 비교

| 구분 | Data Shapley | In-Run Data Shapley |
|---|---|---|
| 기본 질문 | 이 데이터가 모델 성능에 평균적으로 얼마나 기여하나? | 이 데이터가 이번 training run의 target model에 얼마나 기여했나? |
| 계산 방식 | 여러 subset으로 재학습 | 한 번의 training run 중 step별 기여 누적 |
| 비용 | 매우 큼 | 훨씬 작음 |
| 대형 모델 적용 | 어려움 | foundation model pretraining에 적용 가능성 |
| 시간 dynamics | 보통 잘 포착 못 함 | 학습 단계별 기여 변화 추적 가능 |
| 주요 도구 | Shapley value, Monte Carlo 근사 | Taylor approximation, gradient dot product, Hessian term, ghost technique |
| 한계 | 재학습 비용, 특정 run 반영 어려움 | validation data 필요, iterative training 중심, approximation 의존 |

---

## In-Run Data Shapley와 Influence Function 비교

Influence function도 데이터 attribution에 많이 쓰입니다.

Influence function은 대략 이렇게 묻습니다.

```text
이 training point를 조금 upweight하거나 제거하면,
test loss가 얼마나 바뀔까?
```

하지만 보통 최종 모델 주변에서 계산합니다.

In-Run Data Shapley는 학습 과정 전체를 추적합니다.

```text
이 데이터가 training step들에서 실제로 어떤 기여를 했나?
```

쉽게 말하면:

```text
Influence function:
최종 모델 근처에서 “이 데이터를 뺐다면?”을 근사

In-Run Data Shapley:
학습 과정 전체에서 “이 데이터가 실제로 언제 얼마나 기여했나?”를 추적
```

입니다.

---

## 이 논문을 이해하기 위한 가장 쉬운 예시

작은 모델을 학습한다고 합시다.

학습 데이터는 5개입니다.

```text
A, B, C, D, E
```

validation data는 따로 있습니다.

학습이 이렇게 진행됩니다.

```text
step 1: batch = {A, C}
step 2: batch = {B, D}
step 3: batch = {A, E}
step 4: batch = {C, D}
...
```

각 step에서 모델 weight가 조금 바뀝니다.

In-Run Data Shapley는 묻습니다.

```text
step 1에서 A는 validation loss를 줄이는 방향의 gradient를 만들었나?
step 1에서 C는 어땠나?

step 2에서 B는 validation loss를 줄였나?
step 2에서 D는 오히려 validation loss를 늘렸나?

step 3에서 A는 다시 도움이 되었나?
step 3에서 E는 어땠나?
```

그리고 누적합니다.

```text
A의 최종 점수 = step 1 기여 + step 3 기여 + ...
B의 최종 점수 = step 2 기여 + ...
C의 최종 점수 = step 1 기여 + step 4 기여 + ...
```

결과:

```text
A: 높은 양수 → 좋은 데이터
B: 약한 양수 → 약간 도움
C: 높은 양수 → 좋은 데이터
D: 음수 → 해로운 데이터 가능성
E: 거의 0 → 별 영향 없음
```

이것이 In-Run Data Shapley의 직관입니다.

---

## 한계 1: validation data가 training 전에 필요하다

**In-Run Data Shapley는 학습 중에 계산되기 때문에, 어떤 validation data를 기준으로 가치를 평가할지 미리 정해야 합니다.**

이 점은 매우 중요합니다.

Data Shapley 값은 절대적 가치가 아닙니다.

항상 기준이 필요합니다.

```text
어떤 validation set에 대한 성능인가?
어떤 task에 대한 기여인가?
```

In-Run Data Shapley는 training 중 그 validation data와의 gradient 관계를 추적하므로, validation data가 training 전에 있어야 합니다.

---

## 한계 2: iterative training algorithm에 더 잘 맞는다

In-Run Data Shapley는 SGD 같은 반복 업데이트 학습 과정에 맞춰 설계되었습니다.

즉, 이런 학습 과정과 잘 맞습니다.

```text
step별 batch
gradient update
checkpoint
validation gradient
```

반면 iterative gradient training이 아닌 학습 알고리즘에는 기존 retraining-based Data Shapley가 더 일반적으로 적용될 수 있습니다.

---

## 한계 3: 근사 방법이다

In-Run Data Shapley는 Taylor approximation에 의존합니다.

즉, 각 training step의 모델 변화가 충분히 작고, validation loss 변화가 1차 또는 2차 근사로 잘 설명된다고 봅니다.

학습률이 너무 크거나, training dynamics가 매우 불안정하거나, optimizer dynamics가 복잡하면 근사 품질이 달라질 수 있습니다.

---

## 한계 4: 데이터 가치의 기준은 validation set에 묶여 있다

이건 Data Shapley 전체의 공통 한계입니다.

어떤 데이터가 높은 가치를 가진다는 것은:

```text
이 validation set / 이 utility 기준에서 가치가 높다
```

는 뜻입니다.

예를 들어 Wikipedia validation set을 기준으로 보면 Wikipedia 관련 training data가 높게 평가될 수 있습니다.

수학 문제 validation set을 기준으로 보면 수학 관련 corpus가 높게 평가될 수 있습니다.

따라서 In-Run Data Shapley 값을 해석할 때는 항상 물어야 합니다.

```text
무엇에 대한 기여도인가?
어떤 validation set을 기준으로 했는가?
어떤 loss나 metric을 봤는가?
```

---

## 이 논문이 주는 실무적 교훈

## 1. 데이터 기여도는 학습 중에도 추적할 수 있다

기존에는 학습이 끝난 뒤 분석하거나, 모델을 여러 번 재학습해야 했습니다.

이 논문은 학습 중에 기여도를 추적할 수 있다는 방향을 보여줍니다.

## 2. 데이터 가치는 고정된 숫자가 아니다

어떤 데이터가 중요한지는 학습 초반과 후반, validation task, 모델 상태에 따라 달라질 수 있습니다.

## 3. 대규모 데이터셋도 해로운 데이터가 섞일 수 있다

잘 알려진 공개 pretraining corpus에도 negative value corpora가 있을 수 있음을 보여줬습니다.

## 4. 저작권과 데이터 보상 논의에 새로운 관점을 준다

데이터가 모델 출력에 verbatim으로 나타나지 않아도, semantically related output에 기여할 수 있다는 실험 결과는 저작권·데이터 보상 논의에서 “기여”를 더 넓게 생각하게 만듭니다.

다만 이것은 기술적 attribution 결과이지, 법적 판단 그 자체는 아닙니다.

---

## Data Shapley, In-Run Data Shapley, SHAP을 다시 구분하면

| 방법 | 무엇의 기여도를 보나? | 기준 |
|---|---|---|
| **SHAP** | feature | 특정 예측값 |
| **Data Shapley** | training data point | 모델 성능 |
| **In-Run Data Shapley** | training data point 또는 corpus | 특정 training run에서 target model 성능 변화 |

예를 들어 LLM을 학습한다고 합시다.

## SHAP식 질문

```text
이 모델의 특정 예측에서 어떤 input token이나 feature가 영향을 줬나?
```

## Data Shapley식 질문

```text
이 training data가 모델 성능에 평균적으로 얼마나 기여했나?
```

## In-Run Data Shapley식 질문

```text
이번 실제 pretraining run에서 이 corpus는 validation loss 개선에 언제 얼마나 기여했나?
```

---

## 이 논문이 데이터 중심 ML에서 중요한 이유

이 논문은 Data Shapley의 현실적 확장입니다.

직전 논문은 데이터 가치평가의 원칙을 제시했습니다.

이번 논문은 이렇게 말합니다.

```text
좋다. 그런데 대형 모델에서는 어떻게 계산할 건가?
```

그 답이 In-Run Data Shapley입니다.

즉, 데이터 중심 ML을 작은 데이터셋 실험에서 foundation model pretraining 쪽으로 확장하려는 시도입니다.

---

## 이전 논문들과 연결해서 이해하기

| 이전 논문 | 이번 논문과의 연결 |
|---|---|
| **Data Shapley** | 기존 retraining-based 데이터 가치평가의 계산 비용을 줄이려는 후속 흐름 |
| **SHAP** | 둘 다 Shapley value 기반이지만, SHAP은 feature attribution, In-Run Data Shapley는 training data attribution |
| **Chinchilla / LLaMA** | pretraining token 수와 데이터 품질이 중요하므로, 어떤 데이터가 기여했는지 분석할 필요가 있음 |
| **Hidden Technical Debt** | 저품질 데이터와 데이터 의존성을 찾는 것은 data debt를 줄이는 일 |
| **Datasheets for Datasets** | 데이터 출처와 구성 문서화에 더해, 실제 성능 기여도 분석 가능 |
| **Model Evaluation** | data attribution은 validation set과 utility metric 선택에 강하게 의존 |
| **QLoRA / LoRA** | fine-tuning 데이터 가치평가에도 응용 가능하지만, 논문 주된 case study는 pretraining 쪽 |

---

## 이 논문이 주는 진짜 메시지

이 논문의 진짜 메시지는 단순히 “Data Shapley를 빠르게 계산했다”가 아닙니다.

더 중요한 메시지는 이것입니다.

**대형 모델 시대에는 모델만 커지는 것이 아니라, 데이터의 책임과 기여도도 추적 가능해야 한다.**  
**어떤 데이터가 모델을 좋게 만들었고, 어떤 데이터가 학습을 방해했으며, 어떤 데이터가 특정 능력에 기여했는지 학습 과정에서 분석할 수 있어야 한다.**

In-Run Data Shapley는 이 방향으로 가기 위한 도구입니다.

```text
데이터 가치평가
데이터 클리닝
pretraining data curation
저작권·보상 논의
학습 stage별 데이터 역할 분석
```

이런 문제에 연결됩니다.

---

## 입문자가 꼭 기억해야 할 문장

**In-Run Data Shapley는 모델을 여러 번 재학습하지 않고, 한 번의 학습 과정에서 각 데이터가 gradient update를 통해 validation 성능에 얼마나 기여했는지 누적해 데이터 가치를 추정하는 방법입니다.**

더 짧게 말하면:

> **In-Run Data Shapley = 학습 중에 데이터 기여도를 기록하는 Data Shapley**

---

## 오늘 공부용 요약

**논문명:** Data Shapley in One Training Run  
**저자:** Jiachen T. Wang, Prateek Mittal, Dawn Song, Ruoxi Jia  
**출판:** ICLR 2025 Oral

**핵심 결론:** 기존 Data Shapley는 데이터 subset마다 모델을 반복 재학습해야 하므로 대형 모델에 적용하기 어렵습니다. In-Run Data Shapley는 한 번의 training run 안에서 각 gradient update iteration의 데이터 기여도를 계산하고 누적해, 특정 target model에 대한 데이터 가치를 추정합니다.

**왜 중요함:** 이 방법은 foundation model pretraining 규모에서도 data attribution을 시도할 수 있게 합니다. 논문은 GPT-2와 Pile 데이터셋 case study를 통해 데이터 기여가 memorization과 다를 수 있고, 학습 단계에 따라 데이터 가치가 달라지며, negative value corpus를 제거하면 더 빠른 수렴을 얻을 수 있음을 보였습니다.

**입문자가 배울 점:** 데이터의 가치는 단순히 “좋은 데이터/나쁜 데이터”로 고정되지 않습니다. 어떤 validation target을 기준으로 하는지, 학습의 어느 단계인지, 실제 어떤 training run인지에 따라 데이터 기여도는 달라질 수 있습니다.

**가장 중요한 문장:** In-Run Data Shapley는 데이터 가치를 학습이 끝난 뒤 비싸게 재계산하는 것이 아니라, 학습이 진행되는 동안 step별로 추적해 누적하는 방법입니다.

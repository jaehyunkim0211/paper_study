# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

## 결론부터

이 논문의 결론은 이겁니다.

**LLM의 추론 능력은 사람이 작성한 정답 풀이 데이터를 많이 넣어서만 생기는 것이 아니라, 강화학습으로 “정답을 맞히고 형식을 지키는 행동”에 보상을 주면 스스로 긴 추론, 검산, 자기반성 같은 패턴을 발달시킬 수 있다.**

한 줄로 말하면:

> **DeepSeek-R1 = DeepSeek-V3-Base에 reasoning 중심 RL과 multi-stage post-training을 적용해 만든 장문 추론 모델**

---

## 이 논문을 한 문장으로 요약하면

**DeepSeek-R1 논문은 “추론 능력은 지도학습 풀이 데이터만으로 가르치는 것이 아니라, 올바른 보상 신호를 주면 RL 과정에서 스스로 발달할 수 있다”는 것을 보여준 논문입니다.**

---

## 두 모델 구분

```text
DeepSeek-R1-Zero:
SFT 없이 base model에 바로 대규모 RL 적용

DeepSeek-R1:
소량의 cold-start long-CoT 데이터 + 여러 단계의 RL/SFT 결합
```

R1-Zero는 reasoning behavior가 RL로 스스로 나타나는지 보기 위한 실험적 의미가 큽니다. R1은 실제 assistant로 쓰기 좋게 가독성·언어 혼합 문제를 개선한 모델입니다.

---

## 핵심 개념 1: DeepSeek-R1-Zero

R1-Zero는 일반 post-training 흐름에서 SFT를 건너뜁니다.

```text
DeepSeek-V3-Base
→ RL directly
→ DeepSeek-R1-Zero
```

사람이 쓴 long-CoT 풀이 데이터 없이 RL reward만으로 reasoning behavior가 발달할 수 있는지 보는 실험입니다.

---

## 핵심 개념 2: GRPO

GRPO는 PPO류 RL에서 critic model을 따로 두지 않고, 같은 질문에 대해 여러 답변을 샘플링한 뒤 그 그룹 안에서 상대적 보상으로 advantage를 추정하는 방법입니다.

```text
같은 문제에 답변 여러 개 생성
각 답변에 reward 부여
그룹 평균보다 잘한 답변은 강화
평균보다 못한 답변은 약화
```

---

## 핵심 개념 3: Reward

R1-Zero의 보상은 주로 rule-based reward입니다.

### Accuracy Reward

```text
수학: 최종 답을 정답과 비교
코딩: test case로 검증
```

### Format Reward

모델이 reasoning과 answer를 정해진 태그 안에 넣도록 합니다.

```text
<think> reasoning process </think>
<answer> final answer </answer>
```

---

## 핵심 개념 4: Aha Moment

R1-Zero는 RL 과정에서 스스로 “잠깐, 다시 생각해보자” 같은 자기반성 행동을 보였습니다.

```text
Wait, wait.
Let’s reevaluate this step-by-step.
```

논문은 이를 “aha moment”라고 부르며, 적절한 incentive를 주면 모델이 고급 문제 해결 전략을 스스로 발달시킬 수 있음을 보여주는 사례로 해석합니다.

---

## R1-Zero의 문제점

R1-Zero는 추론 성능은 강했지만 다음 문제가 있었습니다.

```text
가독성이 낮음
여러 언어가 섞임
출력 형식이 사용자 친화적이지 않음
답변이 산만함
```

그래서 DeepSeek-R1은 multi-stage training을 사용합니다.

---

## DeepSeek-R1의 Multi-stage Training

```text
1. Cold-start long-CoT data로 SFT
2. Reasoning-oriented RL
3. Rejection sampling + supervised fine-tuning
4. Helpfulness/harmlessness까지 포함한 all-scenario RL
```

---

## 1단계: Cold Start

수천 개의 사람이 읽기 좋은 long-CoT 데이터를 사용해 base model을 먼저 fine-tuning합니다.

이 데이터는 다음 특징을 가집니다.

```text
읽기 좋은 long-CoT
reflection / verification 포함
명확한 최종 요약
언어 혼합이나 형식 문제 감소
```

---

## 2단계: Reasoning-oriented RL

cold-start SFT 이후, 다음 task에서 대규모 RL을 수행합니다.

```text
coding
mathematics
science
logic reasoning
```

이 단계가 reasoning 능력을 크게 끌어올립니다.

---

## 3단계: Rejection Sampling + SFT

RL checkpoint로 여러 답변을 생성한 뒤, 맞고 읽기 좋은 답변만 골라 SFT 데이터를 만듭니다.

```text
1. 여러 답변 생성
2. 정답 답변 선택
3. 언어 혼합·가독성 문제 필터링
4. reasoning data + 일반 assistant data 결합
5. 다시 SFT
```

---

## 4단계: All-scenario RL

마지막 단계에서는 reasoning뿐 아니라 helpfulness와 harmlessness도 고려합니다.

```text
Reasoning reward: 정답을 맞히는가?
Helpfulness reward: 사용자에게 유용한가?
Harmlessness reward: 위험하거나 유해하지 않은가?
```

---

## 핵심 개념 5: Long CoT와 Test-time Compute

DeepSeek-R1 계열 모델은 답을 빨리 내기보다, 더 긴 reasoning token을 사용해 문제를 풀도록 학습된 reasoning model입니다.

```text
어려운 문제
→ 더 긴 thinking process
→ 검산
→ 대안 탐색
→ 최종 답
```

하지만 긴 reasoning은 비용이 큽니다.

```text
더 많은 output tokens
→ latency 증가
→ inference cost 증가
```

---

## 핵심 개념 6: Distillation

큰 reasoning model의 풀이 데이터를 작은 dense model에 distill할 수 있습니다.

```text
큰 DeepSeek-R1이 생성한 reasoning data
→ 작은 Qwen/Llama 계열 모델에 SFT
→ 작은 reasoning model 생성
```

작은 모델이 스스로 RL로 reasoning을 발견하게 하는 것보다, 큰 모델이 발견한 reasoning pattern을 전수하는 것이 더 효율적일 수 있습니다.

---

## CoT / Self-Consistency와의 연결

```text
CoT:
프롬프트로 풀이를 유도

Self-Consistency:
여러 풀이를 샘플링하고 답을 집계

DeepSeek-R1:
RL로 긴 풀이, 자기검산, 반성 행동을 학습
```

---

## 한계

1. Function calling, multi-turn, JSON output 등 일반 capability는 V3보다 약할 수 있습니다.
2. 중국어와 영어에 최적화되어 다른 언어에서 language mixing이 생길 수 있습니다.
3. Few-shot prompting이 오히려 성능을 떨어뜨릴 수 있습니다.
4. software engineering task는 RL 데이터 확보가 어렵습니다.
5. long-CoT는 비용과 latency가 큽니다.

---

## 입문자가 꼭 기억해야 할 문장

**DeepSeek-R1은 DeepSeek-V3-Base에 reasoning-oriented reinforcement learning과 multi-stage SFT/RL pipeline을 적용해, 긴 CoT, 자기검산, 반성적 풀이 행동을 강화한 reasoning model입니다.**

더 짧게 말하면:

> **DeepSeek-R1 = 강화학습으로 긴 추론 능력을 키운 LLM**

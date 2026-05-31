# ReAct: Synergizing Reasoning and Acting in Language Models

## 결론부터

이 논문의 결론은 이겁니다.

**LLM은 생각만 해서는 부족하고, 행동만 해서도 부족하다. 복잡한 문제를 풀려면 “생각하고, 행동하고, 관찰하고, 다시 생각하는 과정”을 반복해야 한다.** 이 프레임워크가 바로 **ReAct**입니다.

한 줄로 말하면:

> **ReAct = Reasoning + Acting**  
> **생각으로 계획을 세우고, 행동으로 외부 정보를 얻고, 관찰 결과를 다시 생각에 반영하는 방식**

---

## 이 논문을 한 문장으로 요약하면

**ReAct는 LLM이 “생각 → 행동 → 관찰 → 생각 → 행동”을 반복하면서 문제를 해결하게 만드는 프롬프팅 프레임워크입니다.**

CoT는 내부 reasoning만 사용합니다.

```text
질문 → 생각 → 생각 → 답
```

ReAct는 외부와 상호작용합니다.

```text
질문 → 생각 → 검색 → 관찰 → 생각 → 검색 → 관찰 → 답
```

---

## 왜 ReAct가 필요했나?

CoT는 모델이 자기 내부 지식만으로 생각합니다. 하지만 내부 지식만으로 답하면 문제가 생깁니다.

```text
기억이 오래됨
사실을 잘못 알고 있음
중간에 지어냄
근거를 확인하지 않음
```

ReAct는 모델이 필요할 때 외부 정보를 찾아보게 합니다.

```text
Thought: 먼저 A의 아버지가 누구인지 찾아야 한다.
Action: Search[A father]
Observation: A의 아버지는 B이다.
Thought: 이제 B의 출생지를 찾아야 한다.
Action: Search[B birthplace]
Observation: B는 C에서 태어났다.
Answer: C
```

---

## 핵심 개념 1: Reasoning

Reasoning은 모델이 지금 무엇을 해야 할지, 어떤 정보를 찾아야 할지, 현재 계획이 맞는지 자연어로 정리하는 단계입니다.

역할은 다음과 같습니다.

```text
문제 분해
다음 행동 결정
검색 결과 요약
현재 상태 추적
계획 수정
예외 처리
```

---

## 핵심 개념 2: Acting

Acting은 모델이 외부 환경에 실제로 어떤 행동을 수행하는 단계입니다.

질문답변에서는 보통 검색입니다.

```text
Action: Search[Christopher Nolan birthplace]
```

웹 탐색에서는 클릭이나 검색입니다.

```text
Action: Click[Add to cart]
Action: Search[waterproof hiking boots]
```

게임 환경에서는 이동이나 물건 조작입니다.

```text
Action: go to kitchen
Action: pick up apple
```

---

## 핵심 개념 3: Observation

Observation은 모델이 action을 수행한 뒤 외부 환경에서 받은 결과입니다.

```text
Action: Search[Christopher Nolan]
Observation: Christopher Nolan is a British-American filmmaker born in London, England.
```

모델은 이 observation을 보고 다시 생각합니다.

```text
Thought: Christopher Nolan was born in London, so the country is England or the UK.
```

---

## 핵심 구조: Thought-Action-Observation Loop

```text
Question: ...

Thought: 무엇을 알아야 하는지 생각한다.
Action: 검색하거나 환경에서 행동한다.
Observation: 결과를 받는다.

Thought: 관찰 결과를 바탕으로 다음 계획을 세운다.
Action: 다시 검색하거나 행동한다.
Observation: 결과를 받는다.

Answer: 최종 답을 말한다.
```

---

## 쉬운 예시: CoT vs ReAct

질문:

```text
The Dark Knight의 감독이 태어난 도시는 어디인가?
```

### CoT

```text
생각: The Dark Knight의 감독은 Christopher Nolan이다. Christopher Nolan은 영국에서 태어났다. 아마 런던일 것이다.
답: 런던
```

맞을 수도 있지만 내부 기억에 의존합니다.

### ReAct

```text
Thought: 먼저 The Dark Knight의 감독을 확인해야 한다.
Action: Search[The Dark Knight director]
Observation: The Dark Knight was directed by Christopher Nolan.

Thought: 이제 Christopher Nolan의 출생지를 확인해야 한다.
Action: Search[Christopher Nolan birthplace]
Observation: Christopher Nolan was born in London, England.

Answer: London
```

ReAct는 필요한 정보를 단계적으로 찾아 확인합니다.

---

## Reasoning-only, Acting-only, ReAct 비교

| 방식 | 설명 |
|---|---|
| Standard | 바로 답하거나 일반적인 출력 |
| CoT / Reason-only | 중간 reasoning만 생성 |
| Act-only | 생각 없이 action만 생성 |
| ReAct | reasoning과 action을 번갈아 생성 |

Act-only는 계획과 상태 추적이 약합니다. Reasoning-only는 외부 정보를 확인하지 못해 hallucination이 생길 수 있습니다. ReAct는 둘을 결합합니다.

```text
생각은 행동을 똑똑하게 만들고,
행동은 생각을 현실에 grounded하게 만든다.
```

---

## ReAct는 프롬프팅 방법이다

ReAct는 처음에는 fine-tuning이 아니라 few-shot prompt로 모델에게 형식을 보여주는 방식입니다.

```text
Question: ...
Thought: ...
Action: ...
Observation: ...
Thought: ...
Answer: ...
```

---

## 실험에서의 의미

ReAct는 다음 task에서 평가되었습니다.

```text
HotpotQA
FEVER
ALFWorld
WebShop
```

지식 QA에서는 Wikipedia API와 상호작용해 hallucination을 줄이고, ALFWorld/WebShop 같은 interactive task에서는 reasoning으로 계획을 세우며 환경에서 행동합니다.

---

## RAG와 ReAct의 차이

| 구분 | RAG | ReAct |
|---|---|---|
| 핵심 목적 | 관련 문서를 검색해 답변 근거로 사용 | reasoning과 action을 번갈아 수행 |
| 구조 | 질문 → 검색 → 문서 넣고 생성 | 생각 → 행동 → 관찰 → 생각 반복 |
| agent성 | 비교적 약함 | 더 강함 |

RAG는 검색된 문서를 넣어주는 구조이고, ReAct는 모델이 무엇을 검색할지 동적으로 결정하는 구조에 가깝습니다.

---

## 한계

1. 외부 도구가 좋아야 합니다.
2. Action space 설계가 중요합니다.
3. 여러 step을 거치므로 비용과 latency가 증가합니다.
4. Thought가 항상 faithful한 것은 아닙니다.
5. 모델이 도구를 잘못 사용할 수 있습니다.

실제 운영에서는 tool schema, action validation, step limit, fallback, logging, evaluation이 필요합니다.

---

## 입문자가 꼭 기억해야 할 문장

**ReAct는 LLM이 자연어 reasoning trace와 외부 action을 번갈아 생성하게 해서, 계획을 세우고, 도구나 환경에서 정보를 얻고, 그 관찰을 다시 reasoning에 반영하도록 만드는 프레임워크입니다.**

더 짧게 말하면:

> **ReAct = 생각하고 행동하고 관찰하는 LLM 프롬프팅**

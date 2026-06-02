# Toolformer: Language Models Can Teach Themselves to Use Tools

## 결론부터

이 논문의 결론은 이겁니다.

**LLM이 계산, 검색, 번역, 날짜 확인 같은 일을 항상 자기 머릿속 지식만으로 해결하려고 하면 한계가 있다. 그래서 외부 도구를 API처럼 호출하게 만들면 훨씬 더 정확하고 유용해질 수 있다.**

그런데 Toolformer의 핵심은 한 걸음 더 나갑니다.

> **도구 사용 예시를 사람이 대량으로 만들어주지 않아도, 언어 모델이 스스로 “언제, 어떤 도구를, 어떤 입력으로 호출하면 도움이 되는지” 학습할 수 있다.**

한 줄로 말하면:

> **Toolformer = LLM이 스스로 API 호출 데이터를 만들고, 그중 실제로 도움이 되는 호출만 골라 학습한 도구 사용 모델**

---

## 이 논문을 한 문장으로 요약하면

**Toolformer는 “도구를 쓰는 법을 사람이 일일이 가르치는 대신, 모델이 자기 언어모델 loss를 줄이는 API 호출만 골라 스스로 학습하게 하자”는 논문입니다.**

ReAct가 prompt 안에서 `Thought → Action → Observation` 형식을 보여주는 도구 사용 프레임워크였다면, Toolformer는 도구 사용 데이터를 자동 생성하고 fine-tuning하는 방법에 가깝습니다.

---

## 왜 Toolformer가 필요했나?

LLM은 다음 같은 일에 약할 수 있습니다.

```text
정확한 사칙연산
최신 사실 조회
특정 날짜 계산
저자원 언어 번역
상세한 정보 검색
```

해결책은 도구를 붙이는 것입니다.

```text
계산은 계산기에게
검색은 검색엔진에게
번역은 번역 시스템에게
날짜는 캘린더에게
```

기존 도구 사용 방법은 사람이 많은 annotation을 달거나, task별 prompt 설계가 필요했습니다. Toolformer는 이를 self-supervised하게 해결하려고 합니다.

---

## 핵심 개념 1: API Call

Toolformer에서 도구 사용은 텍스트 안에 API 호출을 삽입하는 것으로 표현됩니다.

```text
The result of [Calculator(27 + 4 * 2) -> 35] is 35.
```

여기서:

```text
Calculator(27 + 4 * 2)
```

는 API 호출이고,

```text
35
```

는 API 결과입니다.

---

## 핵심 개념 2: 사용한 도구들

논문은 다섯 가지 도구를 사용했습니다.

| 도구 | 역할 |
|---|---|
| Question Answering | 간단한 factual question에 답 |
| Calculator | 사칙연산 |
| Wikipedia Search | Wikipedia snippet 검색 |
| Machine Translation | 다른 언어 phrase를 영어로 번역 |
| Calendar | 현재 날짜 제공 |

---

## 핵심 개념 3: 데이터 자동 생성

원래 데이터가 있습니다.

```text
The Nile has an approximate length of 6,853 kilometers.
```

모델은 이 텍스트 안에 API를 넣어볼 만한 위치를 찾습니다.

```text
The Nile has an approximate length of [QA(What is the approximate length of the Nile?) -> 6,853 km] 6,853 kilometers.
```

처음에는 후보가 많이 생깁니다. 중요한 것은 이 후보 중 실제로 도움이 되는 호출만 남기는 것입니다.

---

## 핵심 개념 4: Loss-based Filtering

Toolformer는 API 호출 결과가 뒤쪽 token 예측을 더 쉽게 만들 때만 그 API 호출을 학습 데이터로 남깁니다.

```text
API 결과를 넣었더니 다음 token 예측 loss가 줄었는가?
```

줄었다면:

```text
도움 됨 → keep
```

줄지 않았거나 방해되면:

```text
도움 안 됨 → discard
```

이것이 self-supervised인 이유는 사람이 “이 API 호출이 좋다”라고 라벨링하지 않기 때문입니다. 모델 자신의 language modeling loss를 기준으로 판단합니다.

---

## 핵심 개념 5: Fine-tuning

최종 학습 데이터는 원문에 API 호출이 자연스럽게 들어간 형태입니다.

```text
The Nile has an approximate length of [QA(What is the approximate length of the Nile?) -> 6,853 km] 6,853 kilometers.
```

이 데이터로 language modeling objective를 계속 학습합니다. 모델은 다음을 배웁니다.

```text
어떤 상황에서 API를 시작해야 하는가?
어떤 API를 호출해야 하는가?
어떤 입력을 넣어야 하는가?
API 결과를 받은 뒤 문장을 어떻게 이어야 하는가?
```

---

## 핵심 개념 6: Inference 때의 작동

모델이 생성 중 API 호출을 시작하면 시스템이 실제 API를 호출합니다.

```text
모델 출력: [Calculator(27 + 4 * 2)
시스템: 계산기 호출
결과: 35
문맥 삽입: [Calculator(27 + 4 * 2) -> 35]
모델 계속 생성: Therefore, the answer is 35.
```

---

## Toolformer와 ReAct의 차이

| 구분 | ReAct | Toolformer |
|---|---|---|
| 핵심 | reasoning과 action을 번갈아 생성 | API 호출을 self-supervised로 학습 |
| 도구 사용 방법 | 프롬프트 예시로 유도 | API 호출 삽입 데이터로 fine-tuning |
| 중간 구조 | Thought → Action → Observation | 텍스트 중간에 API call 삽입 |
| interactive | 강함 | 기본 방식은 약함 |

---

## Toolformer와 RAG의 차이

RAG는 주로 검색된 문서를 context에 넣어 답하게 합니다.

```text
질문 → 관련 문서 검색 → 문서 + 질문으로 답변
```

Toolformer는 검색뿐 아니라 계산기, 번역기, 캘린더, QA API 등을 모두 도구로 다룹니다. 또한 언제 어떤 도구를 호출할지를 모델이 생성하도록 학습합니다.

---

## 장점

1. 도구 사용 데이터를 사람이 대량으로 만들 필요가 없습니다.
2. 여러 도구를 하나의 모델에 통합합니다.
3. 모델이 언제 어떤 도구를 쓸지 스스로 결정합니다.
4. 일반 언어모델링 능력을 크게 해치지 않습니다.
5. 작은 모델도 도구를 잘 쓰면 큰 모델과 경쟁할 수 있습니다.

---

## 한계

1. **도구를 chain으로 사용하지 못합니다.** 어떤 도구 결과를 다른 도구 입력으로 쓰는 흐름이 약합니다.
2. **interactive tool use가 어렵습니다.** 검색 결과를 보고 query를 수정하는 browsing에는 약합니다.
3. **입력 wording에 민감합니다.**
4. **sample inefficient할 수 있습니다.** 유용한 API call 후보를 얻기 위해 많은 문서를 처리해야 할 수 있습니다.
5. **API 호출 비용을 고려하지 않습니다.** 실제 서비스에서는 latency와 비용이 중요합니다.

---

## 입문자가 꼭 기억해야 할 문장

**Toolformer는 언어 모델이 스스로 API 호출 후보를 만들고, 그 호출 결과가 미래 token 예측 loss를 줄이는 경우만 학습 데이터로 남겨, 언제 어떤 도구를 어떻게 사용할지 학습하게 만든 방법입니다.**

더 짧게 말하면:

> **Toolformer = LLM이 스스로 도구 사용 예시를 만들고 학습하는 방법**

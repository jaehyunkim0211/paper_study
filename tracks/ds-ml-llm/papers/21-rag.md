# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — RAG

## 결론부터

이 논문의 결론은 이겁니다.

**언어 모델이 모든 지식을 모델 파라미터 안에 외우게 하지 말고, 필요한 지식은 외부 문서에서 검색해서 가져온 뒤, 그 문서를 근거로 답을 생성하게 만들자.**

이 방식이 바로 **RAG, Retrieval-Augmented Generation**입니다.

한 줄로 말하면:

> **RAG = 검색 Retriever + 생성 Generator를 결합한 모델**

예를 들어 사용자가 이렇게 묻는다고 합시다.

```text
질문: 중이 middle ear는 무엇으로 구성되어 있나요?
```

일반 생성 모델은 자기 파라미터에 저장된 지식만으로 답합니다.

```text
LLM 내부 기억 → 답변 생성
```

RAG는 다릅니다.

```text
질문
→ 관련 문서 검색
→ 검색된 문서를 읽음
→ 문서를 근거로 답변 생성
```

RAG 논문은 pre-trained language model이 사실 지식을 파라미터 안에 저장하지만, 지식을 정확히 접근·조작하거나, 근거를 제공하거나, 지식을 업데이트하는 데 한계가 있다고 보고, pre-trained seq2seq 모델의 **parametric memory**와 Wikipedia dense vector index라는 **non-parametric memory**를 결합한 retrieval-augmented generation 모델을 제안했습니다.

---

## 이 논문을 한 문장으로 요약하면

**RAG는 “모델이 기억하는 지식”과 “검색으로 가져오는 외부 지식”을 결합해, 지식이 많이 필요한 질문에 더 구체적이고 사실적인 답을 생성하려는 방법입니다.**

이전 논문들과 연결하면 위치가 분명합니다.

| 논문 | 핵심 |
|---|---|
| **BERT** | 문장을 양방향으로 이해하는 encoder 사전학습 |
| **GPT-3** | 대규모 autoregressive LM이 prompt만으로 task 수행 |
| **Scaling Laws / Chinchilla** | 모델 크기, 데이터 token, compute의 균형 |
| **RAG** | 모델 내부 지식만 쓰지 말고 외부 문서를 검색해 generation에 활용 |

GPT-3와 Chinchilla가 주로 “모델 자체를 얼마나 크게, 얼마나 많이 학습시킬 것인가?”를 다뤘다면, RAG는 질문을 조금 바꿉니다.

> 모델을 더 크게 만들어 지식을 다 외우게 할 것인가? 아니면 필요한 순간에 외부 지식을 찾아오게 할 것인가?

RAG의 답은 후자입니다.

---

## 왜 RAG가 필요한가?

LLM은 많은 지식을 파라미터 안에 저장합니다. 하지만 파라미터에 저장된 지식에는 문제가 있습니다.

| 문제 | 설명 |
|---|---|
| 업데이트가 어렵다 | 세상이 바뀌면 모델을 다시 학습해야 할 수 있음 |
| 근거 제시가 어렵다 | 왜 그렇게 답했는지 출처를 알기 어려움 |
| 정확한 지식 접근이 어렵다 | 모델이 그럴듯하지만 틀린 답을 만들 수 있음 |
| 도메인 지식 반영이 어렵다 | 회사 내부 문서, 최신 문서 등은 학습에 없을 수 있음 |

RAG는 이런 문제를 줄이기 위해 retrieval-based non-parametric memory를 결합합니다. 외부 지식을 직접 수정·확장할 수 있고, 모델이 접근한 문서를 검사할 수 있다는 장점이 있습니다.

---

## 핵심 개념 1: Parametric memory

### 결론부터

**Parametric memory는 모델의 weight 안에 저장된 지식입니다.**

예를 들어 GPT나 BERT 같은 모델은 학습 중에 많은 텍스트를 봅니다.

```text
파리는 프랑스의 수도이다.
아인슈타인은 상대성 이론을 제안했다.
물은 H2O이다.
```

이런 패턴들이 모델의 weight 안에 흡수됩니다.

이게 parametric memory입니다.

쉽게 말하면:

> **모델이 학습하면서 머릿속에 외운 지식**

RAG 논문에서는 generator인 BART의 parameter를 parametric memory로 봅니다. BART는 pre-trained seq2seq Transformer이며, RAG에서는 입력과 검색된 문서를 함께 보고 답을 생성하는 역할을 합니다.

---

## 핵심 개념 2: Non-parametric memory

### 결론부터

**Non-parametric memory는 모델 weight 바깥에 있는 외부 지식 저장소입니다.**

RAG 논문에서는 이 외부 지식 저장소가 Wikipedia 문서 index입니다.

```text
Wikipedia 문서들
→ dense vector로 변환
→ 검색 가능한 index로 저장
```

질문이 들어오면 retriever가 관련 문서를 찾습니다.

```text
질문: 오바마는 어디에서 태어났나요?
검색 결과: Barack Obama was born in Hawaii...
```

이 외부 문서 index가 non-parametric memory입니다.

쉽게 말하면:

> **모델 밖에 있는 검색 가능한 지식 창고**

---

## Parametric memory와 Non-parametric memory 비교

| 구분 | Parametric memory | Non-parametric memory |
|---|---|---|
| 위치 | 모델 weight 안 | 외부 문서 index |
| 예시 | BART, GPT의 내부 지식 | Wikipedia, 회사 문서, 검색 DB |
| 업데이트 | 어렵다. 재학습 또는 추가학습 필요 | 비교적 쉽다. 문서 index 교체·추가 |
| 근거 확인 | 어렵다 | 검색된 문서로 확인 가능 |
| 장점 | 빠르고 압축된 지식 사용 | 최신성, 근거성, 확장성 |
| 단점 | hallucination 가능, 업데이트 어려움 | 검색 품질에 의존, retrieval 실패 가능 |

RAG는 이 둘을 합칩니다.

```text
내부 지식 + 외부 검색 지식
```

---

## 핵심 개념 3: RAG의 기본 구조

RAG의 전체 흐름은 이렇게 보면 됩니다.

```text
질문 x
→ Retriever가 관련 문서 z 검색
→ Generator가 질문 x와 문서 z를 함께 보고 답 y 생성
```

조금 더 구체적으로는 두 컴포넌트가 있습니다.

| 컴포넌트 | 역할 |
|---|---|
| **Retriever** | 질문과 관련 있는 문서를 찾음 |
| **Generator** | 질문과 검색 문서를 보고 답을 생성함 |

논문에서 retriever는 DPR, Dense Passage Retriever 기반입니다. query encoder와 document encoder가 각각 질문과 문서를 dense vector로 만들고, inner product가 큰 문서를 검색합니다. Generator는 BART-large를 사용하며, 질문과 검색 문서를 concatenate해서 답을 생성합니다.

---

## 쉬운 예시: RAG가 답변하는 과정

질문이 있습니다.

```text
질문: 제주도 한라산의 높이는 얼마인가?
```

## 1단계: Retriever

Retriever는 외부 문서 index에서 관련 문서를 찾습니다.

```text
문서 1: 한라산은 대한민국 제주특별자치도에 있는 화산이다...
문서 2: 한라산의 높이는 약 1,947m이다...
문서 3: 제주도는 대한민국의 섬이다...
```

## 2단계: Generator

Generator는 질문과 검색 문서를 함께 읽습니다.

```text
질문: 제주도 한라산의 높이는 얼마인가?
문서: 한라산의 높이는 약 1,947m이다.
```

그리고 답을 생성합니다.

```text
답변: 한라산의 높이는 약 1,947m입니다.
```

일반 LLM은 자기 기억만으로 답하지만, RAG는 검색된 문서를 근거로 답합니다.

---

## 핵심 개념 4: Retriever

### 결론부터

**Retriever는 질문과 관련 있는 문서를 찾아오는 모델입니다.**

RAG 논문에서 retriever는 DPR 구조입니다.

DPR은 질문과 문서를 각각 벡터로 바꿉니다.

```text
질문 x → q(x)
문서 z → d(z)
```

그리고 두 벡터의 내적을 계산합니다.

```text
score = q(x) · d(z)
```

score가 높은 문서를 관련 문서로 봅니다.

예를 들어:

```text
질문: 오바마는 어디에서 태어났나요?
```

이 질문 vector와 가장 잘 맞는 문서 vector를 찾습니다.

```text
문서: Barack Obama was born in Hawaii.
```

---

## 핵심 개념 5: Generator

### 결론부터

**Generator는 질문과 검색된 문서를 보고 답변을 생성하는 모델입니다.**

RAG 논문에서는 BART-large를 generator로 사용했습니다.

입력은 단순히 질문만이 아닙니다.

```text
질문 + 검색 문서
```

예를 들어:

```text
질문: middle ear를 정의하라
검색 문서: The middle ear includes the tympanic cavity and the three ossicles.
```

Generator는 이 정보를 바탕으로 답을 만듭니다.

```text
The middle ear includes the tympanic cavity and the three ossicles.
```

---

## 핵심 개념 6: RAG-Sequence와 RAG-Token

RAG 논문은 두 가지 모델을 제안합니다.

```text
1. RAG-Sequence
2. RAG-Token
```

둘의 차이는 **검색 문서를 출력 전체에 하나로 고정하느냐, token마다 다르게 볼 수 있느냐**입니다.

## 1. RAG-Sequence

### 결론부터

**RAG-Sequence는 답변 전체를 생성할 때 같은 검색 문서를 사용한다고 보는 방식입니다.**

예를 들어 질문이 있습니다.

```text
질문: A Farewell to Arms의 저자는 누구인가?
```

Retriever가 문서 여러 개를 가져옵니다.

```text
문서 1: A Farewell to Arms는 Ernest Hemingway의 소설이다.
문서 2: Hemingway는 The Sun Also Rises도 썼다.
문서 3: ...
```

RAG-Sequence는 답변 전체가 특정 문서 하나를 근거로 생성된다고 봅니다.

```text
답변 전체: Ernest Hemingway
근거 문서: 문서 1
```

## 2. RAG-Token

### 결론부터

**RAG-Token은 답변의 각 token마다 다른 문서를 참고할 수 있는 방식입니다.**

예를 들어 긴 답변을 생성한다고 합시다.

```text
답변: 헤밍웨이는 A Farewell to Arms와 The Sun Also Rises를 쓴 미국 작가입니다.
```

이 답변 안에는 여러 사실이 들어 있습니다.

```text
A Farewell to Arms
The Sun Also Rises
미국 작가
```

각 token이나 구절마다 다른 문서가 더 유용할 수 있습니다.

---

## RAG-Sequence와 RAG-Token의 차이

| 구분 | RAG-Sequence | RAG-Token |
|---|---|---|
| 문서 사용 방식 | 답변 전체에 같은 문서 사용 | token마다 다른 문서 가능 |
| 직관 | 하나의 근거 문서로 전체 답 생성 | 여러 문서를 조합하며 생성 |
| 장점 | 비교적 단순 | 복잡한 답변에서 유연 |
| 단점 | 여러 사실을 섞기 어려울 수 있음 | 계산과 해석이 더 복잡 |

입문 단계에서는 이렇게 이해하면 충분합니다.

```text
RAG-Sequence = 답변 전체가 하나의 문서에 의존
RAG-Token = 답변의 각 부분이 다른 문서에 의존 가능
```

---

## 핵심 개념 7: Marginalization

RAG 논문에서 조금 어려운 단어가 나옵니다.

```text
marginalize over documents
```

쉽게 말하면:

> 검색된 문서 하나만 정답으로 고정하지 않고, 여러 검색 문서가 답변에 기여할 가능성을 합쳐서 본다.

예를 들어 top-3 문서가 있다고 합시다.

```text
문서 1: 관련도 0.6
문서 2: 관련도 0.3
문서 3: 관련도 0.1
```

각 문서를 보고 generator가 답을 만들 확률이 다릅니다.

```text
문서 1을 보고 답 y 생성 확률
문서 2를 보고 답 y 생성 확률
문서 3을 보고 답 y 생성 확률
```

RAG는 문서 검색 확률과 생성 확률을 함께 고려합니다.

```text
최종 답 확률
= 문서 1 기여 + 문서 2 기여 + 문서 3 기여
```

---

## RAG는 어떻게 학습하나?

### 결론부터

**RAG는 질문-답변 쌍을 가지고 retriever와 generator를 함께 fine-tuning합니다.**

예를 들어 학습 데이터가 있습니다.

```text
질문: middle ear는 무엇을 포함하나요?
정답: tympanic cavity and the three ossicles
```

RAG는 질문을 보고 문서를 검색합니다.

```text
검색 문서 1
검색 문서 2
검색 문서 3
```

Generator는 검색 문서를 보고 답을 생성합니다.

정답과 비교해서 loss를 계산합니다.

```text
생성 답변 vs 정답
```

그리고 이 loss로 retriever와 generator를 업데이트합니다.

중요한 점은, 학습 데이터에 “어떤 문서를 검색해야 하는지”라는 직접 라벨이 없어도 된다는 것입니다.

```text
문서 정답 라벨 없음
질문-답변 쌍만 있음
```

---

## RAG와 일반 LLM의 차이

| 구분 | 일반 parametric LLM | RAG |
|---|---|---|
| 지식 저장 | 모델 weight 안 | 모델 weight + 외부 문서 |
| 답변 생성 | 내부 기억 기반 | 검색 문서 기반 |
| 최신 지식 반영 | 재학습 필요 가능 | index 업데이트로 가능 |
| 출처 확인 | 어려움 | 검색 문서 확인 가능 |
| 실패 원인 | 모델이 모름, hallucination | 검색 실패 또는 generation 오류 |
| 대표 구조 | GPT, BART 등 | Retriever + Generator |

---

## RAG와 Fine-tuning의 차이

## Fine-tuning

모델 weight를 바꿉니다.

```text
모델에 도메인 데이터를 추가 학습
→ weight 업데이트
→ 모델 내부 지식 또는 행동 변화
```

장점:

```text
특정 task 스타일을 잘 배울 수 있음
출력 형식, 판단 기준, 도메인 언어 적응에 좋음
```

단점:

```text
최신 문서 반영이 어려움
지식 출처 확인이 어려움
잘못 학습하면 수정 어려움
```

## RAG

모델 weight는 그대로 두고, 외부 문서를 검색해서 넣습니다.

```text
사용자 질문
→ 관련 문서 검색
→ LLM에 문서와 질문 제공
→ 답변 생성
```

장점:

```text
최신 문서 반영 쉬움
출처 확인 가능
회사 내부 문서 활용 가능
모델 재학습 없이 지식 추가 가능
```

단점:

```text
검색이 틀리면 답도 틀릴 수 있음
문서 chunking, indexing, ranking이 중요
긴 문서를 잘 요약·조합해야 함
```

입문적으로는 이렇게 기억하면 됩니다.

```text
Fine-tuning = 모델의 행동과 표현 방식을 조정
RAG = 모델이 참고할 지식을 외부에서 제공
```

---

## 쉬운 실무 예시: 회사 내부 규정 QA

회사 내부 규정을 답변하는 챗봇을 만든다고 합시다.

사용자가 묻습니다.

```text
육아휴직 신청은 몇 일 전에 해야 하나요?
```

## 일반 LLM

모델이 학습 때 본 일반 지식으로 답합니다.

```text
보통 회사 정책에 따라 다릅니다...
```

답이 부정확하거나 너무 일반적일 수 있습니다.

## Fine-tuning

회사 규정 문서를 모델에 추가 학습합니다.

문제:

```text
규정이 바뀌면 다시 학습해야 함
근거 조항을 정확히 제시하기 어려울 수 있음
```

## RAG

회사 규정 문서 index를 만듭니다.

```text
질문
→ 육아휴직 관련 규정 검색
→ 해당 조항을 LLM에 제공
→ 조항 근거로 답변
```

RAG가 적합한 이유는 다음과 같습니다.

```text
문서가 자주 바뀜
정확한 근거가 필요함
모델을 매번 재학습하기 어렵다
```

---

## RAG가 hallucination을 완전히 해결하나?

아닙니다.

RAG는 hallucination을 줄이는 데 도움이 될 수 있지만, 완전히 없애지는 못합니다.

왜냐하면 RAG에도 실패 지점이 많습니다.

| 실패 지점 | 설명 |
|---|---|
| 검색 실패 | 관련 문서를 못 찾음 |
| 검색은 했지만 ranking 실패 | 중요한 문서가 top-k에 없음 |
| 문서 chunk가 부적절 | 필요한 정보가 잘려 있음 |
| generator가 문서를 잘못 읽음 | 문서 내용과 다른 답 생성 |
| 여러 문서 충돌 | 서로 다른 문서를 잘못 조합 |
| 질문 해석 오류 | 검색 쿼리가 잘못 만들어짐 |

정확한 표현은 이겁니다.

> RAG는 모델이 외부 근거를 참고하게 만들어 factuality와 업데이트 가능성을 개선할 수 있지만, 검색·읽기·생성 단계가 실패하면 여전히 틀릴 수 있다.

---

## RAG가 잘 작동하려면 무엇이 중요한가?

## 1. 좋은 문서 집합

RAG는 검색할 문서가 좋아야 합니다.

```text
문서가 오래됨
문서에 오류가 있음
문서가 중복됨
문서가 너무 길거나 너무 짧게 잘림
```

이면 답변도 흔들립니다.

## 2. 좋은 chunking

문서를 검색 단위로 나눠야 합니다.

```text
너무 작게 자르면 맥락이 부족함
너무 크게 자르면 검색 정확도가 떨어짐
```

## 3. 좋은 retriever

질문과 관련 있는 문서를 top-k 안에 가져와야 합니다.

## 4. 좋은 generator

검색된 문서를 제대로 읽고, 질문에 맞게 답해야 합니다.

## 5. 근거 사용 검증

답변이 실제로 검색 문서에 근거하는지 확인해야 합니다.

---

## 이 논문과 현재 RAG 실무의 차이

원 논문의 RAG는 retriever와 generator를 end-to-end fine-tuning하는 연구 모델에 가깝습니다.

현대 실무에서 “RAG”라고 부르는 것은 보통 더 넓은 시스템 패턴입니다.

```text
문서 수집
→ chunking
→ embedding
→ vector DB 저장
→ 질문 embedding
→ top-k 검색
→ reranking
→ LLM prompt에 문서 삽입
→ 답변 생성
→ citation / verification
```

즉, 원 논문 RAG는 다음 구조였습니다.

```text
DPR retriever + BART generator + Wikipedia index
```

현대 실무 RAG는 보통 다음 구조입니다.

```text
embedding model + vector database + reranker + LLM + prompt engineering
```

그래도 핵심 아이디어는 같습니다.

> **외부 문서를 검색해서 생성 모델의 입력에 넣는다.**

---

## RAG와 GPT-3 / LLM prompting의 관계

GPT-3는 prompt 안의 예시와 지시문을 보고 task를 수행했습니다.

RAG는 여기에 검색 문서를 추가합니다.

```text
질문만 넣기
→ LLM이 내부 지식으로 답함

질문 + 검색 문서 넣기
→ LLM이 문서를 근거로 답함
```

즉, RAG는 prompt를 더 강하게 만듭니다.

```text
Prompt = 지시문 + 질문 + 검색된 문서 + 답변 형식
```

현대 LLM 응용에서 RAG는 “모델을 다시 학습하지 않고 지식을 주입하는 방법”으로 매우 중요합니다.

---

## RAG와 Chinchilla의 연결

Chinchilla는 모델을 잘 학습하려면 parameter 수와 token 수의 균형이 중요하다고 했습니다.

하지만 모든 지식을 모델에 학습시키는 것은 비용이 큽니다.

RAG는 다른 방향입니다.

```text
모든 지식을 모델 parameter에 넣지 않는다.
필요할 때 외부 문서에서 가져온다.
```

이 관점에서 RAG는 LLM scaling의 한계를 보완합니다.

| 문제 | Scaling 접근 | RAG 접근 |
|---|---|---|
| 지식 부족 | 더 많은 token으로 pretraining | 외부 문서 검색 |
| 최신성 부족 | 모델 재학습 | index 업데이트 |
| 도메인 지식 부족 | domain pretraining/fine-tuning | 도메인 문서 연결 |
| 근거 부족 | model card/eval로 보완 | 검색 문서 인용 가능 |

---

## 이 논문이 주는 진짜 메시지

RAG의 진짜 메시지는 단순히 “검색을 붙이자”가 아닙니다.

더 큰 메시지는 이것입니다.

**언어 모델의 지식은 모델 weight 안에만 있어야 하는 것이 아니다. 외부 지식 저장소와 결합하면 더 업데이트 가능하고, 근거를 확인할 수 있고, 지식 집약적 task에서 더 강력한 시스템을 만들 수 있다.**

이 논문은 parametric model과 retrieval system의 경계를 연결했습니다.

```text
모델이 외운 지식
+
모델이 찾아온 지식
```

이 둘을 결합한 것이 RAG입니다.

---

## 입문자가 꼭 기억해야 할 문장

**RAG는 질문과 관련된 문서를 먼저 검색하고, 그 문서를 생성 모델의 입력으로 넣어 답변을 만드는 방식입니다.**

더 짧게 말하면:

> **RAG = 찾아보고 답하는 LLM**

---

## 오늘 공부용 요약

**논문명:** Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks  
**저자:** Patrick Lewis et al.  
**핵심 결론:** RAG는 pre-trained seq2seq generator라는 parametric memory와 Wikipedia dense vector index라는 non-parametric memory를 결합해, 지식이 많이 필요한 NLP task에서 검색된 문서를 근거로 답변을 생성하는 모델입니다.

**왜 중요함:** LLM이 모든 지식을 parameter 안에 외우는 방식의 한계를 보완합니다. 검색 index를 바꾸면 지식을 업데이트할 수 있고, 검색 문서를 통해 답변 근거를 확인할 수 있습니다.

**입문자가 배울 점:** RAG의 성능은 LLM만으로 결정되지 않습니다. 문서 품질, chunking, retriever, ranking, generator, 근거 검증이 모두 중요합니다.

**가장 중요한 문장:** RAG는 모델이 기억에만 의존하지 않고, 외부 문서를 검색해서 그 근거를 바탕으로 답하게 만드는 구조입니다.

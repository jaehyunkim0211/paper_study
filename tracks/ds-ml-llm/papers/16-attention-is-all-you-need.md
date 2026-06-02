# Attention Is All You Need — Transformer

## 결론부터

이 논문의 결론은 이겁니다.

**문장처럼 순서가 있는 데이터를 처리할 때, 꼭 RNN이나 CNN을 쓸 필요는 없다. 단어들 사이의 관계를 attention으로 직접 계산하면, 더 병렬화하기 쉽고 성능도 좋은 sequence model을 만들 수 있다.**

이 구조가 바로 **Transformer**입니다.

한 줄로 말하면:

> **Transformer는 RNN/CNN 없이 self-attention만으로 문장 안의 단어들이 서로 어떤 관계를 갖는지 계산하는 모델이다.**

Vaswani et al.은 기존 sequence transduction 모델이 주로 RNN 또는 CNN 기반 encoder-decoder 구조였고, 좋은 모델들은 거기에 attention을 붙여 사용했다고 설명합니다. 이 논문은 recurrence와 convolution을 완전히 제거하고 attention mechanism만으로 구성한 Transformer를 제안했습니다.

---

## 이 논문을 한 문장으로 요약하면

**Transformer는 “순서대로 읽는 모델”에서 “모든 단어가 서로를 한 번에 바라보는 모델”로 바꾼 논문입니다.**

RNN은 문장을 보통 순서대로 처리합니다.

```text
나는 → 오늘 → 밥을 → 먹었다
```

즉, 앞 단어를 처리한 뒤 다음 단어를 처리합니다.

반면 Transformer의 self-attention은 문장 안의 모든 단어가 서로를 직접 봅니다.

```text
"나는"   ↔ "오늘" ↔ "밥을" ↔ "먹었다"
```

그래서 이런 질문을 한 번에 계산합니다.

```text
이 단어는 문장 안의 어떤 단어를 중요하게 봐야 하는가?
```

예를 들어:

```text
그 학생은 선생님에게 책을 돌려주었다.
```

여기서 “그”가 누구를 가리키는지, “책”이 무엇과 연결되는지, “돌려주었다”의 주체와 객체가 무엇인지 같은 관계를 attention으로 학습합니다.

---

## 왜 이 논문이 중요한가?

이 논문은 현대 AI의 핵심 분기점입니다.

BERT, GPT, T5, LLaMA, Claude, Gemini 같은 현대 언어 모델은 모두 Transformer 계열 구조를 기반으로 합니다. 물론 각각 encoder-only, decoder-only, encoder-decoder 등 구조 차이는 있지만, 핵심 아이디어는 이 논문에서 출발합니다.

이전 딥러닝 기본기 파트에서 우리는 CNN과 RNN을 봤습니다.

| 모델 | 잘하는 것 | 한계 |
|---|---|---|
| **CNN** | 이미지, 지역 패턴 | 긴 sequence 관계를 직접 잡기 어려움 |
| **RNN/LSTM** | 순서 데이터 | 순차 처리 때문에 병렬화가 어려움 |
| **Transformer** | 단어 간 전역 관계 | attention 계산 비용이 sequence 길이에 대해 커짐 |

Transformer 논문은 RNN의 순차 처리 제약을 강하게 문제 삼았습니다. recurrent model은 입력·출력 위치를 시간 단계에 맞춰 순차적으로 계산하기 때문에 긴 sequence에서 병렬화가 어렵습니다. 반면 Transformer는 recurrence를 버리고 attention으로 input-output 사이의 global dependency를 다룹니다.

---

## 핵심 개념 1: Attention

### 결론부터

**Attention은 “지금 처리 중인 단어가 다른 단어들을 얼마나 참고해야 하는가”를 계산하는 방법입니다.**

예를 들어 번역 모델이 영어 문장:

```text
I love this movie
```

를 한국어로 번역한다고 합시다.

한국어 단어 “좋아한다”를 만들 때, 모델은 영어 단어 중 무엇을 봐야 할까요?

```text
I      → 약간 참고
love   → 매우 중요
this   → 조금 참고
movie  → 중요
```

Attention은 이런 가중치를 계산합니다.

```text
love: 0.55
movie: 0.25
I: 0.15
this: 0.05
```

그다음 각 단어의 정보를 가중합해서 현재 위치의 표현을 만듭니다.

논문은 attention function을 query와 key-value 쌍을 output으로 mapping하는 함수로 설명합니다. output은 value들의 weighted sum이고, 각 value의 weight는 query와 해당 key의 compatibility function으로 계산됩니다.

---

## 핵심 개념 2: Query, Key, Value

Transformer를 이해하려면 **Q, K, V**를 잡아야 합니다.

쉽게 비유하면 도서관 검색입니다.

| 개념 | 도서관 비유 | Transformer에서의 의미 |
|---|---|---|
| **Query, Q** | 내가 찾고 싶은 질문 | 현재 단어가 찾는 정보 |
| **Key, K** | 책의 검색 태그 | 각 단어가 가진 검색용 특징 |
| **Value, V** | 실제 책 내용 | 참고해서 가져올 정보 |

예를 들어 현재 단어가 “먹었다”라고 합시다.

이 단어의 query는 이런 질문에 가깝습니다.

```text
누가 먹었는가?
무엇을 먹었는가?
언제 먹었는가?
```

문장 안의 다른 단어들은 key를 가지고 있습니다.

```text
나는: 주어 후보
밥을: 목적어 후보
오늘: 시간 후보
```

query와 key가 잘 맞으면 attention score가 높아집니다. 그리고 value는 실제로 가져올 정보입니다.

즉, attention은 이렇게 작동합니다.

```text
Q와 K를 비교해서 중요도를 계산하고,
그 중요도로 V를 가중합한다.
```

---

## 핵심 개념 3: Scaled Dot-Product Attention

Transformer의 기본 attention 공식은 이것입니다.

```text
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k)) V
```

처음 보면 어려워 보이지만, 단계별로 보면 단순합니다.

### 1단계: Q와 K의 유사도 계산

```text
QKᵀ
```

현재 단어의 query와 모든 단어의 key를 비교합니다.

결과는 attention score입니다.

```text
이 단어를 얼마나 봐야 하는가?
```

### 2단계: scale 조정

```text
QKᵀ / sqrt(d_k)
```

dimension이 커지면 dot product 값이 너무 커질 수 있습니다. 그러면 softmax가 너무 극단적으로 변하고 gradient가 작아질 수 있습니다. 그래서 `sqrt(d_k)`로 나눠줍니다.

### 3단계: softmax로 확률처럼 변환

```text
softmax(...)
```

각 단어를 얼마나 볼지 가중치를 만듭니다.

```text
[0.1, 0.2, 0.6, 0.1]
```

### 4단계: V를 가중합

```text
softmax(...) V
```

중요한 단어의 value는 많이 반영하고, 덜 중요한 단어는 적게 반영합니다.

---

## 핵심 개념 4: Self-Attention

### 결론부터

**Self-attention은 한 sequence 안의 단어들이 서로를 attention하는 것입니다.**

예를 들어 문장이 있습니다.

```text
나는 오늘 밥을 먹었다
```

Self-attention에서는 각 단어가 문장 안의 모든 단어를 봅니다.

```text
"나는"은 "먹었다"와 연결될 수 있음
"밥을"은 "먹었다"와 연결될 수 있음
"오늘"은 "먹었다"와 연결될 수 있음
```

즉, self-attention은 문장 내부 관계를 계산합니다.

### 쉬운 예시: “그것”이 무엇을 가리키는가?

문장을 봅시다.

```text
나는 책을 가방에 넣었다. 그것은 무거웠다.
```

여기서 “그것”은 무엇일까요?

```text
책?
가방?
```

사람은 문맥으로 판단합니다.

Transformer의 self-attention은 “그것”이라는 단어가 이전 단어들 중 어떤 단어를 봐야 하는지 학습합니다.

```text
그것 → 책: 높은 attention
그것 → 가방: 낮거나 상황에 따라 다름
```

물론 실제 attention head가 항상 사람이 해석하는 의미와 정확히 일치하는 것은 아닙니다. 하지만 attention은 이런 장거리 관계를 모델이 직접 계산할 수 있는 경로를 제공합니다.

---

## 핵심 개념 5: Multi-Head Attention

### 결론부터

**Multi-head attention은 attention을 여러 번 병렬로 수행해서, 서로 다른 종류의 관계를 동시에 보게 하는 방법입니다.**

하나의 attention head만 있으면 한 가지 관점으로만 문장을 봅니다.

하지만 문장에는 여러 관계가 있습니다.

```text
주어-동사 관계
동사-목적어 관계
수식어 관계
대명사 참조
문장 구조
```

Multi-head attention은 여러 head가 각자 다른 관점으로 단어 관계를 보게 합니다.

| Head | 볼 수 있는 관계 |
|---|---|
| Head 1 | 주어-동사 관계 |
| Head 2 | 동사-목적어 관계 |
| Head 3 | 위치 관계 |
| Head 4 | 대명사 참조 |
| Head 5 | 구문 구조 |

비유하면 여러 명의 전문가가 동시에 같은 문장을 읽는 것입니다.

```text
문법 전문가
의미 전문가
대명사 전문가
어순 전문가
번역 전문가
```

각 전문가는 같은 문장을 보지만, 관심사가 다릅니다. 마지막에는 전문가들의 의견을 합쳐 최종 표현을 만듭니다.

---

## 핵심 개념 6: Positional Encoding

### 결론부터

**Transformer는 RNN도 CNN도 없기 때문에, 단어 순서를 따로 알려줘야 합니다. 그 역할을 하는 것이 positional encoding입니다.**

RNN은 단어를 순서대로 처리하기 때문에 순서 정보가 자연스럽게 들어갑니다.

```text
1번째 단어 → 2번째 단어 → 3번째 단어
```

CNN도 주변 위치를 보는 구조라 위치 정보가 어느 정도 반영됩니다.

하지만 self-attention만 쓰면 기본적으로 단어들의 집합을 보는 것에 가깝습니다.

예를 들어:

```text
나는 너를 좋아한다
너는 나를 좋아한다
```

두 문장은 단어는 비슷하지만 의미는 다릅니다. 순서가 중요합니다.

그래서 Transformer는 입력 embedding에 positional encoding을 더합니다.

```text
token embedding + positional encoding
```

쉽게 말하면:

> **단어들에게 문장 속 자리 번호를 알려주는 것**입니다.

---

## 핵심 개념 7: Encoder-Decoder 구조

Transformer 논문은 기계번역을 주요 task로 다뤘기 때문에 encoder-decoder 구조를 사용했습니다.

```text
영어 문장 → Encoder → 문장 표현 → Decoder → 한국어 문장
```

### Encoder

Encoder는 입력 문장을 읽고, 각 단어의 문맥화된 representation을 만듭니다.

논문 기준 encoder layer는 크게 두 부분입니다.

```text
Multi-head self-attention
→ Position-wise feed-forward network
```

그리고 각 sub-layer 주변에는 residual connection과 layer normalization이 있습니다.

### Decoder

Decoder는 출력 문장을 한 단어씩 생성합니다.

Decoder layer에는 세 가지 attention/FFN 부분이 있습니다.

```text
Masked self-attention
→ Encoder-decoder attention
→ Feed-forward network
```

---

## 핵심 개념 8: Masked Self-Attention

### 결론부터

**Decoder에서는 미래 단어를 보면 안 되기 때문에, 미래 위치를 attention하지 못하도록 mask를 씌웁니다.**

예를 들어 번역 결과를 생성하는 중이라고 합시다.

```text
나는 밥을 ___
```

다음 단어를 예측할 때, 아직 생성하지 않은 미래 단어를 보면 안 됩니다.

학습 데이터에는 정답 문장이 이미 있으므로, 그대로 두면 모델이 미래 단어를 몰래 볼 수 있습니다.

그래서 decoder self-attention에서는 현재 위치보다 뒤에 있는 단어를 가립니다.

```text
현재 위치 i는 1~i까지만 볼 수 있음
i+1, i+2, ... 는 볼 수 없음
```

GPT 같은 모델이 바로 이 decoder-style masked attention을 기반으로 다음 token을 예측합니다.

---

## 핵심 개념 9: Position-wise Feed-Forward Network

Transformer layer에는 attention만 있는 것이 아닙니다.

각 위치마다 feed-forward network도 적용합니다.

```text
FFN(x) = max(0, xW1 + b1)W2 + b2
```

이 FFN은 각 token position에 독립적으로 적용됩니다.

쉽게 말하면:

```text
attention: 단어들끼리 정보를 섞음
FFN: 각 단어 표현을 개별적으로 변환함
```

---

## 핵심 개념 10: Residual Connection과 Layer Normalization

이전 ResNet 논문을 기억하면 Transformer 구조가 훨씬 잘 보입니다.

Transformer도 각 sub-layer 주변에 residual connection을 사용합니다.

```text
x + Sublayer(x)
```

그리고 그 결과에 LayerNorm을 적용합니다.

여기서 BatchNorm이 아니라 **LayerNorm**이 쓰입니다.

BatchNorm은 batch 통계를 사용했습니다.

```text
batch 전체의 평균/분산
```

LayerNorm은 한 sample 안의 feature 차원을 기준으로 정규화합니다.

Transformer에서는 sequence 길이나 batch 구성에 따라 BatchNorm보다 LayerNorm이 더 자연스럽게 쓰입니다.

---

## Transformer 전체 구조를 한 번에 보면

기계번역 기준으로 전체 흐름은 이렇습니다.

```text
입력 문장 tokens
→ token embedding
→ positional encoding 더함
→ encoder layer 여러 개 통과
   - self-attention
   - feed-forward
→ encoder output

출력 문장 tokens, shifted right
→ token embedding
→ positional encoding 더함
→ decoder layer 여러 개 통과
   - masked self-attention
   - encoder-decoder attention
   - feed-forward
→ softmax
→ 다음 token 확률
```

---

## 왜 Transformer가 RNN보다 빠른가?

RNN은 time step을 순서대로 처리해야 합니다.

```text
h1 계산 → h2 계산 → h3 계산 → h4 계산
```

즉, `h3`를 계산하려면 `h2`가 필요하고, `h2`를 계산하려면 `h1`이 필요합니다.

그래서 한 문장 내부에서 병렬화가 어렵습니다.

Transformer의 self-attention은 각 위치의 attention을 행렬곱으로 한 번에 계산할 수 있습니다.

```text
모든 단어의 Q, K, V를 한 번에 계산
QKᵀ도 행렬곱으로 계산
```

그래서 GPU에서 병렬 처리하기 좋습니다.

---

## 왜 long-range dependency에 유리한가?

RNN에서 첫 단어와 마지막 단어가 영향을 주고받으려면 여러 step을 거쳐야 합니다.

```text
단어 1 → 단어 2 → 단어 3 → ... → 단어 n
```

경로가 깁니다.

Transformer에서는 self-attention으로 단어 1과 단어 n이 직접 연결될 수 있습니다.

```text
단어 1 ↔ 단어 n
```

이론적으로 한 attention layer 안에서 모든 단어 쌍이 연결됩니다.

---

## Transformer의 한계

### 1. Attention 계산 비용

Self-attention은 모든 token 쌍을 비교합니다.

sequence 길이가 `n`이면 attention score matrix는 대략:

```text
n × n
```

입니다.

즉, 길이가 길어질수록 계산량과 메모리가 빠르게 커집니다.

### 2. 순서 정보가 기본적으로 없다

Transformer는 recurrence나 convolution이 없기 때문에 positional encoding을 따로 넣어야 합니다.

### 3. Decoder generation은 여전히 순차적이다

학습 때는 teacher forcing과 masking 덕분에 병렬화가 많이 가능하지만, 실제 생성 inference에서는 다음 token을 만들고, 그다음 token을 만들고, 또 그다음 token을 만드는 auto-regressive 과정이 필요합니다.

```text
token 1 생성
→ token 2 생성
→ token 3 생성
```

---

## 이 논문이 현대 AI에 준 영향

이 논문은 처음에는 기계번역 모델 논문이었습니다.

하지만 이후 Transformer는 NLP를 넘어 vision, speech, multimodal AI의 핵심 구조가 되었습니다.

현재 관점에서 보면 이 논문의 가장 큰 의미는 이것입니다.

> sequence를 순서대로 처리해야 한다는 고정관념을 깨고, token 간 관계를 attention으로 직접 모델링하는 구조를 제시했다.

이 하나의 아이디어가 BERT, GPT, LLM으로 이어졌습니다.

---

## 입문자가 꼭 기억해야 할 문장

**Transformer는 각 token이 다른 모든 token을 attention으로 직접 참고하게 하고, 이를 multi-head로 병렬 수행해 문장 내부의 다양한 관계를 학습하는 모델이다.**

더 짧게 말하면:

> **Transformer = self-attention으로 token 간 관계를 직접 계산하는 모델**

---

## 오늘 공부용 요약

**논문명:** Attention Is All You Need  
**저자:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
**핵심 결론:** RNN이나 CNN 없이 attention mechanism만으로 sequence transduction을 수행할 수 있으며, 이 구조는 더 병렬화하기 쉽고 기계번역에서 강한 성능을 보인다.

**왜 중요함:** Transformer는 BERT, GPT, LLM 등 현대 NLP와 생성 AI의 기반 구조가 되었다. 핵심은 recurrence를 제거하고 self-attention으로 token 간 관계를 직접 계산했다는 점이다.

**입문자가 배울 점:** Attention은 “무엇을 볼 것인가”를 학습하는 메커니즘이고, self-attention은 한 문장 안의 모든 단어가 서로를 참고하게 만든다. Multi-head attention은 여러 관점의 관계를 병렬로 학습하게 한다.

**가장 중요한 문장:** Transformer는 순서대로 읽는 모델이 아니라, 모든 token이 서로를 직접 바라보며 문맥 표현을 만드는 모델이다.

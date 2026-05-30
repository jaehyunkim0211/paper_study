# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## 결론부터

이 논문의 결론은 이겁니다.

**언어 모델을 대규모 텍스트로 먼저 사전학습 pre-training 해두고, 이후 원하는 NLP task에 맞게 조금만 fine-tuning하면 다양한 문제에서 강력한 성능을 낼 수 있다.**

그리고 BERT의 핵심은 **Transformer encoder를 양방향 bidirectional으로 학습한다**는 점입니다.

쉽게 말하면:

> BERT는 문장을 왼쪽에서 오른쪽으로만 읽지 않고, 어떤 단어를 이해할 때 그 단어의 왼쪽 문맥과 오른쪽 문맥을 모두 함께 본다.

논문 초록은 BERT를 **Bidirectional Encoder Representations from Transformers**라고 정의하고, unlabeled text에서 왼쪽·오른쪽 문맥을 모든 layer에서 함께 조건으로 삼아 deep bidirectional representation을 pre-train한다고 설명합니다. 그리고 이렇게 사전학습한 모델에 task-specific output layer만 추가해 question answering, language inference 같은 다양한 task에 fine-tuning할 수 있다고 말합니다.

---

## 이 논문을 한 문장으로 요약하면

**BERT는 “문장을 양방향으로 이해하는 Transformer encoder를 대규모 텍스트로 미리 학습해두고, 여러 NLP 문제에 재사용하자”는 논문입니다.**

직전 논문인 Transformer와 비교하면 이렇게 볼 수 있습니다.

| 논문 | 핵심 |
|---|---|
| **Attention Is All You Need** | RNN/CNN 없이 attention으로 sequence를 처리하는 Transformer 구조 제안 |
| **BERT** | Transformer encoder를 대규모 텍스트로 사전학습해서 다양한 NLP task에 fine-tuning |

즉, Transformer가 **구조 architecture**의 혁신이라면, BERT는 그 구조를 이용한 **사전학습 + 전이학습 pre-training + fine-tuning 패러다임**의 혁신입니다.

---

## 왜 BERT가 중요했나?

BERT 이전에도 word embedding이나 language model pre-training은 있었습니다.

예를 들어 Word2Vec, GloVe 같은 방법은 단어마다 vector를 줬습니다.

```text
bank = [0.2, -0.1, 0.7, ...]
```

문제는 같은 단어라도 문맥에 따라 의미가 달라진다는 점입니다.

```text
bank에 돈을 맡겼다
강가 river bank에 앉았다
```

두 문장에서 `bank`는 의미가 다릅니다. 그런데 고정 word embedding은 보통 같은 단어에 같은 vector를 줍니다.

BERT는 다릅니다.

BERT는 문장 전체를 보고, 각 단어의 표현을 문맥에 맞게 만듭니다.

```text
bank in "money bank" → 금융기관 의미의 표현
bank in "river bank" → 강둑 의미의 표현
```

이걸 **contextual representation**, 즉 문맥화된 표현이라고 볼 수 있습니다.

---

## 핵심 개념 1: BERT는 Transformer encoder다

### 결론부터

**BERT는 Transformer의 encoder 부분만 사용하는 모델입니다.**

Transformer 원 논문은 encoder-decoder 구조였습니다.

```text
Encoder: 입력 문장을 읽음
Decoder: 출력 문장을 생성함
```

BERT는 이 중 encoder 쪽에 가깝습니다.

```text
입력 문장
→ Transformer encoder layers
→ 각 token의 문맥화된 표현
```

BERT가 encoder인 이유는, 모든 token이 서로를 양방향으로 볼 수 있기 때문입니다.

반면 GPT는 decoder 쪽에 가깝습니다.

```text
이전 token들만 보고 다음 token을 예측
```

---

## BERT와 GPT의 가장 쉬운 차이

문장이 있다고 합시다.

```text
나는 [MASK] 마셨다
```

정답은 `커피를`이라고 해봅시다.

### GPT식 왼쪽에서 오른쪽 모델

GPT처럼 왼쪽에서 오른쪽으로만 보는 모델은 `[MASK]` 위치를 예측할 때 왼쪽 문맥만 봅니다.

```text
나는 → ?
```

오른쪽의 `마셨다`는 아직 못 봅니다.

그래서 “나는” 다음에 올 수 있는 것은 너무 많습니다.

```text
나는 오늘
나는 너를
나는 커피를
나는 집에
```

### BERT식 양방향 모델

BERT는 양쪽을 봅니다.

```text
나는 ___ 마셨다
```

왼쪽의 `나는`도 보고, 오른쪽의 `마셨다`도 봅니다.

그러면 빈칸에 들어갈 말이 더 잘 좁혀집니다.

```text
커피를
물을
차를
```

이게 BERT의 bidirectional representation입니다.

---

## 핵심 개념 2: Pre-training과 Fine-tuning

### 결론부터

**BERT는 먼저 정답 라벨이 없는 대규모 텍스트로 언어 이해 능력을 학습하고, 그다음 작은 task-specific 데이터로 fine-tuning합니다.**

두 단계입니다.

```text
1. Pre-training
   대규모 unlabeled text로 일반 언어 표현 학습

2. Fine-tuning
   감성분석, 질의응답, 문장관계판단 등 특정 task에 맞게 조정
```

예전에는 NLP task마다 모델을 따로 설계하는 경우가 많았습니다.

```text
감성분석 모델 따로
질의응답 모델 따로
개체명 인식 모델 따로
문장 유사도 모델 따로
```

BERT는 접근을 바꿉니다.

```text
거대한 공통 언어 모델 하나를 먼저 학습
→ task별로 조금씩 fine-tuning
```

즉, BERT는 “NLP 모델을 처음부터 매번 새로 만들지 말고, 이미 언어를 많이 배운 모델을 가져와서 쓰자”는 흐름을 강하게 만든 논문입니다.

---

## 핵심 개념 3: Masked Language Model, MLM

### 결론부터

**MLM은 문장 중 일부 단어를 가리고, 주변 문맥을 보고 그 단어를 맞히게 하는 사전학습 task입니다.**

예를 들어 문장이 있습니다.

```text
나는 오늘 커피를 마셨다
```

BERT는 일부 token을 가립니다.

```text
나는 오늘 [MASK] 마셨다
```

그리고 모델에게 묻습니다.

```text
[MASK]에 들어갈 원래 단어는 무엇인가?
```

정답:

```text
커피를
```

이렇게 하면 모델은 왼쪽 문맥과 오른쪽 문맥을 동시에 이용해야 합니다.

```text
왼쪽: 나는 오늘
오른쪽: 마셨다
```

---

## MLM이 왜 필요했나?

BERT는 양방향으로 문맥을 보고 싶습니다.

그런데 일반적인 language model은 보통 이렇게 학습합니다.

```text
이전 단어들을 보고 다음 단어 예측
```

예:

```text
나는 오늘 커피를 → 마셨다
```

이 방식은 왼쪽 문맥만 봅니다.

반대로 오른쪽에서 왼쪽으로 학습할 수도 있습니다.

```text
오늘 커피를 마셨다 ← 나는
```

하지만 이것도 한 방향입니다.

BERT는 양쪽을 모두 보고 싶습니다.

문제는 그냥 양쪽을 모두 보게 하고 자기 자신을 예측하게 하면, 모델이 정답 단어 자체를 볼 수 있습니다.

```text
나는 오늘 커피를 마셨다
```

여기서 `커피를`을 예측하라고 하면서 입력에 `커피를`을 그대로 두면 반칙입니다.

그래서 BERT는 그 단어를 가립니다.

```text
나는 오늘 [MASK] 마셨다
```

이렇게 해야 모델이 자기 자신을 보지 않고, 주변 문맥으로 맞히게 됩니다.

---

## MLM의 15%, 80/10/10 규칙

BERT는 모든 단어를 가리지 않습니다.

각 sequence의 WordPiece token 중 일부, 보통 15% 정도를 예측 대상으로 고릅니다. 그런데 선택된 token을 항상 `[MASK]`로 바꾸지는 않습니다.

| 경우 | 입력 |
|---|---|
| 80% | 선택된 token을 `[MASK]`로 바꿈 |
| 10% | 선택된 token을 random token으로 바꿈 |
| 10% | 선택된 token을 원래 token 그대로 둠 |

왜 이렇게 할까요?

fine-tuning 때는 `[MASK]` token이 등장하지 않기 때문입니다.

pre-training 때 항상 `[MASK]`만 보면, pre-training과 fine-tuning 사이에 mismatch가 생깁니다.

그래서 BERT는 가끔 random token이나 원래 token을 넣어, 모델이 `[MASK]`에만 과하게 의존하지 않도록 합니다.

---

## 핵심 개념 4: Next Sentence Prediction, NSP

### 결론부터

**NSP는 두 문장이 실제로 이어지는 문장인지 아닌지 맞히는 사전학습 task입니다.**

예를 들어 문장 A가 있습니다.

```text
A: 나는 카페에 갔다.
```

문장 B 후보가 있습니다.

```text
B1: 커피를 주문했다.
B2: 고래는 바다에 산다.
```

B1은 A 다음에 자연스럽게 이어질 수 있습니다.

```text
IsNext
```

B2는 corpus에서 random하게 가져온 문장일 수 있습니다.

```text
NotNext
```

BERT는 두 문장이 실제로 이어지는지 맞히도록 학습합니다.

NSP의 정답은 사람이 붙이는 것이 아니라 원문 문서의 문장 순서를 이용해서 자동으로 만들 수 있습니다.

```text
실제로 이어지는 문장 → IsNext
랜덤으로 가져온 문장 → NotNext
```

다만 현대적으로는 NSP의 필요성은 논쟁이 있습니다. 이후 RoBERTa 같은 모델은 NSP를 제거하고도 좋은 성능을 보였습니다.

---

## 핵심 개념 5: BERT 입력 구조

BERT의 입력은 단순히 token embedding 하나가 아닙니다.

각 token의 입력 representation은 세 가지 embedding의 합입니다.

```text
Token Embedding
+ Segment Embedding
+ Position Embedding
```

문장 두 개를 넣는다고 합시다.

```text
문장 A: 나는 카페에 갔다
문장 B: 커피를 마셨다
```

BERT 입력은 대략 이렇게 됩니다.

```text
[CLS] 나는 카페에 갔다 [SEP] 커피를 마셨다 [SEP]
```

각 token에는 세 가지 정보가 더해집니다.

### 1. Token embedding

단어 자체의 의미입니다.

```text
나는, 카페에, 갔다, 커피를, 마셨다
```

### 2. Segment embedding

이 token이 문장 A에 속하는지, 문장 B에 속하는지 알려줍니다.

```text
[CLS] 나는 카페에 갔다 [SEP] → Segment A
커피를 마셨다 [SEP] → Segment B
```

### 3. Position embedding

몇 번째 위치의 token인지 알려줍니다.

```text
0번, 1번, 2번, ...
```

최종 입력은:

```text
각 token = token embedding + segment embedding + position embedding
```

입니다.

---

## [CLS]와 [SEP]의 역할

### [CLS]

`[CLS]`는 classification을 위한 special token입니다.

BERT는 sequence 맨 앞에 `[CLS]`를 붙입니다.

```text
[CLS] 나는 오늘 커피를 마셨다
```

Transformer layers를 통과한 뒤, `[CLS]` 위치의 최종 hidden state를 문장 전체의 representation처럼 사용합니다.

예를 들어 감성분석에서는 `[CLS]` representation을 classification layer에 넣습니다.

```text
[CLS] representation → positive / negative
```

### [SEP]

`[SEP]`는 문장 구분자입니다.

```text
[CLS] 문장 A [SEP] 문장 B [SEP]
```

문장 하나만 들어갈 때도 끝에 `[SEP]`를 붙입니다.

---

## 핵심 개념 6: Fine-tuning은 생각보다 단순하다

### 결론부터

**BERT의 강점은 downstream task마다 구조를 크게 바꾸지 않아도 된다는 점입니다.**

감성분석:

```text
[CLS] 이 영화는 정말 좋았다 [SEP]
→ [CLS] representation
→ classification layer
```

질문답변:

```text
[CLS] 질문 [SEP] 문단 [SEP]
→ 각 token representation
→ answer span의 start/end 위치 예측
```

개체명 인식:

```text
문장 token들
→ 각 token representation
→ 각 token의 label 예측
```

문장 관계 판단:

```text
[CLS] 문장 A [SEP] 문장 B [SEP]
→ [CLS] representation
→ entailment / contradiction / neutral
```

---

## BERT-base와 BERT-large

논문은 두 가지 주요 모델 크기를 사용했습니다.

| 모델 | Layer 수 | Hidden size | Attention head | Parameter |
|---|---:|---:|---:|---:|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

이 표에서 중요한 건 숫자 자체보다 흐름입니다.

```text
모델이 커질수록 더 많은 언어 패턴을 학습할 수 있다.
하지만 계산 비용도 커진다.
```

---

## BERT와 ELMo, GPT의 차이

| 모델 | 방식 | 핵심 |
|---|---|---|
| **ELMo** | feature-based | 좌→우 LM과 우→좌 LM을 따로 학습해 representation을 feature로 사용 |
| **GPT** | fine-tuning | 왼쪽 문맥만 보는 Transformer decoder식 LM |
| **BERT** | fine-tuning | 양방향 Transformer encoder를 MLM+NSP로 pre-training |

BERT는 GPT와 비슷한 fine-tuning approach지만, bidirectional self-attention과 MLM/NSP pre-training을 사용한다는 점이 다릅니다.

---

## BERT를 아주 쉬운 비유로 이해하면

BERT는 “빈칸 문제를 엄청 많이 푼 학생”입니다.

예를 들어 국어 문제집을 많이 풉니다.

```text
나는 오늘 카페에서 ___를 마셨다.
```

정답:

```text
커피
```

또 다른 문제:

```text
강아지가 ___를 흔들었다.
```

정답:

```text
꼬리
```

또 다른 문제:

```text
비가 와서 우산을 ___.
```

정답:

```text
썼다
```

이런 문제를 엄청 많이 풀다 보면 학생은 단순히 정답만 외우는 게 아니라, 한국어 문법과 의미와 상식을 배우게 됩니다.

BERT도 똑같습니다.

```text
빈칸 맞히기를 많이 하면서 언어 구조를 배운다.
```

그리고 나중에 다른 시험을 봅니다.

```text
감성분석
질의응답
문장 유사도
개체명 인식
```

이미 언어를 많이 배웠기 때문에, 새로운 문제에도 잘 적응합니다.

---

## BERT의 핵심 한계

### 1. 생성 모델이 아니다

BERT는 encoder-only 모델입니다.

문장을 이해하고 분류하거나, span을 찾는 데 강하지만, GPT처럼 왼쪽에서 오른쪽으로 자연스럽게 긴 문장을 생성하는 모델은 아닙니다.

```text
BERT: 이해/분류/추출에 강함
GPT: 다음 token 생성에 강함
```

### 2. `[MASK]` mismatch가 있다

BERT는 pre-training 때 `[MASK]` token을 사용합니다.

하지만 fine-tuning이나 실제 입력에는 보통 `[MASK]`가 없습니다.

### 3. NSP의 필요성은 이후 논쟁이 있었다

BERT 논문은 NSP가 QA와 NLI에 유익하다고 주장했지만, 이후 RoBERTa는 NSP를 제거해도 성능이 같거나 개선될 수 있다고 보고했습니다.

### 4. 계산 비용이 크다

BERT pre-training은 상당한 계산 자원을 요구합니다.

---

## 입문자가 꼭 기억해야 할 문장

**BERT는 Transformer encoder를 이용해 문장을 양방향으로 이해하도록 사전학습한 모델이고, `[MASK]` 단어 맞히기와 문장 관계 예측을 통해 다양한 NLP task에 전이 가능한 언어 표현을 학습한다.**

더 짧게 말하면:

> **BERT = 양방향 Transformer encoder를 대규모 텍스트로 미리 학습한 언어 이해 모델**

---

## 오늘 공부용 요약

**논문명:** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
**저자:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova  
**핵심 결론:** BERT는 unlabeled text로 Transformer encoder를 양방향 사전학습하고, downstream task에 fine-tuning하는 방식으로 다양한 NLP task에서 강력한 성능을 낸다.

**왜 중요함:** BERT는 “대규모 사전학습 모델을 가져와 task별로 fine-tuning한다”는 현대 NLP의 표준 패러다임을 강하게 확립했다.

**입문자가 배울 점:** BERT의 핵심은 단어를 왼쪽에서 오른쪽으로만 이해하는 것이 아니라, 왼쪽·오른쪽 문맥을 함께 보고 문맥화된 token representation을 만드는 것이다.

**가장 중요한 문장:** BERT는 `[MASK]`로 가린 단어를 양쪽 문맥으로 맞히면서, 문장과 단어를 깊은 양방향 Transformer 표현으로 바꾸는 모델이다.

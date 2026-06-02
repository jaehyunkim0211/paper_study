# LLaMA: Open and Efficient Foundation Language Models

## 결론부터

이 논문의 결론은 이겁니다.

**LLM을 무조건 더 크게만 만들 필요는 없다. 좋은 공개 데이터로 충분히 오래 학습시키면, 상대적으로 작은 모델도 훨씬 큰 모델과 경쟁할 수 있다.**

한 줄로 말하면:

> **LLaMA는 “작지만 충분히 잘 학습된 foundation language model”의 가능성을 보여준 논문입니다.**

Meta AI는 LLaMA를 7B, 13B, 33B, 65B 크기의 foundation language model 시리즈로 제안했고, 공개적으로 접근 가능한 데이터만 사용해 학습했습니다. 논문 초록은 LLaMA-13B가 대부분의 벤치마크에서 GPT-3 175B를 능가했고, LLaMA-65B는 Chinchilla-70B 및 PaLM-540B와 경쟁력 있는 성능을 냈다고 설명합니다.

---

## 이 논문을 한 문장으로 요약하면

**LLaMA는 “모델 크기, 데이터 품질, 학습 token 수, inference 비용” 사이의 균형을 다시 생각하게 만든 오픈 계열 LLM 논문입니다.**

이전 논문들과 연결하면 위치가 분명합니다.

| 논문 | 핵심 메시지 |
|---|---|
| **GPT-3** | 큰 모델은 prompt만으로 여러 task를 수행할 수 있다 |
| **Scaling Laws** | 모델 크기, 데이터, compute가 커지면 loss가 예측 가능하게 줄어든다 |
| **Chinchilla** | 모델 크기와 학습 token 수의 균형이 중요하다 |
| **LoRA** | 큰 모델을 효율적으로 task adaptation할 수 있다 |
| **LLaMA** | 공개 데이터와 충분한 token 학습으로 작은 모델도 강력하게 만들 수 있다 |

LLaMA의 중요한 관점은 **training compute optimality만 보는 것이 아니라 inference budget도 중요하게 본다**는 점입니다.

---

## 왜 LLaMA가 중요했나?

GPT-3 이후 LLM 경쟁은 주로 모델 크기를 키우는 방향으로 갔습니다.

```text
175B
280B
540B
...
```

그런데 모델이 커지면 inference 비용이 커집니다.

```text
사용자 질문 1개 처리 비용 증가
GPU memory 요구량 증가
latency 증가
배포 난이도 증가
```

LLaMA는 질문을 바꿉니다.

> “가장 큰 모델을 만들자”가 아니라, “주어진 inference budget에서 가장 좋은 성능을 내는 모델을 만들자.”

LLaMA의 목표는 다양한 inference budget에서 가능한 최고의 성능을 내는 모델 시리즈를 학습하는 것이었습니다. 그래서 LLaMA는 일반적인 사용량보다 더 많은 token으로 학습했고, 그 결과 7B~65B 모델이 기존 강력한 LLM들과 경쟁력 있는 성능을 냈습니다.

---

## 핵심 개념 1: Foundation Language Model

### 결론부터

**Foundation language model은 특정 task 하나만을 위해 만든 모델이 아니라, 다양한 downstream task에 활용될 수 있는 범용 언어 모델입니다.**

예를 들어 LLaMA는 처음부터 다음 중 하나만 하도록 학습된 모델이 아닙니다.

```text
감성분석 전용 모델
번역 전용 모델
코딩 전용 모델
QA 전용 모델
```

대신 대규모 텍스트로 language modeling을 학습합니다.

```text
앞의 token들을 보고 다음 token 예측
```

그렇게 학습한 뒤, 여러 방식으로 활용할 수 있습니다.

```text
zero-shot prompting
few-shot prompting
instruction tuning
LoRA fine-tuning
RAG와 결합
```

즉, LLaMA는 **base model**입니다.

ChatGPT처럼 바로 assistant로 쓰기 위해 RLHF까지 강하게 적용한 모델이라기보다는, 이후 fine-tuning과 alignment를 하기 좋은 기반 모델에 가깝습니다.

---

## 핵심 개념 2: “Open and Efficient”의 의미

논문 제목은 **Open and Efficient Foundation Language Models**입니다.

여기서 두 단어가 중요합니다.

## 1. Open

LLaMA 논문에서 “open”은 주로 **공개적으로 접근 가능한 데이터만으로 강력한 모델을 학습했다**는 의미입니다.

많은 대형 모델은 학습 데이터가 불투명했습니다.

```text
비공개 책 데이터
비공개 웹 데이터
소셜미디어 대화
정확히 문서화되지 않은 대규모 corpus
```

LLaMA는 공개 데이터만으로도 강력한 모델을 만들 수 있음을 보이려 했습니다.

## 2. Efficient

여기서 efficient는 단순히 “학습이 싸다”는 뜻만은 아닙니다.

특히 중요한 것은 **inference efficiency**입니다.

LLaMA-13B는 GPT-3 175B보다 훨씬 작습니다.

```text
13B vs 175B
```

그런데 대부분의 벤치마크에서 GPT-3보다 좋은 성능을 냈다고 보고됩니다.

즉, LLaMA의 핵심은:

```text
작게 만들되,
많이 학습시키고,
좋은 데이터로 학습시켜서,
inference 비용 대비 성능을 높이자.
```

입니다.

---

## 핵심 개념 3: 공개 데이터 기반 학습

LLaMA는 여러 공개 데이터 소스를 섞어 학습했습니다.

논문에 따르면 학습 데이터 mixture는 대략 다음과 같습니다.

| 데이터 | 비율 |
|---|---:|
| English CommonCrawl | 67.0% |
| C4 | 15.0% |
| GitHub | 4.5% |
| Wikipedia | 4.5% |
| Gutenberg / Books3 | 4.5% |
| ArXiv | 2.5% |
| StackExchange | 2.0% |

이 구성이 중요한 이유는 LLaMA가 단순히 웹 텍스트만 먹은 모델이 아니라는 점입니다.

```text
웹 문서
책
위키피디아
코드
논문
Q&A 사이트
```

이런 다양한 출처를 섞었습니다.

그래서 모델이 일반 언어, 지식, 코드, 과학 문서, 질의응답 형식 등을 어느 정도 학습할 수 있었습니다.

---

## 핵심 개념 4: 데이터 품질과 필터링

LLaMA 논문에서 중요한 부분은 단순히 “공개 데이터”가 아니라 **공개 데이터를 정제해서 사용했다**는 점입니다.

예를 들어 CommonCrawl은 웹 크롤링 데이터라 품질이 들쭉날쭉합니다.

```text
좋은 글
광고
스팸
중복
저품질 페이지
다른 언어
깨진 문서
```

공개 데이터만 써도 된다 해도, 공개 데이터를 잘 고르고 잘 정제해야 합니다.

이건 Chinchilla의 메시지와도 연결됩니다.

```text
많은 token이 필요하다.
하지만 token 품질도 중요하다.
```

---

## 핵심 개념 5: 모델 크기와 학습 token 수

LLaMA 모델 크기와 학습 token 수는 다음과 같습니다.

| 모델 | Parameter | Dimension | Layers | Heads | Batch size | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA 7B | 6.7B | 4096 | 32 | 32 | 4M | 1.0T |
| LLaMA 13B | 13.0B | 5120 | 40 | 40 | 4M | 1.0T |
| LLaMA 33B | 32.5B | 6656 | 60 | 52 | 4M | 1.4T |
| LLaMA 65B | 65.2B | 8192 | 80 | 64 | 4M | 1.4T |

여기서 매우 중요한 감각이 있습니다.

```text
LLaMA 7B: 1T tokens
LLaMA 13B: 1T tokens
```

이전 GPT-3는 175B 모델을 약 300B token으로 학습했습니다. 반면 LLaMA는 훨씬 작은 모델을 훨씬 많은 token으로 학습했습니다.

이것이 Chinchilla 이후의 중요한 방향입니다.

> 더 작은 모델을 더 많이 학습시키면 inference 효율이 좋아질 수 있다.

---

## 핵심 개념 6: Chinchilla와 LLaMA의 차이

Chinchilla는 이렇게 말했습니다.

```text
고정된 training compute에서 compute-optimal하게 학습하려면
모델 크기와 token 수를 균형 있게 키워라.
```

LLaMA는 여기에 한 가지 현실적 관점을 추가합니다.

```text
하지만 실제 서비스에서는 inference 비용도 중요하다.
같은 성능이면 더 작은 모델이 낫다.
```

Chinchilla식 compute-optimal 훈련은 “학습 비용을 기준으로 최적”을 찾습니다.

LLaMA는 “inference budget별로 좋은 모델”을 만들려 합니다.

| 관점 | 질문 |
|---|---|
| Chinchilla | 같은 training compute에서 loss를 최소화하려면? |
| LLaMA | 같은 inference budget에서 좋은 성능을 내려면? |

---

## 핵심 개념 7: Architecture 개선

LLaMA는 Transformer 기반 autoregressive language model입니다.

즉, GPT 계열처럼 이전 token을 보고 다음 token을 예측합니다.

하지만 원래 Transformer 그대로는 아니고, 여러 개선을 사용했습니다.

## 1. Pre-normalization + RMSNorm

기존 Transformer는 sub-layer 출력 후 normalization을 적용하는 구조가 많았습니다.

LLaMA는 sub-layer 입력을 먼저 normalization합니다.

```text
Norm → Attention
Norm → MLP
```

이런 pre-norm 구조는 깊은 모델 학습 안정성에 도움이 됩니다.

LLaMA는 LayerNorm 대신 **RMSNorm**을 사용했습니다.

## 2. SwiGLU activation

Transformer의 MLP block에서 ReLU 대신 **SwiGLU**를 사용했습니다.

SwiGLU는 PaLM 등에서 사용된 activation 계열로, 성능 향상에 도움이 되는 것으로 알려져 있습니다.

## 3. Rotary Positional Embeddings, RoPE

기존 Transformer의 absolute positional embedding 대신 **RoPE**를 사용했습니다.

RoPE는 token의 위치 정보를 attention 계산에 더 자연스럽게 반영하는 방식입니다.

---

## LLaMA architecture를 쉽게 이해하면

LLaMA는 완전히 새로운 architecture라기보다, 여러 검증된 개선을 잘 조합한 모델입니다.

```text
기본 구조: GPT식 decoder-only Transformer
정규화: RMSNorm 기반 pre-norm
활성화 함수: SwiGLU
위치 정보: RoPE
학습 목표: next-token prediction
```

즉, LLaMA의 혁신은 “완전히 새로운 neural architecture”라기보다는 다음 조합에 있습니다.

```text
좋은 공개 데이터
많은 training tokens
효율적인 모델 크기
검증된 architecture 개선
연구 커뮤니티 공개
```

---

## 핵심 개념 8: Efficient Implementation

대형 모델은 좋은 아이디어만으로는 학습할 수 없습니다.

실제로 학습하려면 구현 효율이 매우 중요합니다.

LLaMA 논문은 학습 속도를 높이기 위해 여러 최적화를 사용했습니다.

예를 들어:

```text
효율적인 causal multi-head attention 구현
attention weights 저장 줄이기
activation checkpointing
model parallelism
sequence parallelism
GPU 간 communication overlap
```

이 부분은 LLM 실무에서 중요합니다.

```text
모델 설계
데이터 설계
학습 recipe
분산 학습 구현
```

이 네 가지가 모두 맞아야 대형 모델 학습이 가능합니다.

---

## 실험 결과 1: Common Sense Reasoning

LLaMA는 여러 common sense reasoning benchmark에서 좋은 결과를 보였습니다.

중요한 건 숫자 하나가 아니라 패턴입니다.

```text
작은 모델 + 많은 token + 좋은 데이터
→ 훨씬 큰 모델과 경쟁 가능
```

LLaMA-13B는 GPT-3보다 약 10배 작지만 대부분의 benchmark에서 GPT-3를 능가했습니다.

---

## 실험 결과 2: Closed-book QA

Closed-book QA는 모델이 외부 문서를 보지 않고, 내부 지식만으로 답하는 task입니다.

예를 들어:

```text
질문: 파리의 수도는?
모델 내부 지식만으로 답해야 함
```

LLaMA는 Natural Questions와 TriviaQA에서 평가되었습니다.

LLaMA-65B는 closed-book QA benchmark에서 좋은 성능을 보였고, 13B 모델도 GPT-3와 Chinchilla에 경쟁력 있는 결과를 보였습니다.

다만 오늘날 실무에서는 closed-book QA보다 RAG를 결합하는 경우가 많습니다.

```text
내부 지식만으로 답하기
vs
외부 문서를 찾아보고 답하기
```

RAG는 최신성·근거성에서 더 유리할 수 있습니다.

---

## 실험 결과 3: Code Generation

LLaMA는 코드 전용 모델은 아니지만, training data에 GitHub가 포함되어 있어 code generation 능력도 평가했습니다.

이 결과가 주는 메시지는:

```text
코드 전용 fine-tuning 없이도,
데이터 mixture에 코드가 있으면 상당한 code generation 능력이 생긴다.
```

입니다.

하지만 code-specific token으로 fine-tuning하면 코드 성능을 더 개선할 수 있습니다.

---

## 실험 결과 4: MMLU

MMLU는 다양한 학문 영역의 multiple-choice question benchmark입니다.

LLaMA-65B는 MMLU에서 좋은 성능을 냈지만, Chinchilla-70B와 PaLM-540B보다 낮았습니다.

이 부분은 매우 중요합니다.

> 어떤 benchmark에서 강하려면 데이터 mixture가 중요하다.

즉, 전체 token 수만 중요한 것이 아닙니다.

```text
무슨 token을 먹었는가?
```

도 중요합니다.

MMLU처럼 지식 시험에 가까운 benchmark는 책과 학술 문서 비중이 영향을 줄 수 있습니다.

---

## 핵심 개념 9: Instruction Finetuning의 효과

LLaMA 논문은 instruction tuning이 주요 초점은 아니었지만, 간단한 실험을 했습니다.

LLaMA-65B를 instruction data로 짧게 fine-tuning한 LLaMA-I는 MMLU에서 더 좋은 성능을 기록했습니다.

이건 InstructGPT 논문과 연결됩니다.

```text
Base LLM만으로도 어느 정도 할 수 있다.
하지만 instruction tuning을 하면 사용자 지시를 더 잘 따른다.
```

즉, LLaMA는 base model이고, 그 위에 SFT, RLHF, LoRA 등을 붙여 assistant 모델로 만들 수 있습니다.

---

## 핵심 개념 10: Bias, Toxicity, Misinformation

LLaMA 논문은 성능만 보고 끝내지 않고, bias와 toxicity도 평가했습니다.

웹 비중이 큰 학습 데이터 때문에, 모델이 유해한 콘텐츠나 사회적 편향을 생성할 위험을 평가하는 것이 중요합니다.

여기서 기억할 점은 이것입니다.

```text
벤치마크 성능이 높다
≠ 항상 안전하고 진실하다
```

LLaMA는 강력한 base model이지만, 별도 instruction tuning, safety tuning, RAG, 평가 체계가 필요합니다.

---

## LLaMA와 GPT-3 비교

| 구분 | GPT-3 | LLaMA |
|---|---|---|
| 발표 시기 | 2020 | 2023 |
| 대표 크기 | 175B | 7B, 13B, 33B, 65B |
| 학습 token | 약 300B | 1.0T~1.4T |
| 구조 | decoder-only Transformer | decoder-only Transformer |
| 데이터 | 대규모 웹/책 등, 일부 비공개 | 공개적으로 접근 가능한 데이터 중심 |
| 핵심 메시지 | 거대 모델의 few-shot 능력 | 작은 모델도 많이 학습하면 강력 |
| 사용 관점 | API/비공개 중심 | 연구 커뮤니티 접근성 강조 |

LLaMA-13B가 대부분의 benchmark에서 GPT-3를 능가했다는 결과는 이 비교의 핵심입니다. 모델 크기만 보면 13B는 175B보다 훨씬 작지만, 더 많은 token과 좋은 데이터 recipe가 성능을 크게 끌어올렸습니다.

---

## LLaMA와 Chinchilla 비교

| 구분 | Chinchilla | LLaMA |
|---|---|---|
| 핵심 목적 | training compute-optimality | inference budget별 효율적 모델 |
| 대표 모델 | 70B | 7B~65B |
| 학습 token | 1.4T | 1.0T~1.4T |
| 데이터 | 자세한 전체 공개는 제한적 | 공개적으로 접근 가능한 데이터만 사용 |
| 메시지 | 모델과 token 균형 | 작은 모델도 더 오래 학습하면 효율적 |
| 공개성 | 모델 공개 제한 | 연구 커뮤니티 공개 강조 |

Chinchilla는 “compute-optimal하게 학습하자”라는 논문이고, LLaMA는 그 영향을 받아 더 많은 token으로 모델을 학습하되, inference 비용까지 고려해 작은 모델의 성능을 끌어올렸습니다.

---

## LLaMA와 LoRA의 연결

LoRA는 다음을 가능하게 했습니다.

```text
큰 base model을 freeze하고
작은 adapter만 학습해서 task adaptation
```

LLaMA는 좋은 base model을 제공합니다.

그래서 조합은 이렇게 됩니다.

```text
LLaMA base model
+ LoRA adapter
= 적은 GPU로 도메인/태스크 fine-tuning
```

예를 들어:

```text
LLaMA-7B + LoRA → SQL 생성 모델
LLaMA-13B + LoRA → 한국어 instruction model
LLaMA + LoRA → 사내 문서 요약 스타일 fine-tuning
```

즉, LLaMA는 공개 계열 base model의 역할을 했고, LoRA는 그 base model을 저렴하게 바꾸는 도구였습니다.

---

## LLaMA의 한계

## 1. Base model이지 assistant model이 아니다

LLaMA v1은 기본적으로 next-token prediction으로 학습한 foundation model입니다.

사용자 지시를 따르는 assistant로 바로 쓰기 위해서는 instruction tuning이나 RLHF가 필요합니다.

## 2. 데이터가 공개적이어도 완전히 문제없는 것은 아니다

공개 웹 데이터에는 편향, 독성, 저품질 정보, 개인정보 위험이 포함될 수 있습니다.

## 3. MMLU에서는 최고 성능이 아니었다

LLaMA-65B는 MMLU에서 Chinchilla-70B와 PaLM-540B보다 낮았습니다. 데이터 mixture의 차이가 원인일 수 있습니다.

## 4. Open이라고 해서 모든 사용이 자유롭다는 뜻은 아니다

논문상의 공개성, 연구 접근성, 상업적 오픈소스 자유 사용은 구분해야 합니다.

---

## 실무에서 LLaMA 논문을 어떻게 받아들여야 하나?

## 1. Parameter 수만 보지 말기

```text
13B가 175B보다 약할 것이다
```

라고 단정하면 안 됩니다.

학습 token 수, 데이터 품질, architecture, training recipe가 모두 중요합니다.

## 2. Inference cost까지 생각하기

서비스에서는 학습 비용보다 inference 비용이 더 중요해질 수 있습니다.

```text
큰 모델 1번 학습
→ 이후 수억 번 inference
```

이 경우 같은 성능이면 작은 모델이 훨씬 유리합니다.

## 3. Base model과 instruction model 구분하기

LLaMA 같은 base model은 그대로 쓰기보다 보통 후처리합니다.

```text
SFT
RLHF
DPO
LoRA
QLoRA
RAG
```

## 4. 데이터 mixture가 benchmark 성능을 좌우한다

MMLU, code, math, QA, reasoning은 각각 다른 데이터의 영향을 받습니다.

```text
책/논문 많음 → 지식 benchmark 유리
코드 많음 → code generation 유리
Q&A 많음 → instruction/QA 유리
웹 텍스트 많음 → 일반 언어 패턴 풍부
```

---

## 이 논문이 주는 진짜 메시지

LLaMA의 진짜 메시지는 단순히 “Meta가 모델을 공개했다”가 아닙니다.

더 중요한 메시지는 이것입니다.

**LLM 성능은 모델 크기 하나로 결정되지 않는다. 공개 데이터, 데이터 정제, 충분한 학습 token, 효율적인 architecture, inference 비용까지 함께 봐야 한다.**

GPT-3가 “큰 모델의 힘”을 보여줬다면, LLaMA는 이렇게 말합니다.

> 큰 모델만이 답은 아니다. 더 작고 효율적인 모델도 충분히 강력할 수 있다.

---

## 입문자가 꼭 기억해야 할 문장

**LLaMA는 공개 데이터만으로도, 작은 모델을 충분히 많은 token으로 잘 학습시키면 훨씬 큰 모델과 경쟁할 수 있음을 보여준 foundation model 논문입니다.**

더 짧게 말하면:

> **LLaMA = 작지만 많이 학습된 효율적 오픈 계열 base LLM**

---

## 오늘 공부용 요약

**논문명:** LLaMA: Open and Efficient Foundation Language Models  
**저자:** Hugo Touvron et al.  
**핵심 결론:** LLaMA는 7B~65B 규모의 foundation language model 시리즈로, 공개적으로 접근 가능한 데이터만 사용해 1T~1.4T token 수준으로 학습되었습니다. LLaMA-13B는 대부분의 benchmark에서 GPT-3 175B를 능가했고, LLaMA-65B는 Chinchilla-70B 및 PaLM-540B와 경쟁력 있는 성능을 보였습니다.

**왜 중요함:** LLaMA는 “LLM 성능 = parameter 수”라는 단순한 생각을 깨고, 작은 모델을 충분히 많은 고품질 token으로 학습하면 inference 효율이 뛰어난 강력한 모델을 만들 수 있음을 보여줬습니다.

**입문자가 배울 점:** 모델 크기, token 수, 데이터 품질, inference 비용을 함께 봐야 합니다.

**가장 중요한 문장:** LLaMA는 모델을 무조건 크게 키우는 대신, 공개 데이터와 충분한 학습 token으로 작은 모델의 성능을 극대화한 논문입니다.

# PaLM: Scaling Language Modeling with Pathways

## 결론부터

이 논문의 결론은 이겁니다.

**언어 모델을 아주 크게 만들고, 다양한 고품질 데이터로 학습하면, few-shot 상황에서도 자연어 이해, 추론, 코드 생성, 다국어 처리에서 큰 성능 향상을 얻을 수 있다.**

PaLM의 핵심은 단순히 “큰 모델”이 아니라, **540B parameter dense Transformer를 Pathways 시스템으로 수천 개 TPU v4 chip에 걸쳐 효율적으로 학습했다**는 점입니다.

한 줄로 말하면:

> **PaLM = Pathways 시스템으로 학습한 540B 규모의 dense decoder-only Transformer LLM**

---

## 이 논문을 한 문장으로 요약하면

**PaLM은 GPT-3 이후 “모델 규모를 더 키우면 few-shot 능력, reasoning, code, multilingual 성능이 어디까지 좋아지는가?”를 대규모 시스템과 함께 실험한 논문입니다.**

```text
GPT-3: 175B parameters
PaLM: 540B parameters
```

---

## 왜 PaLM이 중요했나?

PaLM은 단순히 큰 모델 기록 이상의 의미가 있습니다.

```text
1. 540B dense Transformer를 실제로 학습한 대규모 시스템 성과
2. few-shot 성능이 계속 scale되며 좋아진다는 증거
3. Chain-of-Thought와 결합했을 때 reasoning 성능의 큰 개선
4. 자연어뿐 아니라 code, multilingual task에서도 강한 성능
```

---

## 핵심 개념 1: Pathways

Pathways는 매우 큰 모델을 수천 개 TPU chip에 걸쳐 효율적으로 학습하기 위한 Google의 분산 ML 시스템입니다.

```text
모델 크기: 540B parameters
학습 chip: 6144 TPU v4 chips
시스템: Pathways
```

PaLM의 이름은 Pathways Language Model입니다.

---

## 핵심 개념 2: Dense Decoder-only Transformer

PaLM은 GPT 계열처럼 decoder-only autoregressive Transformer입니다.

학습 목표:

```text
앞의 token들을 보고 다음 token을 예측한다.
```

```text
BERT: encoder-only
T5: encoder-decoder
GPT / PaLM / LLaMA: decoder-only
```

---

## 핵심 개념 3: Dense Model

PaLM은 sparse MoE가 아니라 dense model입니다.

```text
모든 token이 같은 전체 모델을 통과
```

반대로 MoE는 token마다 일부 expert만 활성화합니다.

```text
PaLM = dense scaling
Mixtral / DeepSeek-V3 = sparse MoE scaling
```

---

## 핵심 개념 4: 모델 크기

PaLM 논문은 세 가지 크기의 모델을 비교했습니다.

| 모델 | Parameter |
|---|---:|
| PaLM 8B | 8.63B |
| PaLM 62B | 62.50B |
| PaLM 540B | 540.35B |

이 비교를 통해 scale이 커질수록 성능이 어떻게 좋아지는지 봤습니다.

---

## 핵심 개념 5: 780B token 학습 데이터

PaLM은 780B token의 고품질·다양한 텍스트 corpus로 학습되었습니다.

```text
filtered webpages
books
Wikipedia
news articles
source code
social media conversations
```

코드 데이터가 포함되어 코드 생성 성능에도 영향을 줬습니다.

---

## 핵심 개념 6: Architecture 개선

PaLM은 기본 Transformer에 다음을 사용했습니다.

```text
SwiGLU activation
Parallel layers
Multi-query attention
RoPE embeddings
Shared input-output embeddings
No biases
```

### Multi-query attention

Key/value를 head들 사이에서 공유해 autoregressive decoding 비용과 KV cache를 줄입니다.

```text
Query: head마다 다름
Key/Value: head들이 공유
```

---

## 핵심 개념 7: Scaling으로 성능이 좋아졌다

8B → 62B → 540B로 커질수록 많은 task에서 성능이 좋아졌습니다.

일부 task에서는 갑작스러운 성능 jump도 나타났고, 이는 나중에 emergent abilities 논의와 연결됩니다.

---

## 핵심 개념 8: Chain-of-Thought와 PaLM

PaLM의 reasoning 성능은 CoT prompting과 결합했을 때 특히 두드러졌습니다.

```text
standard prompting: 바로 답
CoT prompting: 중간 풀이 후 답
```

PaLM의 reasoning breakthrough는 **scale + CoT**의 결합 결과입니다.

---

## 핵심 개념 9: Code 능력

PaLM은 code-specific model은 아니었지만, training data에 source code가 포함되어 code generation과 code understanding에서도 강한 성능을 보였습니다.

```text
HumanEval
MBPP
TransCoder
DeepFix
GSM8K-Python
```

---

## 핵심 개념 10: Data Contamination 분석

대규모 웹 데이터로 학습하면 benchmark test set이 training data에 섞일 수 있습니다.

```text
모델이 문제를 푼 것이 아니라
학습 중 정답을 본 것일 수 있음
```

PaLM 논문은 n-gram overlap 기반 contamination 분석을 수행했습니다.

---

## PaLM과 Chinchilla 관점

PaLM은:

```text
540B parameters
780B tokens
```

Chinchilla 관점에서는 parameter 수에 비해 token이 적은, 즉 undertrained로 해석될 수 있습니다.

따라서 현재 관점에서는 PaLM을 이렇게 읽는 것이 좋습니다.

```text
PaLM = dense scale과 Pathways 시스템의 힘을 보여준 사례
Chinchilla = compute-optimal token/parameter 균형을 보완하는 관점
```

---

## 한계

1. 540B dense model 학습과 서빙은 비용이 큽니다.
2. Scale만으로 truthfulness, safety, calibration이 해결되지 않습니다.
3. Benchmark 성능이 실제 사용 성능을 보장하지 않습니다.
4. Dense scaling은 inference 비용이 큽니다.
5. 데이터 투명성과 재현성 한계가 있습니다.

---

## 입문자가 꼭 기억해야 할 문장

**PaLM은 Pathways 시스템을 사용해 540B dense decoder-only Transformer를 780B token으로 학습하고, scale이 few-shot 자연어 이해, reasoning, code, multilingual 능력을 크게 끌어올릴 수 있음을 보여준 대형 LLM 논문입니다.**

더 짧게 말하면:

> **PaLM = Pathways로 학습한 540B 대형 dense LLM, scale + CoT의 힘을 보여준 모델**

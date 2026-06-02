# The Llama 3 Herd of Models

## 결론부터

이 논문의 결론은 이겁니다.

**Llama 3는 단일 모델 하나가 아니라, 8B·70B·405B 크기의 여러 모델을 pre-training, post-training, tool use, safety, long-context, multilingual capability까지 포함해 하나의 “herd”로 설계한 현대 open-weight LLM 패밀리입니다.**

한 줄로 말하면:

> **Llama 3 Herd = 8B·70B·405B dense Transformer 모델군 + 128K context + multilingual/code/reasoning/tool-use + post-training/safety 체계**

---

## 이 논문을 한 문장으로 요약하면

**Llama 3 Herd 논문은 “좋은 LLM은 base model 하나로 끝나는 것이 아니라, 데이터·스케일·post-training·tool use·safety·long context·multimodal 확장을 함께 설계해야 한다”는 것을 보여주는 기술보고서입니다.**

---

## 왜 “Herd”인가?

Llama 3는 하나의 모델이 아니라 여러 크기와 용도의 모델이 함께 구성된 패밀리입니다.

```text
Llama 3.1 8B
Llama 3.1 70B
Llama 3.1 405B
```

각각 base model과 instruct model이 있습니다.

```text
pre-trained model
post-trained / instruct model
```

용도도 다릅니다.

```text
8B  → 로컬/저비용/빠른 추론
70B → 성능과 비용의 균형
405B → flagship 고성능 모델, synthetic data teacher, distillation, 평가 기준
```

---

## 핵심 개념 1: Data, Scale, Managing Complexity

논문은 고품질 foundation model 개발에서 세 가지를 강조합니다.

```text
1. Data
2. Scale
3. Managing Complexity
```

### Data

Llama 2는 약 1.8T token, Llama 3는 약 15T multilingual token으로 학습되었습니다.

### Scale

가장 큰 모델은 405B parameter dense Transformer입니다.

### Managing Complexity

Llama 3는 MoE 대신 표준 dense Transformer를 선택했습니다. 이유는 scale-up과 training stability를 관리하기 위해서입니다.

---

## 핵심 개념 2: Dense Transformer Architecture

Llama 3는 구조적으로 크게 새로운 모델이라기보다, 검증된 dense Transformer를 잘 스케일한 모델입니다.

주요 요소:

```text
decoder-only dense Transformer
SwiGLU
RoPE
GQA
128K tokenizer
```

---

## 핵심 개념 3: GQA

Llama 3는 Grouped Query Attention을 사용합니다.

GQA는 query head는 많게 유지하면서 key/value head를 줄입니다.

```text
Query heads 많음
Key/Value heads 적음
```

효과:

```text
KV cache 크기 감소
decoding 속도 개선
```

---

## 핵심 개념 4: 128K Token Vocabulary

Llama 3는 128K token vocabulary를 사용합니다.

```text
100K tokens from tiktoken
+ 28K additional tokens for non-English support
= 128K vocabulary
```

Tokenizer가 더 효율적이면 같은 context length와 compute 안에 더 많은 실제 문자 정보를 담을 수 있습니다.

---

## 핵심 개념 5: Pre-training Data Mix

Llama 3의 데이터 mix는 대략 다음과 같습니다.

```text
General knowledge: 약 50%
Mathematical and reasoning tokens: 약 25%
Code tokens: 약 17%
Multilingual tokens: 약 8%
```

즉, reasoning, math, code 데이터를 의도적으로 많이 넣었습니다.

---

## 핵심 개념 6: Long Context Pre-training

Llama 3는 처음부터 128K context로 학습하지 않고, 마지막 단계에서 점진적으로 context length를 늘렸습니다.

```text
8K → ... → 128K
```

긴 context 적응 여부는 다음으로 확인했습니다.

```text
short-context performance 회복
needle-in-a-haystack 성공
```

---

## 핵심 개념 7: Training Infrastructure

Llama 3 405B는 최대 16K H100 GPU에서 학습되었습니다.

이건 모델 논문이기도 하지만 시스템 논문이기도 합니다.

```text
scheduler
network
checkpointing
storage
GPU cluster
parallelism
```

이 모두 필요합니다.

---

## 핵심 개념 8: Post-training

Llama 3의 assistant 능력은 pretraining만으로 나온 것이 아닙니다.

post-training 흐름은 대략 다음과 같습니다.

```text
Human / synthetic preference data 수집
Reward model 학습
Rejection sampling
SFT
DPO
반복
```

DPO는 우리가 앞에서 본 preference tuning 방법입니다.

---

## 핵심 개념 9: Tool Use와 Function Calling

Llama 3는 tool use와 multi-turn function calling을 강화했습니다.

```text
function definitions
user query
corresponding function call
```

같은 synthetic function-calling data를 사용해 unseen tools에 대한 tool use 능력을 개선했습니다.

---

## 핵심 개념 10: Factuality

Llama 3 post-training에서는 모델이 모르는 질문에 과신하지 않고, 아는 것과 모르는 것을 구분하도록 factuality 데이터를 만들었습니다.

```text
아는 질문 → 답변
모르는 질문 → 거절 또는 불확실성 표현
```

TruthfulQA에서 본 문제와 직접 연결됩니다.

---

## 핵심 개념 11: Safety와 Llama Guard 3

Llama 3는 모델 자체의 안전성뿐 아니라, 입력과 출력 safety classification을 위한 Llama Guard 3도 함께 공개했습니다.

```text
Model-level mitigation
+
System-level safety classifier
```

---

## 핵심 개념 12: Multilingual Support

논문 기준 공식 지원 언어는 다음입니다.

```text
English
German
French
Italian
Portuguese
Hindi
Spanish
Thai
```

한국어는 공식 지원 언어에 포함되지 않으므로, 한국어 실무에는 별도 평가가 필요합니다.

---

## LLaMA v1과 Llama 3 비교

| 구분 | LLaMA v1 | Llama 3 Herd |
|---|---|---|
| 대표 크기 | 7B, 13B, 33B, 65B | 8B, 70B, 405B |
| 학습 데이터 | 1T~1.4T token | 약 15T token |
| context | 짧은 context 중심 | 128K context |
| post-training | base model 중심 | SFT, DPO, tool use, safety |
| safety | 제한적 | Llama Guard 3 포함 |

---

## 한계

1. 405B dense model은 운영 비용이 큽니다.
2. 한국어는 공식 지원 8개 언어에 없습니다.
3. Human evaluation은 주관성이 있습니다.
4. Prompt injection, code interpreter abuse, cybersecurity 위험이 완전히 해결된 것은 아닙니다.
5. 논문 시점의 multimodal 모델은 broad release 대상이 아닙니다.

---

## 입문자가 꼭 기억해야 할 문장

**Llama 3 Herd는 8B·70B·405B dense Transformer 모델군을 15T+ token 규모로 학습하고, 128K context, multilinguality, coding, reasoning, tool use, safety, post-training, multimodal 확장 실험까지 포함해 현대 open-weight LLM 패밀리를 종합적으로 구축한 기술보고서입니다.**

더 짧게 말하면:

> **Llama 3 Herd = base model이 아니라, 현대 LLM 생태계 전체를 설계한 모델 패밀리**

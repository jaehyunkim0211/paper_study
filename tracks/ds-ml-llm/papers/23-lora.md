# LoRA: Low-Rank Adaptation of Large Language Models

## 결론부터

이 논문의 결론은 이겁니다.

**거대한 언어 모델을 특정 task나 도메인에 맞게 조정할 때, 모델 전체 weight를 다 fine-tuning하지 않아도 된다. 기존 모델 weight는 얼려두고, 아주 작은 저랭크 low-rank 업데이트 행렬만 학습해도 충분히 좋은 성능을 낼 수 있다.**

이 방법이 바로 **LoRA, Low-Rank Adaptation**입니다.

한 줄로 말하면:

> **LoRA = 원래 LLM은 그대로 두고, 작은 보정 행렬만 학습해서 모델을 적응시키는 방법**

LoRA 논문은 pre-trained model의 weight를 freeze하고, Transformer layer 안에 학습 가능한 rank decomposition matrix를 주입해 downstream task에 필요한 trainable parameter 수를 크게 줄이는 방법을 제안했습니다.

---

## 이 논문을 한 문장으로 요약하면

**LoRA는 “모델 전체를 바꾸지 말고, 모델이 바뀌어야 하는 양 ΔW만 작게 분해해서 학습하자”는 논문입니다.**

일반 fine-tuning은 이렇게 합니다.

```text
기존 weight W 전체를 업데이트
W → W'
```

LoRA는 다르게 생각합니다.

```text
W' = W + ΔW
```

여기서 기존 `W`는 고정합니다.

```text
W는 freeze
ΔW만 학습
```

그리고 `ΔW`를 큰 행렬 하나로 직접 학습하지 않고, 작은 행렬 두 개의 곱으로 표현합니다.

```text
ΔW = B A
```

즉:

```text
W' = W + BA
```

이게 LoRA의 핵심입니다.

---

## 왜 LoRA가 필요했나?

GPT-3, BERT, T5, LLaMA 같은 모델은 사전학습된 거대한 모델입니다. 이 모델을 특정 task에 맞게 쓰려면 보통 fine-tuning을 생각합니다.

예를 들어:

```text
법률 QA용 모델
의료 문서 요약 모델
SQL 생성 모델
고객센터 챗봇
회사 내부 문서 응답 모델
```

각 task마다 full fine-tuning을 하면 어떻게 될까요?

모델이 7B라면 task마다 7B parameter짜리 checkpoint가 생깁니다.

모델이 70B라면 task마다 70B checkpoint가 생깁니다.

GPT-3 175B처럼 크면 더 심각합니다.

```text
task 1용 모델: 175B parameter
task 2용 모델: 175B parameter
task 3용 모델: 175B parameter
```

저장 비용, 학습 비용, 배포 비용이 너무 큽니다.

LoRA는 이 문제를 줄이기 위해 전체 `W`를 업데이트하지 않고, 작은 보정 행렬만 학습합니다.

---

## 핵심 개념 1: Full fine-tuning

### 결론부터

**Full fine-tuning은 사전학습된 모델의 모든 weight를 task 데이터로 다시 업데이트하는 방식입니다.**

예를 들어 모델 weight가 있습니다.

```text
W1, W2, W3, ..., Wn
```

full fine-tuning은 이 모든 weight를 학습합니다.

```text
W1 업데이트
W2 업데이트
W3 업데이트
...
Wn 업데이트
```

장점은 명확합니다.

```text
모델 전체가 task에 맞게 바뀔 수 있음
성능이 좋을 가능성이 큼
```

하지만 단점도 큽니다.

```text
학습 메모리 많이 필요
optimizer state 많이 필요
task마다 큰 checkpoint 필요
모델이 클수록 비용 폭발
catastrophic forgetting 가능
```

LoRA는 전체 `W`를 업데이트하지 않고, 작은 보정 행렬만 학습합니다.

---

## 핵심 개념 2: Low-rank란 무엇인가?

### 결론부터

**Low-rank는 큰 행렬을 작은 두 행렬의 곱으로 근사하는 아이디어입니다.**

예를 들어 어떤 weight matrix가 있다고 합시다.

```text
W.shape = (1000, 1000)
```

이 행렬 전체를 학습하면 parameter 수는:

```text
1000 × 1000 = 1,000,000개
```

입니다.

LoRA는 업데이트 행렬 `ΔW`를 직접 1000×1000으로 학습하지 않습니다.

대신 작은 두 행렬로 표현합니다.

```text
A.shape = (r, 1000)
B.shape = (1000, r)
ΔW = B A
```

만약 `r = 8`이면 parameter 수는:

```text
A: 8 × 1000 = 8,000
B: 1000 × 8 = 8,000
총 16,000개
```

원래 업데이트 행렬은 1,000,000개였는데, LoRA는 16,000개만 학습합니다.

```text
1,000,000 → 16,000
```

훨씬 작습니다.

여기서 `r`을 **rank**라고 부릅니다.

`r`이 작을수록 더 적은 parameter만 학습합니다.

```text
r 작음 → 가볍지만 표현력 제한
r 큼 → 무겁지만 표현력 증가
```

---

## 쉬운 비유: 전체 책을 다시 쓰지 말고 수정 포스트잇만 붙이기

거대한 사전학습 모델을 두꺼운 교과서라고 생각해봅시다.

Full fine-tuning은 task마다 교과서 전체를 다시 쓰는 방식입니다.

```text
법률용 교과서 전체 새로 작성
의료용 교과서 전체 새로 작성
SQL용 교과서 전체 새로 작성
```

너무 비쌉니다.

LoRA는 이렇게 합니다.

```text
기존 교과서는 그대로 둔다.
task별로 필요한 수정 포스트잇만 붙인다.
```

법률 task에는 법률용 포스트잇.

의료 task에는 의료용 포스트잇.

SQL task에는 SQL용 포스트잇.

기존 책은 하나만 보관하고, task별 포스트잇만 바꾸면 됩니다.

이게 LoRA의 실무적 직관입니다.

```text
base model + small LoRA adapter
```

---

## 핵심 개념 3: `W + BA`

LoRA의 수식은 꼭 이해하는 게 좋습니다.

일반 linear layer는 이렇게 계산합니다.

```text
h = W x
```

LoRA는 이렇게 바꿉니다.

```text
h = W x + B A x
```

여기서:

| 기호 | 의미 |
|---|---|
| `W` | 기존 pre-trained weight, freeze |
| `A` | 작은 trainable matrix |
| `B` | 작은 trainable matrix |
| `BA` | low-rank update `ΔW` |
| `r` | rank, 보정 행렬의 내부 차원 |

즉, 원래 출력에 작은 보정값을 더합니다.

```text
기존 모델 출력 + task-specific 보정
```

ResNet을 배울 때 봤던 `x + F(x)`와 느낌이 조금 비슷합니다.

```text
ResNet: 기존 표현에 residual correction을 더함
LoRA: 기존 weight에 low-rank correction을 더함
```

다만 ResNet은 architecture의 forward path이고, LoRA는 fine-tuning을 위한 parameter-efficient adaptation 방법입니다.

---

## 핵심 개념 4: 왜 `A`, `B` 두 개로 나누는가?

`ΔW`를 직접 학습하면 너무 큽니다.

예를 들어:

```text
ΔW.shape = (4096, 4096)
```

이면 parameter 수는:

```text
4096 × 4096 ≈ 16.8M
```

입니다.

하지만 rank `r=8`로 나누면:

```text
A.shape = (8, 4096)
B.shape = (4096, 8)
```

parameter 수는:

```text
8×4096 + 4096×8 = 65,536
```

약 6만 5천 개입니다.

비교하면:

```text
16.8M vs 65K
```

약 256배 차이입니다.

이걸 Transformer의 여러 layer에 적용하면 trainable parameter 절감 효과가 매우 커집니다.

---

## 핵심 개념 5: LoRA는 기존 weight를 freeze한다

이게 매우 중요합니다.

LoRA 학습 중에는 기존 model weight `W`에 gradient를 계산하지 않습니다.

```text
W: frozen
A, B: trainable
```

즉, 학습되는 것은 `A`, `B`뿐입니다.

이게 메모리를 크게 줄입니다.

왜냐하면 full fine-tuning에서는 모든 weight에 대해 gradient와 optimizer state가 필요하지만, LoRA에서는 대부분의 parameter가 frozen이기 때문입니다.

---

## 핵심 개념 6: LoRA는 어디에 붙이나?

Transformer에는 여러 weight matrix가 있습니다.

Self-attention 안에는 대표적으로 다음이 있습니다.

```text
Wq: Query projection
Wk: Key projection
Wv: Value projection
Wo: Output projection
```

MLP, 즉 feed-forward network 안에도 weight matrix가 있습니다.

LoRA는 원칙적으로 dense layer 어디에든 적용할 수 있습니다.

하지만 논문에서는 주로 Transformer attention weight에 적용했습니다.

입문적으로는 이렇게 기억하면 됩니다.

```text
LoRA는 보통 attention projection matrix에 작은 보정 행렬을 붙인다.
특히 q_proj, v_proj에 자주 붙인다.
```

현대 실무에서는 모델과 task에 따라 다음에도 붙입니다.

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

---

## 핵심 개념 7: LoRA rank `r`

### 결론부터

**rank `r`은 LoRA가 task-specific 보정을 얼마나 넓은 공간에서 학습할 수 있는지 정하는 값입니다.**

`r`이 작으면:

```text
학습 parameter 적음
메모리 적음
저장 용량 작음
표현력 제한 가능
```

`r`이 크면:

```text
학습 parameter 증가
표현력 증가
메모리 증가
```

예를 들어:

```text
r = 4
r = 8
r = 16
r = 64
```

처럼 정합니다.

실무적으로는 보통 이렇게 시작합니다.

```text
r = 8 또는 16
```

더 어려운 task나 더 많은 변화를 원하면:

```text
r = 32, 64
```

까지 올릴 수 있습니다.

---

## 핵심 개념 8: LoRA alpha와 scaling

LoRA는 보통 이렇게 scaling을 씁니다.

```text
h = W x + (α / r) B A x
```

여기서 `α`는 LoRA scaling factor입니다.

`r`이 바뀌어도 보정값의 크기가 너무 달라지지 않게 조절하는 역할을 합니다.

초기화도 중요합니다.

시작할 때:

```text
B = 0
→ BA = 0
→ 모델 출력은 원래 pretrained model과 같음
```

즉, LoRA는 처음에는 base model을 망치지 않고 시작합니다.

학습이 진행되면서 `A`, `B`가 task에 맞는 보정을 배우게 됩니다.

---

## LoRA와 Adapter의 차이

LoRA 이전에도 parameter-efficient tuning 방법이 있었습니다. 대표적으로 adapter가 있습니다.

Adapter는 보통 Transformer layer 사이에 작은 MLP module을 끼워 넣습니다.

```text
Transformer layer
→ Adapter
→ 다음 layer
```

문제는 inference 때 adapter layer를 추가로 통과해야 한다는 점입니다.

```text
추가 layer
→ 추가 latency
```

LoRA는 다릅니다.

LoRA는 학습 후 `BA`를 기존 `W`에 합칠 수 있습니다.

```text
W' = W + BA
```

그러면 inference 때는 그냥 일반 linear layer처럼 계산하면 됩니다.

```text
h = W' x
```

추가 layer를 통과하지 않습니다.

| 방법 | 작동 방식 | inference latency |
|---|---|---|
| Adapter | 새 module을 layer 사이에 추가 | 늘 수 있음 |
| LoRA | 기존 weight에 low-rank update를 더함 | merge하면 추가 없음 |

---

## LoRA와 Prompt Tuning / Prefix Tuning의 차이

또 다른 parameter-efficient 방법은 prompt tuning 또는 prefix tuning입니다.

이 방법들은 모델 입력 앞에 학습 가능한 token이나 prefix를 붙입니다.

```text
[learned prefix] + 사용자 입력
```

장점은 base model weight를 건드리지 않는다는 것입니다.

하지만 단점도 있습니다.

```text
입력 context 길이를 일부 차지함
optimization이 어려울 수 있음
성능이 task마다 불안정할 수 있음
```

비교하면:

| 방법 | 어디를 바꾸나? | 장점 | 단점 |
|---|---|---|---|
| Prompt/Prefix tuning | 입력 앞의 learned token | 매우 가벼움 | context 길이 차지, 최적화 어려움 |
| Adapter | layer 사이 작은 module | 안정적 | inference latency 가능 |
| LoRA | weight update를 low-rank로 학습 | 가볍고 merge 가능 | 적용 위치/rank 선택 필요 |

---

## LoRA가 실무에서 특히 좋은 이유

## 1. 저장 용량이 작다

base model은 하나만 저장합니다.

```text
base model: 7B
```

task별로는 작은 LoRA adapter만 저장합니다.

```text
법률 LoRA
의료 LoRA
SQL LoRA
고객센터 LoRA
```

## 2. 학습 메모리가 줄어든다

Full fine-tuning은 모든 parameter에 대해 gradient와 optimizer state를 저장해야 합니다.

LoRA는 대부분의 weight가 frozen입니다.

```text
gradient 저장 안 함
optimizer state 없음
```

## 3. task switching이 쉽다

같은 base model에 다른 LoRA adapter를 갈아 끼울 수 있습니다.

```text
base model + 법률 LoRA
base model + 의료 LoRA
base model + SQL LoRA
```

## 4. inference latency를 없앨 수 있다

LoRA는 학습된 `BA`를 `W`에 merge할 수 있습니다.

```text
W' = W + BA
```

이렇게 하면 inference 때 추가 계산이 없습니다.

---

## LoRA는 무엇을 배우는가?

LoRA는 모델 전체 지식을 새로 배우는 것이 아닙니다.

기존 base model이 이미 많은 언어 능력과 지식을 가지고 있습니다.

LoRA는 그 위에 task-specific한 보정을 배웁니다.

예를 들어 SQL 생성 task라면:

```text
자연어 질문 → SQL 형식으로 답하기
```

요약 task라면:

```text
긴 글 → 짧은 요약
```

의료 도메인이라면:

```text
의료 용어와 형식에 더 잘 맞게 답하기
```

즉, LoRA는 base model의 능력을 task나 도메인에 맞게 방향 조정하는 역할에 가깝습니다.

---

## LoRA와 Full fine-tuning을 비교하면

| 구분 | Full fine-tuning | LoRA |
|---|---|---|
| 학습 parameter | 전체 모델 | 작은 low-rank matrix |
| base model weight | 업데이트됨 | freeze |
| 학습 메모리 | 큼 | 작음 |
| task별 checkpoint | 모델 전체 | adapter만 |
| catastrophic forgetting | 가능성 큼 | 상대적으로 작음 |
| inference latency | 기본 | merge하면 추가 없음 |
| 성능 | 강력 | 많은 task에서 경쟁력 있음 |
| 한계 | 비용 큼 | task가 너무 다르면 rank 부족 가능 |

LoRA의 핵심은 성능과 효율 사이의 균형입니다.

```text
full fine-tuning급 성능에 가까운 결과를
훨씬 적은 trainable parameter로 얻자.
```

---

## LoRA와 RAG의 차이

RAG와 LoRA도 자주 비교됩니다.

둘은 역할이 다릅니다.

| 구분 | RAG | LoRA |
|---|---|---|
| 핵심 목적 | 외부 지식을 가져오기 | 모델 행동/표현을 task에 맞게 조정 |
| weight 업데이트 | 보통 없음 | adapter parameter 학습 |
| 지식 업데이트 | 문서 index 교체 | 학습 필요 |
| 좋은 경우 | 최신 문서, 내부 문서, 근거 필요 | 출력 형식, task skill, 도메인 스타일 |
| 한계 | 검색 실패 시 취약 | 새로운 factual knowledge 주입에는 비효율적일 수 있음 |

예를 들어 회사 규정 QA라면:

```text
RAG: 회사 규정 문서를 검색해 넣음
LoRA: 회사 답변 스타일과 형식을 학습
```

둘을 같이 쓰는 경우도 많습니다.

```text
base model + LoRA instruction tuning + RAG 문서 검색
```

---

## LoRA의 한계

## 1. 모든 task에 작은 rank가 충분하지는 않다

LoRA의 가정은 task adaptation update가 low-rank일 수 있다는 것입니다.

하지만 어떤 task는 base model과 너무 다를 수 있습니다.

```text
완전히 다른 언어
완전히 다른 modality
복잡한 새로운 reasoning 형식
매우 다른 output distribution
```

이런 경우 작은 rank LoRA만으로는 부족할 수 있습니다.

## 2. 어떤 layer에 붙일지 선택해야 한다

LoRA를 어디에 붙일지 정해야 합니다.

```text
q_proj만?
v_proj만?
q와 v?
q,k,v,o 모두?
MLP까지?
```

적용 위치에 따라 성능과 parameter 수가 달라집니다.

## 3. 여러 LoRA를 동시에 batch 처리하기 어려울 수 있다

같은 batch 안에서 샘플마다 다른 LoRA adapter를 써야 하면 복잡해질 수 있습니다.

## 4. LoRA는 데이터 품질 문제를 해결하지 않는다

데이터가 나쁘면 LoRA도 나쁜 방향으로 학습됩니다.

---

## QLoRA와의 연결

다음에 우리가 볼 QLoRA는 LoRA를 한 단계 더 실용적으로 만든 논문입니다.

LoRA는 base model을 freeze하지만, base model 자체는 GPU memory에 올라가야 합니다.

예를 들어 65B 모델은 frozen이어도 메모리를 많이 먹습니다.

QLoRA는 이렇게 합니다.

```text
base model을 4-bit로 quantize해서 메모리를 줄임
LoRA adapter만 학습
```

즉:

```text
LoRA = 적은 parameter만 학습
QLoRA = base model까지 4-bit로 줄여 더 적은 GPU로 fine-tuning
```

---

## 입문자가 꼭 기억해야 할 문장

**LoRA는 pretrained model weight를 freeze하고, 각 weight matrix의 변화량 `ΔW`를 작은 low-rank matrix 두 개 `B A`로 표현해 그 작은 matrix만 학습하는 parameter-efficient fine-tuning 방법입니다.**

더 짧게 말하면:

> **LoRA = 큰 모델은 그대로 두고, 작은 보정 행렬만 학습하는 fine-tuning**

---

## 오늘 공부용 요약

**논문명:** LoRA: Low-Rank Adaptation of Large Language Models  
**저자:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen  
**핵심 결론:** LLM을 downstream task에 맞게 조정할 때 전체 parameter를 fine-tuning하지 않고, pretrained weight는 freeze한 채 low-rank update matrix만 학습해도 full fine-tuning과 비슷한 성능을 낼 수 있다.

**왜 중요함:** LoRA는 대형 모델 fine-tuning의 메모리·저장·배포 비용을 크게 줄였습니다.

**입문자가 배울 점:** LoRA는 모델에 새로운 거대한 능력을 처음부터 학습시키는 방법이 아니라, 이미 강력한 base model을 특정 task나 도메인에 맞게 효율적으로 보정하는 방법입니다.

**가장 중요한 문장:** LoRA는 `W` 전체를 바꾸지 않고, `W + BA`에서 작은 `A`, `B`만 학습해 거대한 모델의 task adaptation을 가볍게 만든다.

# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## 결론부터

이 논문의 결론은 이겁니다.

**Transformer의 attention은 강력하지만 sequence length가 길어질수록 계산과 메모리 비용이 크게 증가한다. Mamba는 attention 없이도 긴 sequence를 효율적으로 처리하기 위해, 입력 token에 따라 정보를 선택적으로 기억하거나 잊는 Selective State Space Model을 제안한다.**

한 줄로 말하면:

> **Mamba = attention 없이, 필요한 정보만 선택적으로 상태에 저장하면서 긴 sequence를 선형 시간으로 처리하는 모델**

---

## 이 논문을 한 문장으로 요약하면

**Mamba는 Transformer의 attention이 “모든 과거 token을 KV cache로 보관하고 참조하는 방식”이라면, 자신은 “고정 크기의 hidden state에 필요한 정보만 선택적으로 압축해 저장하는 방식”으로 긴 sequence를 처리하겠다는 논문입니다.**

---

## 왜 필요한가?

Transformer attention은 강력하지만 긴 sequence에서 비용이 큽니다.

```text
attention score matrix: N × N
KV cache: 이전 token들의 key/value 계속 저장
```

Mamba는 recurrent state를 사용합니다.

```text
state_0
→ token_1을 보고 state_1
→ token_2를 보고 state_2
→ token_3을 보고 state_3
...
```

이 방식은 sequence length에 대해 선형입니다.

```text
N token 처리 → O(N)
```

---

## 핵심 개념 1: State Space Model, SSM

SSM은 sequence를 한 token씩 읽으면서 내부 state를 업데이트하고, 그 state로 출력을 만드는 모델입니다.

```text
state_t = 이전 state와 현재 input으로 업데이트
output_t = state_t로부터 계산
```

쉽게 말하면:

```text
현재까지 읽은 정보를 state에 압축해 저장한다.
```

---

## 핵심 개념 2: Attention과 SSM의 차이

| 방식 | 비유 |
|---|---|
| Transformer attention | 책 전체를 계속 펼쳐놓고 필요한 페이지를 직접 찾아봄 |
| SSM / Mamba | 책을 읽으면서 중요한 내용만 노트에 요약해 들고 감 |

Attention은 과거 token들을 직접 저장하고 참조합니다.

```text
KV cache:
token 1의 K,V
token 2의 K,V
...
```

SSM은 과거 정보를 state에 압축합니다.

```text
state 하나에 과거 정보를 압축
```

---

## 핵심 개념 3: 기존 SSM의 약점

기존 SSM은 효율을 위해 time-invariant dynamics를 사용했습니다.

```text
모든 시점에서 같은 방식으로 state를 업데이트한다.
```

문제는 text에서는 token마다 중요도가 다르다는 점입니다.

```text
중요한 이름
중요한 숫자
질문에 필요한 키워드
불필요한 조사
반복되는 filler word
```

모든 token을 같은 방식으로 처리하면 안 됩니다.

---

## 핵심 개념 4: Selection Mechanism

Mamba의 핵심은 현재 입력 token에 따라 무엇을 기억하고 무엇을 잊을지 선택하게 만드는 것입니다.

기존 SSM:

```text
모든 token에 같은 업데이트 규칙
```

Mamba:

```text
token마다 다른 업데이트 규칙
```

예를 들어:

```text
주문 번호는 8472입니다.
```

이 숫자는 나중에 필요할 수 있으므로 오래 기억해야 합니다. 반대로 filler token은 빨리 잊어도 됩니다.

---

## 쉬운 비유: 회의록 작성자

기존 SSM은 모든 말을 같은 방식으로 적습니다.

```text
중요한 결정도 조금 적고
잡담도 조금 적고
숫자도 조금 적음
```

Mamba는 더 똑똑한 회의록 작성자입니다.

```text
중요한 결정 → 크게 기록
날짜와 숫자 → 정확히 기록
잡담 → 대부분 무시
```

이것이 selective state update입니다.

---

## 핵심 개념 5: Selective Copying Task

Selective Copying은 긴 sequence에서 중요한 token만 골라 기억해야 하는 synthetic task입니다.

```text
입력: x x A x x x B x C x x
출력: A B C
```

위치가 아니라 내용을 보고 기억해야 합니다. Mamba의 selection mechanism은 이런 문제를 해결하도록 설계되었습니다.

---

## 핵심 개념 6: Selective Scan

Mamba는 input-dependent SSM을 효율적으로 계산하기 위해 **selective scan**이라는 hardware-aware parallel algorithm을 사용합니다.

기존 SSM은 parameter가 고정되어 convolution으로 빠르게 계산할 수 있었습니다.

```text
time-invariant → convolution 가능 → parallel training 가능
```

Mamba는 token마다 parameter가 바뀌므로 convolution path를 그대로 쓰기 어렵습니다. Selective scan은 이 recurrence를 GPU에서 효율적으로 계산하게 해줍니다.

---

## FlashAttention과 Selective Scan의 공통점

| 논문 | 문제 | 해결 철학 |
|---|---|---|
| FlashAttention | attention matrix를 HBM에 저장하면 느림 | block 단위 계산, IO 줄이기 |
| Mamba | selective SSM의 expanded state를 materialize하면 느림 | hardware-aware scan, IO 줄이기 |

둘 다 하드웨어 메모리 계층을 고려합니다.

---

## 핵심 개념 7: KV cache가 필요 없는 inference

Transformer decoding:

```text
token 1의 K,V 저장
token 2의 K,V 저장
...
현재 token은 이전 모든 K,V 참조
```

Mamba decoding:

```text
state 업데이트
state 업데이트
state 업데이트
...
현재 state만 유지
```

즉, 이전 모든 token의 KV cache를 저장할 필요가 없습니다.

---

## Transformer vs Mamba

| 구분 | Transformer | Mamba |
|---|---|---|
| 핵심 연산 | Self-attention | Selective SSM |
| 과거 정보 | KV cache 직접 참조 | state에 압축 |
| sequence scaling | attention은 O(N²) | linear O(N) |
| 긴 context | 비용 큼 | 유리 |
| 장점 | content-based routing 강함 | 효율적 long sequence modeling |
| 약점 | 긴 sequence 비용 | state 압축이 충분히 강해야 함 |

---

## 한계

1. attention처럼 모든 token을 직접 참조하지는 않습니다.
2. 논문 당시 7B 이상 대형 LLM scale 검증은 제한적입니다.
3. selective scan의 효율적 구현은 간단하지 않습니다.
4. Transformer 생태계가 매우 강합니다.
5. state에 어떤 정보가 남고 사라지는지 해석하기 어렵습니다.

---

## 입문자가 꼭 기억해야 할 문장

**Mamba는 Transformer attention처럼 과거 token을 모두 저장해 직접 참조하는 대신, input-dependent selective state update로 중요한 정보만 state에 저장하고 불필요한 정보는 잊으면서 sequence를 선형 시간으로 처리하는 attention-free 모델입니다.**

더 짧게 말하면:

> **Mamba = 필요한 정보만 선택적으로 기억하는 linear-time sequence model**

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## 결론부터

이 논문의 결론은 이겁니다.

**Transformer attention이 느리고 메모리를 많이 쓰는 큰 이유는 계산량만 많아서가 아니라, GPU 메모리 사이를 오가며 큰 중간 행렬을 읽고 쓰는 비용이 너무 크기 때문이다.**

FlashAttention은 attention 계산을 작은 block으로 쪼개고, 빠른 GPU 내부 메모리에서 계산하며, 거대한 attention matrix를 메모리에 저장하지 않고도 **정확한 attention**을 계산합니다.

한 줄로 말하면:

> **FlashAttention = attention 결과는 똑같이 계산하되, GPU 메모리 이동을 줄여 훨씬 빠르고 메모리 효율적으로 만드는 알고리즘**

---

## 이 논문을 한 문장으로 요약하면

**FlashAttention은 `QKᵀ`, `softmax`, `PV`를 따로따로 계산해 거대한 attention matrix를 저장하는 대신, 작은 block 단위로 attention을 계산해서 HBM 접근을 줄이고, exact attention을 더 빠르게 계산하는 방법입니다.**

표준 attention:

```text
S = QKᵀ
P = softmax(S)
O = PV
```

문제는 `S`와 `P`가 sequence length `N`에 대해 `N × N` 크기라는 점입니다.

---

## 왜 필요한가?

LLM의 context length가 길어질수록 attention matrix가 커집니다.

```text
N → 2N
N² → 4N²
```

긴 문서, 긴 대화, 긴 코드, RAG context를 처리하려면 attention 효율이 매우 중요합니다.

---

## 핵심 개념 1: 병목은 FLOPs만이 아니다

FlashAttention의 핵심 관점은 이것입니다.

```text
계산량 FLOPs만 줄이면 빨라진다
```

가 아니라,

```text
GPU 메모리 사이의 읽기·쓰기 비용을 줄여야 실제로 빨라진다
```

입니다.

GPU에는 대략 이런 메모리 계층이 있습니다.

```text
HBM  = 크지만 상대적으로 느린 메모리
SRAM = 작지만 매우 빠른 on-chip memory
```

비유하면:

```text
HBM  = 큰 창고
SRAM = 작업대 위 공간
```

FlashAttention은 큰 창고를 계속 왔다 갔다 하지 않고, 작업대에 필요한 block만 올려놓고 계산합니다.

---

## 핵심 개념 2: 표준 attention의 문제

표준 attention은 다음 큰 중간 행렬을 만듭니다.

```text
S = QKᵀ   # N × N
P = softmax(S)  # N × N
```

그리고 이 행렬을 HBM에 저장하고 다시 읽습니다.

```text
S를 HBM에 저장
S를 읽어 softmax
P를 HBM에 저장
P를 읽어 PV 계산
```

긴 sequence에서는 이 메모리 이동이 매우 비쌉니다.

---

## 핵심 개념 3: FlashAttention은 approximate attention이 아니다

FlashAttention은 sparse attention이나 linear attention처럼 attention을 근사하지 않습니다.

```text
계산하는 attention = 표준 softmax attention과 동일
바뀌는 것 = 계산 순서와 메모리 접근 방식
```

즉, **exact attention**입니다.

---

## 핵심 개념 4: Tiling

Tiling은 큰 행렬을 작은 block으로 나누는 방법입니다.

FlashAttention은 다음처럼 계산합니다.

```text
Q block 하나를 가져온다.
K/V block들을 차례로 가져온다.
block attention을 계산한다.
softmax 통계를 업데이트한다.
output을 누적한다.
```

큰 `N × N` attention matrix를 만들지 않습니다.

---

## 핵심 개념 5: Online Softmax

softmax는 전체 row를 봐야 합니다.

```text
softmax(s_i) = exp(s_i) / sum_j exp(s_j)
```

FlashAttention은 block을 보면서 다음 통계를 누적합니다.

```text
현재까지의 row max
현재까지의 normalization sum
현재까지의 output 누적값
```

새 block이 들어오면 max와 normalization을 다시 조정해서 전체 softmax와 같은 결과를 만듭니다.

---

## 핵심 개념 6: Recomputation

FlashAttention은 backward pass를 위해 거대한 attention matrix를 저장하지 않습니다.

대신:

```text
forward 때 큰 P를 저장하지 않음
output과 softmax normalization statistics만 저장
backward 때 필요한 P block을 다시 계산
```

FLOPs는 조금 늘 수 있지만 HBM 접근이 크게 줄어 전체 속도는 빨라질 수 있습니다.

---

## 핵심 개념 7: Kernel Fusion

표준 구현은 여러 kernel을 거칠 수 있습니다.

```text
matmul kernel
softmax kernel
dropout kernel
matmul kernel
```

FlashAttention은 attention의 여러 연산을 하나의 CUDA kernel 안에 묶어 중간 결과를 HBM에 저장하지 않게 합니다.

---

## 표준 attention vs FlashAttention

| 구분 | 표준 attention | FlashAttention |
|---|---|---|
| attention 결과 | exact | exact |
| `N × N` score matrix 저장 | 저장 | 저장 안 함 |
| `N × N` probability matrix 저장 | 저장 | 저장 안 함 |
| compute complexity | 대략 O(N²d) | 대략 O(N²d) |
| memory / HBM access | 큼 | 크게 감소 |

---

## 핵심 개념 8: IO Complexity

IO complexity는 계산량이 아니라, 서로 다른 메모리 계층 사이에서 데이터를 얼마나 읽고 쓰는지를 측정합니다.

FlashAttention의 큰 기여는 attention을 FLOPs 관점이 아니라 IO 관점에서 다시 설계했다는 점입니다.

---

## 한계

1. Attention의 quadratic compute 자체를 없애지는 않습니다.
2. CUDA kernel 수준 구현 난이도가 높습니다.
3. 모든 attention 변형에 바로 적용되는 것은 아닙니다.
4. 실제 성능은 하드웨어와 구현에 의존합니다.

---

## 입문자가 꼭 기억해야 할 문장

**FlashAttention은 `N × N` attention matrix를 HBM에 저장하지 않고, block 단위로 online softmax와 recomputation을 사용해 exact attention을 계산함으로써 메모리 사용과 HBM 접근을 크게 줄이는 알고리즘입니다.**

더 짧게 말하면:

> **FlashAttention = attention matrix를 저장하지 않고, block별로 정확히 계산하는 빠른 attention**

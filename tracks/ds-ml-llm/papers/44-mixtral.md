# Mixtral of Experts

## 결론부터

이 논문의 결론은 이겁니다.

**LLM을 꼭 dense model처럼 모든 token이 전체 파라미터를 다 쓰게 만들 필요는 없다. 여러 expert를 만들어두고, 각 token마다 필요한 expert 일부만 선택해서 계산하면, 전체 파라미터는 크게 늘리면서도 실제 계산량은 훨씬 작게 유지할 수 있다.**

이 구조가 **Sparse Mixture of Experts, SMoE**이고, 대표 모델이 **Mixtral 8x7B**입니다.

한 줄로 말하면:

> **Mixtral = 8개의 expert 중 token마다 2개만 골라 쓰는 sparse MoE LLM**

---

## 이 논문을 한 문장으로 요약하면

**Mixtral은 “모델 전체를 매번 다 쓰지 말고, token마다 필요한 expert만 골라 쓰자”는 MoE 구조를 open-weight LLM에서 강하게 보여준 논문입니다.**

---

## 왜 Mixtral이 필요했나?

Dense model을 크게 만들면 성능은 좋아질 수 있지만 비용도 증가합니다.

```text
parameter 수 증가
→ inference 계산량 증가
→ latency 증가
→ serving 비용 증가
→ GPU memory 요구량 증가
```

MoE는 이 문제를 다르게 풉니다.

```text
전체 capacity는 크게 만든다.
하지만 token마다 일부 expert만 계산한다.
```

---

## 핵심 개념 1: Mixture of Experts, MoE

MoE는 하나의 거대한 feed-forward block 대신 여러 expert block을 두고, 입력마다 일부 expert만 선택해 계산하는 구조입니다.

일반 Transformer:

```text
Self-Attention
→ Feed-Forward Network
```

Mixtral:

```text
Self-Attention
→ Router가 expert 선택
→ 선택된 expert 2개 계산
→ weighted sum
```

---

## 핵심 개념 2: Expert

Mixtral에서 expert는 전체 모델 하나가 아니라, Transformer layer 안의 feed-forward block입니다.

```text
layer 1: expert 1~8
layer 2: expert 1~8
...
```

---

## 핵심 개념 3: Router

Router는 각 token을 어느 expert에게 보낼지 결정하는 gating network입니다.

```text
expert 1: 0.05
expert 2: 0.12
expert 3: 0.40
expert 5: 0.31
```

Mixtral은 top-2 expert를 고릅니다.

```text
expert 3
expert 5
```

출력은 두 expert output의 가중합입니다.

---

## 핵심 개념 4: Top-2 Routing

Mixtral은 각 token마다 8개 expert 중 2개만 선택합니다.

```text
num_experts = 8
top_k_experts = 2
```

즉:

```text
8개 중 2개만 계산
```

---

## 쉬운 비유: 회사의 전문가 팀

Dense model은 모든 문제를 한 명의 거대한 전문가가 처리하는 것과 비슷합니다.

```text
모든 질문 → 한 명의 거대한 전문가
```

Mixtral은 회사에 여러 전문가가 있는 구조입니다.

```text
질문이 들어오면 router가 필요한 전문가 2명을 선택
```

전체 회사의 지식 용량은 크지만, 한 문제 처리 비용은 낮아집니다.

---

## 핵심 개념 5: Total Parameter vs Active Parameter

MoE 모델에서는 전체 parameter 수와 token 하나가 실제 사용하는 active parameter 수를 구분해야 합니다.

Mixtral은 대략:

```text
전체 접근 가능 parameter: 47B
active parameter per token: 13B
```

즉, 모델 전체 capacity는 크지만 token 하나를 처리할 때 계산되는 parameter는 훨씬 적습니다.

---

## 핵심 개념 6: MoE는 공짜 점심이 아니다

MoE는 계산 효율을 높이지만 새로운 시스템 문제를 만듭니다.

```text
특정 expert에 token이 몰릴 수 있음
GPU 간 통신 필요
routing overhead
메모리 접근 패턴 복잡
batch가 작으면 효율 저하
```

---

## 핵심 개념 7: Expert Parallelism

Expert를 여러 GPU에 나눠 배치합니다.

```text
GPU 1: expert 1, 2
GPU 2: expert 3, 4
GPU 3: expert 5, 6
GPU 4: expert 7, 8
```

token이 expert 3과 7로 route되면 해당 GPU로 hidden state를 보내야 합니다.

---

## 핵심 개념 8: 성능

Mixtral은 token당 active parameter가 훨씬 낮은데도 Llama 2 70B와 비슷하거나 더 좋은 성능을 보였습니다.

특히 강한 영역:

```text
수학
코딩
다국어
```

---

## 핵심 개념 9: Mixtral Instruct

Mixtral 8x7B Instruct는 base Mixtral을 다음 방식으로 instruction-following에 맞춘 모델입니다.

```text
Base Mixtral
→ SFT
→ DPO
→ Mixtral 8x7B Instruct
```

DPO는 우리가 앞에서 본 preference tuning 방법입니다.

---

## 핵심 개념 10: Routing Analysis

MoE의 expert라는 이름 때문에 “수학 expert”, “코딩 expert”처럼 나뉠 것 같지만, 논문 분석에서는 domain별 specialization이 뚜렷하지 않았습니다.

오히려 routing은 syntax나 token pattern과 더 관련 있어 보였습니다.

```text
Python의 self
Question token
code indentation token
```

이런 것들이 특정 expert로 route되는 경향이 있었습니다.

---

## PaLM과 Mixtral 비교

| 구분 | PaLM | Mixtral |
|---|---|---|
| 구조 | Dense Transformer | Sparse MoE Transformer |
| parameter 사용 | 모든 token이 dense model 통과 | token마다 8 expert 중 2개만 사용 |
| 효율 철학 | 거대한 dense scale | 큰 capacity + 낮은 active compute |
| 단점 | inference 비용 큼 | routing, load balancing, system 복잡도 |

---

## 한계

1. MoE serving은 dense model보다 복잡합니다.
2. Memory cost는 active parameter가 아니라 total parameter에 가깝습니다.
3. Expert가 사람이 해석하기 쉬운 domain expert로 나뉘지는 않았습니다.
4. Benchmark 비교는 evaluation protocol에 민감합니다.
5. 당시 leaderboard 비교는 특정 시점의 모델 버전에 대한 결과입니다.

---

## 입문자가 꼭 기억해야 할 문장

**Mixtral은 각 Transformer layer의 FFN을 8개의 expert로 바꾸고, token마다 router가 2개 expert만 선택하게 해서, 전체 47B parameter capacity를 가지면서도 token당 약 13B active parameter만 사용하는 sparse MoE LLM입니다.**

더 짧게 말하면:

> **Mixtral = 큰 모델 용량을 갖지만, token마다 일부 expert만 계산하는 효율적 MoE 모델**

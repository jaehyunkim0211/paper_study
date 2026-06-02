# Efficient Memory Management for Large Language Model Serving with PagedAttention — vLLM

## 결론부터

이 논문의 결론은 이겁니다.

**LLM 서빙에서 병목은 모델 계산만이 아니라, 생성 중 계속 커지는 KV cache 메모리를 얼마나 효율적으로 관리하느냐에 크게 좌우된다.**

기존 시스템은 각 요청의 KV cache를 연속된 메모리 공간에 크게 잡아두기 때문에 메모리 낭비가 심했고, vLLM은 이를 운영체제의 virtual memory와 paging 아이디어처럼 **작은 block 단위로 나눠 관리**해서 처리량을 크게 높였습니다.

한 줄로 말하면:

> **PagedAttention = LLM의 KV cache를 OS의 page처럼 쪼개서 관리하는 방법**  
> **vLLM = PagedAttention 위에 만든 고처리량 LLM serving engine**

---

## 이 논문을 한 문장으로 요약하면

**vLLM/PagedAttention은 LLM serving에서 가장 비싼 자원 중 하나인 KV cache 메모리를 “연속된 큰 덩어리”가 아니라 “작은 page block들의 조합”으로 관리해서 batch size를 키우고 throughput을 높이는 시스템 논문입니다.**

---

## LLM Serving은 Prefill과 Decode로 나뉜다

### Prefill

prompt 전체를 한 번에 처리합니다.

```text
prompt tokens 전체 처리
→ 첫 번째 생성 token 확률 계산
→ prompt token들의 KV cache 생성
```

### Decode

그다음부터 token을 하나씩 생성합니다.

```text
새 token 생성
→ 그 token의 KV cache 추가
→ 다음 token 생성
→ 반복
```

decode는 순차적이고 KV cache가 계속 커집니다.

---

## 핵심 개념 1: KV Cache

KV cache는 이전 token들의 attention key와 value vector를 저장해둔 메모리입니다.

```text
token 1의 K/V 저장
token 2의 K/V 저장
token 3의 K/V 저장
...
```

새 token을 생성할 때 현재 query는 이전 모든 token의 K/V를 참고합니다.

KV cache는 요청마다 다르고, 생성 길이에 따라 동적으로 커집니다.

---

## 핵심 개념 2: 기존 시스템의 문제

기존 시스템은 각 요청의 KV cache를 연속된 큰 메모리 공간에 미리 잡아두는 방식이었습니다.

```text
요청 A:
최대 2048 token까지 생성될 수 있음
→ 2048 token용 KV cache 공간을 미리 예약
```

실제 생성이 200 token이면:

```text
실제 사용: 200 token
예약: 2048 token
낭비: 1848 token 공간
```

---

## 핵심 개념 3: Fragmentation

### Internal Fragmentation

요청 하나에 너무 큰 공간을 잡아놓고 실제로 일부만 쓰는 경우입니다.

```text
예약: 2048 token
실제 생성: 300 token
남은 공간 낭비
```

### External Fragmentation

메모리 전체 빈 공간은 충분하지만, 연속된 큰 덩어리가 아니라 여기저기 쪼개져 있어 새 요청을 못 넣는 경우입니다.

---

## 핵심 개념 4: PagedAttention

PagedAttention은 요청의 KV cache를 고정 크기 block으로 나눕니다.

```text
logical block 0
logical block 1
logical block 2
```

이 logical block들은 실제 GPU 메모리의 비연속 physical block에 저장될 수 있습니다.

```text
logical block 0 → physical block 7
logical block 1 → physical block 1
logical block 2 → physical block 3
```

이 아이디어는 OS virtual memory와 paging에서 온 것입니다.

---

## 핵심 개념 5: Block Table

Block table은 각 요청의 logical KV block이 실제 GPU 메모리의 어떤 physical KV block에 저장되어 있는지 알려주는 매핑표입니다.

PagedAttention kernel은 attention 계산 중 block table을 보고 필요한 KV block을 찾아 읽습니다.

---

## 쉬운 비유: 연속 좌석 예약 vs 페이지 단위 예약

기존 방식은 극장 좌석을 이렇게 예약하는 것과 비슷합니다.

```text
손님이 몇 명 올지 모르니까,
한 그룹당 100석을 연속으로 미리 잡아두자.
```

실제로 10명만 오면 90석이 낭비됩니다.

PagedAttention은 이렇게 합니다.

```text
4석짜리 작은 묶음으로 필요한 만큼만 배정한다.
그 묶음들이 극장 안에서 떨어져 있어도 된다.
어디에 앉혔는지 표로 관리한다.
```

---

## 핵심 개념 6: Batching과 Throughput

LLM serving throughput을 높이려면 많은 요청을 batch로 묶어야 합니다.

```text
request 1
request 2
request 3
→ batch 처리
```

batch가 커지면 GPU utilization이 좋아지지만, 각 요청의 KV cache도 같이 커집니다.

```text
요청 수 증가
→ KV cache memory 증가
→ GPU 메모리 부족
→ batch size 제한
```

PagedAttention은 KV cache 낭비를 줄여 더 큰 batch를 가능하게 합니다.

---

## 핵심 개념 7: Iteration-level Scheduling

LLM generation은 token 단위로 진행됩니다. 따라서 요청 단위가 아니라 generation iteration 단위로 scheduling하면 효율이 좋아집니다.

```text
iteration 1 끝
→ 완료된 요청 제거
→ 새 요청 추가
→ 다음 iteration 실행
```

---

## 핵심 개념 8: Memory Sharing

같은 prompt에서 여러 답변을 sampling하거나 beam search를 하면 prefix가 공유됩니다.

기존 방식:

```text
sample 1 prompt KV cache
sample 2 prompt KV cache
sample 3 prompt KV cache
```

PagedAttention:

```text
prompt 부분 physical block을 공유
다른 부분만 따로 저장
```

---

## 핵심 개념 9: Copy-on-Write

공유 block을 수정해야 할 때만 복사합니다.

```text
sample A와 B가 같은 physical block 공유
sample A가 수정하려 함
→ 새 physical block 할당
→ 필요한 내용 복사
→ sample A는 새 block에 씀
```

---

## 핵심 개념 10: Preemption

GPU KV block이 부족하면 일부 요청을 잠시 중단하고 나중에 재개할 수 있습니다.

복구 방법은 두 가지입니다.

```text
1. Swapping: KV cache를 CPU RAM으로 옮김
2. Recomputation: KV cache를 버리고 나중에 다시 계산
```

---

## FlashAttention과 PagedAttention 비교

| 구분 | FlashAttention | PagedAttention |
|---|---|---|
| 주요 상황 | attention 계산 kernel | LLM serving KV cache 관리 |
| 병목 | `N × N` attention matrix와 HBM 접근 | KV cache fragmentation과 duplication |
| 해결 | tiling, online softmax, recomputation | paging, block table, copy-on-write |

짧게 말하면:

```text
FlashAttention = attention 계산을 잘한다.
PagedAttention = attention에 필요한 과거 KV memory를 잘 관리한다.
```

---

## 한계

1. PagedAttention kernel 자체에는 block table 접근 overhead가 있습니다.
2. 모든 GPU workload에 paging이 좋은 것은 아닙니다.
3. Scheduler, block allocator, copy-on-write, preemption 등 시스템 복잡도가 커집니다.
4. 모델 구조가 MoE, MLA, MQA/GQA 등으로 변하면 최적 설계도 달라질 수 있습니다.

---

## 입문자가 꼭 기억해야 할 문장

**PagedAttention은 LLM 생성 중 계속 커지는 KV cache를 고정 크기 block으로 나누고, logical block과 physical block을 block table로 매핑해서 비연속 GPU 메모리에도 저장할 수 있게 만든 attention serving 알고리즘입니다.**

더 짧게 말하면:

> **PagedAttention = KV cache를 page처럼 나눠 관리해서 LLM serving throughput을 높이는 방법**

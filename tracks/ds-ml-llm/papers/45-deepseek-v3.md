# DeepSeek-V3 Technical Report

## 결론부터

이 논문의 결론은 이겁니다.

**DeepSeek-V3는 “모델을 무작정 dense하게 크게 만드는 것”이 아니라, MoE 구조, KV cache 절감 attention, FP8 저정밀 학습, 통신 최적화, multi-token prediction, post-training까지 모두 결합해서 성능 대비 학습·추론 비용을 크게 낮추려 한 대형 LLM입니다.**

한 줄로 말하면:

> **DeepSeek-V3 = 671B 전체 파라미터를 가진 MoE 모델이지만, token 하나를 처리할 때는 37B만 활성화하는 고효율 대형 LLM**

---

## 이 논문을 한 문장으로 요약하면

**DeepSeek-V3는 “MoE로 전체 capacity는 크게, token당 계산량은 작게, MLA로 KV cache는 작게, FP8과 DualPipe로 학습 비용은 낮게” 만든 대형 open-weight 계열 LLM 기술보고서입니다.**

---

## 왜 중요했나?

DeepSeek-V3의 핵심 메시지는 이것입니다.

```text
대형 LLM의 성능은 model size 하나가 아니라,
architecture,
routing,
attention memory,
training precision,
parallelism,
communication overlap,
post-training recipe
가 함께 결정된다.
```

---

## 핵심 개념 1: MoE — 전체 671B, token당 37B 활성화

DeepSeek-V3는 전체 parameter는 671B이지만 token 하나를 처리할 때는 그중 37B만 사용합니다.

Dense model:

```text
token 하나 → 전체 모델 계산
```

MoE model:

```text
전체 capacity는 크지만
token마다 일부 expert만 계산
```

DeepSeek-V3의 MoE layer는:

```text
1 shared expert
256 routed experts
token마다 routed expert 8개 활성화
```

---

## 핵심 개념 2: Shared Expert와 Routed Expert

Shared expert는 모든 token이 공통으로 사용하는 expert입니다.

```text
문법
기본 문장 처리
공통 언어 패턴
```

Routed expert는 router가 token마다 선택하는 expert입니다.

```text
특정 token/문맥에 따라 선택
```

---

## 핵심 개념 3: Auxiliary-Loss-Free Load Balancing

MoE에서는 특정 expert에 token이 몰리면 안 됩니다.

전통적으로는 load balancing auxiliary loss를 사용합니다.

```text
routing이 불균형하면 loss로 벌점
```

DeepSeek-V3는 expert별 bias를 동적으로 조정합니다.

```text
과부하 expert의 bias 낮춤
덜 쓰이는 expert의 bias 높임
```

이렇게 loss에 강한 penalty를 넣지 않고 expert load를 맞추려 합니다.

---

## 핵심 개념 4: Node-Limited Routing과 No Token Dropping

MoE에서는 token을 expert가 있는 node로 보내야 합니다.

통신 비용을 줄이기 위해 DeepSeek-V3는 token이 최대 4개 node에만 보내지도록 제한합니다.

```text
각 token → 최대 4개 node
```

또 load balancing이 잘 되기 때문에 training과 inference에서 token dropping을 하지 않는다고 설명합니다.

---

## 핵심 개념 5: MLA, Multi-head Latent Attention

MLA는 key/value를 low-rank latent vector로 압축해 KV cache를 줄이는 attention 구조입니다.

일반 attention:

```text
K cache
V cache
```

MLA:

```text
compressed latent vector만 cache
필요할 때 K/V 복원
```

vLLM/PagedAttention이 KV cache를 잘 관리한다면, MLA는 KV cache 자체를 줄이려는 접근입니다.

---

## 핵심 개념 6: Multi-Token Prediction, MTP

일반 language model은 다음 token 하나를 예측합니다.

```text
현재 위치 → 다음 token
```

MTP는 추가 미래 token도 함께 예측합니다.

```text
현재 위치 → 다음 token + 그 다음 token
```

이는 training signal을 더 풍부하게 만들고, speculative decoding에도 활용될 수 있습니다.

---

## 핵심 개념 7: FP8 Mixed Precision Training

DeepSeek-V3는 대규모 모델 학습에 FP8 mixed precision을 적용했습니다.

```text
FP8 = 더 적은 메모리, 더 높은 처리량 가능
하지만 numerical stability 관리 필요
```

모든 것을 FP8로 하는 것은 아닙니다.

```text
어디는 FP8
어디는 BF16
어디는 FP32
정밀도 민감한 곳은 별도 처리
```

---

## 핵심 개념 8: DualPipe와 통신-계산 overlap

MoE training은 all-to-all communication이 많습니다.

```text
token dispatch
expert 계산
output combine
```

DualPipe는 computation과 communication을 겹쳐서 pipeline bubble을 줄입니다.

```text
어떤 micro-batch는 계산
다른 micro-batch는 통신
동시에 진행
```

---

## 핵심 개념 9: 14.8T token pre-training

DeepSeek-V3는 14.8T token으로 pre-training되었습니다.

데이터는 특히 다음을 강화했습니다.

```text
수학
프로그래밍
다국어 coverage
고품질 다양성
```

---

## 핵심 개념 10: Long Context Extension

DeepSeek-V3는 pre-training 중에는 4K sequence length를 사용하고, 이후 context extension으로 32K, 다시 128K까지 확장했습니다.

```text
4K → 32K → 128K
```

---

## 핵심 개념 11: Post-training

DeepSeek-V3는 base model pre-training 이후 다음을 수행합니다.

```text
SFT
RL
R1 reasoning capability distillation
```

수학처럼 정답 검증 가능한 문제는 rule-based reward를 사용하고, 자유형 답변은 model-based reward model을 사용합니다.

---

## DeepSeek-V3와 Mixtral 비교

| 구분 | Mixtral 8x7B | DeepSeek-V3 |
|---|---:|---:|
| 전체 parameter | 약 47B | 671B |
| token당 active parameter | 약 13B | 37B |
| expert 구조 | 8 experts, top-2 | 1 shared + 256 routed, top-8 |
| attention | 일반 Transformer 계열 | MLA |
| context | 32K | 128K |
| post-training | SFT + DPO | SFT + RL, R1 distillation |

---

## 장점

1. 큰 capacity와 낮은 active compute의 결합.
2. MLA로 inference KV cache 절감.
3. auxiliary-loss-free load balancing.
4. FP8와 DualPipe로 training efficiency 강화.
5. code, math, long-context에서 강한 성능.

---

## 한계

1. Deployment unit이 큽니다.
2. Serving 최적화 여지가 남아 있습니다.
3. 제시 비용은 official training 기준이며 R&D 비용은 제외됩니다.
4. MoE system complexity가 큽니다.
5. Benchmark 결과는 평가 설정에 의존합니다.

---

## 입문자가 꼭 기억해야 할 문장

**DeepSeek-V3는 671B 전체 파라미터를 가진 MoE 모델이지만 token당 37B만 활성화하고, MLA로 KV cache를 줄이며, FP8·DualPipe·통신 최적화로 training cost를 낮춘 대형 LLM system co-design 사례입니다.**

더 짧게 말하면:

> **DeepSeek-V3 = MoE + MLA + FP8 + DualPipe + post-training을 결합한 고효율 대형 LLM**

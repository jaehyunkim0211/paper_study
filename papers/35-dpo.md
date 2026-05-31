# Direct Preference Optimization: Your Language Model is Secretly a Reward Model — DPO

## 결론부터

이 논문의 결론은 이겁니다.

**사람이 선호한 답변과 선호하지 않은 답변 쌍이 있다면, 별도의 reward model을 만들고 PPO 같은 강화학습을 돌리지 않아도, 언어 모델을 직접 선호 데이터에 맞게 학습시킬 수 있다.**

이 방법이 바로 **DPO, Direct Preference Optimization**입니다.

한 줄로 말하면:

> **DPO = RLHF를 더 단순하게 만든 preference fine-tuning 방법**

---

## 기존 RLHF와 DPO

기존 RLHF는 보통 이렇게 복잡했습니다.

```text
1. SFT 모델 준비
2. 여러 답변 생성
3. 사람이 어떤 답변이 더 좋은지 비교
4. reward model 학습
5. PPO로 언어 모델을 reward가 높아지는 방향으로 강화학습
```

DPO는 이걸 단순하게 바꿉니다.

```text
선호된 답변 chosen
선호되지 않은 답변 rejected
이 둘의 확률 차이를 직접 학습
```

---

## 이 논문을 한 문장으로 요약하면

**DPO는 “사람이 더 좋아한 답변의 확률은 높이고, 덜 좋아한 답변의 확률은 낮추되, 기존 모델에서 너무 멀어지지 않게 하는” preference optimization 방법입니다.**

예를 들어:

```text
prompt: p-value를 초보자에게 쉽게 설명해줘.
chosen: 쉬운 예시와 함께 설명한 답변
rejected: 너무 전문적이고 딱딱한 답변
```

DPO는 같은 prompt에서 chosen의 확률을 rejected보다 높게 만들도록 학습합니다.

---

## 핵심 개념 1: Preference Data

DPO에 필요한 데이터는 다음 형식입니다.

```text
prompt x
chosen response y_w
rejected response y_l
```

`w`는 winner, `l`은 loser라고 보면 됩니다.

DPO는 이 pairwise preference를 학습합니다.

```text
chosen이 rejected보다 더 선호되어야 한다.
```

---

## 핵심 개념 2: Reference Model

DPO는 현재 학습 중인 모델이 원래 모델에서 너무 멀어지지 않도록 reference model을 기준으로 삼습니다.

왜 필요할까요?

```text
diversity 감소
특정 문체 과도 최적화
preference data 과적합
모델 붕괴
```

을 막기 위해서입니다.

기존 RLHF의 KL penalty와 비슷한 역할을 합니다.

---

## 핵심 개념 3: DPO Loss

DPO의 직관은 단순합니다.

```text
chosen의 확률을 rejected보다 높여라.
하지만 reference model 대비 상대적으로 조정하라.
```

대표 수식은 다음과 같은 형태입니다.

```text
L_DPO = - log σ(
  β [
    log πθ(y_w|x) - log πθ(y_l|x)
    - log πref(y_w|x) + log πref(y_l|x)
  ]
)
```

입문적으로는 이렇게 읽으면 됩니다.

```text
현재 모델이 reference model보다
chosen/rejected 차이를 더 선호 방향으로 만들도록 학습한다.
```

---

## 핵심 개념 4: “Your Language Model is Secretly a Reward Model”

DPO의 핵심 통찰은 **현재 policy와 reference policy의 log probability ratio를 reward처럼 볼 수 있다**는 것입니다.

기존 RLHF:

```text
Reward Model:
prompt + response → reward score
```

DPO:

```text
reward ≈ β × log(πθ(y|x) / πref(y|x))
```

즉, 명시적 reward model을 따로 만들지 않아도, 언어 모델 자체의 확률 변화가 implicit reward처럼 작동합니다.

---

## 핵심 개념 5: Bradley-Terry Preference Model

Bradley-Terry 모델은 두 답변 중 하나를 더 선호할 확률을 reward 차이로 표현합니다.

```text
reward(A) > reward(B)
→ 사람이 A를 선호할 확률이 높다.
```

DPO는 이 preference likelihood를 reward model이 아니라 policy log ratio로 직접 표현합니다.

---

## RLHF와 DPO 비교

| 구분 | RLHF | DPO |
|---|---|---|
| 선호 데이터 | 사용 | 사용 |
| reward model | 따로 학습 | 없음 |
| 강화학습 | PPO 사용 | 사용하지 않음 |
| reference model | KL 제약에 사용 | log-ratio에 사용 |
| 구현 난이도 | 높음 | 낮음 |
| 학습 안정성 | 튜닝 필요 | 상대적으로 단순 |

---

## SFT와 DPO의 차이

### SFT

```text
prompt → chosen answer
```

목표:

```text
chosen answer의 likelihood를 높여라.
```

### DPO

```text
prompt
chosen answer
rejected answer
```

목표:

```text
chosen을 rejected보다 더 선호하게 하라.
```

SFT는 모범답안 따라 하기이고, DPO는 상대적 선호 학습입니다.

---

## DPO의 장점

1. 구현이 단순합니다.
2. PPO보다 안정적인 supervised-style 학습에 가깝습니다.
3. reward model을 따로 만들 필요가 없습니다.
4. training 중 on-policy sampling loop가 필요 없습니다.
5. LoRA/QLoRA와 결합하기 쉽습니다.

---

## DPO의 한계

1. Preference data 품질에 민감합니다.
2. Pairwise preference는 다차원 선호를 단순화합니다.
3. Reference model 설정이 중요합니다.
4. Over-optimization 문제가 완전히 사라지는 것은 아닙니다.
5. 새로운 지식을 넣는 방법은 아닙니다.

---

## DPO와 LoRA/QLoRA 연결

DPO는 학습 objective입니다. LoRA/QLoRA는 효율적으로 fine-tuning하는 방법입니다.

같이 쓸 수 있습니다.

```text
Base model
→ LoRA adapter 삽입
→ DPO loss로 preference pair 학습
```

또는:

```text
4-bit base model
→ QLoRA adapter
→ DPO loss
```

---

## 입문자가 꼭 기억해야 할 문장

**DPO는 chosen/rejected preference pair를 이용해, 선호된 답변의 상대적 확률을 높이고 선호되지 않은 답변의 상대적 확률을 낮추는 방식으로 언어 모델을 직접 alignment하는 방법입니다.**

더 짧게 말하면:

> **DPO = reward model 없이 preference pair로 직접 학습하는 RLHF 대안**

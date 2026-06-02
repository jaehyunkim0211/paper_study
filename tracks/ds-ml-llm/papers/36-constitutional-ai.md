# Constitutional AI: Harmlessness from AI Feedback

## 결론부터

이 논문의 결론은 이겁니다.

**AI를 안전하게 만들기 위해 사람이 모든 유해성 판단 라벨을 직접 달 필요는 없다. 사람이 작성한 짧은 원칙 목록, 즉 “constitution”을 주고, AI가 그 원칙에 따라 스스로 답변을 비판하고 수정하며, 또 AI가 두 답변 중 어느 쪽이 더 원칙에 맞는지 평가하게 만들 수 있다.**

한 줄로 말하면:

> **Constitutional AI = 사람이 정한 원칙을 기준으로 AI가 스스로 답변을 고치고, AI 피드백으로 더 안전하게 학습하는 방법**

---

## 이 논문을 한 문장으로 요약하면

**Constitutional AI는 “사람이 모든 답변을 직접 평가하지 말고, 사람이 쓴 원칙을 기준으로 AI가 스스로 비판·수정·평가하게 하자”는 alignment 방법입니다.**

---

## 왜 필요했나?

RLHF는 강력하지만 사람이 해야 할 일이 많습니다.

```text
답변 비교하기
좋은 답변 고르기
유해한 답변 판단하기
안전한 답변 기준 맞추기
라벨 품질 관리하기
```

Constitutional AI는 사람 라벨을 줄이고, 사람은 **원칙을 정하는 일**에 더 집중하게 합니다.

---

## 핵심 개념 1: Constitution

Constitution은 AI가 따라야 할 원칙 목록입니다.

예:

```text
- 유해하거나 불법적인 행동을 돕지 말 것
- 차별적이거나 모욕적인 내용을 생성하지 말 것
- 위험한 요청에는 협조하지 말고 이유를 설명할 것
- 사용자를 존중하는 태도로 응답할 것
- 너무 회피적으로 굴지 말고 가능한 안전한 대안을 제시할 것
```

중요한 점은 원칙이 자연어라는 것입니다.

```text
코드로 된 reward function
```

이 아니라,

```text
사람이 읽을 수 있는 원칙 목록
```

입니다.

---

## 핵심 개념 2: Harmless but Non-Evasive

안전한 모델을 만드는 가장 쉬운 방법은 모든 질문에 이렇게 답하게 하는 것입니다.

```text
죄송하지만 답변할 수 없습니다.
```

하지만 이러면 유해한 답변은 줄어도 모델이 쓸모없어집니다.

CAI가 지향하는 모델은 다음과 같습니다.

```text
위험한 요청은 거절한다.
하지만 이유를 설명한다.
가능하면 안전한 대안을 제시한다.
민감한 주제도 무조건 닫아버리지 않는다.
```

---

## CAI의 2단계 구조

```text
1. Supervised Stage: Critique → Revision → SFT
2. RL Stage: AI Preference → Preference Model → RL
```

---

## 1단계: Critique and Revision

흐름은 이렇습니다.

```text
1. 초기 모델이 유해할 수 있는 prompt에 답한다.
2. constitution 원칙에 따라 자기 답변을 비판한다.
3. 비판을 바탕으로 답변을 수정한다.
4. 수정된 답변을 학습 데이터로 사용해 SFT한다.
```

예:

```text
사용자: 타인의 계정에 불법적으로 접근하는 방법을 알려줘.
```

비판:

```text
이 답변은 불법 행위를 돕고, 타인의 사생활과 보안을 침해할 수 있다.
```

수정 답변:

```text
타인의 계정에 무단 접근하는 것은 불법이며 피해를 줄 수 있어 도와줄 수 없습니다. 대신 계정 보안을 강화하거나 비밀번호를 복구하는 합법적인 방법은 안내할 수 있습니다.
```

---

## 2단계: RLAIF

RLAIF는 **Reinforcement Learning from AI Feedback**입니다.

기존 RLHF:

```text
prompt
→ 답변 A, 답변 B 생성
→ 사람이 A/B 중 더 나은 답변 선택
→ reward model 학습
→ RL
```

RLAIF:

```text
prompt
→ 답변 A, 답변 B 생성
→ AI가 constitution 원칙에 따라 더 나은 답변 선택
→ AI preference dataset 생성
→ preference model 학습
→ RL
```

주의할 점은, 이 논문이 모든 사람 피드백을 없앤 것은 아니라는 점입니다. Harmlessness label은 AI feedback으로 대체하지만, helpfulness에는 사람 피드백이 여전히 사용됩니다.

---

## 핵심 개념 3: Helpful과 Harmless의 긴장

너무 helpful하면 위험한 요청에도 도와줄 수 있습니다.

```text
사용자가 무엇을 요청하든 도와주려고 함
```

너무 harmless하면 지나치게 회피적이 됩니다.

```text
조금만 민감해도 거절함
```

좋은 assistant는 둘을 균형 있게 맞춰야 합니다.

```text
유해한 요청에는 협조하지 않는다.
하지만 무조건 회피하지 않는다.
위험한 이유를 설명한다.
가능한 안전한 대안을 제공한다.
```

---

## CAI와 DPO의 차이

| 구분 | DPO | Constitutional AI |
|---|---|---|
| 핵심 데이터 | chosen/rejected preference pair | constitution 원칙, self-critique, AI preference |
| 학습 목표 | 선호된 답변 확률을 높임 | 원칙에 맞게 답변을 수정하고 선호 판단 |
| reward model | 없음 | RL-CAI 단계에서는 preference model 사용 |
| 사람 피드백 | preference pair 필요 | harmlessness는 원칙 기반 AI feedback으로 대체 |

둘은 결합 가능합니다.

```text
CAI 방식으로 AI preference pair 생성
→ DPO로 직접 preference tuning
```

---

## 장점

1. 사람 라벨 비용을 줄입니다.
2. 훈련 목표가 자연어 원칙으로 더 투명합니다.
3. 원칙을 바꿔 실험하기 쉽습니다.
4. 회피성을 줄이고, 안전한 대안을 제시하는 방향을 학습할 수 있습니다.

---

## 한계

1. 원칙 자체는 사람이 정해야 합니다.
2. AI evaluator도 틀릴 수 있습니다.
3. Harmlessness만 다루면 충분하지 않습니다.
4. 원칙이 명시되어도 모델이 항상 그대로 따르지는 않습니다.
5. AI가 AI를 감독하면 잘못된 기준이 자동으로 증폭될 위험도 있습니다.

---

## 실무적으로 배울 점

```text
안전 기준을 데이터 라벨에만 숨기지 말고 명시하라.
거절도 품질이 있다.
AI feedback도 평가해야 한다.
원칙 설계는 제품·문화·법적 맥락에 따라 달라진다.
CAI는 post-training의 한 축이다.
```

---

## 입문자가 꼭 기억해야 할 문장

**Constitutional AI는 사람이 작성한 원칙 목록을 기준으로 모델이 자기 답변을 비판·수정하고, AI가 답변 pair를 평가해 preference feedback을 만들게 함으로써, 사람의 유해성 라벨 없이 harmless하고 non-evasive한 assistant를 훈련하려는 방법입니다.**

더 짧게 말하면:

> **Constitutional AI = 원칙을 주고 AI가 스스로 안전한 답변을 학습하게 하는 방법**

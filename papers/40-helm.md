# Holistic Evaluation of Language Models — HELM

## 결론부터

이 논문의 결론은 이겁니다.

**LLM을 평가할 때 accuracy 하나만 보면 안 된다.** 언어 모델은 여러 상황에서 쓰이고, 모델 품질도 정확도뿐 아니라 **calibration, robustness, fairness, bias, toxicity, efficiency** 같은 여러 기준으로 봐야 합니다.

한 줄로 말하면:

> **HELM = LLM을 여러 사용 시나리오와 여러 평가 기준으로 종합적으로 평가하려는 프레임워크**

---

## 이 논문을 한 문장으로 요약하면

**HELM은 “LLM 평가는 단일 점수 경쟁이 아니라, 다양한 task와 다양한 metric을 함께 보는 다차원 평가여야 한다”는 논문입니다.**

---

## 왜 HELM이 필요했나?

LLM 평가는 어렵습니다. 모델이 할 수 있는 일이 너무 많기 때문입니다.

```text
질문답변
요약
번역
상식 추론
수학 추론
코드 생성
대화
정보 검색
독해
유해성 대응
편향 없는 응답
긴 문맥 처리
```

기존 평가는 좁은 benchmark 점수에 의존하는 경우가 많았습니다.

```text
MMLU 몇 점?
GSM8K 몇 점?
TruthfulQA 몇 점?
HumanEval 몇 점?
```

이런 점수는 유용하지만 모델 전체를 설명하지는 못합니다.

---

## 핵심 개념 1: Holistic Evaluation

Holistic evaluation은 모델을 한 가지 task나 한 가지 metric으로 평가하지 않고, 여러 사용 상황과 여러 평가 기준을 함께 보는 방식입니다.

정확도만 보면 안 됩니다.

```text
정확한가?
확신도는 믿을 만한가?
입력 변형에 강한가?
편향적이지 않은가?
유해한 말을 만들지 않는가?
얼마나 빠르고 저렴한가?
```

---

## 핵심 개념 2: Scenario

Scenario는 모델이 사용될 구체적인 상황 또는 task입니다.

```text
질문답변
요약
정보 검색
독해
상식 추론
수학 추론
언어 이해
텍스트 분류
대화 응답
```

HELM은 benchmark dataset 이름만 나열하지 않고, 먼저 어떤 사용 상황을 평가해야 하는지 정리하려 합니다.

---

## 핵심 개념 3: Metric

HELM은 core scenario에서 가능한 경우 다음 7개 metric을 함께 봅니다.

```text
accuracy
calibration
robustness
fairness
bias
toxicity
efficiency
```

---

## 7개 metric 쉽게 이해하기

### Accuracy

```text
정답을 맞혔는가?
```

### Calibration

모델의 확신도와 실제 정답률이 잘 맞는지 봅니다.

```text
90% 확신한 문제 중 실제로 약 90%를 맞히는가?
```

### Robustness

입력이 조금 바뀌어도 성능이 유지되는지 봅니다.

```text
What is the capital of France?
Could you tell me the capital city of France?
```

### Fairness

집단별 성능 차이가 과도하지 않은지 봅니다.

### Bias

특정 집단이나 속성에 대한 편향된 연상이나 출력을 보는 기준입니다.

### Toxicity

모욕적, 공격적, 유해한 내용을 생성하는지 봅니다.

### Efficiency

latency, throughput, inference cost, model size, energy use를 봅니다.

---

## 핵심 개념 4: Multi-metric Evaluation

하나의 점수로 모델을 줄 세우면 중요한 trade-off가 가려집니다.

| 모델 | Accuracy | Robustness | Toxicity | Efficiency |
|---|---:|---:|---:|---:|
| 모델 A | 높음 | 낮음 | 높음 | 보통 |
| 모델 B | 중간 | 높음 | 낮음 | 좋음 |

어느 모델이 더 좋은지는 사용 상황에 따라 다릅니다.

---

## 핵심 개념 5: Standardized Evaluation

모델들을 같은 조건에서 비교해야 합니다.

```text
scenario
metric
prompting strategy
inference setting
model output 기록
evaluation pipeline
```

이것이 표준화되지 않으면 공정한 비교가 어렵습니다.

---

## 핵심 개념 6: Transparency

점수만 저장하면 부족합니다. 다음도 함께 남겨야 합니다.

```text
prompt
completion
metric
model version
decoding parameter
```

HELM은 raw prompts와 completions를 공개해 평가 투명성을 높이려 했습니다.

---

## HELM과 MMLU 차이

| 구분 | MMLU | HELM |
|---|---|---|
| 성격 | 지식·문제풀이 benchmark | 종합 평가 프레임워크 |
| 주 관심 | 57개 과목 객관식 accuracy | scenario × metric 다차원 평가 |
| metric | 주로 accuracy | accuracy, calibration, robustness, fairness, bias, toxicity, efficiency |

```text
MMLU = 하나의 중요한 시험
HELM = 여러 시험과 평가 기준을 함께 운영하는 평가 체계
```

---

## 한계

1. Holistic하다고 해서 모든 사용 상황과 metric을 포함하는 것은 아닙니다.
2. bias, toxicity, fairness metric 자체도 완벽하지 않습니다.
3. 어떤 scenario를 포함하느냐가 결과를 좌우합니다.
4. 자동 평가와 사람 평가 사이에는 간극이 있습니다.
5. living benchmark는 지속적인 유지보수가 필요합니다.

---

## 실무적 교훈

```text
단일 benchmark 점수로 모델을 고르지 마라.
Accuracy 외 metric을 반드시 포함하라.
평가 조건을 표준화하라.
Raw outputs를 저장하라.
평가도 계속 업데이트해야 한다.
```

---

## 입문자가 꼭 기억해야 할 문장

**HELM은 LLM을 accuracy 하나로 줄 세우지 말고, 다양한 사용 scenario와 accuracy·calibration·robustness·fairness·bias·toxicity·efficiency 같은 여러 metric으로 평가해야 한다는 종합 평가 프레임워크입니다.**

더 짧게 말하면:

> **HELM = LLM을 여러 상황과 여러 기준으로 공정하게 평가하려는 프레임워크**

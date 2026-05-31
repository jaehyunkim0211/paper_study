# Measuring Massive Multitask Language Understanding — MMLU

## 결론부터

이 논문의 결론은 이겁니다.

**LLM이 정말 “언어를 이해한다”고 말하려면, 단순한 문장 분류나 독해 문제만 잘 풀어서는 부족하다. 수학, 역사, 법, 의학, 철학, 경제학, 컴퓨터과학처럼 사람이 실제로 공부하는 다양한 과목의 문제를 폭넓게 풀 수 있어야 한다.**

이런 능력을 평가하기 위해 만든 benchmark가 **MMLU, Massive Multitask Language Understanding**입니다.

한 줄로 말하면:

> **MMLU = LLM에게 57개 과목의 객관식 시험을 보게 해서, 세계지식과 문제 해결 능력을 평가하는 benchmark**

---

## 이 논문을 한 문장으로 요약하면

**MMLU는 “LLM이 여러 학문·전문 분야의 객관식 시험을 얼마나 잘 푸는가”를 측정해서, 모델의 지식 폭과 문제 해결 능력의 빈틈을 드러내는 benchmark입니다.**

---

## 왜 MMLU가 필요했나?

기존 NLP benchmark는 대체로 다음 문제를 봤습니다.

```text
문장 감성분석
문장 유사도
자연어 추론
독해
상식 추론
```

하지만 LLM은 인터넷, 책, 위키피디아, 코드, 논문 등 엄청난 텍스트로 학습됩니다. 그러면 궁금해집니다.

```text
법률 문제를 이해할까?
의학 문제를 풀 수 있을까?
수학 절차를 적용할 수 있을까?
역사·경제·철학 문제를 다룰 수 있을까?
```

MMLU는 이 질문에 답하려고 했습니다.

---

## 핵심 개념 1: Massive Multitask

MMLU의 핵심은 하나의 task가 아니라 57개 과목을 넓게 평가하는 것입니다.

범주는 대략 다음과 같습니다.

```text
Humanities
Social Sciences
STEM
Other professional / specialized subjects
```

예:

```text
수학
물리
컴퓨터과학
미국사
세계사
법
철학
윤리
경제학
심리학
의학
마케팅
회계
종교
국제관계
```

---

## 핵심 개념 2: 객관식 문제 형식

MMLU는 대부분 A/B/C/D 중 하나를 고르는 multiple-choice question 형식입니다.

장점:

```text
자동 채점이 쉽다.
모델 간 비교가 쉽다.
accuracy로 단순 평가할 수 있다.
```

단점:

```text
선택지 패턴을 이용할 수 있다.
정답을 찍을 수 있다.
실제 생성형 답변 품질을 직접 평가하지 않는다.
풀이 과정이 맞는지는 알 수 없다.
```

---

## 핵심 개념 3: Zero-shot과 Few-shot 평가

MMLU는 모델을 새로 fine-tuning하지 않고 평가합니다.

### Zero-shot

예시 없이 바로 문제를 줍니다.

```text
Question: ...
A. ...
B. ...
C. ...
D. ...
Answer:
```

### Few-shot

과목별로 몇 개 예시를 먼저 보여주고 새 문제를 풉니다.

```text
Question 1: ...
Answer: B

Question 2: ...
Answer: D

Question 3: ...
Answer:
```

---

## 핵심 개념 4: Pretraining에서 배운 지식 평가

MMLU는 task-specific fine-tuning으로 문제 유형을 외우는지보다, 대규모 pretraining에서 배운 지식을 실제 문제에 적용할 수 있는지를 보려 합니다.

```text
대규모 텍스트 pretraining
→ few-shot 예시
→ 시험 문제 풀이
```

---

## 데이터셋 구성

MMLU는 총 15,908개 객관식 질문으로 구성됩니다.

```text
57개 과목
총 15,908개 질문
few-shot development set: 과목당 5개 질문
validation set: 1,540개 질문
test set: 14,079개 질문
각 과목 test example 최소 100개
```

질문들은 GRE, USMLE, 전문 자격시험 practice questions, undergraduate course questions 등 다양한 공개 출처에서 수집되었습니다.

---

## 핵심 결과

### 1. 당시 모델들은 대부분 random chance 수준이었다

4지선다 random baseline은 25%입니다. MMLU가 처음 나왔을 때 많은 모델은 이 근처였습니다.

```text
언어 benchmark를 잘 푼다
≠ 세계지식과 전문 문제 해결을 잘한다
```

### 2. 모델 크기가 커질수록 성능이 좋아졌다

GPT-3 계열에서도 작은 모델은 random chance 근처였지만, 가장 큰 모델은 의미 있게 좋아졌습니다.

### 3. Fine-tuned QA 모델도 강했다

UnifiedQA처럼 QA에 fine-tuning된 모델은 parameter가 더 작아도 MMLU에서 좋은 성능을 보였습니다.

### 4. 과목별로 성능이 불균형했다

```text
history는 강하지만 math는 약함
law는 약함
STEM 계산 문제는 약함
```

MMLU는 이런 blind spot을 드러냅니다.

### 5. Calibration 문제가 있었다

모델이 자신 있게 답했지만 실제로 틀리는 경우가 많았습니다.

---

## MMLU와 HELM의 차이

| 구분 | MMLU | HELM |
|---|---|---|
| 성격 | 하나의 benchmark | 평가 프레임워크 |
| 핵심 질문 | 57개 과목 객관식 문제를 얼마나 잘 푸는가? | 다양한 scenario와 metric에서 trade-off는? |
| 주요 metric | accuracy | accuracy, calibration, robustness, fairness, bias, toxicity, efficiency |

---

## MMLU와 TruthfulQA의 차이

| 구분 | MMLU | TruthfulQA |
|---|---|---|
| 평가 대상 | 학문·전문 과목 지식과 문제 해결 | 흔한 오해를 따라하지 않고 진실하게 답하는가 |
| 형식 | 주로 multiple-choice | misconception 유도 질문 |
| 핵심 실패 | 지식 부족, 추론 실패 | 그럴듯한 거짓 모방 |

---

## 현대적 한계

1. 상위 모델들이 점점 포화되었습니다.
2. 일부 문항 오류와 noise 문제가 지적되었습니다.
3. 객관식 형식은 실제 업무 능력을 완전히 대표하지 않습니다.
4. 영어 중심·텍스트 중심입니다.

따라서 현재는 MMLU만 보지 않고 MMLU-Pro, GPQA, GSM8K, TruthfulQA, HumanEval, HELM, domain-specific eval을 함께 봅니다.

---

## 입문자가 꼭 기억해야 할 문장

**MMLU는 LLM이 57개 학문·전문 과목의 객관식 문제를 얼마나 잘 푸는지 평가해, 모델의 세계지식과 문제 해결 능력, 그리고 과목별 약점을 드러내는 benchmark입니다.**

더 짧게 말하면:

> **MMLU = LLM에게 보는 종합 학력고사**

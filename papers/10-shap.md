# A Unified Approach to Interpreting Model Predictions — SHAP

## 결론부터

이 논문의 결론은 이겁니다.

**복잡한 머신러닝 모델의 예측을 설명하려면, 각 feature가 “이번 예측값”에 얼마나 기여했는지를 공정하게 나눠 계산해야 한다. SHAP은 그 기여도를 Shapley value라는 게임이론 개념으로 계산하는 통합 프레임워크다.**

쉽게 말하면:

> 모델이 어떤 고객을 “이탈 위험 높음”이라고 예측했다면,  
> SHAP은 “가입기간, 월요금, 최근 사용량, 불만 횟수 각각이 이 예측을 얼마나 올리거나 내렸는지”를 나눠서 설명해준다.

Lundberg와 Lee의 논문은 SHAP을 **SHapley Additive exPlanations**라고 정의하고, 특정 예측 하나에 대해 각 feature에 importance value를 부여하는 통합 설명 프레임워크로 제안합니다. 이 논문은 여러 기존 설명 기법을 **additive feature attribution method**라는 하나의 틀로 묶고, 그 안에서 바람직한 성질을 만족하는 유일한 해가 Shapley value임을 보입니다.

---

## 이 논문을 한 문장으로 요약하면

**SHAP은 “모델 예측값”을 baseline에서 출발해 feature별 기여도의 합으로 분해하는 방법이다.**

형태는 이렇게 이해하면 됩니다.

```text
모델 예측값
= 평균적인 예측값
+ feature 1의 기여
+ feature 2의 기여
+ feature 3의 기여
+ ...
```

예를 들어 고객 이탈 모델이 있다고 해봅시다.

```text
전체 평균 이탈 위험도: 20%
이 고객의 최종 예측 위험도: 65%
```

SHAP은 이 45%p 상승을 feature별로 나눠 설명합니다.

| Feature | SHAP 해석 |
|---|---:|
| 가입기간 짧음 | +18%p |
| 월요금 높음 | +12%p |
| 최근 사용량 감소 | +20%p |
| 장기 할인 적용 중 | -5%p |
| 최종 예측 | 65% |

즉, SHAP은 단순히 “이 feature가 중요하다”가 아니라:

> **이 고객의 이번 예측에서 이 feature가 예측값을 얼마나 올렸거나 내렸는가?**

를 설명합니다.

---

## 왜 이 논문이 중요한가?

지금까지 우리는 Random Forest, Gradient Boosting, XGBoost, LightGBM을 봤습니다.

이 모델들은 성능이 좋지만, 단일 선형회귀처럼 쉽게 설명되지 않습니다.

| 모델 | 성능 | 해석 난이도 |
|---|---:|---:|
| Linear Regression | 보통 | 쉬움 |
| Decision Tree 하나 | 중간 | 비교적 쉬움 |
| Random Forest | 좋음 | 어려움 |
| XGBoost | 매우 좋음 | 어려움 |
| LightGBM | 매우 좋음 | 어려움 |

이때 생기는 문제가 있습니다.

> 모델은 잘 맞추는데, 왜 그렇게 예측했는지 설명하기 어렵다.

데이터사이언스 실무에서는 이 문제가 굉장히 큽니다.

| 상황 | 왜 설명이 필요한가 |
|---|---|
| 고객 이탈 예측 | 마케팅팀이 어떤 고객에게 어떤 조치를 할지 알아야 함 |
| 신용평가 | 대출 거절 이유를 설명해야 할 수 있음 |
| 의료 예측 | 의사가 모델 판단 근거를 이해해야 함 |
| 사기 탐지 | 운영팀이 경고의 근거를 확인해야 함 |
| 채용 모델 | 차별적 판단이 아닌지 점검해야 함 |

그래서 SHAP은 단순한 시각화 도구가 아니라, **복잡한 모델을 검토하고 설명하고 운영하기 위한 핵심 도구**입니다.

---

## 핵심 개념 1: Local explanation

**SHAP은 기본적으로 “개별 예측 하나”를 설명하는 local explanation 방법입니다.**

예를 들어 LightGBM 모델의 전체 성능이 AUC 0.91이라고 합시다.

이건 전체 성능입니다.

하지만 실무자가 궁금해하는 질문은 종종 이겁니다.

> 왜 이 고객은 이탈 위험이 높게 나왔지?  
> 왜 이 거래는 사기라고 판단됐지?  
> 왜 이 환자는 고위험군으로 분류됐지?

SHAP은 이런 질문에 답합니다.

```text
모델 전체가 어떻게 작동하는가?
```

보다 먼저:

```text
이 특정 입력 x에 대해 모델이 왜 이런 예측을 했는가?
```

를 설명합니다.

---

## 핵심 개념 2: Additive feature attribution

**SHAP은 예측을 feature별 기여도의 합으로 설명합니다.**

복잡한 모델 `f(x)`가 있다고 합시다.

SHAP은 이 모델의 예측을 설명하기 위해 단순한 설명 모델 `g`를 둡니다.

```text
g = base value + feature별 기여도의 합
```

쉽게 쓰면:

```text
예측 설명값 = φ₀ + φ₁ + φ₂ + φ₃ + ... + φ_M
```

여기서:

| 기호 | 의미 |
|---|---|
| `φ₀` | baseline, 평균적인 예측값 |
| `φ₁` | feature 1의 기여도 |
| `φ₂` | feature 2의 기여도 |
| `φ_M` | feature M의 기여도 |

쉽게 이해하면 됩니다.

> SHAP은 복잡한 모델의 예측 하나를,  
> “기준값 + feature별 영향”이라는 단순한 덧셈 구조로 바꿔 설명한다.

---

## 핵심 개념 3: Shapley value

**Shapley value는 여러 feature가 함께 만든 예측값을 각 feature에게 공정하게 나눠주는 방법입니다.**

원래 Shapley value는 게임이론에서 나왔습니다.

예를 들어 세 사람이 같이 프로젝트를 해서 100만 원의 이익을 냈다고 합시다.

```text
A, B, C가 함께 100만 원을 벌었다.
```

문제는 이것입니다.

> A, B, C 각각에게 공을 얼마나 배분해야 공정할까?

단순히 1/3씩 나누면 쉬울 수 있지만, 현실은 다릅니다.

| 사람 | 기여 |
|---|---|
| A | 혼자서도 꽤 많이 기여 |
| B | A와 같이 있을 때 시너지가 큼 |
| C | 특정 조건에서만 도움 |

Shapley value는 각 사람이 여러 조합에 들어갔을 때 추가로 얼마나 기여했는지를 평균내서, 공정한 기여도를 계산합니다.

SHAP에서는 이걸 feature에 적용합니다.

| 게임이론 | SHAP |
|---|---|
| 플레이어 | feature |
| 게임의 보상 | 모델 예측값 |
| 플레이어의 기여 | feature의 예측 기여도 |
| Shapley value | SHAP value |

즉:

> feature들이 함께 만든 예측값을, feature별로 공정하게 나눠 갖게 하는 방법이 SHAP입니다.

---

## 쉬운 예시: 고객 이탈 예측

고객 이탈 예측 모델이 있습니다.

입력 feature는 다음과 같습니다.

| Feature | 값 |
|---|---|
| 가입기간 | 2개월 |
| 월요금 | 90,000원 |
| 최근 사용량 | 1GB |
| 불만 접수 | 2회 |
| 할인 적용 | 예 |

모델의 평균 예측, 즉 baseline은 이렇습니다.

```text
전체 평균 이탈 위험도 = 20%
```

이 고객에 대한 모델 예측은:

```text
이탈 위험도 = 70%
```

그럼 SHAP은 이렇게 설명합니다.

| Feature | 기여도 |
|---|---:|
| 가입기간 2개월 | +15%p |
| 월요금 90,000원 | +10%p |
| 최근 사용량 1GB | +20%p |
| 불만 접수 2회 | +18%p |
| 할인 적용 | -3%p |

합치면:

```text
20% + 15% + 10% + 20% + 18% - 3% = 70%
```

이게 SHAP의 핵심입니다.

**각 feature의 기여도를 모두 더하면 모델의 예측값이 됩니다.**

이 성질이 바로 아래에서 말할 **local accuracy**와 연결됩니다.

---

## 핵심 개념 4: SHAP의 세 가지 중요한 성질

논문은 additive feature attribution 방법에서 바람직한 성질로 **local accuracy, missingness, consistency**를 제시합니다. 그리고 이 세 성질을 만족하는 유일한 additive feature attribution 방법이 Shapley value라고 보입니다.

### 1. Local Accuracy

**feature별 기여도를 다 더하면 원래 모델의 예측값과 같아야 한다는 성질입니다.**

예를 들어:

```text
base value = 20
feature A의 SHAP value = +10
feature B의 SHAP value = -5
feature C의 SHAP value = +15
```

그러면:

```text
20 + 10 - 5 + 15 = 40
```

모델 예측값도 40이어야 합니다.

쉽게 말하면:

> 설명을 다 더했는데 원래 예측값과 다르면, 좋은 설명이 아니다.

### 2. Missingness

**존재하지 않는 feature는 기여도가 0이어야 한다는 성질입니다.**

예를 들어 어떤 설명에서 feature가 빠져 있거나, 단순화된 입력에서 해당 feature가 존재하지 않는다고 합시다.

그러면 그 feature는 예측에 영향을 줬다고 말하면 안 됩니다.

```text
없는 feature의 기여도 = 0
```

### 3. Consistency

**어떤 feature의 실제 기여가 커졌다면, 그 feature의 attribution이 줄어들면 안 된다는 성질입니다.**

예를 들어 모델 A와 모델 B가 있다고 합시다.

모델 B에서 `최근 사용량 감소`라는 feature가 이탈 위험을 더 강하게 올리도록 바뀌었습니다.

그렇다면 설명 방법도 그 feature의 기여도를 더 크게 주거나 최소한 줄이지는 않아야 합니다.

쉽게 말하면:

> 모델에서 feature의 영향이 커졌는데 설명값이 작아지면 이상하다.  
> SHAP은 이런 모순을 피하려고 한다.

---

## 핵심 개념 5: 왜 “Unified Approach”인가?

논문 제목이 **A Unified Approach**입니다.

왜 통합 접근일까요?

이 논문 이전에도 설명 기법은 많았습니다.

| 방법 | 설명 |
|---|---|
| LIME | 특정 예측 주변에서 단순한 선형 모델로 근사 |
| DeepLIFT | 딥러닝 모델에서 reference 대비 activation 차이를 전파 |
| Layer-wise Relevance Propagation | 딥러닝 예측 기여도를 layer별로 역전파 |
| Shapley sampling values | Shapley value를 샘플링으로 근사 |
| Shapley regression values | 회귀 모델에서 Shapley 방식으로 feature contribution 계산 |
| Quantitative Input Influence | 입력 feature의 영향력 측정 |

SHAP 논문은 이 여러 방법이 사실 같은 형태의 설명 모델, 즉 additive feature attribution method로 볼 수 있다고 정리합니다.

즉, 이 논문의 기여는 단순히 새로운 기법 하나를 낸 게 아닙니다.

> 기존 설명 방법들을 하나의 수학적 틀로 묶고,  
> 그 틀 안에서 어떤 설명이 바람직한지 기준을 제시했다.

이게 핵심입니다.

---

## 핵심 개념 6: Base value에서 예측값까지 가는 설명

SHAP을 가장 쉽게 이해하는 문장은 이것입니다.

> SHAP은 평균 예측값에서 시작해, 각 feature가 예측값을 얼마나 밀고 당겼는지를 보여준다.

예를 들어 집값 예측 모델을 봅시다.

```text
평균 집값 예측: 5억
현재 집의 예측값: 8억
```

SHAP 설명:

| Feature | 기여 |
|---|---:|
| 강남 위치 | +2.0억 |
| 전용면적 큼 | +1.0억 |
| 구축 아파트 | -0.5억 |
| 역세권 | +0.8억 |
| 저층 | -0.3억 |

합:

```text
5억 + 2.0억 + 1.0억 - 0.5억 + 0.8억 - 0.3억 = 8억
```

이렇게 보면 모델 예측이 훨씬 이해됩니다.

---

## SHAP value는 양수와 음수로 해석한다

| SHAP value | 의미 |
|---:|---|
| 양수 | 예측값을 높이는 방향으로 기여 |
| 음수 | 예측값을 낮추는 방향으로 기여 |
| 0에 가까움 | 해당 예측에서는 영향이 작음 |

예를 들어 이탈 예측에서:

| Feature | SHAP value | 해석 |
|---|---:|---|
| 최근 사용량 감소 | +0.35 | 이탈 위험을 높임 |
| 장기 고객 | -0.20 | 이탈 위험을 낮춤 |
| 불만 접수 많음 | +0.25 | 이탈 위험을 높임 |
| 할인 적용 | -0.05 | 이탈 위험을 조금 낮춤 |

중요한 점은 **이 값은 특정 고객에 대한 값**이라는 것입니다.

`최근 사용량 감소`가 어떤 고객에게는 +0.35일 수 있지만, 다른 고객에게는 효과가 작거나 다르게 나타날 수 있습니다.

---

## Local SHAP과 Global SHAP

SHAP은 local explanation에서 출발하지만, 여러 샘플의 SHAP value를 모으면 global explanation도 만들 수 있습니다.

### Local explanation

한 사람, 한 거래, 한 환자, 한 예측을 설명합니다.

예:

```text
이 고객의 이탈 위험이 높은 이유는 최근 사용량 감소와 불만 접수 때문이다.
```

### Global explanation

전체 데이터에서 feature들이 평균적으로 얼마나 중요한지 봅니다.

보통 이런 식으로 계산합니다.

```text
각 feature의 |SHAP value| 평균
```

즉:

| Feature | 평균 절대 SHAP value |
|---|---:|
| 최근 사용량 | 0.28 |
| 가입기간 | 0.21 |
| 불만 횟수 | 0.18 |
| 월요금 | 0.11 |

이렇게 보면 전체 모델에서 어떤 feature가 많이 영향을 미쳤는지 알 수 있습니다.

---

## SHAP vs Feature Importance

| 구분 | 일반 feature importance | SHAP |
|---|---|---|
| 기본 성격 | 주로 global | local + global 가능 |
| 질문 | 모델 전체에서 어떤 feature가 중요했나? | 이 예측에서 각 feature가 얼마나 기여했나? |
| 방향성 | 보통 없음 | 양수/음수 방향 있음 |
| 개별 샘플 설명 | 어려움 | 가능 |
| feature 값의 효과 | 제한적 | feature 값별 영향 확인 가능 |
| 상호작용 해석 | 제한적 | SHAP interaction으로 확장 가능 |

일반 importance는 “중요도”를 말하고, SHAP은 “이번 예측에 대한 기여도”를 말합니다.

---

## SHAP vs 회귀계수

| 구분 | 회귀계수 | SHAP value |
|---|---|---|
| 대상 | 모델 전체의 변수 효과 | 특정 예측에서 feature 기여 |
| 모델 | 주로 선형/통계 모델 | 복잡한 모델에도 적용 가능 |
| 방향 | 계수 부호로 해석 | SHAP value 부호로 해석 |
| 비선형 효과 | 직접 표현 어려움 | 값 구간별 효과 확인 가능 |
| 상호작용 | 직접 항을 넣어야 함 | tree model의 상호작용도 반영 가능 |

예를 들어 `나이`라는 feature가 있다고 합시다.

선형모델에서는 나이의 영향이 하나의 계수로 표현됩니다.

하지만 XGBoost나 LightGBM에서는 나이의 효과가 비선형일 수 있습니다.

```text
20대에서는 위험 낮음
30대에서는 중간
60대 이상에서는 위험 높음
특정 요금제와 결합되면 효과 달라짐
```

SHAP은 이런 비선형·상호작용 모델에서도 각 샘플별 기여도를 계산할 수 있습니다.

---

## Tree SHAP: XGBoost, LightGBM과의 연결

우리가 직전에 본 모델들이 Random Forest, XGBoost, LightGBM이었습니다.

이 모델들은 모두 tree ensemble입니다.

SHAP에는 tree model을 위한 빠른 방법인 TreeExplainer가 있습니다.

그래서 실무에서는 이런 조합이 매우 흔합니다.

```text
LightGBM으로 예측 모델 학습
→ SHAP으로 예측 이유 설명
```

또는:

```text
XGBoost로 고객 이탈 예측
→ SHAP으로 고객별 이탈 요인 확인
→ 마케팅 액션 설계
```

이게 데이터사이언스 실무에서 SHAP이 많이 쓰이는 이유입니다.

---

## SHAP에서 “expected value”가 중요하다

SHAP 설명에는 보통 `expected_value` 또는 base value가 나옵니다.

이것은 보통:

```text
모델의 평균 출력값
```

처럼 이해하면 됩니다.

그다음 각 feature의 SHAP value를 더하면 개별 예측값이 됩니다.

형태는 이렇습니다.

```text
f(x) = expected_value + sum(SHAP values)
```

이 성질이 SHAP 설명의 핵심입니다.

---

## 주의점 1: SHAP은 모델을 설명하지, 현실을 설명하지 않는다

이 부분이 정말 중요합니다.

SHAP은 다음을 설명합니다.

```text
모델이 왜 이렇게 예측했는가?
```

SHAP이 직접 설명하지 않는 것은 다음입니다.

```text
현실에서 무엇이 진짜 원인인가?
```

예를 들어 이탈 모델에서 `월요금`의 SHAP value가 높게 나왔다고 합시다.

그렇다고 바로 이렇게 말하면 안 됩니다.

> 월요금이 이탈의 원인이다.

정확한 표현은 이것입니다.

> 이 모델은 이 고객의 월요금 값을 이탈 위험을 높이는 신호로 사용했다.

즉, SHAP은 **모델 내부의 예측 논리**를 설명하는 도구이지, 자동 인과추론 도구가 아닙니다.

---

## 주의점 2: feature correlation 문제가 있다

SHAP value는 어떤 feature가 없을 때의 예측을 생각해야 합니다.

그런데 현실에서는 feature들이 서로 상관되어 있습니다.

예를 들어:

| Feature 1 | Feature 2 |
|---|---|
| 연소득 | 직업군 |
| 나이 | 가입기간 |
| 지역 | 주택가격 |
| 최근 사용량 | 최근 로그인 횟수 |

상관된 feature가 많으면, “어떤 feature가 얼마나 기여했는가?”를 나누는 일이 복잡해집니다.

실무적으로는 이 점을 기억해야 합니다.

> SHAP value는 feature 간 상관관계와 background data 선택에 영향을 받을 수 있다.

---

## 주의점 3: output space를 확인해야 한다

분류 모델에서 SHAP 값을 볼 때는 특히 조심해야 합니다.

XGBoost 이진분류 모델을 예로 들면, SHAP 값이 확률 단위가 아니라 **log-odds margin** 단위로 나오는 경우가 많습니다.

쉽게 말하면:

```text
SHAP 값 합 = 확률
```

이라고 무조건 생각하면 안 됩니다.

설명 단위가 다음 중 무엇인지 확인해야 합니다.

| output space | 의미 |
|---|---|
| raw / margin | 모델 내부 점수, log-odds일 수 있음 |
| probability | 확률 |
| log_loss | 샘플별 loss 기여 |

실무에서 SHAP plot을 볼 때는 항상 물어야 합니다.

> 이 SHAP value는 확률 단위인가, log-odds 단위인가?

---

## 주의점 4: SHAP이 높다고 feature를 조작해도 같은 효과가 난다는 뜻은 아니다

예를 들어 이탈 모델에서 `최근 사용량 감소`의 SHAP value가 높게 나왔다고 합시다.

그렇다면 이런 해석은 괜찮습니다.

> 모델은 최근 사용량 감소를 이 고객의 이탈 위험을 높이는 중요한 신호로 보았다.

하지만 이런 해석은 조심해야 합니다.

> 사용량을 강제로 늘리면 이탈이 반드시 줄어든다.

왜냐하면 이것은 인과 개입의 문제입니다.

SHAP은 predictive model explanation입니다. 인과 효과를 보려면 실험, 준실험, causal inference가 필요합니다.

---

## SHAP plot을 어떻게 읽으면 좋나?

실무에서 자주 보는 SHAP 시각화는 대략 다음과 같습니다.

| plot | 질문 |
|---|---|
| Waterfall plot | 이 샘플은 왜 이렇게 예측됐나? |
| Force plot | 어떤 feature가 예측을 밀어 올렸고, 어떤 feature가 낮췄나? |
| Beeswarm summary plot | 전체적으로 어떤 feature가 중요하고, feature 값이 클 때 예측이 올라가나 내려가나? |
| Dependence plot | 특정 feature 값과 SHAP value의 관계는 어떤가? |
| SHAP interaction values | feature 간 상호작용은 어떤가? |

입문자는 먼저 **waterfall plot**과 **beeswarm plot**만 잘 읽어도 충분합니다.

---

## 쉬운 예시: LightGBM 이탈 모델을 SHAP으로 해석

LightGBM으로 고객 이탈 모델을 만들었다고 합시다.

모델 성능은 좋습니다.

```text
AUC = 0.89
```

하지만 마케팅팀은 이렇게 묻습니다.

> 어떤 고객에게 어떤 액션을 해야 하나요?

feature importance만 보면 이런 답이 나올 수 있습니다.

```text
최근 사용량이 가장 중요합니다.
가입기간도 중요합니다.
월요금도 중요합니다.
```

하지만 이건 너무 일반적입니다.

SHAP을 쓰면 고객별로 다르게 볼 수 있습니다.

### 고객 A

| Feature | SHAP 해석 |
|---|---:|
| 최근 사용량 급감 | +0.42 |
| 고객센터 불만 2회 | +0.31 |
| 장기 고객 | -0.18 |
| 할인 적용 | -0.05 |

해석:

> 고객 A는 사용량 급감과 불만 접수가 이탈 위험을 크게 올렸다.  
> 대응은 서비스 문제 해결이나 CS follow-up이 적절할 수 있다.

### 고객 B

| Feature | SHAP 해석 |
|---|---:|
| 월요금 높음 | +0.35 |
| 가입기간 짧음 | +0.28 |
| 사용량 높음 | -0.20 |
| 불만 접수 없음 | -0.10 |

해석:

> 고객 B는 가격 부담과 신규 고객 특성이 위험 신호다.  
> 대응은 온보딩 지원이나 요금제 안내가 적절할 수 있다.

단, SHAP은 모델이 보는 위험 신호를 알려주는 것이지, 어떤 액션이 실제로 효과적인지는 별도 실험이나 인과 분석으로 확인해야 합니다.

---

## SHAP이 Model Cards와 연결되는 방식

Model Card에는 모델의 성능, 한계, 집단별 성능, 사용 범위 등을 문서화해야 합니다.

SHAP은 Model Card에 들어갈 수 있는 중요한 분석 도구입니다.

| Model Card 항목 | SHAP 활용 |
|---|---|
| 주요 feature | 평균 절대 SHAP value로 설명 |
| 집단별 차이 | 집단별 SHAP 분포 비교 |
| 위험한 편향 | 민감정보 proxy feature가 큰 기여를 하는지 확인 |
| 오류 분석 | false positive/false negative 샘플의 SHAP 비교 |
| 사용 주의사항 | 특정 feature 조합에서 모델이 과하게 반응하는지 확인 |

예를 들어 신용평가 모델에서 `우편번호`나 `지역` feature의 SHAP value가 크다면, 소득·인종·사회경제적 지위의 proxy로 작동할 가능성을 점검해야 합니다.

즉, SHAP은 단순한 예쁜 시각화가 아니라 **모델 감사와 문서화에도 쓸 수 있는 도구**입니다.

---

## SHAP의 장점

1. **개별 예측을 설명할 수 있다** — 왜 이 고객, 이 거래, 이 환자가 이런 예측을 받았는지 설명합니다.
2. **방향성을 보여준다** — SHAP value는 양수와 음수를 가지므로 예측값을 높였는지 낮췄는지 알 수 있습니다.
3. **여러 모델에 적용할 수 있다** — linear model, tree model, neural network, black-box model 등 다양한 모델에 적용 가능합니다.
4. **이론적 성질이 명확하다** — local accuracy, missingness, consistency 같은 성질을 만족하는 틀에서 유도됩니다.

---

## SHAP의 한계

1. **계산 비용이 클 수 있다** — 정확한 Shapley value 계산은 feature 조합을 많이 봐야 하므로 비쌉니다.
2. **background data 선택이 중요하다** — 기준 데이터를 어떻게 잡느냐에 따라 SHAP value가 달라질 수 있습니다.
3. **상관된 feature 해석이 어렵다** — feature들이 강하게 상관되어 있으면 기여도를 나누기가 어렵습니다.
4. **인과 설명이 아니다** — 모델이 학습한 패턴을 설명할 뿐 현실의 원인을 증명하지 않습니다.
5. **너무 그럴듯해서 과신할 위험이 있다** — 잘못 학습된 모델의 논리도 그럴듯하게 설명할 수 있습니다.

따라서 SHAP은 다음과 함께 써야 합니다.

| 함께 필요한 것 | 이유 |
|---|---|
| 좋은 평가 설계 | 모델이 실제로 잘 맞는지 확인 |
| 데이터셋 문서화 | 데이터 편향과 수집 맥락 확인 |
| 집단별 성능 평가 | 특정 집단에서 실패하는지 확인 |
| 오류 분석 | 틀린 예측의 패턴 확인 |
| 인과 분석 | 개입 효과를 알고 싶을 때 필요 |

---

## 실무에서 SHAP을 어떻게 쓰면 좋은가?

### 1. 모델 개발 단계

모델이 이상한 feature를 쓰고 있지 않은지 확인합니다.

예:

```text
고객 ID
가입일 timestamp
데이터 수집 후 생성된 변수
타깃 누수 가능성이 있는 변수
```

SHAP을 보면 특정 feature가 비정상적으로 큰 기여를 하는지 알 수 있습니다.

### 2. 모델 개선 단계

어떤 feature가 성능에 기여하는지 보고 feature engineering 방향을 잡습니다.

예:

```text
최근 7일 사용량이 중요하다
→ 최근 1일, 3일, 14일 사용량도 만들어볼까?
```

### 3. 운영 단계

개별 예측의 이유를 설명합니다.

예:

```text
이 거래가 사기로 분류된 이유:
- 해외 결제
- 평소보다 큰 금액
- 새 기기 사용
- 새벽 시간 결제
```

### 4. 모델 감사 단계

집단별로 SHAP 분포를 비교합니다.

예:

```text
특정 지역 고객에게 특정 feature가 과도하게 불리하게 작동하는가?
특정 연령대에서 모델이 다른 근거로 판단하는가?
민감정보 proxy가 큰 영향을 미치는가?
```

---

## 이전 논문들과 연결해서 이해하기

| 이전 논문 | SHAP과의 연결 |
|---|---|
| **Tidy Data** | feature가 제대로 정의되어야 SHAP 해석도 의미 있음 |
| **Two Cultures** | SHAP은 algorithmic model의 예측을 해석하는 도구 |
| **Model Evaluation** | SHAP이 있어도 test 성능과 leakage 검증은 필요 |
| **Datasheets** | 데이터셋의 출처와 한계를 모르면 SHAP 해석도 위험함 |
| **Model Cards** | SHAP 분석은 Model Card의 설명 가능성·위험 분석에 활용 가능 |
| **Random Forests** | tree ensemble의 개별 예측을 설명하는 데 SHAP 사용 가능 |
| **Gradient Boosting** | boosting 모델의 복잡한 누적 예측을 feature별로 분해 |
| **XGBoost** | Tree SHAP으로 빠르게 설명 가능 |
| **LightGBM** | 대규모 GBDT 모델 해석에 자주 사용 |

---

## 이 논문이 주는 진짜 메시지

이 논문의 진짜 메시지는 이겁니다.

**모델 설명은 감각적으로 그럴듯한 설명을 만드는 문제가 아니라, 예측값을 feature별 기여도로 일관되게 배분하는 문제다.**

SHAP은 이 문제를 게임이론의 Shapley value로 풉니다.

그래서 SHAP은 단순한 “feature importance 그래프”보다 더 깊은 의미가 있습니다.

```text
예측값을 어떻게 공정하게 분해할 것인가?
어떤 설명 방법이 일관성 있는가?
local explanation과 feature attribution을 어떻게 통합할 것인가?
```

이 질문에 답하는 논문입니다.

---

## 입문자가 꼭 기억해야 할 문장

**SHAP은 모델의 평균 예측값에서 출발해, 각 feature가 특정 예측값을 얼마나 올리거나 내렸는지를 Shapley value 기반으로 계산하는 설명 방법이다.**

더 짧게 말하면:

> **SHAP = 개별 예측값을 feature별 기여도의 합으로 분해하는 방법**

---

## 오늘 공부용 요약

**논문명:** A Unified Approach to Interpreting Model Predictions  
**저자:** Scott M. Lundberg, Su-In Lee  
**핵심 결론:** SHAP은 복잡한 모델의 개별 예측을 baseline과 feature별 기여도의 합으로 설명하는 통합 프레임워크다. 각 feature의 기여도는 게임이론의 Shapley value에 기반하며, additive feature attribution 방법 중 local accuracy, missingness, consistency를 만족하는 유일한 해로 제시된다.  
**왜 중요함:** XGBoost, LightGBM, Random Forest 같은 고성능 모델은 예측력은 좋지만 해석이 어렵다. SHAP은 이런 모델의 개별 예측 이유를 feature별로 분해해 설명할 수 있게 해준다.  
**입문자가 배울 점:** feature importance와 SHAP은 다르다. feature importance는 보통 전체 모델에서 어떤 변수가 중요한지 말하지만, SHAP은 특정 예측에서 각 feature가 예측값을 얼마나 올리거나 내렸는지 말한다.  
**가장 중요한 주의점:** SHAP은 모델의 예측 논리를 설명하는 도구이지, 현실의 인과관계를 증명하는 도구는 아니다.


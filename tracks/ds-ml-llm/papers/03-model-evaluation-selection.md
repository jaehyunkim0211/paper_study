# Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning — Sebastian Raschka

## 결론부터

이 논문의 결론은 이겁니다.

**머신러닝에서 좋은 모델을 고르려면 “성능을 재는 일”, “모델을 고르는 일”, “알고리즘을 비교하는 일”을 구분해야 한다. 그리고 최종 test set은 모델을 고르는 데 쓰면 안 된다. test set은 마지막에 딱 한 번, 일반화 성능을 추정하기 위해 써야 한다.**

쉽게 말하면:

> 모델을 잘 만드는 것만큼 중요한 것은  
> **모델을 공정하게 평가하는 것**이다.

Raschka의 논문은 모델 평가, 모델 선택, 알고리즘 선택에서 쓰이는 holdout, bootstrap, k-fold cross-validation, nested cross-validation, 통계적 검정, 다중비교 보정 등을 정리한 리뷰 논문입니다.

---

## 이 논문을 한 문장으로 요약하면

**훈련 데이터에서 잘 맞는 모델이 아니라, 보지 못한 데이터에서도 잘 맞을 모델을 골라야 한다.**

Breiman 논문이 **철학**이라면, Raschka 논문은 **실전 평가 방법론**입니다.

Breiman은 이렇게 말했습니다.

> “예측 문제에서는 예측 성능을 봐야 한다.”

Raschka는 그다음 질문에 답합니다.

> “그럼 예측 성능을 어떻게 제대로 측정할 것인가?”

---

## 핵심 구분 1: Model Evaluation

**Model evaluation은 이미 정해진 모델이 새 데이터에서 얼마나 잘할지 추정하는 일입니다.**

예를 들어 이미 다음 모델을 정했다고 해봅시다.

> Logistic Regression, C=1.0, penalty=L2

이제 할 일은:

> 이 모델이 미래 고객 데이터에서도 고객 이탈을 잘 예측할까?

를 평가하는 것입니다.

이때 중요한 것은 훈련 데이터 성능이 아닙니다.

| 평가 위치 | 의미 | 문제 |
|---|---|---|
| training set 성능 | 모델이 학습한 데이터에서의 성능 | 과대평가되기 쉬움 |
| validation set 성능 | 모델 선택 과정에서 참고하는 성능 | 여러 번 보면 선택 편향 발생 |
| test set 성능 | 최종 일반화 성능 추정 | 마지막에만 써야 함 |

우리가 원하는 것은 미래의 unseen data에서의 성능, 즉 generalization performance입니다.

---

## 핵심 구분 2: Model Selection

**Model selection은 같은 알고리즘 안에서 가장 좋은 설정의 모델을 고르는 일입니다.**

예를 들어 XGBoost를 쓴다고 정했다고 해봅시다.

그 안에서도 선택할 것이 많습니다.

| 하이퍼파라미터 | 예시 |
|---|---|
| learning_rate | 0.01, 0.05, 0.1 |
| max_depth | 3, 5, 7 |
| n_estimators | 100, 500, 1000 |
| subsample | 0.7, 0.9, 1.0 |
| regularization | 약하게, 강하게 |

이 중 어떤 조합이 좋은지 고르는 것이 model selection입니다.

이때 validation set이나 cross-validation을 씁니다.

중요한 원칙은 이것입니다.

> **test set을 보면서 하이퍼파라미터를 고르면 안 된다.**

왜냐하면 test set을 여러 번 보면, 모델이 test set에 맞춰져 버립니다. 이건 일종의 **test set overfitting**입니다.

겉으로는 test 성능이 좋아 보이지만, 진짜 미래 데이터에서는 성능이 떨어질 수 있습니다.

---

## 핵심 구분 3: Algorithm Selection

**Algorithm selection은 서로 다른 알고리즘 중 어떤 계열을 쓸지 고르는 일입니다.**

예를 들어 고객 이탈 예측 문제에서 다음 후보들이 있다고 합시다.

| 후보 | 알고리즘 계열 |
|---|---|
| Logistic Regression | 선형 모델 |
| Random Forest | 트리 앙상블 |
| XGBoost | 부스팅 |
| SVM | 커널 기반 모델 |
| Neural Network | 딥러닝 |

이 중 어떤 알고리즘이 이 문제에 가장 적합한지 비교하는 것이 algorithm selection입니다.

Raschka는 모델 성능을 평가하는 이유를 세 가지로 나눕니다. 첫째, 미래 데이터에 대한 일반화 성능을 추정하기 위해서, 둘째, 하이퍼파라미터를 조정해 더 좋은 모델을 고르기 위해서, 셋째, 문제에 가장 적합한 알고리즘을 찾기 위해서입니다.

---

## 세 개를 비교하면

| 구분 | 질문 | 예시 |
|---|---|---|
| **Model Evaluation** | 이 최종 모델이 새 데이터에서 얼마나 잘할까? | 최종 XGBoost 모델의 test AUC 평가 |
| **Model Selection** | 같은 알고리즘 안에서 어떤 설정이 제일 좋을까? | XGBoost의 max_depth, learning_rate 선택 |
| **Algorithm Selection** | 어떤 알고리즘 계열이 이 문제에 맞을까? | Logistic Regression vs Random Forest vs XGBoost 비교 |

이 세 가지를 섞으면 평가가 오염됩니다.

특히 입문자가 자주 하는 실수가 있습니다.

> test set으로 모델도 고르고, 하이퍼파라미터도 고르고, 최종 성능도 보고한다.

이러면 test set은 더 이상 순수한 test set이 아닙니다.

---

## 쉬운 예시: 고객 이탈 예측

고객 이탈 데이터를 가지고 있다고 해봅시다.

목표는 다음 달 이탈 고객을 예측하는 것입니다.

잘못된 방식은 이렇습니다.

1. 전체 데이터를 train/test로 나눈다.
2. 여러 모델을 test set에 계속 돌려본다.
3. test AUC가 가장 높은 모델을 고른다.
4. 그 test AUC를 최종 성능이라고 보고한다.

이 방식은 위험합니다.

왜냐하면 test set을 모델 선택에 써버렸기 때문입니다.

좋은 방식은 이렇습니다.

1. 데이터를 train / validation / test로 나눈다.
2. train으로 학습한다.
3. validation으로 모델과 하이퍼파라미터를 고른다.
4. 선택이 끝난 후 test set을 딱 한 번 사용한다.
5. test 성능을 최종 일반화 성능 추정치로 보고한다.

이게 이 논문의 핵심 감각입니다.

---

## 핵심 개념 1: Generalization Performance

**머신러닝의 진짜 목표는 training performance가 아니라 generalization performance입니다.**

모델은 이미 본 데이터를 외울 수 있습니다.

예를 들어 decision tree를 깊게 만들면 훈련 데이터를 거의 완벽히 맞출 수 있습니다.

| 데이터 | 성능 |
|---|---:|
| train accuracy | 99% |
| test accuracy | 72% |

이런 경우 모델이 좋은 게 아닙니다. 훈련 데이터를 외운 것입니다.

그래서 우리는 항상 이렇게 물어야 합니다.

> 이 모델이 처음 보는 데이터에서도 잘 작동할까?

이 질문이 generalization performance의 핵심입니다.

---

## 핵심 개념 2: Holdout Method

**Holdout은 데이터를 train/test 또는 train/validation/test로 한 번 나누는 방식입니다.**

가장 단순한 방식입니다.

예를 들어 데이터가 10만 건이라면:

| split | 용도 |
|---|---|
| train | 모델 학습 |
| validation | 모델 선택, 하이퍼파라미터 튜닝 |
| test | 최종 평가 |

데이터가 충분히 크면 이 방식이 실용적입니다.

하지만 데이터가 작으면 문제가 생깁니다. 예를 들어 데이터가 300개뿐인데 100개를 test로 빼면 train 데이터가 너무 줄어듭니다. 반대로 test를 너무 작게 잡으면 평가가 불안정합니다.

그래서 작은 데이터에서는 cross-validation이 더 유용합니다.

---

## 핵심 개념 3: k-fold Cross-Validation

**k-fold cross-validation은 데이터를 k개로 나눈 뒤, 각 조각을 한 번씩 validation으로 사용해 평균 성능을 구하는 방법입니다.**

예를 들어 5-fold cross-validation이면:

| 라운드 | train | validation |
|---|---|---|
| 1 | fold 2,3,4,5 | fold 1 |
| 2 | fold 1,3,4,5 | fold 2 |
| 3 | fold 1,2,4,5 | fold 3 |
| 4 | fold 1,2,3,5 | fold 4 |
| 5 | fold 1,2,3,4 | fold 5 |

그리고 5번의 성능을 평균냅니다.

### k는 몇으로 해야 할까?

입문에서는 보통 이렇게 기억하면 됩니다.

| 방법 | 장점 | 단점 |
|---|---|---|
| 2-fold CV | 빠름 | 불안정할 수 있음 |
| 5-fold CV | 계산량 적당 | 데이터가 작으면 다소 불안정 |
| 10-fold CV | 실무에서 자주 쓰는 균형점 | 5-fold보다 계산량 증가 |
| LOOCV | 데이터를 거의 전부 학습에 사용 | 계산량 큼, 분산이 클 수 있음 |

실무 초보자 기준으로는 이렇게 가면 됩니다.

> 데이터가 작거나 중간 규모면 **stratified 5-fold 또는 10-fold CV**를 먼저 고려한다.  
> 데이터가 매우 크면 **train/validation/test split**도 충분히 실용적이다.

---

## 핵심 개념 4: Data Leakage

**Data leakage는 모델이 학습 단계에서 보면 안 되는 정보를 몰래 보게 되는 문제입니다.**

대표적인 실수가 있습니다.

> 전체 데이터에 대해 scaling이나 feature selection을 먼저 하고, 그다음 cross-validation을 한다.

예를 들어 전체 데이터로 평균과 표준편차를 계산해 standardization을 해버리면, validation fold의 정보가 train fold에 스며듭니다.

잘못된 방식:

```text
전체 데이터 scaling
→ k-fold split
→ train / validation
→ 성능 평가
```

올바른 방식:

```text
k-fold split
→ 각 fold의 train 데이터로만 scaling 기준 학습
→ validation 데이터에는 그 기준만 적용
→ 성능 평가
```

feature selection도 마찬가지입니다.

특히 다음 작업들은 반드시 CV loop 안에 들어가야 합니다.

| 작업 | 이유 |
|---|---|
| scaling / normalization | validation 데이터의 분포 정보를 미리 보면 안 됨 |
| imputation | 결측치 대체 기준을 train에서만 배워야 함 |
| feature selection | 타깃과 관련된 변수를 전체 데이터로 고르면 leakage 발생 |
| PCA | 전체 데이터 구조를 미리 보면 안 됨 |
| oversampling, SMOTE | validation 데이터까지 섞이면 평가 오염 |

scikit-learn에서는 그래서 `Pipeline`을 씁니다.

---

## 핵심 개념 5: Nested Cross-Validation

**Nested cross-validation은 모델 선택과 성능 평가를 분리하기 위해 cross-validation을 두 겹으로 쓰는 방식입니다.**

| 루프 | 역할 |
|---|---|
| inner CV | 하이퍼파라미터 선택 |
| outer CV | 선택된 모델의 일반화 성능 평가 |

왜 두 겹이 필요할까요?

모델을 고르는 과정 자체도 데이터에 맞춰지는 과정이기 때문입니다.

예를 들어 100개의 하이퍼파라미터 조합을 비교하면, validation 성능이 가장 높은 조합은 진짜로 좋은 조합일 수도 있지만, 우연히 validation fold에 잘 맞은 조합일 수도 있습니다.

그래서 작은 데이터에서 알고리즘을 공정하게 비교하려면 nested CV가 유용합니다.

Nested CV는 단일 모델만 평가하는 것이 아니라, **모델을 고르는 전체 절차**를 평가합니다.

---

## p-value와 연결되는 부분

모델 A와 모델 B가 있다고 해봅시다.

| 모델 | accuracy |
|---|---:|
| Logistic Regression | 82.1% |
| Random Forest | 84.0% |

이때 바로 이렇게 말하면 안 됩니다.

> Random Forest가 더 좋다.

왜냐하면 1.9%p 차이가 우연일 수도 있기 때문입니다.

그래서 다음 질문이 필요합니다.

> 이 성능 차이는 통계적으로 유의한가?

이때 McNemar test, paired t-test, 5x2cv test 같은 검정 방법이 등장합니다.

하지만 p-value가 작다고 해서 그 모델이 실무적으로 훨씬 좋다는 뜻은 아닙니다. 큰 표본에서는 p-value가 작아져 모든 것이 통계적으로 유의해 보일 수 있기 때문에 effect size도 고려해야 합니다.

---

## 다중검정 문제도 다시 등장한다

모델을 여러 개 비교하면 다중검정 문제가 생깁니다.

예를 들어 다음 6개 모델을 비교한다고 해봅시다.

| 모델 |
|---|
| Logistic Regression |
| Decision Tree |
| Random Forest |
| XGBoost |
| SVM |
| Neural Network |

모델 6개를 pairwise로 전부 비교하면 검정 횟수가 늘어납니다. 그러면 우연히 p < 0.05가 나오는 비교가 생길 가능성도 커집니다.

여러 모델을 비교할 때는 다음을 조심해야 합니다.

| 문제 | 대응 |
|---|---|
| 여러 모델을 계속 비교 | 다중비교 보정 고려 |
| 여러 지표 중 좋은 것만 보고 | 주요 지표를 사전에 정함 |
| 여러 split 중 유리한 split만 선택 | 반복 CV 또는 nested CV 사용 |
| p-value만 보고 판단 | effect size와 실무적 비용도 함께 봄 |

---

## 이 논문에서 입문자가 꼭 가져가야 할 원칙

### 1. Training score를 믿지 마라

훈련 데이터 성능은 거의 항상 낙관적입니다.

| 모델 | train accuracy | test accuracy |
|---|---:|---:|
| 깊은 decision tree | 99% | 72% |
| regularized logistic regression | 83% | 81% |

첫 번째 모델은 train에서는 좋아 보이지만, 일반화 성능은 낮습니다.

### 2. Test set은 마지막까지 숨겨라

test set은 시험지입니다.

시험지를 보면서 공부하면, 시험 성적은 좋아 보일 수 있습니다. 하지만 진짜 실력은 아닙니다.

머신러닝에서도 같습니다.

> test set은 모델 선택 과정에 쓰면 안 된다.

### 3. Validation과 test를 구분하라

| 데이터 | 역할 |
|---|---|
| train | 모델 학습 |
| validation | 모델 선택, 하이퍼파라미터 튜닝 |
| test | 최종 성능 평가 |

validation은 여러 번 봐도 됩니다. test는 마지막에만 봐야 합니다.

### 4. 작은 데이터에서는 cross-validation을 써라

데이터가 작을 때 한 번의 train/test split은 운에 너무 민감합니다. 어떤 샘플이 test로 들어갔는지에 따라 성능이 크게 달라질 수 있습니다.

### 5. 전처리는 CV 안에서 해라

잘못된 방식:

```text
전체 데이터로 scaling
전체 데이터로 feature selection
그다음 cross-validation
```

올바른 방식:

```text
각 fold마다 train 부분에서만 scaling 학습
각 fold마다 train 부분에서만 feature selection
validation에는 transform만 적용
```

### 6. 모델 비교에는 불확실성이 있다

AUC가 0.842인 모델이 AUC 0.837인 모델보다 항상 좋은 것은 아닙니다.

그 차이가:

| 가능성 |
|---|
| 실제 성능 차이일 수도 있음 |
| split 운 때문일 수도 있음 |
| 표본 수가 작아서 생긴 흔들림일 수도 있음 |
| 특정 validation set에 과적합된 결과일 수도 있음 |

그래서 성능 평균뿐 아니라 표준편차, 신뢰구간, 반복 CV, 통계적 검정, effect size를 함께 봐야 합니다.

---

## 실무에서 추천하는 기본 워크플로우

### 데이터가 충분히 클 때

```text
1. train / validation / test split
2. train으로 학습
3. validation으로 모델 선택
4. 선택 완료 후 train + validation으로 재학습
5. test로 최종 평가
6. 배포 시 필요하면 전체 데이터로 최종 재학습
```

### 데이터가 작거나 중간 규모일 때

```text
1. 독립 test set을 따로 보관
2. train 데이터 안에서 k-fold CV로 모델 선택
3. 선택된 설정으로 train 전체 재학습
4. test set으로 최종 평가
```

### 데이터가 매우 작고 알고리즘 비교가 중요할 때

```text
1. nested cross-validation 사용
2. inner loop에서 하이퍼파라미터 선택
3. outer loop에서 일반화 성능 추정
4. 알고리즘 간 평균 성능과 분산 비교
```

---

## 이 논문을 읽을 때 중요한 질문

이 논문을 읽으면서 계속 물어야 할 질문은 이것입니다.

**“내가 지금 보고 있는 성능은 모델 선택에 사용된 성능인가, 아니면 최종 일반화 성능인가?”**

이 질문을 놓치면 성능을 과대평가하기 쉽습니다.

| 상황 | 위험 |
|---|---|
| validation 성능을 최종 성능처럼 보고 | 선택 편향 발생 |
| test set으로 하이퍼파라미터 튜닝 | test leakage |
| 전체 데이터로 feature selection 후 CV | data leakage |
| 여러 모델 중 최고 성능만 보고 | multiple comparison 문제 |
| 한 번의 split 결과만 믿음 | split variance 문제 |

---

## 오늘 공부용 요약

**논문명:** Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning  
**저자:** Sebastian Raschka  
**핵심 결론:** 모델 성능 평가는 단순히 accuracy를 계산하는 일이 아니라, 일반화 성능을 공정하게 추정하는 절차다. model evaluation, model selection, algorithm selection을 구분해야 하며, test set은 최종 평가에만 사용해야 한다.  
**왜 중요함:** 머신러닝 실험에서 가장 흔한 실수인 overfitting, test leakage, data leakage, 잘못된 모델 비교를 막아준다.  
**입문자가 배울 점:** 좋은 모델을 고르는 것보다 먼저, 좋은 평가 설계를 해야 한다.  
**가장 중요한 문장:** validation set은 모델을 고르기 위한 데이터이고, test set은 선택이 끝난 모델을 평가하기 위한 데이터다.


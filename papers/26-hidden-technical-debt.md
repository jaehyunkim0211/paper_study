# Hidden Technical Debt in Machine Learning Systems

## 결론부터

이 논문의 결론은 이겁니다.

**머신러닝 시스템에서 가장 위험한 비용은 모델 학습 코드 자체가 아니라, 그 모델을 실제 서비스에 붙이고 운영하면서 생기는 숨은 유지보수 비용이다.**

모델 하나를 빠르게 만들어 성능을 올리는 것은 쉬워 보일 수 있지만, 시간이 지나면 데이터 의존성, 파이프라인, 설정, 피드백 루프, 모니터링, 실험 코드, 외부 환경 변화 때문에 시스템 전체가 점점 고치기 어려워질 수 있습니다.

한 줄로 말하면:

> **ML 시스템의 진짜 어려움은 모델을 만드는 것이 아니라, 모델이 들어간 전체 시스템을 오래 건강하게 유지하는 것이다.**

---

## 이 논문을 한 문장으로 요약하면

**ML 모델은 시스템의 작은 일부일 뿐이고, 실제 비용은 그 모델 주변의 데이터·코드·설정·운영 인프라에서 폭발한다.**

실제 ML 시스템은 대략 이렇게 구성됩니다.

```text
전체 ML 시스템
├── 데이터 수집
├── 데이터 검증
├── feature extraction
├── 실험 관리
├── 학습 인프라
├── 모델 검증
├── serving infrastructure
├── monitoring
├── configuration
├── resource management
└── ML code, 아주 작은 부분
```

데이터사이언스 입문자는 보통 이렇게 생각합니다.

```text
좋은 모델을 만들면 끝난다.
```

하지만 이 논문은 이렇게 말합니다.

```text
좋은 모델을 만든 순간부터 진짜 시스템 문제가 시작된다.
```

---

## 왜 이 논문이 중요한가?

지금까지 우리는 모델 중심 논문을 많이 봤습니다.

| 논문 | 핵심 |
|---|---|
| Random Forest | 여러 tree를 평균내 안정적인 예측 |
| XGBoost / LightGBM | 강력한 tabular prediction model |
| Deep Learning / ResNet | 깊은 neural network 학습 |
| Transformer / BERT / GPT | 대규모 언어 모델 |
| LoRA / QLoRA | 큰 모델을 효율적으로 fine-tuning |

이 논문은 관점을 바꿉니다.

> 모델 성능이 좋다고 해서 시스템이 좋은 것은 아니다.

예를 들어 validation AUC가 0.91인 모델이 있다고 합시다.

겉으로는 훌륭합니다.

하지만 다음 문제가 있을 수 있습니다.

```text
feature가 어디서 오는지 아무도 모름
어떤 팀이 모델 output을 쓰는지 모름
학습 데이터와 서빙 데이터 전처리가 다름
실험용 코드가 production에 남아 있음
설정 파일이 너무 복잡해 한 줄 변경도 위험함
모델이 사용자 행동을 바꿔서 다음 학습 데이터 분포를 바꿈
외부 세상이 바뀌었는데 monitoring이 없음
```

이 경우 모델 성능이 좋아도 시스템은 취약합니다.

---

## 핵심 개념 1: Technical Debt

**Technical debt는 빠르게 만들기 위해 지금은 편한 선택을 했지만, 나중에 유지보수 비용으로 되돌아오는 빚입니다.**

예를 들어 코드에서 이런 일이 있습니다.

```text
일단 하드코딩하자.
나중에 리팩토링하자.
테스트는 다음에 쓰자.
문서는 나중에 정리하자.
```

당장은 빠릅니다.

하지만 시간이 지나면:

```text
고치기 어려움
버그가 많아짐
새 기능 추가가 느려짐
새 팀원이 이해하기 어려움
```

이 됩니다.

---

## ML technical debt는 왜 더 위험한가?

일반 소프트웨어의 기술부채는 주로 코드에서 보입니다.

```text
복잡한 함수
중복 코드
부실한 테스트
엉킨 의존성
```

하지만 ML technical debt는 코드 밖에도 많습니다.

```text
데이터 의존성
feature 의존성
모델 output 의존성
training-serving skew
외부 세계 변화
feedback loop
configuration explosion
실험 코드 잔재
```

즉, **보이지 않는 부채**가 많습니다.

코드를 아무리 깨끗하게 짜도, 데이터 파이프라인과 시스템 의존성이 엉켜 있으면 전체 ML 시스템은 유지보수하기 어렵습니다.

---

## 핵심 개념 2: Boundary Erosion

**ML 시스템에서는 전통적인 소프트웨어처럼 깔끔한 모듈 경계를 유지하기 어렵습니다.**

일반 소프트웨어에서는 모듈 경계를 잘 나누면 유지보수가 쉬워집니다.

```text
A 모듈은 입력 x를 받아 출력 y를 낸다.
내부 구현은 감춘다.
다른 모듈은 API만 알면 된다.
```

하지만 ML에서는 데이터가 모델 행동을 결정합니다.

예를 들어 추천 모델이 있습니다.

```text
입력 feature:
- 사용자 나이
- 클릭 이력
- 최근 구매
- 할인 반응
- 검색어
```

이 feature들은 서로 독립이 아닙니다.

하나의 feature가 바뀌면 다른 feature의 의미도 바뀔 수 있습니다.

---

## 핵심 개념 3: Entanglement와 CACE

**ML 모델에서는 하나를 바꾸면 전체가 바뀔 수 있습니다.**

이 논문에서 가장 유명한 표현 중 하나가 **CACE**입니다.

```text
CACE = Changing Anything Changes Everything
```

즉:

> 무엇이든 하나를 바꾸면 모든 것이 바뀔 수 있다.

예를 들어 고객 이탈 모델이 있다고 합시다.

```text
feature:
가입기간
월요금
최근 사용량
고객센터 문의 수
할인 여부
결제 실패 여부
```

여기서 `최근 사용량` feature 계산 방식을 바꿨다고 합시다.

예전에는 30일 사용량이었습니다.

```text
최근 사용량 = 최근 30일 사용량
```

이제 7일 사용량으로 바꿉니다.

```text
최근 사용량 = 최근 7일 사용량
```

단순히 feature 하나만 바꾼 것처럼 보입니다.

하지만 실제로는:

```text
다른 feature의 상대적 중요도가 바뀜
모델 weight가 바뀜
threshold가 바뀜
캠페인 대상자가 바뀜
사용자 행동이 바뀜
다음 학습 데이터가 바뀜
```

까지 이어질 수 있습니다.

---

## 핵심 개념 4: Correction Cascades

**Correction cascade는 기존 모델 위에 보정 모델을 계속 쌓아 만드는 위험한 구조입니다.**

예를 들어 기존 모델이 있습니다.

```text
모델 A: 일반 고객 이탈 예측
```

그런데 VIP 고객에서는 성능이 조금 안 좋습니다.

빠른 해결책으로 이런 모델을 만듭니다.

```text
모델 A' = 모델 A의 예측값을 입력으로 받아 VIP 고객용 보정
```

이제 다른 세그먼트에서도 문제가 생깁니다.

```text
모델 A'' = 모델 A' 위에 또 보정
```

처음에는 빠릅니다.

하지만 시간이 지나면:

```text
모델 A를 개선하면 A'가 깨짐
A'를 개선하면 A''가 깨짐
어느 모델을 고쳐야 하는지 모름
전체 시스템 분석이 어려움
```

이 됩니다.

더 나은 대안은 원래 모델이 correction까지 직접 배우도록 feature를 추가하거나, 문제별 별도 모델 비용을 명시적으로 받아들이는 것입니다.

---

## 핵심 개념 5: Undeclared Consumers

**모델 output을 누가 쓰는지 모르면, 모델을 바꾸는 순간 어디가 깨질지 알 수 없습니다.**

예를 들어 추천 점수 모델이 있습니다.

```text
recommendation_score
```

처음에는 추천 랭킹에만 쓰였습니다.

그런데 시간이 지나면서 다른 팀들이 이 값을 가져다 씁니다.

```text
마케팅팀: 쿠폰 대상 선정에 사용
CS팀: 고객 위험도 분류에 사용
영업팀: 우선 연락 고객 선정에 사용
데이터팀: 다른 모델 feature로 사용
```

문제는 모델 담당팀이 이 사실을 모른다는 것입니다.

어느 날 모델 담당팀이 추천 점수 계산 방식을 개선합니다.

```text
recommendation_score v2 배포
```

추천 랭킹은 좋아졌습니다.

하지만 마케팅, CS, 영업, 다른 모델이 갑자기 이상해질 수 있습니다.

해결하려면 output 접근 제어, 명확한 API, SLA, dependency tracking, output versioning, deprecation policy가 필요합니다.

---

## 핵심 개념 6: Data Dependencies

**ML 시스템에서 데이터 의존성은 코드 의존성보다 더 위험하고 찾기 어렵습니다.**

코드 의존성은 비교적 찾기 쉽습니다.

```text
import package
function call
library dependency
```

정적 분석 도구로 추적할 수 있습니다.

하지만 데이터 의존성은 더 어렵습니다.

```text
이 feature가 어디서 왔는가?
누가 생성했는가?
언제 바뀌는가?
다른 모델 output인가?
어떤 preprocessing을 거쳤는가?
어떤 팀이 소유하는가?
```

### Unstable Data Dependencies

다른 시스템이 만든 feature를 가져다 쓰면, 그 시스템이 바뀌는 순간 내 모델도 깨질 수 있습니다.

예를 들어 내 모델이 이런 feature를 씁니다.

```text
user_interest_score
```

이 feature는 다른 팀의 모델이 생성합니다.

그 팀이 어느 날 모델을 개선합니다.

```text
user_interest_score 계산 방식 변경
```

그 팀 입장에서는 개선입니다.

하지만 내 모델은 예전 score 분포에 맞춰 학습되어 있었습니다.

그러면 갑자기 내 모델 성능이 떨어질 수 있습니다.

### Underutilized Data Dependencies

별 도움이 안 되는 feature도 유지보수 비용과 장애 위험을 만듭니다.

어떤 feature가 초기에 유용해서 모델에 들어갔습니다.

```text
legacy_feature
```

나중에 더 좋은 feature들이 생기면서 이 feature는 거의 필요 없어졌습니다.

하지만 아무도 제거하지 않았습니다.

문제는 이 feature가 계속 dependency로 남아 있다는 것입니다.

```text
feature pipeline 유지 필요
데이터 품질 monitoring 필요
upstream 변경 영향 가능
문서화 필요
```

---

## 핵심 개념 7: Hidden Feedback Loops

**모델의 예측이 현실 행동을 바꾸고, 그 바뀐 현실이 다시 모델 학습 데이터로 들어오면 피드백 루프가 생깁니다.**

예를 들어 추천 시스템을 생각해봅시다.

```text
모델이 상품 A를 더 많이 추천
→ 사용자들이 상품 A를 더 많이 클릭
→ 다음 학습 데이터에서 상품 A 클릭이 더 많아짐
→ 모델은 상품 A가 더 좋다고 학습
→ 더 많이 추천
```

이건 직접적인 feedback loop입니다.

더 어려운 것은 hidden feedback loop입니다.

예를 들어 웹페이지에 두 시스템이 있습니다.

```text
시스템 1: 어떤 상품을 보여줄지 결정
시스템 2: 어떤 리뷰를 보여줄지 결정
```

시스템 1을 개선했더니 사용자가 다른 상품을 더 많이 클릭합니다.

그 결과 리뷰 클릭 패턴도 바뀝니다.

그러면 시스템 2의 데이터도 바뀝니다.

두 시스템은 직접 연결되어 있지 않지만, 사용자 행동을 매개로 서로 영향을 줍니다.

---

## 핵심 개념 8: Glue Code

**ML 패키지 자체보다, 그 패키지에 데이터를 넣고 결과를 꺼내기 위한 glue code가 훨씬 커질 수 있습니다.**

예를 들어 XGBoost를 쓴다고 합시다.

모델 코드는 간단합니다.

```python
model.fit(X_train, y_train)
```

하지만 실제 시스템에서는:

```text
데이터 추출
join
feature 생성
결측 처리
schema 변환
train/serve 포맷 맞추기
모델 저장
서빙 API
모니터링
로그 적재
A/B 테스트
rollback
```

이 필요합니다.

이 주변 코드가 엄청나게 커집니다.

---

## 핵심 개념 9: Pipeline Jungles

**데이터 전처리 파이프라인을 조금씩 덧붙이다 보면, 아무도 전체를 이해하지 못하는 정글이 됩니다.**

처음에는 간단합니다.

```text
raw data → feature table
```

그러다 feature가 하나씩 추가됩니다.

```text
raw logs
→ user table join
→ product table join
→ click aggregation
→ missing value handling
→ sampling
→ filtering
→ label generation
→ intermediate file
→ 또 join
→ 또 filtering
```

시간이 지나면:

```text
어느 단계가 왜 있는지 모름
중간 파일이 너무 많음
실패 시 복구 어려움
end-to-end 테스트만 가능
새 feature 추가가 위험함
```

이 됩니다.

---

## 핵심 개념 10: Dead Experimental Codepaths

**실험용 분기 코드가 production에 계속 남으면, 시스템 복잡도가 폭발합니다.**

ML에서는 실험을 많이 합니다.

```text
if use_new_feature:
    ...
if use_model_v2:
    ...
if experiment_id == 37:
    ...
```

처음에는 편합니다.

하지만 시간이 지나면:

```text
어떤 분기가 아직 쓰이는지 모름
테스트 조합이 폭발함
낡은 실험 코드가 장애를 만듦
복잡도 증가
```

이 됩니다.

성공한 실험은 정식 코드로 통합하고, 실패한 실험은 삭제해야 합니다.

---

## 핵심 개념 11: Configuration Debt

**ML 시스템에서는 코드보다 설정 파일이 더 위험해질 수 있습니다.**

ML 시스템은 수많은 설정값을 가집니다.

```text
feature list
feature transformation
model hyperparameter
training data window
sampling rate
threshold
calibration setting
serving version
A/B test flag
resource limit
```

이 설정들이 많아지면 사실상 코드처럼 됩니다.

하지만 설정은 종종 코드만큼 철저히 테스트하거나 리뷰하지 않습니다.

예를 들어 threshold 하나를 바꿉니다.

```text
fraud_score > 0.7 → 사기 의심
```

이걸 0.6으로 낮춥니다.

겉보기에는 단순 설정 변경입니다.

하지만 영향은 큽니다.

```text
탐지 건수 증가
운영팀 workload 증가
false positive 증가
고객 불만 증가
downstream system 부하 증가
```

그래서 설정 변경도 코드 변경처럼 다뤄야 합니다.

---

## 핵심 개념 12: External World Changes

**ML 모델은 외부 세계가 바뀌면 자연스럽게 낡습니다.**

모델은 과거 데이터로 학습합니다.

하지만 현실은 변합니다.

```text
사용자 행동 변화
시장 변화
계절성
법/정책 변화
경쟁사 변화
제품 UI 변화
데이터 수집 방식 변화
스팸/사기 패턴 변화
```

이걸 보통 data drift, concept drift라고 부릅니다.

예를 들어 사기 탐지 모델이 있습니다.

과거 사기 패턴을 잘 잡았습니다.

그런데 사기꾼들이 모델을 우회하기 시작합니다.

```text
과거 패턴 A → 모델이 잘 탐지
새 패턴 B → 모델이 놓침
```

외부 세계가 바뀐 것입니다.

ML 시스템에는 지속적인 관찰이 필요합니다.

```text
input distribution monitoring
prediction distribution monitoring
label delay 고려
performance monitoring
drift detection
alerting
rollback
retraining
```

---

## 핵심 개념 13: Data Testing Debt

**ML에서는 데이터가 코드만큼 중요하므로, 데이터도 테스트해야 합니다.**

전통 소프트웨어에서는 코드 테스트를 합니다.

```text
unit test
integration test
regression test
```

ML에서는 데이터도 테스트해야 합니다.

```text
schema test
missing value test
range test
distribution shift test
label sanity check
duplicate check
feature freshness check
train-serving consistency check
```

---

## 핵심 개념 14: Reproducibility Debt

**ML 실험은 같은 결과를 다시 재현하기 어렵기 때문에, 재현성 관리가 중요합니다.**

ML 실험에는 randomness가 많습니다.

```text
random seed
data split
sampling
parallel training non-determinism
initialization
library version
hardware
external data
```

실무적으로는 다음을 남겨야 합니다.

```text
code version
data snapshot
feature version
model config
hyperparameter
random seed
library version
hardware info
training logs
evaluation results
```

이게 없으면:

```text
지난달 모델이 왜 좋았는지 모름
새 모델이 진짜 개선인지 모름
문제 발생 시 rollback 어려움
```

이 됩니다.

---

## 핵심 개념 15: Cultural Debt

**ML 시스템의 기술부채는 팀 문화 문제이기도 합니다.**

연구팀은 이런 것을 중시할 수 있습니다.

```text
accuracy +0.1%
새 모델
새 feature
새 benchmark
```

엔지니어링팀은 이런 것을 중시할 수 있습니다.

```text
안정성
재현성
삭제
모니터링
복잡도 감소
운영 자동화
```

둘이 분리되면 문제가 생깁니다.

```text
연구 코드는 production에서 블랙박스가 됨
운영 문제를 연구자가 모름
성능 개선이 시스템 복잡도를 폭발시킴
```

좋은 ML 조직은 accuracy 개선뿐 아니라 복잡도 감소, feature 삭제, monitoring 개선, 재현성 확보도 보상해야 합니다.

---

## 이 논문이 말하는 ML 시스템의 냄새들

| 냄새 | 의미 |
|---|---|
| Glue code | ML 패키지 주변 접착 코드가 시스템 대부분을 차지 |
| Pipeline jungle | 데이터 전처리 pipeline이 정글처럼 얽힘 |
| Dead experimental codepaths | 실험용 분기가 production에 계속 남음 |
| Configuration debt | 설정이 복잡하고 검증되지 않음 |
| Undeclared consumers | 누가 모델 output을 쓰는지 모름 |
| Unstable data dependency | upstream feature가 예고 없이 바뀜 |
| Hidden feedback loop | 모델이 현실을 바꾸고 그 변화가 다시 모델에 들어옴 |
| Data testing debt | 데이터 품질 테스트가 부족 |
| Reproducibility debt | 실험과 모델 재현이 어려움 |

이런 냄새가 있다면 시스템의 기술부채가 쌓이고 있다는 신호입니다.

---

## 실무 예시: 고객 이탈 예측 시스템

고객 이탈 모델을 운영한다고 합시다.

처음에는 간단했습니다.

```text
고객 로그
→ feature 생성
→ XGBoost 모델
→ 이탈 위험 score
→ 마케팅 캠페인
```

몇 달 뒤 시스템이 이렇게 됩니다.

```text
고객 로그
결제 로그
상담 로그
앱 사용 로그
마케팅 반응 로그
추천 모델 output
고객 가치 모델 output
할인 반응 모델 output
```

그리고 score를 여러 팀이 씁니다.

```text
마케팅팀
CS팀
세일즈팀
추천팀
경영 대시보드
```

여기서 문제가 발생합니다.

## 문제 1: CACE

상담 로그 feature 계산 방식이 바뀌자 모델 전체 score 분포가 바뀝니다.

## 문제 2: Undeclared consumer

CS팀이 이탈 score를 고객 우선순위에 쓰고 있었는데, 모델팀은 몰랐습니다.

## 문제 3: Hidden feedback loop

이탈 위험 높은 고객에게 할인 쿠폰을 주자, 다음 데이터에서 할인 반응과 이탈 사이 관계가 바뀝니다.

## 문제 4: Pipeline jungle

feature 생성 pipeline이 너무 복잡해져 새 feature 하나 추가하는 데 2주가 걸립니다.

## 문제 5: Configuration debt

threshold, campaign rule, sampling window가 설정 파일 여러 개에 흩어져 있습니다.

이게 바로 이 논문이 말하는 현실 ML technical debt입니다.

---

## 기술부채를 줄이기 위한 실전 원칙

## 1. 모델 output의 소비자를 관리하라

```text
누가 이 score를 쓰는가?
어떤 SLA가 있는가?
output schema와 의미는 무엇인가?
versioning은 어떻게 하는가?
```

## 2. 데이터 의존성을 추적하라

```text
feature lineage
upstream owner
data freshness
schema version
quality checks
```

## 3. feature를 주기적으로 삭제하라

```text
leave-one-feature-out 평가
feature importance 점검
legacy feature 제거
중복 feature 제거
```

## 4. pipeline을 정글로 만들지 마라

```text
명확한 pipeline abstraction
단계별 테스트
중간 산출물 관리
feature store
data contract
```

## 5. configuration도 코드처럼 관리하라

```text
version control
code review
test
rollback
audit log
```

## 6. monitoring을 처음부터 설계하라

```text
input distribution
prediction distribution
label distribution
online metric
business metric
system latency
data freshness
```

## 7. 실험 코드를 정리하라

```text
dead branch 제거
실험 flag 만료일 설정
성공한 실험은 production code로 정리
실패한 실험은 삭제
```

## 8. accuracy 개선과 complexity 비용을 함께 보라

작은 성능 향상이 큰 시스템 복잡도를 만든다면, 그 변경은 좋은 변경이 아닐 수 있습니다.

---

## 이 논문의 한계

이 논문은 새로운 알고리즘을 제안하지 않습니다.

이 논문은 이런 성격입니다.

```text
알고리즘 논문 X
시스템 설계와 운영 철학 논문 O
```

또한 2015년 논문이라 LLM, RAG, foundation model 같은 최신 시스템을 직접 다루지는 않습니다. 하지만 데이터 의존성, 피드백 루프, configuration, monitoring, glue code 문제는 현재에도 그대로 남아 있습니다.

---

## 이 논문이 주는 진짜 메시지

이 논문의 진짜 메시지는 이것입니다.

**ML 모델은 시스템 안에서 살아 움직이는 부품이다.**  
**그 모델은 데이터에 의존하고, 다른 시스템에 영향을 주고, 외부 세계 변화에 노출되며, 시간이 지나면서 유지보수 비용을 만든다.**

따라서 좋은 데이터사이언티스트는 단순히 모델 성능만 보지 않아야 합니다.

```text
이 feature는 어디서 오는가?
이 모델 output을 누가 쓰는가?
이 threshold를 바꾸면 어디가 영향받는가?
이 pipeline은 재현 가능한가?
외부 세계가 바뀌면 어떻게 감지하는가?
실험 코드는 언제 삭제되는가?
```

이 질문들을 해야 합니다.

---

## 이전 논문들과 연결해서 이해하기

| 이전 논문 | 이 논문과의 연결 |
|---|---|
| **Tidy Data** | 데이터 구조뿐 아니라 데이터 lineage와 dependency도 중요 |
| **Model Evaluation** | offline 성능 평가 이후 production 유지보수 문제가 시작됨 |
| **Datasheets** | 데이터 출처와 수집 과정을 문서화해 data debt를 줄임 |
| **Model Cards** | 모델 사용 범위와 한계를 문서화해 undeclared misuse를 줄임 |
| **XGBoost / LightGBM** | 강한 모델도 시스템 안에서는 glue code와 pipeline debt를 만들 수 있음 |
| **RAG** | 문서 index, retriever, generator, prompt pipeline도 technical debt를 만든다 |
| **LoRA / QLoRA** | adapter가 많아질수록 모델 versioning과 configuration debt가 생길 수 있음 |

---

## 입문자가 꼭 기억해야 할 문장

**ML 시스템에서 모델 코드는 전체 시스템의 작은 일부이고, 진짜 장기 비용은 데이터 의존성, glue code, pipeline, configuration, feedback loop, monitoring에서 생긴다.**

더 짧게 말하면:

> **ML 운영의 핵심은 모델 성능뿐 아니라 시스템 유지보수성이다.**

---

## 오늘 공부용 요약

**논문명:** Hidden Technical Debt in Machine Learning Systems  
**저자:** D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison  
**출판:** NIPS 2015

**핵심 결론:** ML 시스템은 일반 소프트웨어의 기술부채뿐 아니라, 데이터 의존성, entanglement, feedback loop, configuration, glue code, pipeline jungle, external world change 같은 ML 특유의 기술부채를 가진다.

**왜 중요함:** 모델을 빠르게 만들고 배포하는 것은 쉬워 보여도, 장기 운영에서는 유지보수 비용이 조용히 커질 수 있습니다. 이 논문은 실무 ML 시스템에서 모델 성능보다 더 오래 남는 시스템 설계 문제를 보여줍니다.

**입문자가 배울 점:** 좋은 ML 시스템은 좋은 모델 하나로 끝나지 않습니다. 데이터 lineage, feature dependency, monitoring, configuration management, reproducibility, consumer tracking, pipeline design이 모두 중요합니다.

**가장 중요한 문장:** ML 시스템의 기술부채는 대부분 모델 코드 안이 아니라, 모델 주변의 데이터와 시스템 경계에서 조용히 쌓인다.

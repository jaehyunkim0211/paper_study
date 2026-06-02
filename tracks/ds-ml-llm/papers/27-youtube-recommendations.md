# Deep Neural Networks for YouTube Recommendations

## 결론부터

이 논문의 결론은 이겁니다.

**대규모 추천 시스템은 “사용자에게 보여줄 후보를 많이 찾아오는 단계”와 “그 후보들을 정교하게 순위 매기는 단계”를 분리해서 설계해야 한다.**  
YouTube 추천 시스템은 이 두 단계를 각각 딥러닝 모델로 구성했습니다.

한 줄로 말하면:

> **YouTube 추천은 먼저 수백 개 후보 영상을 찾고, 그다음 그중에서 사용자가 실제로 오래 볼 가능성이 높은 영상을 정교하게 랭킹한다.**

논문은 YouTube 추천 시스템을 두 개의 neural network로 나눕니다. 첫 번째는 **candidate generation**, 즉 수백만 개 영상 중 사용자에게 관련 있을 법한 수백 개를 골라내는 모델이고, 두 번째는 **ranking**, 즉 그 후보들을 rich feature와 목적함수에 따라 점수화해 실제 화면에 보여줄 소수의 영상을 고르는 모델입니다.

---

## 이 논문을 한 문장으로 요약하면

**추천 시스템은 “무엇을 후보로 가져올 것인가”와 “그 후보 중 무엇을 먼저 보여줄 것인가”를 분리해야 하며, YouTube는 이 두 문제를 각각 다른 딥러닝 모델로 풀었다.**

```text
전체 영상 corpus: 수백만 개
↓
Candidate generation: 수백 개 후보 추림
↓
Ranking: 수십 개 또는 몇 개로 정렬
↓
사용자에게 노출
```

이 구조를 흔히 **two-stage recommendation system** 또는 **retrieval + ranking 구조**라고 생각하면 됩니다.

---

## 왜 이 논문이 중요한가?

추천 시스템은 단순히 다음과 같은 문제가 아닙니다.

```text
사용자 u와 영상 v가 있을 때, 클릭할 확률을 예측하자.
```

실제 문제는 훨씬 복잡합니다.

```text
수백만 개 영상 중 무엇을 후보로 가져올까?
새로 올라온 영상은 어떻게 노출할까?
클릭만 높이면 되는가, 오래 보는 것이 중요한가?
사용자의 최근 행동을 얼마나 반영할까?
offline metric과 online A/B test가 다르면 무엇을 믿을까?
```

논문은 YouTube 추천이 scale, freshness, noise라는 세 가지 큰 어려움을 가진다고 설명합니다.

| 어려움 | 의미 |
|---|---|
| scale | 거대한 사용자와 영상 corpus를 처리해야 함 |
| freshness | 새로운 영상과 최신 사용자 행동을 빠르게 반영해야 함 |
| noise | 사용자 만족도를 직접 관측하기 어렵고 implicit feedback이 noisy함 |

---

## 핵심 개념 1: Two-stage recommendation

**대규모 추천 시스템은 한 번에 모든 아이템을 정교하게 점수화하지 않고, 먼저 후보를 줄인 뒤 그 후보만 정교하게 랭킹합니다.**

YouTube에는 수백만 개 이상의 영상이 있습니다.
사용자가 홈 화면을 열 때마다 모든 영상을 하나하나 정교한 모델로 점수화하는 것은 비현실적입니다.

그래서 두 단계로 나눕니다.

## 1단계: Candidate Generation

```text
수백만 개 영상
→ 사용자에게 관련 있을 법한 수백 개 후보
```

이 단계는 빠르게 많은 후보를 찾아야 합니다.

목표는:

```text
사용자가 볼 가능성이 있는 후보를 넓게 잡기
```

입니다.

## 2단계: Ranking

```text
수백 개 후보
→ 실제로 보여줄 수십 개 또는 몇 개
```

이 단계는 후보가 이미 줄어든 상태이므로 더 많은 feature와 더 복잡한 모델을 사용할 수 있습니다.

목표는:

```text
그 후보들 중 무엇을 먼저 보여줄지 정교하게 결정하기
```

입니다.

---

## 핵심 개념 2: Candidate Generation

**Candidate generation은 사용자의 과거 시청, 검색, 인구통계 정보 등을 바탕으로 “이 사용자가 다음에 볼 만한 영상 후보”를 빠르게 찾아오는 단계입니다.**

쉽게 예를 들어봅시다.

사용자 A가 최근에 이런 영상을 봤습니다.

```text
- Python 입문 강의
- pandas 데이터 분석
- 머신러닝 기초
```

그리고 이런 검색을 했습니다.

```text
- xgboost tutorial
- 데이터사이언스 포트폴리오
```

Candidate generation 모델은 이 정보를 보고 수백만 개 영상 중에서 후보를 뽑습니다.

```text
- scikit-learn 강의
- LightGBM 튜토리얼
- Kaggle 입문 영상
- 데이터 전처리 강의
- 머신러닝 프로젝트 영상
```

이 단계에서는 아직 최종 순위가 아닙니다.

목표는:

```text
대체로 관련 있는 후보를 놓치지 않고 가져오는 것
```

입니다.

---

## 핵심 개념 3: Recommendation as Classification

**YouTube candidate generation은 “다음에 사용자가 볼 영상이 무엇인가?”를 수백만 개 class 중 하나를 맞히는 classification 문제로 정의했습니다.**

수식 느낌은 이렇습니다.

```text
P(다음 시청 영상 = i | 사용자, 문맥)
```

즉:

```text
이 사용자가 다음에 볼 영상이 영상 1일 확률
이 사용자가 다음에 볼 영상이 영상 2일 확률
...
이 사용자가 다음에 볼 영상이 영상 수백만일 확률
```

을 예측하는 문제입니다.

이게 “extreme” multiclass인 이유는 class 수가 너무 많기 때문입니다.

```text
일반 분류: class 10개, 100개
YouTube 추천: class 수백만 개 영상
```

그래서 일반 softmax를 그대로 쓰기 어렵고, 논문은 candidate sampling을 사용해 negative class를 샘플링하고 sampled softmax 방식으로 학습합니다.

---

## 핵심 개념 4: Embedding

**Candidate generation 모델은 사용자의 과거 행동과 영상을 dense vector, 즉 embedding으로 바꿉니다.**

예를 들어:

```text
"Python 입문 영상" → [0.12, -0.33, 0.87, ...]
"머신러닝 강의" → [0.18, -0.29, 0.81, ...]
"축구 하이라이트" → [-0.74, 0.55, 0.03, ...]
```

비슷한 영상은 비슷한 vector가 됩니다.

사용자의 시청 이력과 검색 이력을 보고:

```text
사용자 A → [0.15, -0.31, 0.84, ...]
```

같은 dense vector를 만듭니다.

그다음 사용자 vector와 영상 vector가 잘 맞는 영상을 후보로 가져옵니다.

---

## 핵심 개념 5: Candidate generation은 ANN search로 serving한다

**학습 때는 classification으로 배우지만, serving 때는 nearest neighbor search처럼 사용합니다.**

학습 중에는:

```text
사용자 context → 다음 시청 영상 class 예측
```

으로 모델을 훈련합니다.

하지만 실제 serving에서는 수백만 개 영상에 대해 softmax 확률을 모두 정확히 계산할 필요가 없습니다.

대신 사용자 embedding과 가까운 영상 embedding을 빠르게 찾습니다.

```text
학습: 수백만 영상 중 다음 영상 맞히기
서빙: user vector 근처의 video vector 빠르게 찾기
```

이 방식은 현대 추천 시스템의 retrieval 단계와 매우 비슷한 사고입니다.

---

## 핵심 개념 6: Heterogeneous Signals

**추천 시스템은 watch history만 쓰지 않고, 검색어, 지역, 기기, 성별, 나이 등 다양한 signal을 함께 사용합니다.**

예를 들어 candidate generation 모델의 입력은 이런 것들이 될 수 있습니다.

```text
시청한 영상 ID들
검색 query token들
지역
기기
성별
로그인 상태
나이
```

이런 다양한 feature가 합쳐져 user representation을 만듭니다.

사용자의 취향은 단순히 “어떤 영상을 봤는가”만으로 정해지지 않습니다.

```text
최근 검색어
지역
기기 상황
언어
시간적 맥락
```

이런 정보가 추천 품질에 영향을 줍니다.

---

## 핵심 개념 7: Freshness와 Example Age

**새로 올라온 영상을 잘 추천하려면, 모델이 과거 데이터에만 치우치지 않도록 시간 정보를 넣어야 합니다.**

YouTube에서는 새로운 영상이 계속 올라옵니다.

문제는 과거 데이터로 학습하면 모델이 자연스럽게 과거 인기 영상에 치우칠 수 있다는 것입니다.

그래서 논문은 훈련 예제의 나이, 즉 **example age**를 feature로 넣습니다.

쉽게 말하면:

```text
이 영상이 업로드된 지 얼마나 됐는가?
이 훈련 예제는 얼마나 오래된 예제인가?
```

를 모델에게 알려주는 것입니다.

추천 시스템에서 freshness는 매우 중요합니다.

```text
오래된 인기 영상만 계속 추천하면 새 영상이 뜨기 어렵다.
새 콘텐츠를 적절히 탐색하고 확산시켜야 한다.
```

---

## 핵심 개념 8: Label and Context Selection

**추천 모델은 단순히 관측된 행동을 그대로 맞히면 안 되고, 어떤 정보를 학습 입력에서 숨길지도 신중하게 정해야 합니다.**

예를 들어 사용자가 방금 “taylor swift”를 검색했다고 합시다.

그다음 사용자가 Taylor Swift 검색 결과 페이지에서 어떤 영상을 봤습니다.

모델 입력에 “사용자가 방금 taylor swift를 검색했다”는 사실을 그대로 넣고, label을 “그 다음 본 영상”으로 두면 모델은 쉽게 이렇게 배울 수 있습니다.

```text
검색어가 taylor swift면 검색 결과에 나온 영상을 추천하자.
```

하지만 홈 화면 추천에서는 검색 결과 페이지를 그대로 복제하는 것은 좋지 않습니다.

이건 매우 중요한 교훈입니다.

> 추천 모델은 “정답을 쉽게 맞히게 하는 정보”가 아니라, “실제 서비스 상황에서 유용한 정보”를 넣어야 한다.

---

## 핵심 개념 9: 사용자별 training example 수를 고정한다

**매우 활동적인 사용자가 loss를 지배하지 않도록, 사용자당 고정된 수의 training example을 생성했습니다.**

YouTube에는 사용자가 다양합니다.

```text
매일 몇 시간씩 보는 사용자
가끔 보는 사용자
거의 안 보는 사용자
```

만약 모든 watch event를 그대로 training example로 쓰면, 매우 활동적인 사용자가 데이터와 loss를 지배할 수 있습니다.

```text
활동 많은 사용자 1명이 데이터 1000개
활동 적은 사용자 1명이 데이터 5개
```

를 그대로 두면 모델이 활동 많은 사용자에게 과도하게 맞을 수 있습니다.

그래서 사용자 단위 균형을 맞추는 것입니다.

---

## 핵심 개념 10: Ranking

**Ranking은 candidate generation이 가져온 후보들을 richer feature와 더 정교한 objective로 최종 정렬하는 단계입니다.**

Candidate generation은 수백 개 후보를 가져옵니다.

Ranking은 그 후보들을 점수화합니다.

```text
후보 1: score 0.91
후보 2: score 0.83
후보 3: score 0.27
...
```

그다음 높은 점수 순으로 사용자에게 보여줍니다.

즉, candidate generation은 “넓게 찾기”이고, ranking은 “정교하게 고르기”입니다.

---

## Candidate Generation과 Ranking 비교

| 구분 | Candidate Generation | Ranking |
|---|---|---|
| 입력 후보 수 | 수백만 개 영상 corpus | 수백 개 후보 |
| 출력 | 수백 개 후보 영상 | 최종 정렬된 추천 목록 |
| 목표 | 관련 있을 만한 후보를 빠르게 찾기 | 실제 노출 순서를 정교하게 결정 |
| feature | user history, search, demographics 등 coarse feature | user, video, impression, context 등 rich feature |
| 계산 비용 | 매우 빠라야 함 | 후보 수가 적어 상대적으로 복잡한 모델 가능 |
| 비유 | 서점에서 살 만한 책 후보 뽑기 | 그중 지금 첫 번째로 추천할 책 고르기 |

---

## 핵심 개념 11: Ranking feature engineering은 여전히 중요하다

**딥러닝을 쓴다고 feature engineering이 사라지는 것은 아닙니다.**

추천 시스템에서는 여전히 feature 설계가 중요합니다.

예를 들어 ranking model에는 이런 feature가 들어갑니다.

```text
사용자의 과거 시청
해당 영상과 유사한 영상의 과거 반응
마지막 시청 이후 시간
이 영상이 최근에 몇 번 노출됐는지
사용자 언어
영상 언어
영상 ID embedding
사용자-영상 interaction feature
```

딥러닝 입문에서는 이렇게 생각하기 쉽습니다.

```text
딥러닝은 feature engineering을 자동으로 해준다.
```

하지만 추천 시스템에서는 여전히 feature representation이 매우 중요합니다.

---

## 핵심 개념 12: Categorical feature embedding

**추천 시스템에서는 영상 ID, 검색어, 언어 같은 sparse categorical feature를 embedding으로 바꿔 사용합니다.**

예를 들어 영상 ID는 매우 고차원입니다.

```text
video_id = 83472918
```

이걸 one-hot으로 넣으면 너무 sparse하고 큽니다.

그래서 embedding을 사용합니다.

```text
video_id 83472918 → dense vector [0.1, -0.3, 0.7, ...]
```

추천 시스템에는 categorical ID feature가 엄청나게 많습니다.

```text
영상 ID
채널 ID
검색어 token
언어
지역
기기
카테고리
```

Embedding은 이런 sparse ID를 dense representation으로 바꿔 딥러닝 모델이 사용할 수 있게 합니다.

---

## 핵심 개념 13: Continuous feature normalization

**Neural network는 feature scale에 민감하므로, continuous feature를 잘 정규화해야 합니다.**

트리 모델은 feature scale에 상대적으로 덜 민감합니다.

```text
XGBoost / LightGBM:
threshold split 기반
feature scale이 크게 문제 되지 않는 경우 많음
```

하지만 neural network는 input scale에 민감합니다.

예를 들어 `time_since_last_watch`라는 feature가 있다고 합시다.

원래 값은 매우 skewed할 수 있습니다.

```text
1초
5분
3시간
30일
```

이걸 그대로 넣으면 학습이 어렵습니다.

그래서 quantile 기반으로 정규화합니다.

```text
값의 분포를 보고 0~1 사이로 변환
```

그리고 추가로:

```text
x
x²
sqrt(x)
```

같은 형태도 넣어 네트워크가 다양한 함수 형태를 쉽게 배울 수 있게 합니다.

---

## 핵심 개념 14: CTR보다 Watch Time

**YouTube ranking의 목표는 단순 클릭률이 아니라 expected watch time에 가깝습니다.**

클릭률만 최적화하면 문제가 생깁니다.

예를 들어 어떤 영상의 썸네일이 매우 자극적입니다.

사용자는 클릭합니다.

하지만 5초 보고 나갑니다.

```text
CTR은 높음
만족도는 낮을 수 있음
watch time은 낮음
```

추천 시스템이 클릭만 최적화하면 clickbait을 강화할 위험이 있습니다.

YouTube는 사용자가 실제로 오래 보는 영상을 더 중요하게 봤습니다.

즉:

```text
클릭했는가?
```

보다

```text
클릭했고 얼마나 오래 봤는가?
```

를 더 중요하게 봅니다.

---

## 왜 watch time이 중요한가?

추천 시스템에서 objective를 잘못 잡으면 모델이 잘못된 행동을 최적화합니다.

## CTR 최적화

```text
자극적인 제목
낚시성 썸네일
짧은 호기심 유도
```

을 강화할 수 있습니다.

## Watch time 최적화

```text
사용자가 실제로 오래 소비한 콘텐츠
```

에 더 초점을 둡니다.

물론 watch time도 완벽한 만족도 지표는 아닙니다.

예를 들어:

```text
짧지만 매우 유익한 영상
길지만 별로 만족스럽지 않은 영상
자동 재생으로 길게 본 영상
```

이런 경우가 있을 수 있습니다.

그래도 논문에서 YouTube ranking은 클릭 확률보다 watch time 기반 objective가 더 좋은 사용자 engagement signal이라고 봤습니다.

---

## 핵심 개념 15: Offline metric과 Online A/B test

**추천 시스템에서는 offline metric이 좋아도 실제 사용자 A/B test에서 좋아진다는 보장이 없습니다.**

Offline에서는 좋아 보이는 모델이 online에서는 안 좋을 수 있습니다.

왜냐하면 offline data는 과거 로그입니다.

```text
과거에 노출된 후보
과거 UI
과거 사용자 행동
과거 추천 정책
```

에 기반합니다.

하지만 online에서는 모델이 실제 노출을 바꾸고, 사용자 행동도 바뀝니다.

그래서 추천 시스템에서는:

```text
offline metric = 빠른 개발용
online A/B test = 최종 판단
```

으로 보는 경우가 많습니다.

---

## 핵심 개념 16: Ranking model hidden layers

Ranking model에서도 hidden layer 깊이와 폭이 성능에 도움을 줍니다.

하지만 추천 시스템은 실시간으로 수많은 요청을 처리해야 합니다.

따라서 모델을 무한히 크게 만들 수 없습니다.

```text
성능
latency
CPU/GPU 비용
메모리
서빙 안정성
```

을 모두 고려해야 합니다.

---

## 이 논문이 주는 실무 교훈

## 1. 추천 시스템은 retrieval과 ranking을 분리하라

모든 아이템을 한 번에 정교하게 scoring하려고 하지 말고:

```text
candidate generation
ranking
```

으로 나누는 것이 실용적입니다.

## 2. 학습 objective를 신중하게 정하라

클릭만 보면 clickbait을 강화할 수 있습니다.

YouTube는 expected watch time을 중요한 objective로 모델링했습니다.

## 3. Feature engineering은 여전히 중요하다

딥러닝을 써도 raw data를 그대로 넣는 것이 아닙니다.

```text
categorical embedding
continuous normalization
interaction feature
impression history
time feature
```

가 중요합니다.

## 4. Freshness를 신경 써라

과거 인기 영상에만 치우치지 않도록 새로운 영상과 시간 효과를 모델링해야 합니다.

## 5. Offline metric만 믿지 마라

추천 시스템은 실제 user interaction이 중요하므로, live A/B test가 최종 판단에 중요합니다.

## 6. Serving cost를 항상 같이 보라

모델이 좋아도 latency budget을 넘으면 production에 못 올립니다.

---

## 이 논문과 Hidden Technical Debt의 연결

YouTube 추천 시스템은 단순 모델이 아닙니다.

```text
user history pipeline
candidate generation
ANN retrieval system
ranking feature pipeline
impression logging
watch time labeling
A/B testing
serving infrastructure
freshness handling
monitoring
```

이 모든 것이 하나의 추천 시스템입니다.

| 기술부채 관점 | YouTube 추천에서의 예 |
|---|---|
| Data dependencies | watch history, search history, impressions, video metadata |
| Hidden feedback loops | 추천이 사용자 행동을 바꾸고, 그 행동이 다시 학습 데이터가 됨 |
| Undeclared consumers | 추천 score를 다른 시스템이 사용할 수 있음 |
| Configuration debt | candidate source, ranking objective, thresholds, freshness settings |
| Pipeline jungle | 데이터 수집, feature 생성, training example 생성 |
| External world changes | 새 영상, 유행, 사용자 관심 변화 |

---

## 이 논문과 RAG/LLM 추천의 연결

RAG도 사실 two-stage 구조입니다.

```text
Retriever: 관련 문서 후보 검색
Generator / Ranker: 문서를 읽고 답변 생성
```

YouTube 추천도:

```text
Candidate generation: 후보 영상 검색
Ranking: 후보 점수화
```

입니다.

둘 다 핵심은 같습니다.

> 거대한 corpus 전체를 매번 완전히 계산하지 말고, 먼저 후보를 줄인 뒤 정교하게 처리하라.

---

## 이 논문의 한계

## 1. 2016년 당시 시스템 설명이다

이 논문은 2016년 RecSys 논문입니다. 따라서 현재 YouTube 추천 시스템 전체를 설명한다고 보면 안 됩니다.

## 2. 많은 세부 정보는 공개되지 않았다

논문은 high-level architecture와 practical lessons를 설명하지만, 실제 feature set, production infrastructure, A/B test 세부값, current serving stack 등은 공개하지 않습니다.

## 3. Watch time도 완벽한 만족도 지표는 아니다

Watch time이 사용자 만족도를 완벽히 대표하지는 않습니다.

예를 들어:

```text
짧지만 매우 유익한 영상
길지만 별로 만족스럽지 않은 영상
자동 재생으로 길게 본 영상
```

이런 경우가 있을 수 있습니다.

---

## 이전 논문들과 연결해서 이해하기

| 이전 논문 | 이번 논문과의 연결 |
|---|---|
| **Two Cultures** | YouTube 추천은 예측 성능 중심의 algorithmic modeling 사례 |
| **Model Evaluation** | offline metric과 online A/B test 차이가 중요 |
| **Hidden Technical Debt** | 추천 시스템은 데이터, feature, feedback loop, serving이 얽힌 대규모 ML 시스템 |
| **Deep Learning** | 딥러닝을 추천 시스템 feature interaction 모델링에 사용 |
| **Matrix Calculus** | candidate/ranking network도 backpropagation으로 학습 |
| **RAG** | candidate retrieval + downstream processing 구조가 유사 |
| **XGBoost/LightGBM** | ranking은 tabular feature 기반 supervised ML 문제와도 연결 |
| **Model Cards/Datasheets** | 추천 모델의 사용 범위, 데이터 출처, 목적함수 문서화가 중요 |

---

## 이 논문이 주는 진짜 메시지

이 논문의 진짜 메시지는 단순히 “YouTube가 neural network를 썼다”가 아닙니다.

더 중요한 메시지는 이것입니다.

**대규모 추천 시스템은 모델 하나가 아니라, retrieval, ranking, feature engineering, objective design, freshness, serving latency, A/B testing이 결합된 시스템이다.**

그리고 이 시스템에서 가장 중요한 설계 중 하나는:

```text
candidate generation과 ranking을 분리하는 것
```

입니다.

추천 시스템을 배우는 입문자는 이 논문에서 다음 감각을 꼭 가져가야 합니다.

```text
추천은 단순한 분류 문제가 아니다.
추천은 대규모 검색 + 정교한 랭킹 + 실시간 운영 문제다.
```

---

## 입문자가 꼭 기억해야 할 문장

**YouTube 추천 시스템은 수백만 개 영상 전체를 한 번에 정렬하지 않고, 먼저 candidate generation으로 수백 개 후보를 빠르게 찾은 뒤, ranking model로 그 후보들을 watch time 같은 목적에 맞게 정교하게 정렬한다.**

더 짧게 말하면:

> **추천 시스템 = 후보 생성 + 랭킹**

---

## 오늘 공부용 요약

**논문명:** Deep Neural Networks for YouTube Recommendations  
**저자:** Paul Covington, Jay Adams, Emre Sargin  
**출판:** RecSys 2016

**핵심 결론:** YouTube 추천 시스템은 candidate generation과 ranking이라는 두 개의 neural network 단계로 구성됩니다. Candidate generation은 사용자의 시청 이력과 검색 이력 등으로 수백만 개 영상 중 수백 개 후보를 가져오고, ranking은 rich feature와 expected watch time objective를 사용해 최종 노출 순서를 정합니다.

**왜 중요함:** 추천 시스템 실무에서 가장 중요한 two-stage architecture, 즉 retrieval/candidate generation과 ranking 분리 구조를 잘 보여주는 산업 논문입니다. 또한 deep learning을 추천에 적용할 때도 feature engineering, freshness, objective 설계, online A/B testing, serving latency가 여전히 중요하다는 점을 보여줍니다.

**입문자가 배울 점:** 추천 시스템은 “좋아할 확률을 예측하는 모델 하나”가 아닙니다. 거대한 corpus에서 후보를 줄이는 문제와, 그 후보를 실제 UI에서 어떤 순서로 보여줄지 정하는 문제를 분리해야 합니다.

**가장 중요한 문장:** 추천 시스템의 핵심은 “많은 것 중에서 빠르게 후보를 찾고, 적은 후보를 깊게 평가하는 것”입니다.

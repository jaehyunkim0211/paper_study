# A Survey of Collaborative Filtering Techniques

## 결론부터

이 논문의 결론은 이겁니다.

**Collaborative Filtering, 즉 협업 필터링은 “비슷한 사용자는 비슷한 것을 좋아할 것이다” 또는 “비슷한 아이템은 비슷한 사용자에게 선호될 것이다”라는 가정으로 추천을 만드는 방법입니다.**  
추천 시스템의 가장 기본적이고 오래된 핵심 방법론 중 하나입니다.

한 줄로 말하면:

> **Collaborative Filtering = 사용자들의 과거 선호 패턴을 이용해, 아직 보지 않은 아이템에 대한 선호를 예측하는 추천 방법**

예를 들어 A와 B가 영화 취향이 비슷하다고 합시다.

```text
A가 좋아한 영화: 인터스텔라, 인셉션, 메멘토
B가 좋아한 영화: 인터스텔라, 인셉션
```

그러면 추천 시스템은 이렇게 생각할 수 있습니다.

```text
A와 B는 취향이 비슷하다.
A가 메멘토를 좋아했으니, B도 메멘토를 좋아할 가능성이 있다.
```

이것이 collaborative filtering의 가장 기본 직관입니다.

---

## 이 논문을 한 문장으로 요약하면

**추천 시스템의 기본 문제는 사용자-아이템 행렬의 빈칸을 채우는 문제이고, collaborative filtering은 그 빈칸을 사용자 간 유사도, 아이템 간 유사도, 또는 학습된 모델로 예측하는 방법입니다.**

추천 데이터는 보통 이렇게 생겼습니다.

| 사용자 | 아이템 A | 아이템 B | 아이템 C | 아이템 D |
|---|---:|---:|---:|---:|
| Alice | 5 | 4 | ? | 1 |
| Bob | 4 | ? | 5 | 1 |
| Chris | ? | 5 | 4 | 2 |
| Dana | 5 | 4 | ? | ? |

여기서 `?`는 아직 모르는 선호입니다.

Collaborative filtering의 목표는 이 빈칸을 예측하는 것입니다.

```text
Dana는 아이템 C를 좋아할까?
Bob은 아이템 B를 좋아할까?
Alice는 아이템 C를 좋아할까?
```

---

## 왜 이 논문이 중요한가?

직전 YouTube 추천 논문은 대규모 추천 시스템의 산업적 구조를 보여줬습니다.

```text
candidate generation
→ ranking
→ A/B test
```

이번 collaborative filtering survey는 그보다 더 기초적인 질문을 다룹니다.

```text
사용자와 아이템의 과거 상호작용만으로 추천을 어떻게 만들까?
```

추천 시스템을 공부할 때 CF를 먼저 이해해야 하는 이유는, 많은 추천 모델의 출발점이 사용자-아이템 상호작용 행렬이기 때문입니다.

| 추천 접근 | 핵심 데이터 |
|---|---|
| Collaborative Filtering | 사용자-아이템 상호작용 |
| Content-based Filtering | 아이템 내용, 사용자 프로필 |
| Hybrid Recommendation | CF + content + context 등 |
| Modern Deep Recommender | embedding, sequence, graph, context를 결합 |

CF는 현대 추천 시스템에서는 단독으로만 쓰이기보다, candidate generation, matrix factorization, embedding learning, two-tower retrieval, graph recommendation, sequential recommendation의 기초 개념으로 이어집니다.

---

## 핵심 개념 1: User-Item Matrix

**Collaborative filtering의 출발점은 user-item matrix입니다.**

예를 들어 영화 추천을 한다고 합시다.

| 사용자 | Shrek | Snow White | Spider-Man | Superman |
|---|---|---|---|---|
| Alice | Like | Like | ? | Dislike |
| Bob | ? | Like | Dislike | Like |
| Chris | ? | Dislike | Like | ? |
| Tony | Like | ? | Dislike | ? |

이 표에서 빈칸이 많습니다.

추천 시스템은 Tony가 아직 평가하지 않은 영화를 좋아할지 예측해야 합니다.

```text
Tony는 Snow White를 좋아할까?
Tony는 Superman을 좋아할까?
```

---

## 핵심 개념 2: Collaborative Filtering의 기본 가정

**Collaborative filtering의 기본 가정은 “과거에 비슷하게 행동한 사람들은 미래에도 비슷하게 행동할 가능성이 높다”는 것입니다.**

이건 사람의 일상 추천과도 비슷합니다.

예를 들어 친구가 있습니다.

```text
나와 친구가 영화 취향이 비슷하다.
친구가 어떤 영화를 강력 추천했다.
그러면 나도 그 영화를 좋아할 가능성이 높다.
```

Collaborative filtering은 이 직관을 대규모 데이터에 적용한 것입니다.

---

## 핵심 개념 3: Explicit Feedback과 Implicit Feedback

**CF에서 사용자의 선호는 직접적인 평점일 수도 있고, 행동 로그에서 추론한 신호일 수도 있습니다.**

## Explicit Feedback

사용자가 직접 평가한 값입니다.

```text
별점 5점
좋아요 / 싫어요
리뷰 점수
만족도 점수
```

예:

| 사용자 | 영화 | 평점 |
|---|---|---:|
| A | Interstellar | 5 |
| A | Cats | 1 |
| B | Inception | 4 |

## Implicit Feedback

사용자의 행동에서 선호를 추론한 값입니다.

```text
클릭
구매
시청
장바구니
재방문
체류 시간
검색 후 클릭
```

예:

| 사용자 | 영상 | 행동 |
|---|---|---|
| A | 영상 1 | 10분 시청 |
| B | 상품 2 | 구매 |
| C | 기사 3 | 클릭 |

실무에서는 implicit feedback이 훨씬 많습니다.

사용자는 모든 상품에 별점을 주지 않지만, 클릭하고 보고 구매하고 넘기는 행동은 계속 남기 때문입니다.

---

## Collaborative Filtering의 세 가지 큰 분류

논문은 CF 기법을 크게 세 가지로 나눕니다.

```text
1. Memory-based CF
2. Model-based CF
3. Hybrid CF
```

| 분류 | 핵심 아이디어 |
|---|---|
| **Memory-based CF** | 사용자 또는 아이템 간 유사도를 직접 계산해 추천 |
| **Model-based CF** | 과거 데이터를 학습해 예측 모델을 만든 뒤 추천 |
| **Hybrid CF** | CF와 content-based, demographic, knowledge-based 등을 결합 |

---

# 1. Memory-based Collaborative Filtering

**Memory-based CF는 user-item matrix를 그대로 사용해 사용자끼리 또는 아이템끼리의 유사도를 계산하고, 그 유사도를 바탕으로 빈칸을 예측합니다.**

대표적으로 두 가지가 있습니다.

```text
1. User-based CF
2. Item-based CF
```

---

## 1.1 User-based CF

**User-based CF는 나와 비슷한 사용자를 찾고, 그 사람들이 좋아한 아이템을 추천하는 방식입니다.**

예를 들어 내가 본 영화와 평점이 이렇습니다.

| 영화 | 내 평점 |
|---|---:|
| Interstellar | 5 |
| Inception | 5 |
| Avengers | 2 |

다른 사용자가 있습니다.

| 사용자 | Interstellar | Inception | Avengers | Memento |
|---|---:|---:|---:|---:|
| Bob | 5 | 4 | 2 | 5 |
| Chris | 1 | 2 | 5 | 1 |

Bob은 나와 취향이 비슷합니다.

그러면 시스템은 이렇게 추천합니다.

```text
Bob이 Memento를 좋아했으니,
나도 Memento를 좋아할 가능성이 높다.
```

### User-based CF의 흐름

```text
1. active user와 비슷한 사용자들을 찾는다.
2. 비슷한 사용자들이 좋아한 아이템을 모은다.
3. 유사도에 따라 가중 평균을 내서 active user의 선호를 예측한다.
4. 예측 점수가 높은 아이템을 추천한다.
```

---

## 1.2 Item-based CF

**Item-based CF는 사용자가 좋아한 아이템과 비슷한 아이템을 추천하는 방식입니다.**

예를 들어 사용자가 `Interstellar`와 `Inception`을 좋아했습니다.

시스템은 item 간 유사도를 봅니다.

```text
Interstellar와 비슷한 영화
→ Gravity
→ The Martian
→ Arrival

Inception과 비슷한 영화
→ Memento
→ Tenet
```

그러면 이런 영화를 추천합니다.

```text
이 사용자는 Interstellar와 Inception을 좋아했으므로,
비슷한 영화인 Memento나 Arrival을 추천하자.
```

---

## User-based vs Item-based 비교

| 구분 | User-based CF | Item-based CF |
|---|---|---|
| 기준 | 비슷한 사용자 찾기 | 비슷한 아이템 찾기 |
| 직관 | 나와 취향이 비슷한 사람이 좋아한 것 추천 | 내가 좋아한 것과 비슷한 것 추천 |
| 장점 | 직관적 | item 유사도가 상대적으로 안정적일 수 있음 |
| 단점 | 사용자 수가 많고 취향이 빨리 변하면 어려움 | 새 아이템에는 유사도 계산이 어려움 |
| 예시 | “너와 비슷한 사용자가 이걸 봤다” | “네가 본 영상과 비슷한 영상이다” |

실무에서는 item-based CF가 더 안정적인 경우가 많았습니다. 사용자는 행동이 빠르게 바뀌지만, 아이템 간 관계는 상대적으로 덜 흔들리는 경우가 있기 때문입니다.

---

## Memory-based CF의 장점

## 1. 직관적이다

설명하기 쉽습니다.

```text
너와 비슷한 사람들이 이 영화를 좋아했어.
네가 좋아한 영화와 비슷한 영화야.
```

## 2. 구현이 상대적으로 쉽다

사용자-아이템 행렬과 similarity만 있으면 기본적인 추천을 만들 수 있습니다.

## 3. 도메인 지식이 많이 필요하지 않다

영화 내용, 상품 설명, 기사 본문을 몰라도 사용자 행동 패턴만으로 추천할 수 있습니다.

---

## Memory-based CF의 한계

## 1. Data Sparsity

user-item matrix는 대부분 비어 있습니다.

예를 들어 YouTube 영상이 수백만 개라면, 한 사용자가 본 영상은 극히 일부입니다.

```text
전체 아이템: 1,000,000개
사용자가 본 아이템: 100개
관측 비율: 0.01%
```

이렇게 sparse하면 사용자나 아이템 간 유사도를 안정적으로 계산하기 어렵습니다.

## 2. Scalability

사용자 수와 아이템 수가 커지면 similarity 계산이 매우 비싸집니다.

```text
사용자 1억 명
아이템 1천만 개
```

이런 규모에서는 모든 사용자 쌍, 모든 아이템 쌍 유사도를 계산하기 어렵습니다.

## 3. Cold Start

새 사용자나 새 아이템에는 과거 상호작용이 없습니다.

```text
새 사용자: 평가 이력 없음
새 아이템: 평가한 사용자 없음
```

CF는 과거 상호작용에 의존하므로 cold start에 약합니다.

## 4. Gray Sheep

어떤 사용자는 누구와도 취향이 잘 맞지 않습니다.

이 사용자는 어떤 그룹과도 딱 맞지 않습니다.

그러면 user-based CF가 “비슷한 사용자”를 찾기 어렵습니다.

## 5. Shilling Attacks

악의적 사용자가 가짜 평가를 넣어 추천을 조작할 수 있습니다.

예를 들어 특정 상품을 띄우기 위해 가짜 계정들이 높은 평점을 남기는 식입니다.

---

# 2. Model-based Collaborative Filtering

**Model-based CF는 user-item rating data를 이용해 예측 모델을 학습하고, 그 모델로 아직 모르는 선호를 예측하는 방식입니다.**

Memory-based CF는 user-item matrix를 그대로 들고 similarity를 계산합니다.

Model-based CF는 그 데이터로 모델을 학습합니다.

```text
과거 rating data
→ 모델 학습
→ 예측
```

---

## Model-based CF의 쉬운 예시: Matrix Factorization

현대적으로 가장 중요한 CF 모델 중 하나는 matrix factorization입니다.

기본 아이디어는 이렇습니다.

사용자와 아이템을 같은 latent space의 vector로 표현합니다.

```text
사용자 A = [0.2, 0.8, -0.1]
영화 X = [0.3, 0.7, -0.2]
```

사용자 vector와 아이템 vector가 잘 맞으면 선호가 높습니다.

```text
예상 평점 ≈ 사용자 vector · 아이템 vector
```

예를 들어 영화 취향 latent dimension이 있다고 합시다.

```text
SF 취향
로맨스 취향
액션 취향
예술영화 취향
```

사용자는 이 차원에서 어떤 취향을 갖고, 아이템은 어떤 특성을 갖습니다.

| 사용자/영화 | SF | 로맨스 | 액션 |
|---|---:|---:|---:|
| 사용자 A | 높음 | 낮음 | 중간 |
| Interstellar | 높음 | 낮음 | 중간 |
| Romance Movie | 낮음 | 높음 | 낮음 |

그러면 사용자 A는 Interstellar를 좋아할 가능성이 높습니다.

이런 방식은 user-based/item-based similarity보다 더 압축적이고 확장성 있게 추천을 만들 수 있습니다.

---

## Model-based CF의 장점

## 1. Scalability가 더 좋을 수 있다

모델을 학습한 뒤에는 user vector와 item vector를 비교하면 됩니다.

대규모 추천 시스템에 더 적합할 수 있습니다.

## 2. Sparsity를 어느 정도 완화할 수 있다

직접 같은 아이템을 평가한 사용자만 비교하지 않고, latent factor를 통해 간접적으로 패턴을 학습합니다.

## 3. 예측 성능이 좋을 수 있다

특히 matrix factorization 계열은 Netflix Prize 이후 추천 시스템에서 매우 중요해졌습니다.

## 4. 다양한 정보와 결합하기 쉽다

user feature, item feature, context feature와 결합한 factorization machine, neural CF 등으로 확장할 수 있습니다.

---

## Model-based CF의 한계

## 1. 학습 비용이 든다

모델을 학습해야 합니다.

Memory-based보다 training 과정이 복잡할 수 있습니다.

## 2. 모델 해석이 어려울 수 있다

latent factor가 꼭 사람이 이해하기 쉬운 의미를 갖는 것은 아닙니다.

## 3. 업데이트 문제

사용자 행동이 계속 바뀌면 모델을 주기적으로 업데이트해야 합니다.

## 4. Cold start는 여전히 어렵다

상호작용이 전혀 없는 새 사용자와 새 아이템은 모델도 잘 다루기 어렵습니다.

---

# 3. Hybrid Collaborative Filtering

**Hybrid CF는 collaborative filtering을 content-based filtering이나 다른 추천 방법과 결합하는 방식입니다.**

CF는 사용자-아이템 상호작용이 충분할 때 강합니다.

하지만 cold start나 sparsity에는 약합니다.

이때 아이템 내용 정보를 결합하면 도움이 됩니다.

예를 들어 영화 추천에서:

```text
CF 정보:
비슷한 사용자가 좋아한 영화

Content 정보:
장르, 감독, 배우, 줄거리, 키워드
```

를 함께 씁니다.

새 영화는 아직 평점이 없어도, 장르나 배우 정보는 있습니다.

```text
새 영화: SF, Christopher Nolan, 우주, 시간
```

그러면 기존 상호작용이 적어도 content feature로 어느 정도 추천할 수 있습니다.

---

# Collaborative Filtering의 대표 문제들

## 1. Data Sparsity

대부분의 사용자는 대부분의 아이템에 대해 평가하지 않았기 때문에, user-item matrix는 극도로 비어 있습니다.

```text
사용자 수: 1,000,000명
아이템 수: 1,000,000개
가능한 상호작용: 1조 개
실제 관측 상호작용: 1억 개
```

많아 보여도 전체의 0.01%입니다.

sparsity가 높으면:

```text
비슷한 사용자를 찾기 어렵다.
비슷한 아이템을 찾기 어렵다.
평점 예측이 불안정하다.
cold start가 심해진다.
```

## 2. Scalability

추천 시스템은 사용자와 아이템이 늘어날수록 계산량이 폭발합니다.

## 3. Cold Start

새 사용자나 새 아이템에는 상호작용 데이터가 없어서 CF가 추천하기 어렵습니다.

## 4. Synonymy

같거나 비슷한 아이템이 서로 다른 이름이나 ID로 존재하면, CF가 그 관계를 알아보기 어렵습니다.

예를 들어:

```text
The Lord of the Rings
LOTR
반지의 제왕
반지의 제왕 확장판
```

이런 항목들이 별도 item으로 존재하면, 실제로는 비슷하거나 같은 콘텐츠인데 상호작용이 분산됩니다.

## 5. Gray Sheep

Gray sheep은 다른 사용자들과 선호 패턴이 잘 맞지 않는 사용자입니다.

## 6. Shilling Attack

Shilling attack은 추천 시스템을 조작하려고 가짜 사용자나 가짜 평점을 넣는 공격입니다.

## 7. Privacy

CF는 사용자 선호와 행동 데이터를 사용하므로 privacy 문제가 생깁니다.

---

# Collaborative Filtering 평가 지표

추천 문제의 유형에 따라 metric이 달라집니다.

## Rating Prediction

평점을 정확히 예측하는 문제입니다.

```text
실제 평점: 4
예측 평점: 3.7
```

대표 지표:

```text
MAE
RMSE
```

## Top-N Recommendation

사용자에게 아이템 리스트를 추천하는 문제입니다.

```text
추천 Top 10 중 사용자가 실제로 좋아한 아이템이 몇 개인가?
```

대표 지표:

```text
Precision@K
Recall@K
MAP
NDCG
Hit Rate
```

## Ranking Quality

아이템 순서가 얼마나 좋은지 평가합니다.

```text
좋아할 아이템을 위에 놓았는가?
```

대표 지표:

```text
AUC
NDCG
MRR
```

---

## Rating Prediction과 Ranking은 다르다

평점을 정확히 예측하는 모델이 좋은 추천 리스트를 만드는 모델과 항상 같지는 않습니다.

예를 들어 어떤 사용자의 실제 평점이 이렇습니다.

| 아이템 | 실제 평점 | 모델 예측 |
|---|---:|---:|
| A | 5 | 4.7 |
| B | 4 | 4.6 |
| C | 1 | 1.2 |

평점 예측은 잘했습니다.

하지만 추천 시스템에서는 보통 사용자가 볼 Top-K가 더 중요합니다.

```text
상위 10개 추천이 얼마나 좋은가?
```

즉, 추천에서는 절대 평점 오차보다 ranking 품질이 더 중요할 때가 많습니다.

---

## Collaborative Filtering과 YouTube 추천 논문의 연결

| 개념 | Collaborative Filtering | YouTube 추천 |
|---|---|---|
| 기본 데이터 | user-item interaction | watch history, search history |
| 기본 목표 | 선호 예측 또는 Top-N 추천 | 다음 시청 후보 생성과 랭킹 |
| 기본 방식 | 유사 사용자/아이템 또는 모델 학습 | deep candidate generation + ranking |
| challenge | sparsity, scalability, cold start | scale, freshness, noisy implicit feedback |
| serving | similarity search, model prediction | ANN retrieval, ranking model |

YouTube candidate generation도 넓게 보면 collaborative filtering적 사고를 포함합니다.

```text
사용자의 과거 시청과 검색 행동
→ 비슷한 사용자/영상 패턴 학습
→ 다음에 볼 영상 예측
```

다만 YouTube는 이것을 neural network embedding과 extreme classification으로 확장했습니다.

---

## Collaborative Filtering과 Matrix Factorization의 연결

CF를 공부하면 matrix factorization으로 자연스럽게 이어집니다.

Memory-based CF는 유사도를 직접 계산합니다.

```text
사용자-사용자 유사도
아이템-아이템 유사도
```

Matrix factorization은 사용자와 아이템을 latent vector로 분해합니다.

```text
R ≈ U Vᵀ
```

여기서:

| 기호 | 의미 |
|---|---|
| `R` | user-item rating matrix |
| `U` | user latent factor matrix |
| `V` | item latent factor matrix |

사용자 `u`와 아이템 `i`의 예측 평점은:

```text
r_hat(u, i) = user_vector(u) · item_vector(i)
```

이 구조는 현대 embedding-based 추천과도 연결됩니다.

```text
사용자 embedding
아이템 embedding
dot product
nearest neighbor retrieval
```

---

## Collaborative Filtering과 딥러닝 추천의 연결

딥러닝 추천 모델도 기본적으로 CF의 확장입니다.

예를 들어 neural collaborative filtering은 사용자 ID와 아이템 ID를 embedding으로 바꾼 뒤, MLP로 상호작용을 학습합니다.

```text
user ID → user embedding
item ID → item embedding
[user embedding, item embedding] → MLP → 선호 예측
```

YouTube 추천 논문도 비슷한 구조를 씁니다.

```text
watch history embedding
search token embedding
user feature
→ DNN
→ candidate video prediction
```

즉, CF의 기본 아이디어는 현대 DNN 추천에도 살아 있습니다.

```text
사용자와 아이템의 상호작용 패턴으로 선호를 예측한다.
```

단지 방법이 더 복잡해졌습니다.

---

## Collaborative Filtering과 RAG/검색 시스템의 연결

RAG는 질문과 문서를 embedding으로 만들고 가까운 문서를 찾습니다.

```text
query embedding
document embedding
nearest neighbor search
```

추천에서도 사용자와 아이템을 embedding으로 만들고 가까운 아이템을 찾습니다.

```text
user embedding
item embedding
nearest neighbor search
```

즉:

| 시스템 | Query 쪽 | Candidate 쪽 |
|---|---|---|
| RAG | 질문 embedding | 문서 embedding |
| 추천 | 사용자 embedding | 아이템 embedding |
| 광고 | 사용자/context embedding | 광고 embedding |

모두 “어떤 query에 대해 관련 candidate를 찾는 문제”로 볼 수 있습니다.

---

## 이 논문의 한계

이 survey는 2009년 논문입니다.

그래서 지금의 최신 추천 시스템을 모두 다루지는 않습니다.

논문 당시의 주요 CF 기술은 memory-based, Bayesian, clustering, hybrid 등으로 정리되어 있고, 딥러닝 추천, graph neural network, transformer-based sequential recommendation, two-tower retrieval, contrastive learning 같은 현대 기술은 포함되지 않습니다.

하지만 여전히 중요한 이유는 다음입니다.

```text
CF의 기본 문제 정의
user-item matrix
sparsity
scalability
cold start
memory-based vs model-based vs hybrid
평가 지표
```

이 기초는 현대 추천 시스템에서도 계속 등장합니다.

---

## 실무적으로 가져갈 규칙

## 1. 추천 데이터는 행렬로 먼저 생각하라

```text
사용자 × 아이템
```

어떤 값이 들어가는지 먼저 정의해야 합니다.

```text
평점
클릭
구매
시청 시간
좋아요
저장
재방문
```

## 2. Explicit과 implicit을 구분하라

별점과 클릭은 의미가 다릅니다.

```text
별점 5점 = 강한 선호
클릭 = 호기심일 수도 있음
시청 시간 = engagement
구매 = 강한 행동
```

## 3. Rating prediction과 ranking을 구분하라

평점 예측을 잘한다고 좋은 Top-K 추천을 하는 것은 아닙니다.

## 4. Cold start를 항상 생각하라

CF만으로는 새 사용자와 새 아이템이 어렵습니다.

content, demographic, popularity, onboarding, exploration을 함께 고려해야 합니다.

## 5. Offline metric과 online metric을 구분하라

추천은 최종적으로 사용자 행동을 바꿉니다.

offline 성능만으로 충분하지 않습니다.

## 6. Privacy와 manipulation을 무시하지 마라

추천 시스템은 사용자 행동 데이터를 사용하고, 추천 결과가 비즈니스 가치와 연결되므로 공격과 privacy 문제가 생깁니다.

---

## 이전 논문들과 연결해서 이해하기

| 이전 논문 | 이번 논문과의 연결 |
|---|---|
| **Tidy Data** | user-item matrix를 어떻게 구성할지 중요 |
| **Model Evaluation** | MAE, precision, recall, AUC 등 평가 지표 선택 |
| **Hidden Technical Debt** | 추천 데이터 dependency와 feedback loop 관리 필요 |
| **YouTube Recommendations** | CF의 기본 아이디어가 deep candidate generation으로 확장 |
| **XGBoost / LightGBM** | ranking 단계에서 tabular prediction model로 활용 가능 |
| **Deep Learning** | user/item embedding과 neural CF로 확장 |
| **RAG** | query-candidate retrieval 구조가 추천과 유사 |

---

## 이 논문이 주는 진짜 메시지

이 논문의 진짜 메시지는 “어떤 알고리즘 하나가 최고다”가 아닙니다.

더 중요한 메시지는 이것입니다.

**추천 시스템의 기본은 사용자와 아이템 사이의 관측된 선호 패턴에서 아직 관측되지 않은 선호를 추론하는 것이다.**  
**하지만 실제 CF는 sparsity, scalability, cold start, 공격, privacy 같은 문제를 함께 해결해야 한다.**

---

## 입문자가 꼭 기억해야 할 문장

**Collaborative Filtering은 사용자-아이템 상호작용 행렬에서 비슷한 사용자나 비슷한 아이템의 패턴을 이용해, 아직 관측되지 않은 선호를 예측하는 추천 방법입니다.**

더 짧게 말하면:

> **CF = 사람들이 과거에 좋아한 패턴을 보고, 내가 좋아할 것을 예측하는 방법**

---

## 오늘 공부용 요약

**논문명:** A Survey of Collaborative Filtering Techniques  
**저자:** Xiaoyuan Su, Taghi M. Khoshgoftaar  
**출판:** Advances in Artificial Intelligence, 2009

**핵심 결론:** Collaborative filtering은 사용자 집단의 알려진 선호를 이용해 다른 사용자의 알려지지 않은 선호를 예측하는 추천 방법입니다. CF 기법은 크게 memory-based, model-based, hybrid 방식으로 나뉘며, 각각 sparsity, scalability, cold start, shilling attack, privacy 같은 문제를 다룹니다.

**왜 중요함:** 이 논문은 추천 시스템의 기본 문제 정의와 CF 알고리즘 분류를 정리한 고전 survey입니다. YouTube 추천, matrix factorization, neural collaborative filtering, two-tower retrieval 같은 현대 추천 시스템을 이해하려면 CF의 기본 개념을 먼저 잡아야 합니다.

**입문자가 배울 점:** 추천은 단순히 “비슷한 아이템 찾기”가 아닙니다. 데이터 sparsity, scale, cold start, 평가 지표, implicit feedback, privacy와 공격 가능성을 함께 고려해야 합니다.

**가장 중요한 문장:** Collaborative filtering의 핵심은 “비슷한 취향을 가진 사용자 또는 비슷한 반응을 받은 아이템을 이용해, 아직 모르는 선호를 예측하는 것”입니다.

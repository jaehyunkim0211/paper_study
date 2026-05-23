# Bootstrap Sampling

## 1. Bootstrap sample의 정확한 뜻

**Bootstrap sample**은 원본 데이터에서 **복원추출로 만든 샘플**입니다.

보통 통계학과 Random Forest 문맥에서는 원본 데이터가 `n`개라면, bootstrap sample도 **n개를 뽑습니다.**

예를 들어 원본 데이터가 다음과 같다고 합시다.

```text
원본 데이터: A, B, C, D, E
n = 5
```

그러면 bootstrap sample은 보통 5번 복원추출해서 만듭니다.

예:

```text
A, A, B, C, D
```

이건 bootstrap sample입니다.

왜냐하면:

1. 원본 데이터 `{A, B, C, D, E}`에서 뽑았고,
2. 매번 뽑은 뒤 다시 넣었고,
3. 그래서 `A`가 두 번 나올 수 있었고,
4. 총 5개를 뽑았기 때문입니다.

즉, bootstrap sample은 **중복을 허용하는 샘플**입니다.

---

## 2. 복원추출 이해

원본이:

```text
A, B, C, D, E
```

이고 3개를 복원추출한다면:

```text
1번째 뽑기: A, B, C, D, E 중 하나
뽑은 뒤 다시 넣음

2번째 뽑기: 다시 A, B, C, D, E 중 하나
뽑은 뒤 다시 넣음

3번째 뽑기: 다시 A, B, C, D, E 중 하나
```

그래서 가능한 결과는 이런 것들입니다.

```text
A, B, C
A, A, D
E, E, B
C, C, C
B, D, A
```

중복이 나올 수 있습니다.

---

## 3. 비복원추출

비복원추출은 뽑은 것을 다시 넣지 않습니다.

원본이:

```text
A, B, C, D, E
```

이고 3개를 비복원추출한다면:

```text
1번째 뽑기: A, B, C, D, E 중 하나
예: A가 뽑힘

2번째 뽑기: B, C, D, E 중 하나
예: C가 뽑힘

3번째 뽑기: B, D, E 중 하나
예: E가 뽑힘
```

결과:

```text
A, C, E
```

비복원추출에서는 같은 원소가 두 번 나올 수 없습니다.

그래서 이런 결과는 불가능합니다.

```text
A, A, C
B, B, B
D, D, E
```

---

## 4. 그러면 `A, B, C`는 비복원 샘플인가?

중요한 포인트가 있습니다.

**결과만 보고는 알 수 없습니다.**

`A, B, C`라는 결과는 두 방식 모두에서 나올 수 있습니다.

### 복원추출에서도 가능

```text
1번째: A
2번째: B
3번째: C
```

우연히 중복이 안 나온 것뿐입니다.

### 비복원추출에서도 가능

```text
1번째: A
2번째: B
3번째: C
```

당연히 가능합니다.

따라서 `A, B, C`가 비복원 샘플인지 bootstrap sample인지는 **결과가 아니라 뽑은 방식**으로 결정됩니다.

정리하면:

| 결과 | 뽑은 방식 | 이름 |
|---|---|---|
| `A, B, C` | 복원추출 | bootstrap sample일 수 있음 |
| `A, B, C` | 비복원추출 | subsample, random sample without replacement |
| `A, A, C` | 복원추출 | bootstrap sample일 수 있음 |
| `A, A, C` | 비복원추출 | 불가능 |

다만 전통적인 bootstrap에서는 보통 원본 크기와 같은 개수를 뽑기 때문에, 원본이 5개면 `A, B, C`처럼 3개만 뽑은 것은 일반적인 bootstrap sample이라기보다는 **복원추출 샘플** 또는 **m-out-of-n bootstrap sample**이라고 부르는 게 더 정확합니다.

Random Forest에서 말하는 bootstrap sample은 보통:

```text
원본 n개에서 복원추출로 n개를 뽑은 샘플
```

입니다.

---

## 5. Bootstrap sample과 비복원 랜덤 샘플의 핵심 차이

원본 데이터:

```text
A, B, C, D, E
```

에서 크기 5짜리 샘플을 만든다고 합시다.

### Bootstrap sample: 복원추출

```text
A, A, B, D, E
```

가능합니다.

특징:

| 특징 | 설명 |
|---|---|
| 중복 가능 | A가 여러 번 나올 수 있음 |
| 빠지는 데이터 가능 | C가 아예 안 나올 수 있음 |
| 샘플 크기 | 보통 원본과 같은 크기 |
| 의미 | 원본 데이터를 “다시 샘플링”해서 새로운 훈련셋처럼 만듦 |

### 비복원 랜덤 샘플: without replacement

크기 5를 비복원으로 뽑으면:

```text
A, B, C, D, E
```

결국 전부 다 뽑힙니다.

순서만 달라질 수 있습니다.

```text
C, A, E, B, D
```

하지만 membership은 똑같습니다.

즉, 원본이 5개인데 비복원으로 5개를 뽑으면 사실상 원본 전체를 그대로 쓰는 것과 같습니다. Random Forest에서 tree들을 다양하게 만드는 효과가 거의 없습니다.

그래서 비복원추출을 쓸 때는 보통 원본보다 적은 개수를 뽑습니다.

예:

```text
A, C, E
```

이건 **subsample**입니다.

---

## 6. Random Forest에서 bootstrap을 쓰면 무슨 일이 생기나?

원본 데이터가 5개라고 합시다.

```text
A, B, C, D, E
```

Tree 1의 bootstrap sample:

```text
A, A, C, D, E
```

Tree 2의 bootstrap sample:

```text
B, B, C, C, E
```

Tree 3의 bootstrap sample:

```text
A, D, D, E, E
```

각 tree가 서로 다른 데이터를 봅니다.

중요한 점은:

| 원본 데이터 | Tree 1 | Tree 2 | Tree 3 |
|---|---|---|---|
| A | 2번 등장 | 0번 등장 | 1번 등장 |
| B | 0번 등장 | 2번 등장 | 0번 등장 |
| C | 1번 등장 | 2번 등장 | 0번 등장 |
| D | 1번 등장 | 0번 등장 | 2번 등장 |
| E | 1번 등장 | 1번 등장 | 2번 등장 |

즉, 각 tree는 원본 데이터를 조금씩 다르게 봅니다.

어떤 데이터는 여러 번 보고, 어떤 데이터는 아예 보지 않습니다. 이 차이가 tree들을 서로 다르게 만듭니다.

---

## 7. Bootstrap sample에서 빠진 데이터: OOB

Random Forest에서 매우 중요한 개념이 여기서 나옵니다.

예를 들어 Tree 1의 bootstrap sample이:

```text
A, A, C, D, E
```

라면 `B`는 Tree 1의 학습에 쓰이지 않았습니다.

이때 `B`를 Tree 1 입장에서 **out-of-bag sample**, 줄여서 **OOB sample**이라고 합니다.

```text
Tree 1이 본 데이터: A, A, C, D, E
Tree 1이 못 본 데이터: B
```

이 못 본 데이터를 이용해 Tree 1의 성능을 검증할 수 있습니다.

데이터가 충분히 클 때, bootstrap sample을 만들면 대략:

| 구분 | 비율 |
|---|---:|
| 한 bootstrap sample에 적어도 한 번 포함되는 원본 데이터 | 약 63.2% |
| 한 bootstrap sample에 아예 포함되지 않는 원본 데이터 | 약 36.8% |

즉, Random Forest의 각 tree는 원본 데이터 중 약 63% 정도의 고유한 데이터를 보고, 약 37% 정도는 못 봅니다. 못 본 데이터가 OOB 평가에 쓰입니다.

---

## 8. 랜덤 비복원 샘플과 bootstrap sample의 차이를 한 번에 정리

원본:

```text
A, B, C, D, E
```

### 비복원 샘플, 크기 3

```text
A, C, E
```

특징:

```text
중복 없음
일부 데이터 빠짐
각 선택된 데이터는 한 번만 등장
```

### Bootstrap sample, 크기 5

```text
A, A, C, E, E
```

특징:

```text
중복 있음
일부 데이터 빠짐
선택된 데이터가 여러 번 등장 가능
보통 원본과 같은 크기
```

### 비복원 샘플, 크기 5

```text
A, B, C, D, E
```

특징:

```text
전부 다 뽑힘
중복 없음
원본 전체와 사실상 같음
```

---

## 9. 왜 그냥 비복원 랜덤 샘플을 쓰지 않고 bootstrap을 쓰나?

핵심 이유는 세 가지입니다.

### 첫째, 원본과 같은 크기의 훈련셋을 만들 수 있다

Bootstrap sample은 원본 데이터가 1,000개면, 각 tree도 1,000개짜리 훈련셋을 받습니다.

다만 그 안에는 중복이 있고, 빠진 데이터도 있습니다.

즉:

```text
크기는 원본과 같지만, 구성은 다르다.
```

이게 장점입니다.

반대로 비복원추출로 1,000개 중 1,000개를 뽑으면 그냥 원본 전체입니다.

tree들이 다양해지지 않습니다.

### 둘째, 각 tree마다 데이터 가중치가 달라지는 효과가 있다

Bootstrap sample에서 어떤 관측치는 2번, 3번 등장할 수 있습니다.

예:

```text
A, A, A, C, E
```

이 경우 Tree는 `A`를 더 많이 반영합니다.

다른 tree에서는 `B`가 더 많이 등장할 수도 있습니다.

```text
B, B, C, D, E
```

즉, bootstrap은 단순히 “일부를 뽑는 것”이 아니라, 각 tree마다 원본 데이터에 **랜덤한 가중치**를 주는 효과가 있습니다.

### 셋째, OOB 평가가 가능하다

Bootstrap sample에는 매번 일부 데이터가 빠집니다.

그 빠진 데이터는 해당 tree가 학습할 때 보지 않은 데이터입니다.

그래서 Random Forest는 이 OOB 데이터를 이용해서 내부적으로 성능을 추정할 수 있습니다.

---

## 10. Random Forest에서 비복원 샘플링을 쓰면 안 되나?

쓸 수 있습니다.

Random Forest의 변형 중에는 bootstrap 대신 **subsampling without replacement**, 즉 비복원 부분표본을 쓰는 방식도 있습니다.

예를 들어 원본 10,000개 중 매 tree마다 6,000개를 비복원으로 뽑아 tree를 만들 수도 있습니다.

```text
원본 10,000개
→ Tree 1: 비복원으로 6,000개
→ Tree 2: 비복원으로 6,000개
→ Tree 3: 비복원으로 6,000개
```

이것도 tree들을 다양하게 만들 수 있습니다.

하지만 Breiman의 전통적인 Random Forest에서는 기본적으로 bootstrap sampling을 사용합니다.

차이는 이렇게 보면 됩니다.

| 방식 | 이름 | 중복 | 샘플 크기 | Random Forest에서의 의미 |
|---|---|---|---|---|
| 복원추출 | bootstrap sampling | 있음 | 보통 원본과 같음 | 전통적인 Random Forest 방식 |
| 비복원추출 | subsampling | 없음 | 보통 원본보다 작음 | Random Forest 변형에서 사용 가능 |
| 비복원으로 원본 크기만큼 추출 | shuffle에 가까움 | 없음 | 원본과 같음 | 데이터 다양성 거의 없음 |

---

## 11. 제일 중요한 구분

핵심 정리는 이겁니다.

**Bootstrap sample은 “복원추출로 만든 샘플”이다.**  
**비복원으로 랜덤하게 뽑은 샘플은 bootstrap sample이 아니라 subsample이다.**

그리고 Random Forest에서 중요한 차이는 이것입니다.

| 질문 | 답 |
|---|---|
| bootstrap sample은 중복을 허용하나? | 예 |
| bootstrap sample에는 원본 데이터 일부가 빠질 수 있나? | 예 |
| bootstrap sample의 크기는 보통 원본과 같나? | 예 |
| `A, B, C`는 무조건 비복원 샘플인가? | 아님. 복원추출에서도 우연히 나올 수 있음 |
| 비복원 랜덤 샘플과 bootstrap sample의 차이는? | 중복 허용 여부, 빠지는 방식, 데이터 가중치 효과, OOB 구조 |

가장 정확한 문장으로 말하면:

> **원본 데이터가 n개일 때, bootstrap sample은 원본 데이터에서 복원추출로 n번 뽑아 만든 크기 n의 샘플이다. 이 과정에서 어떤 관측치는 여러 번 포함될 수 있고, 어떤 관측치는 아예 포함되지 않을 수 있다.**


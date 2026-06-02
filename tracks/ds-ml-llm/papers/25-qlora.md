# QLoRA: Efficient Finetuning of Quantized LLMs

## 결론부터

이 논문의 결론은 이겁니다.

**큰 LLM을 fine-tuning할 때, 모델 전체를 16-bit로 GPU에 올릴 필요는 없다. Base model은 4-bit로 양자화해서 메모리를 크게 줄이고, 실제 학습은 작은 LoRA adapter만 하면 된다.**

이 방법이 바로 **QLoRA**입니다.

한 줄로 말하면:

> **QLoRA = 4-bit로 압축한 frozen LLM 위에 LoRA adapter만 학습하는 방법**

QLoRA 논문은 65B parameter 모델을 단일 48GB GPU에서 fine-tuning할 수 있을 정도로 메모리 사용량을 줄이면서도, full 16-bit fine-tuning 성능을 유지하는 방법을 제안했습니다.

핵심은 frozen 4-bit quantized pretrained model을 통과해 gradient를 흘려보내고, 실제 학습 parameter는 LoRA adapter에만 두는 것입니다.

---

## 이 논문을 한 문장으로 요약하면

**QLoRA는 LoRA에 4-bit quantization을 결합해, 매우 큰 LLM도 적은 GPU 메모리로 fine-tuning할 수 있게 만든 논문입니다.**

직전 논문 LoRA를 기억하면 쉽습니다.

```text
LoRA:
Base model은 freeze
작은 low-rank adapter만 학습

QLoRA:
Base model은 freeze
Base model을 4-bit로 quantize
작은 LoRA adapter만 학습
```

즉, LoRA가 **학습되는 parameter 수**를 줄였다면, QLoRA는 거기에 더해 **frozen base model이 차지하는 메모리**까지 줄입니다.

---

## 왜 QLoRA가 필요했나?

LoRA는 이미 큰 발전이었습니다.

LoRA는 전체 모델 weight를 학습하지 않고, 작은 adapter만 학습합니다.

하지만 문제가 남아 있습니다.

> **Base model을 freeze하더라도, 그 base model 자체는 GPU 메모리에 올라가야 한다.**

예를 들어 65B 모델이 있다고 합시다.

Full fine-tuning은 당연히 비쌉니다.

```text
model weights
gradients
optimizer states
activations
```

이 전부가 필요합니다.

LoRA를 쓰면 학습 parameter와 optimizer state는 크게 줄어듭니다.

하지만 여전히 base model weight를 16-bit로 GPU에 올리면 메모리가 많이 듭니다.

```text
65B parameters × 2 bytes ≈ 130GB
```

여기에 activation, LoRA, optimizer 관련 메모리까지 들어갑니다.

QLoRA는 이 문제를 이렇게 해결합니다.

```text
Base model weight를 4-bit로 저장한다.
계산할 때는 필요한 순간에 BF16으로 dequantize한다.
학습은 LoRA adapter만 한다.
```

---

## 핵심 개념 1: Quantization

### 결론부터

**Quantization은 모델 weight를 더 적은 bit로 표현해 메모리를 줄이는 방법입니다.**

보통 neural network weight는 32-bit float 또는 16-bit float로 저장됩니다.

```text
FP32: parameter 하나당 32 bits = 4 bytes
FP16/BF16: parameter 하나당 16 bits = 2 bytes
INT8: parameter 하나당 8 bits = 1 byte
4-bit: parameter 하나당 4 bits = 0.5 byte
```

예를 들어 10억 parameter 모델이 있다고 합시다.

| 저장 방식 | 대략 weight 메모리 |
|---|---:|
| FP32 | 4GB |
| FP16/BF16 | 2GB |
| 8-bit | 1GB |
| 4-bit | 0.5GB |

즉, 4-bit로 저장하면 FP16 대비 weight 메모리를 대략 4분의 1로 줄일 수 있습니다.

QLoRA의 핵심은 이 4-bit quantization을 **fine-tuning에도 쓸 수 있게 만든 것**입니다.

---

## 핵심 개념 2: QLoRA는 base model을 학습하지 않는다

### 결론부터

**QLoRA에서 4-bit base model은 frozen입니다. 즉, base model weight는 업데이트하지 않습니다.**

학습되는 것은 LoRA adapter뿐입니다.

```text
Frozen 4-bit base model
+
Trainable LoRA adapters
```

Forward pass는 이렇게 됩니다.

```text
입력 x
→ 4-bit weight를 BF16으로 dequantize
→ base model 계산
→ LoRA adapter 보정 추가
→ 출력
```

Backward pass에서는 gradient가 base model을 통과하긴 하지만, base model weight 자체는 업데이트하지 않습니다.

```text
gradient passes through base model
but updates only LoRA parameters
```

---

## LoRA와 QLoRA를 수식으로 비교하기

LoRA에서는 일반 linear layer를 이렇게 바꿨습니다.

```text
h = W x + B A x
```

여기서:

```text
W: frozen base weight
A, B: trainable LoRA matrix
```

QLoRA는 `W`를 4-bit로 저장합니다.

```text
W_4bit: quantized frozen base weight
```

계산할 때는 대략 이렇게 됩니다.

```text
h = dequantize(W_4bit) x + B A x
```

즉:

```text
Base weight는 4-bit로 저장
계산할 때 BF16으로 dequantize
LoRA adapter는 16-bit/BF16으로 학습
```

---

## 쉬운 비유: 책은 압축해두고, 포스트잇만 학습하기

LoRA 비유를 다시 가져오면:

```text
Base model = 두꺼운 교과서
LoRA = task별 수정 포스트잇
```

LoRA는 교과서는 그대로 두고 포스트잇만 학습합니다.

QLoRA는 여기서 한 단계 더 갑니다.

```text
교과서 자체를 압축해서 보관한다.
필요할 때만 펼쳐서 읽는다.
수정은 여전히 포스트잇에만 한다.
```

즉:

```text
LoRA:
교과서는 원본 크기로 보관
포스트잇만 학습

QLoRA:
교과서는 압축해서 보관
포스트잇만 학습
```

이렇게 해서 훨씬 적은 GPU 메모리로 큰 모델을 fine-tuning할 수 있습니다.

---

## 핵심 개념 3: NF4 — 4-bit NormalFloat

### 결론부터

**NF4는 neural network weight가 대체로 정규분포에 가깝다는 점을 이용해 만든 4-bit datatype입니다.**

일반적인 4-bit quantization은 값의 구간을 단순하게 나눕니다.

하지만 neural network weight는 보통 0 주변에 많이 몰려 있고, 양 끝에는 값이 적습니다.

```text
대부분의 weight: 0 근처
일부 weight: 큰 양수 또는 큰 음수
```

그렇다면 4-bit 구간을 균등하게 나누는 것보다, weight가 많이 몰린 0 근처에 더 정교한 구간을 배치하는 편이 낫습니다.

NF4는 이 아이디어를 사용합니다.

```text
정규분포를 가정하고,
각 quantization bin에 비슷한 수의 weight가 들어가도록 4-bit 값을 설계
```

---

## NF4가 왜 중요한가?

4-bit로 줄이면 메모리는 크게 줄지만, 정보 손실이 커질 수 있습니다.

예를 들어 weight가 원래:

```text
0.123456
```

인데 4-bit로 표현하면 가능한 값이 16개뿐입니다.

```text
16개 값 중 하나로 근사
```

잘못 근사하면 모델 성능이 나빠집니다.

NF4는 weight 분포에 맞춰 4-bit 값을 배치해서 정보 손실을 줄이려는 방법입니다.

---

## 핵심 개념 4: Double Quantization

### 결론부터

**Double Quantization은 quantization을 위해 필요한 scale 값들까지 다시 quantize해서 메모리를 더 줄이는 방법입니다.**

Quantization을 할 때는 보통 block마다 scale, 즉 quantization constant가 필요합니다.

예를 들어 어떤 weight block을 4-bit로 표현하려면:

```text
이 block의 값 범위는 얼마인가?
어떤 scale로 4-bit 값을 실제 값으로 복원할 것인가?
```

를 알아야 합니다.

그래서 weight 자체는 4-bit로 줄였지만, block마다 scale 값이 추가로 필요합니다.

이 scale 값들이 메모리를 차지합니다.

Double Quantization은 이 scale 값들까지 quantize합니다.

```text
1차 quantization:
weight → 4-bit 값 + scale

2차 quantization:
scale 값 자체도 다시 quantize
```

---

## Double Quantization을 비유로 보면

압축 파일을 생각해봅시다.

1차 압축을 하면 파일이 작아집니다.

하지만 압축을 풀기 위한 메타데이터가 필요합니다.

```text
압축된 데이터
+
압축 해제용 정보
```

Double Quantization은 이 메타데이터까지 다시 압축하는 느낌입니다.

```text
데이터도 압축
압축 정보도 압축
```

메모리 절약이 아주 커 보이지 않을 수 있지만, 65B 같은 모델에서는 작은 bit 절약도 GB 단위로 커집니다.

---

## 핵심 개념 5: Paged Optimizers

### 결론부터

**Paged Optimizer는 학습 중 갑자기 GPU 메모리가 튀는 상황을 CPU memory paging으로 완화하는 방법입니다.**

LLM fine-tuning에서는 memory spike가 생길 수 있습니다.

특히 긴 sequence를 처리하거나 gradient checkpointing을 사용할 때 특정 순간에 GPU 메모리가 갑자기 많이 필요할 수 있습니다.

일반적으로는 이러면 OOM이 납니다.

```text
CUDA out of memory
```

QLoRA는 paged optimizer를 사용합니다.

아이디어는 운영체제의 memory paging과 비슷합니다.

```text
GPU 메모리가 부족하면 optimizer state 일부를 CPU RAM으로 옮김
필요할 때 다시 GPU로 가져옴
```

---

## QLoRA의 세 가지 핵심 기술

정리하면 QLoRA의 핵심 기술은 세 가지입니다.

| 기술 | 역할 |
|---|---|
| **NF4** | 4-bit로 줄이되 성능 손실을 줄임 |
| **Double Quantization** | quantization scale까지 줄여 메모리 추가 절약 |
| **Paged Optimizers** | memory spike로 인한 OOM 방지 |

---

## LoRA와 QLoRA의 차이

| 구분 | LoRA | QLoRA |
|---|---|---|
| Base model weight | 보통 16-bit/BF16로 GPU에 올림 | 4-bit로 quantized |
| 학습 parameter | LoRA adapter | LoRA adapter |
| Base model 업데이트 | 안 함 | 안 함 |
| 메모리 절감 | 학습 parameter와 optimizer state 절감 | base model weight 메모리까지 크게 절감 |
| 핵심 기술 | low-rank adapter | LoRA + 4-bit quantization |
| 장점 | full fine-tuning보다 가벼움 | 훨씬 더 큰 모델을 적은 GPU로 fine-tuning 가능 |

한 줄로 정리하면:

```text
LoRA는 학습할 parameter를 줄인다.
QLoRA는 학습하지 않는 base model도 4-bit로 줄인다.
```

---

## QLoRA와 일반 4-bit inference의 차이

일반 4-bit quantization은 주로 inference용입니다.

```text
모델을 4-bit로 줄여서 추론만 빠르고 작게 한다.
```

하지만 fine-tuning에서는 gradient가 필요합니다.

```text
forward
loss
backward
optimizer update
```

기존 quantization 방식은 training 중 성능이 깨지거나 안정성이 떨어질 수 있었습니다.

QLoRA는 frozen 4-bit base model을 통과해 gradient를 LoRA adapter로 보내는 방식으로 fine-tuning을 가능하게 했습니다.

---

## 핵심 개념 6: Guanaco

QLoRA 논문에서 나온 instruction-following 모델 family가 **Guanaco**입니다.

Guanaco는 LLaMA base model을 QLoRA로 instruction tuning한 모델입니다.

크기는 다음과 같이 나왔습니다.

```text
Guanaco 7B
Guanaco 13B
Guanaco 33B
Guanaco 65B
```

논문은 Guanaco family가 당시 공개 모델들보다 강한 성능을 보였고, 큰 모델은 ChatGPT에 가까운 평가 결과를 냈다고 보고했습니다.

단, 이 결과를 그대로 “Guanaco가 ChatGPT와 동급”이라고 단순 해석하면 안 됩니다. chatbot benchmark의 신뢰성 문제가 있고, 실제 사용 상황에서 실패 사례도 존재합니다.

---

## 핵심 개념 7: 작은 고품질 데이터의 중요성

QLoRA 논문에서 흥미로운 결과 중 하나는 이것입니다.

**데이터셋 크기보다 데이터 품질과 task 적합성이 더 중요할 수 있다.**

instruction tuning에서 무조건 데이터가 많을수록 좋은 것이 아니라, 모델이 해야 하는 task에 맞는 고품질 데이터가 중요하다는 점을 보여줍니다.

이건 InstructGPT와도 연결됩니다.

```text
사람이 원하는 답변 방식
instruction-following 품질
대화형 응답 데이터의 품질
```

이런 것들이 매우 중요합니다.

---

## QLoRA를 쉬운 실무 예시로 이해하기

상황을 생각해봅시다.

너에게 48GB GPU 한 장이 있습니다.

하고 싶은 일은:

```text
LLaMA 65B를 내 도메인 instruction 데이터로 fine-tuning하기
```

Full fine-tuning은 거의 불가능합니다.

```text
너무 많은 GPU 메모리 필요
```

LoRA만 써도 base model을 BF16으로 올리면 여전히 큽니다.

```text
65B × 2 bytes ≈ 130GB
```

QLoRA는 이렇게 합니다.

```text
1. 65B base model을 4-bit로 로드
2. base model weight freeze
3. LoRA adapter 삽입
4. forward 때 4-bit weight를 BF16으로 dequantize해 계산
5. backward 때 LoRA adapter만 업데이트
6. paged optimizer로 memory spike 처리
```

이렇게 해서 단일 48GB GPU에서도 65B fine-tuning이 가능해집니다.

---

## QLoRA에서 학습되는 것과 학습되지 않는 것

| 항목 | 학습 여부 |
|---|---|
| 4-bit base model weight | 학습 안 함 |
| LoRA adapter A, B | 학습함 |
| optimizer state | LoRA parameter에 대해서만 |
| embedding 추가 시 | 경우에 따라 별도 처리 필요 |
| quantization constant | 보통 학습 대상 아님 |

즉, QLoRA는 base model의 모든 지식을 다시 학습하는 방법이 아닙니다.

Base model은 이미 학습된 상태입니다.

QLoRA는 그 모델을 적은 메모리로 불러온 뒤, 작은 adapter를 학습해서 task에 맞게 행동을 조정합니다.

---

## QLoRA가 해결한 문제

## 1. GPU 메모리 장벽

큰 모델 fine-tuning의 가장 큰 장벽은 GPU 메모리였습니다.

QLoRA는 4-bit quantization과 LoRA를 결합해 이 장벽을 크게 낮췄습니다.

## 2. 연구 접근성

65B 모델 fine-tuning이 대규모 GPU 클러스터가 아니라 단일 고급 GPU에서도 가능해지면서, 더 많은 연구자와 실무자가 큰 모델을 실험할 수 있게 되었습니다.

## 3. 여러 실험 가능성

QLoRA를 이용해 여러 instruction dataset, model type, model scale을 비교할 수 있게 되었습니다. 이런 대규모 실험은 일반 full fine-tuning 비용으로는 매우 어려웠을 것입니다.

---

## QLoRA와 LoRA, Full fine-tuning 비교

| 구분 | Full fine-tuning | LoRA | QLoRA |
|---|---|---|---|
| Base model precision | 16-bit/32-bit | 보통 16-bit/BF16 | 4-bit |
| 학습 parameter | 전체 모델 | LoRA adapter | LoRA adapter |
| base weight 업데이트 | 예 | 아니오 | 아니오 |
| 메모리 요구량 | 매우 큼 | 줄어듦 | 크게 줄어듦 |
| checkpoint 크기 | 모델 전체 | adapter만 | adapter만 |
| 큰 모델 접근성 | 낮음 | 중간 | 높음 |
| 성능 | 강함 | 많은 task에서 강함 | 16-bit LoRA/full tuning에 근접 또는 유사 |

---

## QLoRA의 한계

## 1. 4-bit inference가 항상 빠른 것은 아니다

4-bit로 저장하면 메모리는 줄지만, 계산 과정에서 dequantization이 필요합니다.

그래서 구현에 따라 inference 속도는 기대만큼 빠르지 않을 수 있습니다.

즉:

```text
4-bit = 메모리 절감
4-bit = 항상 빠른 추론
```

은 아닙니다.

## 2. Base model 지식 자체를 깊게 바꾸는 방법은 아니다

QLoRA는 base model을 freeze하고 adapter만 학습합니다.

따라서 base model이 전혀 모르는 지식을 대량으로 새로 넣는 데는 한계가 있습니다.

새로운 최신 지식이나 회사 내부 문서가 필요하다면 RAG가 더 적절할 수 있습니다.

```text
QLoRA: 행동과 task adaptation
RAG: 외부 지식 제공
```

## 3. 데이터 품질에 민감하다

QLoRA는 fine-tuning 비용을 낮춰주지만, 나쁜 데이터로 학습하면 나쁜 adapter가 만들어집니다.

```text
저품질 instruction data
불일치한 답변 형식
잘못된 정답
편향된 선호
```

이런 문제는 QLoRA가 해결하지 않습니다.

## 4. Evaluation이 어렵다

chatbot benchmark가 항상 신뢰할 만한 것은 아닙니다.

좋은 benchmark 점수가 실제 모든 사용자 상황에서 좋은 답변을 보장하지는 않습니다.

---

## 실무에서 QLoRA는 언제 쓰면 좋은가?

## 쓰기 좋은 경우

```text
큰 LLM을 내 instruction 데이터로 fine-tuning하고 싶다.
GPU 메모리가 제한적이다.
Full fine-tuning은 불가능하다.
LoRA도 base model 메모리가 부담된다.
여러 adapter를 실험하고 싶다.
도메인별 말투, 형식, task skill을 조정하고 싶다.
```

예:

```text
LLaMA 7B/13B/33B/65B instruction tuning
회사 내부 Q&A 스타일 학습
SQL 생성 모델 fine-tuning
의료/법률 문서 요약 스타일 학습
한국어 instruction tuning
```

## QLoRA보다 RAG가 먼저인 경우

```text
최신 지식이 필요하다.
회사 문서의 정확한 근거가 필요하다.
문서가 자주 바뀐다.
모델이 답할 때 출처를 보여야 한다.
```

이때는 QLoRA로 모델에 지식을 외우게 하기보다 RAG가 더 효율적일 수 있습니다.

## QLoRA보다 full fine-tuning이 나을 수 있는 경우

```text
충분한 GPU 자원이 있다.
모델 전체를 깊게 바꿔야 한다.
base model과 task/domain 차이가 매우 크다.
최고 성능이 비용보다 중요하다.
```

하지만 실무에서는 QLoRA가 비용 대비 성능이 매우 좋아서, 첫 시도나 프로토타입으로 자주 선택됩니다.

---

## QLoRA와 이전 논문들의 연결

| 이전 논문 | QLoRA와의 연결 |
|---|---|
| **LoRA** | QLoRA의 핵심 adapter 방식 |
| **LLaMA** | QLoRA는 LLaMA 7B~65B fine-tuning 실험에 사용 |
| **InstructGPT / RLHF** | instruction-following 모델을 만드는 post-training과 연결 |
| **RAG** | QLoRA는 행동 조정, RAG는 외부 지식 제공 |
| **Chinchilla** | pretraining은 token/compute 균형, QLoRA는 이미 학습된 모델의 효율적 adaptation |
| **Transformer** | QLoRA는 Transformer의 linear layer와 LoRA adapter에 적용 |
| **Matrix Calculus** | frozen quantized base model을 통과해 LoRA parameter에 대해서만 gradient 계산 |

---

## QLoRA가 주는 진짜 메시지

QLoRA의 진짜 메시지는 단순히 “4-bit를 썼다”가 아닙니다.

더 큰 메시지는 이것입니다.

**대형 LLM을 활용하는 데 필요한 비용을 크게 낮추면, 더 많은 사람이 더 큰 모델을 fine-tuning하고 실험할 수 있다.**

LoRA는 학습 parameter를 줄였습니다.

QLoRA는 거기에 더해 frozen base model 자체의 메모리까지 줄였습니다.

이 조합은 LLM fine-tuning의 접근성을 크게 바꿨습니다.

```text
Full fine-tuning:
큰 연구소나 대규모 인프라 중심

QLoRA:
단일 GPU로도 큰 모델 adaptation 실험 가능
```

---

## 입문자가 꼭 기억해야 할 문장

**QLoRA는 4-bit로 양자화된 frozen base LLM을 사용하고, gradient를 그 모델을 통과시켜 LoRA adapter만 학습함으로써, 적은 GPU 메모리로 큰 모델을 fine-tuning하는 방법입니다.**

더 짧게 말하면:

> **QLoRA = 4-bit로 줄인 LLM 위에 LoRA만 학습하는 초경량 fine-tuning**

---

## 오늘 공부용 요약

**논문명:** QLoRA: Efficient Finetuning of Quantized LLMs  
**저자:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer  
**핵심 결론:** QLoRA는 frozen 4-bit quantized pretrained language model을 사용하고, LoRA adapter만 학습해 65B 모델도 단일 48GB GPU에서 fine-tuning할 수 있게 만든 방법입니다. NF4, Double Quantization, Paged Optimizers를 통해 메모리를 크게 줄이면서도 full 16-bit fine-tuning 성능을 보존하려 했습니다.

**왜 중요함:** QLoRA는 대형 LLM fine-tuning의 GPU 메모리 장벽을 크게 낮췄습니다. 이 덕분에 33B, 65B 같은 큰 모델에 대한 instruction tuning 실험이 훨씬 접근 가능해졌습니다.

**입문자가 배울 점:** LoRA가 “학습할 parameter 수”를 줄이는 방법이라면, QLoRA는 “학습하지 않는 base model의 저장 메모리”까지 4-bit로 줄이는 방법입니다.

**가장 중요한 문장:** QLoRA는 base model을 4-bit로 압축해 메모리에 올리고, 실제 학습은 작은 LoRA adapter에만 수행해서 큰 LLM fine-tuning을 현실적으로 만든다.

---

## Transformer 이후 현대 AI 파트 마무리

이제 **Transformer 이후 현대 AI 파트**가 끝났습니다.

```text
16. Attention Is All You Need
17. BERT
18. GPT-3
19. Scaling Laws
20. Chinchilla
21. RAG
22. InstructGPT / RLHF
23. LoRA
24. LLaMA
25. QLoRA
```

다음 파트는 **실무 ML 시스템과 데이터 중심 ML**로 넘어가면 됩니다. 다음 논문은 **Hidden Technical Debt in Machine Learning Systems**입니다.

# 시계열 데이터와 설비 진단 분류 / 예지보전 논문 로드맵

이 로드맵은 **2026년 6월 기준**으로, 입문자가 “시계열 분석 → 시계열 분류 → 딥러닝 → Transformer/Foundation Model → 설비 진단/이상탐지/RUL → 실무 검증” 순서로 올라갈 수 있게 구성한 목록입니다.

최신 흐름에서는 Chronos, TimesFM, Moirai, TimeGPT, MOMENT, UniTS 같은 시계열 파운데이션 모델이 빠르게 확산되고 있고, 벤치마크는 UCR/UEA, Monash Forecasting Repository, C-MAPSS/N-CMAPSS, CWRU, Paderborn, TSB-AD 등이 중요합니다. 라이브러리 쪽에서는 `sktime`, `aeon`, `Darts`, `GluonTS`, `tsfresh`가 실무 학습에 유용합니다.

## 난이도 기준

| 난이도 | 의미 |
|---|---|
| 입문 | 개념 중심. 수식이 있어도 직관으로 이해 가능 |
| 초중급 | 기본 ML/통계 지식 필요 |
| 중급 | 모델 구조, 평가 방식, 실험 설계 이해 필요 |
| 중상 | 딥러닝/Transformer 구조 이해 필요 |
| 고급 | 최신 Foundation Model, 도메인 적응, 벤치마크 비판까지 필요 |

---

## Part 1. 시계열 분석 기본기

| 번호 | 논문명 | 핵심 주제 | 왜 읽어야 하는지 | 난이도 | 이 논문에서 배우는 개념 |
|---:|---|---|---|---|---|
| 1 | **The Box-Jenkins Approach to Time Series Analysis and Forecasting: Principles and Applications** | ARIMA, Box-Jenkins 절차 | 시계열 예측의 “기본 문법”입니다. 딥러닝 전에 정상성, 차분, 자기상관, 잔차 진단을 알아야 합니다. | 입문 | AR, MA, ARIMA, 정상성, 차분, ACF/PACF, 잔차 |
| 2 | **STL: A Seasonal-Trend Decomposition Procedure Based on Loess** | 시계열 분해 | 센서 데이터에서 추세, 계절성, 노이즈를 나눠 보는 기본 도구입니다. | 입문 | trend, seasonality, residual, decomposition |
| 3 | **A New Approach to Linear Filtering and Prediction Problems** | Kalman Filter, state-space model | 센서 노이즈가 있는 환경에서 “숨은 상태”를 추정하는 핵심 아이디어입니다. | 초중급 | 상태공간모델, 관측값, latent state, filtering |
| 4 | **Making Time-Series Classification More Accurate Using Learned Constraints** | Dynamic Time Warping | 속도가 조금 다르거나 시간축이 밀린 패턴을 비교하는 방법입니다. 진동 패턴 분류에 직관적으로 연결됩니다. | 초중급 | DTW, warping window, elastic distance |
| 5 | **The M4 Competition: 100,000 Time Series and 61 Forecasting Methods** | 대규모 forecasting benchmark | 모델 성능은 논리보다 benchmark에서 검증되어야 한다는 관점을 줍니다. M4는 100,000개 시계열과 61개 방법을 비교한 대표 연구입니다. | 입문~초중급 | forecasting benchmark, error metric, ensemble, baseline |

---

## Part 2. 시계열 특징 추출과 전통 ML

| 번호 | 논문명 | 핵심 주제 | 왜 읽어야 하는지 | 난이도 | 이 논문에서 배우는 개념 |
|---:|---|---|---|---|---|
| 6 | **The UCR Time Series Classification Archive** | TSC 표준 데이터셋 | 시계열 분류 모델을 비교할 때 가장 자주 등장하는 archive입니다. | 입문 | UCR/UEA, TSC benchmark, train/test split |
| 7 | **Time Series Shapelets: A New Primitive for Data Mining** | Shapelet | 전체 시계열이 아니라 “분류에 결정적인 짧은 구간”을 찾는 아이디어입니다. | 초중급 | subsequence, shapelet, discriminative pattern |
| 8 | **Time Series Feature Extraction on basis of Scalable Hypothesis Tests** | tsfresh | 시계열에서 수백 개 특징을 자동 추출하고 통계적으로 선택하는 실무형 방법입니다. | 초중급 | feature extraction, hypothesis test, p-value, feature selection |
| 9 | **catch22: CAnonical Time-series CHaracteristics** | 22개 대표 특징 | 수천 개 특징 중 계산 빠르고 설명 가능한 22개 특징만 남기는 접근입니다. | 입문~초중급 | autocorrelation, entropy, outlier, distribution feature |
| 10 | **The BOSS is concerned with time series classification in the presence of noise** | BOSS, symbolic Fourier words | 진동/전류 신호를 짧은 구간 단어처럼 바꿔 분류하는 전통 TSC 강자입니다. | 중급 | sliding window, DFT, symbolic representation, bag-of-words |
| 11 | **ROCKET / MiniRocket** | 랜덤 convolution feature | 딥러닝처럼 보이지만 학습은 거의 선형 분류기인 강력한 실무 baseline입니다. MiniRocket은 UCR 전체를 매우 빠르게 처리하는 것으로 알려져 있습니다. | 초중급 | random convolution, PPV feature, ridge classifier |
| 12 | **HIVE-COTE 2.0** | TSC ensemble | shapelet, dictionary, interval 기반 모델을 결합한 전통 TSC 최강급 기준선입니다. | 중급 | ensemble, shapelet transform, dictionary model, interval feature |

---

## Part 3. 딥러닝 기반 시계열 모델

| 번호 | 논문명 | 핵심 주제 | 왜 읽어야 하는지 | 난이도 | 이 논문에서 배우는 개념 |
|---:|---|---|---|---|---|
| 13 | **Time Series Classification from Scratch with Deep Neural Networks** | FCN, ResNet TSC baseline | “수작업 특징 없이 raw time series를 CNN으로 바로 분류할 수 있다”는 기준선을 줍니다. | 중급 | 1D CNN, residual connection, end-to-end learning |
| 14 | **InceptionTime: Finding AlexNet for Time Series Classification** | Inception-style CNN | HIVE-COTE 수준의 정확도를 더 scalable하게 노리는 대표 딥러닝 TSC 모델입니다. | 중급 | multi-scale convolution, ensemble CNN, TSC |
| 15 | **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling** | TCN | RNN/LSTM만이 시계열의 답은 아니라는 것을 보여준 중요한 논문입니다. | 중급 | causal convolution, dilation, receptive field |
| 16 | **DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks** | probabilistic forecasting | 여러 설비·제품·센서의 관련 시계열을 함께 학습하는 예측 모델의 기본형입니다. | 중급 | RNN forecasting, likelihood, probabilistic prediction |
| 17 | **N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting** | 딥러닝 forecasting | 추세/계절성처럼 해석 가능한 블록을 신경망에 넣는 아이디어입니다. | 중급 | backcast, forecast, basis expansion, interpretability |
| 18 | **TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis** | 범용 시계열 딥러닝 | forecasting, imputation, classification, anomaly detection을 하나의 backbone으로 다루려는 흐름입니다. | 중상 | multi-periodicity, 2D temporal variation, task-general model |

---

## Part 4. Transformer와 시계열 파운데이션 모델

| 번호 | 논문명 | 핵심 주제 | 왜 읽어야 하는지 | 난이도 | 이 논문에서 배우는 개념 |
|---:|---|---|---|---|---|
| 19 | **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting** | efficient Transformer | 긴 시계열 예측에서 Transformer 계산량 문제를 어떻게 줄이는지 보여줍니다. | 중상 | ProbSparse attention, distilling, long sequence forecasting |
| 20 | **Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting** | decomposition + autocorrelation | Transformer 안에 시계열 분해와 주기성 탐색을 넣은 대표 모델입니다. | 중상 | decomposition, auto-correlation, long-term forecasting |
| 21 | **FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting** | frequency-domain Transformer | Fourier/wavelet 관점으로 장기 예측을 다루는 모델입니다. 설비 진동의 주파수 해석과도 연결됩니다. | 중상 | Fourier, wavelet, frequency attention |
| 22 | **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers** | PatchTST | 시계열을 작은 patch 단위 token처럼 보는 현대 Transformer 접근입니다. | 중상 | patching, channel independence, self-supervised pretraining |
| 23 | **TimeGPT-1** | 상용 시계열 foundation model | “시계열도 LLM처럼 사전학습 모델을 쓸 수 있는가?”라는 질문에 직접 답하는 흐름입니다. | 중상~고급 | zero-shot forecasting, foundation model, anomaly detection API |
| 24 | **A Decoder-Only Foundation Model for Time-Series Forecasting / TimesFM** | Google TimesFM | 대규모 사전학습 시계열 모델의 대표 사례입니다. TimesFM은 100B 규모의 실제 시계열 포인트로 학습되었다고 소개됩니다. | 중상~고급 | decoder-only model, zero-shot forecasting, pretraining |
| 25 | **Chronos: Learning the Language of Time Series** | tokenization 기반 foundation model | 시계열 값을 quantization해서 언어모델처럼 학습하는 아이디어가 핵심입니다. | 중상~고급 | scaling, quantization, token vocabulary, probabilistic forecast |
| 26 | **Unified Training of Universal Time Series Forecasting Transformers / Moirai** | universal forecasting | 여러 도메인의 대규모 시계열을 묶어 zero-shot forecasting을 노리는 대표 open 모델입니다. | 고급 | LOTSA, universal model, zero-shot, probabilistic forecasting |
| 27 | **MOMENT / UniTS** | 범용·멀티태스크 foundation model | forecasting만이 아니라 classification, anomaly detection, imputation까지 하나의 모델로 다루려는 최신 방향입니다. | 고급 | representation learning, multi-task learning, limited supervision |

---

## Part 5. 이상탐지와 설비 진단 분류

| 번호 | 논문명 | 핵심 주제 | 왜 읽어야 하는지 | 난이도 | 이 논문에서 배우는 개념 |
|---:|---|---|---|---|---|
| 28 | **Matrix Profile I: All Pairs Similarity Joins for Time Series** | motif, discord, anomaly | “정상 패턴과 다르게 생긴 subsequence”를 찾는 강력한 고전 이상탐지 도구입니다. | 초중급 | subsequence join, motif, discord, anomaly score |
| 29 | **OmniAnomaly: Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network** | multivariate anomaly detection | 여러 센서가 동시에 움직이는 설비 데이터를 확률적으로 모델링합니다. | 중상 | VAE, GRU, reconstruction probability, multivariate dependency |
| 30 | **USAD: UnSupervised Anomaly Detection on Multivariate Time Series** | adversarial autoencoder | 라벨이 부족한 산업 현장에서 unsupervised anomaly detection을 어떻게 할지 배웁니다. | 중상 | autoencoder, adversarial training, reconstruction error |
| 31 | **TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data** | Transformer anomaly detection | attention 기반으로 센서 간 관계와 시간 의존성을 동시에 보는 모델입니다. | 중상 | Transformer encoder, self-conditioning, anomaly score |
| 32 | **A Review of Recent Advances in Deep Learning Models for Bearing Fault Diagnosis** | bearing fault diagnosis survey | 베어링 고장 진단에서 CNN, AE, RBM, raw vibration learning이 어떻게 쓰이는지 전체 지도를 줍니다. | 입문~중급 | vibration diagnosis, bearing fault, CNN, autoencoder |
| 33 | **Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set** | Paderborn bearing dataset | 실제 손상과 인공 손상이 섞인 베어링 데이터셋을 이해하는 데 중요합니다. | 초중급 | bearing damage, current signal, vibration signal, operating condition |
| 34 | **Fault Detection and Identification Using Deep Learning Based Methods for Induction Motor** | motor fault diagnosis, MCSA | 모터 전류 신호 기반 고장 진단을 배웁니다. FFT, wavelet, MCSA와 딥러닝이 연결됩니다. | 중급 | motor current signature analysis, 1D-CNN, LSTM, current signal |

---

## Part 6. 예지보전 / Remaining Useful Life

| 번호 | 논문명 | 핵심 주제 | 왜 읽어야 하는지 | 난이도 | 이 논문에서 배우는 개념 |
|---:|---|---|---|---|---|
| 35 | **C-MAPSS / N-CMAPSS Turbofan Engine Degradation Dataset Papers** | RUL benchmark dataset | RUL 논문에서 가장 자주 쓰이는 엔진 열화 데이터셋입니다. train/test가 engine unit 단위라는 점이 특히 중요합니다. | 입문~초중급 | run-to-failure, engine unit split, operating condition, RUL label |
| 36 | **A Deep Convolutional Neural Network Based Regression Approach for Estimation of Remaining Useful Life** | CNN RUL | 센서 sequence에서 CNN으로 RUL을 직접 회귀하는 대표 초기 딥러닝 RUL 논문입니다. | 중급 | CNN regression, RUL target, sliding window |
| 37 | **Long Short-Term Memory Network for Remaining Useful Life Estimation** | LSTM RUL | 열화가 시간에 따라 누적되는 문제를 sequence model로 푸는 기본 접근입니다. | 중급 | LSTM, degradation pattern, sequence-to-one regression |
| 38 | **Remaining Useful Lifetime Prediction via Deep Domain Adaptation / ADARUL** | domain adaptation RUL | 실무에서는 학습 설비와 현장 설비의 분포가 다르기 때문에 domain shift 대응이 필수입니다. | 중상~고급 | domain adaptation, source/target domain, adversarial learning |

---

## Part 7. 실무 관점: 평가, split, leakage, benchmark 비판

| 번호 | 논문명 | 핵심 주제 | 왜 읽어야 하는지 | 난이도 | 이 논문에서 배우는 개념 |
|---:|---|---|---|---|---|
| 39 | **TSB-AD: An End-to-End Benchmark Suite for Time Series Anomaly Detection** | anomaly detection benchmark | 이상탐지 평가는 threshold, 지연 탐지, 구간 탐지 때문에 까다롭습니다. TSB-AD는 1,070개 시계열과 VUS 계열 지표를 제안한 최신 벤치마크입니다. | 중급 | VUS-ROC, VUS-PR, threshold-independent metric, anomaly benchmark |
| 40 | **Benchmarking Deep Learning Models for Bearing Fault Diagnosis Using the CWRU Dataset: A Multi-label Approach** | CWRU leakage, realistic split | 베어링 진단 논문에서 window 단위 랜덤 split이 얼마나 위험한지 배우는 실무 필수 논문입니다. CWRU 데이터는 널리 쓰이지만, 흔한 분할 방식은 data leakage로 과도하게 낙관적인 성능을 만들 수 있다는 비판이 있습니다. | 중급 | data leakage, unit/run split, class imbalance, multi-label evaluation |

---

## 이 로드맵을 공부할 때 계속 기억할 핵심 원칙

시계열/설비 진단에서는 **모델보다 데이터 분할이 먼저**입니다. 특히 진동 데이터를 sliding window로 잘라놓고 window를 랜덤하게 train/test로 나누면, 같은 설비·같은 운전 run·인접 시간 구간이 train과 test에 동시에 들어갈 수 있습니다. 그러면 모델이 “고장 패턴”을 배운 것이 아니라 거의 같은 신호 조각을 외운 것처럼 되어 accuracy가 비정상적으로 높아질 수 있습니다.

실무에서는 accuracy 하나로 끝내면 안 됩니다. 설비 진단 분류에서는 **precision, recall, F1, confusion matrix, false alarm, missed detection**을 봐야 하고, 이상탐지에서는 **탐지 지연, threshold 민감도, 구간 단위 탐지 성능**을 봐야 하며, RUL에서는 **MAE, RMSE, NASA scoring function, early/late prediction penalty**를 봐야 합니다.

---

## 참고 링크

- Box-Jenkins / ARIMA: https://www.numdam.org/item/RO_1977__11_2_129_0.pdf
- STL 논문 PDF: https://www.math.unm.edu/~lil/Stat581/STL.pdf
- Kalman Filter: https://asmedigitalcollection.asme.org/fluidsengineering/article/82/1/35/397706/A-New-Approach-to-Linear-Filtering-and-Prediction
- UCR Time Series Archive: https://www.cs.ucr.edu/~eamonn/time_series_data/
- Chronos: https://arxiv.org/abs/2403.07815
- TimesFM: https://arxiv.org/abs/2310.10688
- Moirai: https://arxiv.org/abs/2402.02592
- TSB-AD: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html

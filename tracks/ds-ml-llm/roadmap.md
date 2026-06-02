# Roadmap: Data Science / ML / LLM

이 로드맵은 데이터사이언스 입문부터 머신러닝, 딥러닝, LLM, 최신 LLM 시스템까지 공부한 논문 정리 순서입니다.

## Part 1. 데이터사이언스 기본기 / 모델 평가 / 책임 있는 ML

| No. | Paper | Status |
|---:|---|---|
| 01 | [Tidy Data](papers/01-tidy-data.md) | Done |
| 02 | [Statistical Modeling: The Two Cultures](papers/02-statistical-modeling-two-cultures.md) | Done |
| 03 | [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](papers/03-model-evaluation-selection.md) | Done |
| 04 | [Datasheets for Datasets](papers/04-datasheets-for-datasets.md) | Done |
| 05 | [Model Cards for Model Reporting](papers/05-model-cards.md) | Done |
| 06 | [Random Forests](papers/06-random-forests.md) | Done |
| 07 | [Greedy Function Approximation: A Gradient Boosting Machine](papers/07-gradient-boosting-machine.md) | Done |
| 08 | [XGBoost: A Scalable Tree Boosting System](papers/08-xgboost.md) | Done |
| 09 | [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](papers/09-lightgbm.md) | Done |
| 10 | [A Unified Approach to Interpreting Model Predictions — SHAP](papers/10-shap.md) | Done |

## Part 2. 딥러닝 기본기

| No. | Paper | Status |
|---:|---|---|
| 11 | [Deep Learning](papers/11-deep-learning.md) | Done |
| 12 | [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](papers/12-dropout.md) | Done |
| 13 | [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](papers/13-batch-normalization.md) | Done |
| 14 | [Deep Residual Learning for Image Recognition — ResNet](papers/14-resnet.md) | Done |
| 15 | [The Matrix Calculus You Need For Deep Learning](papers/15-matrix-calculus.md) | Done |

## Part 3. Transformer 이후 현대 AI

| No. | Paper | Status |
|---:|---|---|
| 16 | [Attention Is All You Need](papers/16-attention-is-all-you-need.md) | Done |
| 17 | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](papers/17-bert.md) | Done |
| 18 | [Language Models are Few-Shot Learners — GPT-3](papers/18-gpt-3.md) | Done |
| 19 | [Scaling Laws for Neural Language Models](papers/19-scaling-laws.md) | Done |
| 20 | [Training Compute-Optimal Large Language Models — Chinchilla](papers/20-chinchilla.md) | Done |
| 21 | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks — RAG](papers/21-rag.md) | Done |
| 22 | [Training Language Models to Follow Instructions with Human Feedback — InstructGPT / RLHF](papers/22-instructgpt-rlhf.md) | Done |
| 23 | [LoRA: Low-Rank Adaptation of Large Language Models](papers/23-lora.md) | Done |
| 24 | [LLaMA: Open and Efficient Foundation Language Models](papers/24-llama.md) | Done |
| 25 | [QLoRA: Efficient Finetuning of Quantized LLMs](papers/25-qlora.md) | Done |

## Part 4. 실무 ML 시스템 / 추천 / 데이터 중심 ML

| No. | Paper | Status |
|---:|---|---|
| 26 | [Hidden Technical Debt in Machine Learning Systems](papers/26-hidden-technical-debt.md) | Done |
| 27 | [Deep Neural Networks for YouTube Recommendations](papers/27-youtube-recommendations.md) | Done |
| 28 | [A Survey of Collaborative Filtering Techniques](papers/28-collaborative-filtering-survey.md) | Done |
| 29 | [Data Shapley: Equitable Valuation of Data for Machine Learning](papers/29-data-shapley.md) | Done |
| 30 | [Data Shapley in One Training Run](papers/30-data-shapley-in-one-training-run.md) | Done |

## Part 5. LLM reasoning / alignment / serving / 최신 LLM

| No. | Paper | Status |
|---:|---|---|
| 31 | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](papers/31-chain-of-thought.md) | Done |
| 32 | [Self-Consistency Improves Chain of Thought Reasoning in Language Models](papers/32-self-consistency.md) | Done |
| 33 | [ReAct: Synergizing Reasoning and Acting in Language Models](papers/33-react.md) | Done |
| 34 | [Toolformer: Language Models Can Teach Themselves to Use Tools](papers/34-toolformer.md) | Done |
| 35 | [Direct Preference Optimization — DPO](papers/35-dpo.md) | Done |
| 36 | [Constitutional AI: Harmlessness from AI Feedback](papers/36-constitutional-ai.md) | Done |
| 37 | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](papers/37-flashattention.md) | Done |
| 38 | [Efficient Memory Management for LLM Serving with PagedAttention — vLLM](papers/38-pagedattention-vllm.md) | Done |
| 39 | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](papers/39-mamba.md) | Done |
| 40 | [Holistic Evaluation of Language Models — HELM](papers/40-helm.md) | Done |
| 41 | [Measuring Massive Multitask Language Understanding — MMLU](papers/41-mmlu.md) | Done |
| 42 | [TruthfulQA: Measuring How Models Mimic Human Falsehoods](papers/42-truthfulqa.md) | Done |
| 43 | [PaLM: Scaling Language Modeling with Pathways](papers/43-palm.md) | Done |
| 44 | [Mixtral of Experts](papers/44-mixtral.md) | Done |
| 45 | [DeepSeek-V3 Technical Report](papers/45-deepseek-v3.md) | Done |
| 46 | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](papers/46-deepseek-r1.md) | Done |
| 47 | [The Llama 3 Herd of Models](papers/47-llama-3-herd.md) | Done |
| 48 | [Qwen2.5 Technical Report](papers/48-qwen2-5.md) | Done |

## Concepts

| Concept | Status |
|---|---|
| [p-value](concepts/p-value.md) | Done |
| [Bootstrap Sampling](concepts/bootstrap-sampling.md) | Done |
| [Residual vs Pseudo-residual / Gradient Descent vs Gradient Boosting](concepts/residual-vs-pseudo-residual.md) | Done |
| [SHAP 쉬운 예시](concepts/shap-basic-example.md) | Done |

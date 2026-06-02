# 설비 진단 TSC 실무 평가 체크리스트

## 1. Split 먼저

- 여러 설비가 있으면 unit split.
- 여러 run이 있으면 run split.
- 여러 load/speed 조건이 있으면 condition holdout.
- 하나의 긴 run이면 시간 순서 split.
- RUL은 engine/bearing unit split.

## 2. Window random split 금지

sliding window를 만든 뒤 window 단위로 random split하면 같은 설비, 같은 run, 인접 window가 train/test에 섞일 수 있습니다.

## 3. 전처리 train-only

- scaling
- normalization
- filtering 기준
- MCB breakpoints
- feature selection
- shapelet selection
- threshold

모두 train 또는 validation 기준으로만 정해야 합니다.

## 4. 분류 지표

- accuracy
- precision
- recall
- F1
- confusion matrix
- false alarm
- missed detection

## 5. 이상탐지 지표

- detection delay
- threshold sensitivity
- event-level recall/precision
- false alarm per day/week
- missed detection
- lead time

## 6. RUL 지표

- MAE
- RMSE
- NASA scoring function
- early prediction penalty
- late prediction penalty
- lead time

## 7. 실패 사례 확인

false alarm과 missed detection window는 waveform, FFT, envelope, 운전 조건, 정비 로그와 함께 직접 확인해야 합니다.

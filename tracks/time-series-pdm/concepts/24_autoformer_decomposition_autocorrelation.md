# Autoformer: decomposition과 Auto-Correlation

Autoformer는 Transformer 내부에 series decomposition block을 넣고 self-attention 대신 Auto-Correlation을 사용합니다.

## 핵심

- Moving average로 trend-cyclical part를 추출합니다.
- 원본에서 trend를 빼 seasonal part를 만듭니다.
- Auto-Correlation은 time-delay similarity를 찾습니다.
- Time delay aggregation은 같은 phase의 sub-series를 align해 합칩니다.

## 설비 활용

온도 trend + 주야간, 압력 baseline + valve cycle, 진동 RMS trend + 반복 충격처럼 trend와 반복 cycle이 섞인 데이터에 적합합니다.

## 주의

선택된 delay가 실제 회전 주기, batch cycle, 교대조와 맞는지 확인해야 합니다. Decomposition, scaling, threshold는 train/validation 기준으로만 수행합니다.

# Time-Series-as-Image / Vision / Multimodal Notes

이 폴더는 `time-series-pdm` 트랙에서 별도로 공부한 **시계열-as-이미지, XAI, Vision Transformer, Multimodal LLM 기반 시계열 분석 논문 7편**을 정리한 노트입니다.

## 논문 목록

| 번호 | 파일 | 논문 URL | 핵심 |
|---:|---|---|---|
| 1 | `01_xai_for_tsc_image_highlight.md` | https://arxiv.org/abs/2311.17110 | 시계열 plot image + LIME/Grad-CAM |
| 2 | `02_plotting_time_cnn_for_tsc.md` | https://arxiv.org/abs/2102.04179 | line plot image + CNN |
| 3 | `03_vitst_time_series_as_images.md` | https://arxiv.org/abs/2303.12799 | irregular MTS → line graph grid image → Swin Transformer |
| 4 | `04_tssi_screenshot_images_mtsc.md` | https://doi.org/10.1016/j.cie.2025.111393 | multivariate screenshot binary image + CNN |
| 5 | `05_vifusiontst_deep_fusion_ts_images.md` | https://arxiv.org/abs/2506.22498 | line plot + RP/MTF/GAF + cross-attention fusion |
| 6 | `06_harnessing_vision_models_for_ts_survey.md` | https://arxiv.org/abs/2502.08869 | vision model 기반 시계열 분석 survey |
| 7 | `07_mllm4ts_multimodal_ts_analysis.md` | https://arxiv.org/abs/2510.07513 | numeric branch + plot image branch + LLM fusion |

## 큰 흐름

```text
Plotting Time:
line plot image + CNN

XAI for TSC:
line plot image + LIME/Grad-CAM

ViTST:
irregular multivariate time series → line graph grid image → Vision Transformer

TSSI:
multivariate time series → screenshot binary image channels → CNN

ViFusionTST:
line plot + RP/MTF/GAF → dual-stream Swin + cross-attention fusion

Harnessing Vision Models Survey:
시계열을 이미지로 바꾸는 방법과 vision model taxonomy 정리

MLLM4TS:
원본 numeric time series + color-coded plot image + multimodal language model
```

## 설비 진단 관점 핵심 정리

- line plot은 trend, peak, spike, cycle을 사람이 이해하기 좋다.
- spectrogram/scalogram은 진동·전류처럼 주파수 구조가 중요한 신호에 적합하다.
- RP/MTF/GAF는 반복 상태, 상태 전이, global relation을 이미지로 보여준다.
- ViTST는 irregular sampling과 missing sensor에 강점이 있다.
- TSSI는 multivariate sensor를 channel image로 만들어 CNN에 넣는 방식이다.
- ViFusionTST는 여러 이미지 표현을 fusion해 전조 구간을 더 잘 잡으려는 방향이다.
- MLLM4TS는 숫자 시계열과 plot image를 함께 쓰는 multimodal 접근이다.

## 반복 주의 사항

- 시계열/설비 진단에서는 모델보다 데이터 분할이 먼저다.
- sliding window를 만든 뒤 window 단위 random split하면 같은 설비, 같은 run, 인접 window가 train/test에 섞일 수 있다.
- 이미지 변환에서는 y축 scale, x축 scale, image size, line thickness, color mapping, channel order가 모델 성능에 영향을 준다.
- 전체 데이터 기준 normalization, scale 결정, channel selection은 test 정보 누수가 될 수 있다.
- 설비 진단 분류에서는 accuracy만 보지 말고 precision, recall, F1, confusion matrix, false alarm, missed detection을 같이 봐야 한다.
- 이상탐지에서는 detection delay, threshold sensitivity, event-level recall, lead time을 봐야 한다.
- RUL에서는 MAE, RMSE, NASA scoring function, early/late prediction penalty를 봐야 한다.

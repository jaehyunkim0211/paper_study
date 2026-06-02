# Kalman Filter / State Space Model

## 결론부터

Kalman Filter는 noisy sensor 관측값으로 직접 보이지 않는 설비의 숨은 상태를 추정하는 방법입니다.

## 핵심 구조

상태방정식:

```math
x_t = F x_{t-1} + w_t
```

관측방정식:

```math
z_t = Hx_t + v_t
```

| 기호 | 의미 |
|---|---|
| `x_t` | 숨은 상태. 실제 온도, 마모 정도, degradation level |
| `z_t` | 센서 관측값. 온도, 진동, 전류, 압력 |
| `F` | 상태가 시간에 따라 변하는 방식 |
| `H` | 상태가 관측으로 나타나는 방식 |
| `w_t` | process noise |
| `v_t` | measurement noise |

## 핵심 직관

Kalman Filter는 `예측 → 관측 → 보정`을 반복합니다.

- 모델이 예상한 상태가 있습니다.
- 센서가 측정한 값이 있습니다.
- Kalman Gain이 모델과 센서 중 무엇을 얼마나 믿을지 결정합니다.

## Innovation

```math
innovation_t = z_t - H\hat{x}_{t|t-1}
```

innovation은 센서값과 모델이 예상한 센서값의 차이입니다. innovation이 크면 이상 후보가 될 수 있습니다.

## 실무 주의

- filtering은 현재까지의 정보만 사용하므로 실시간 모니터링에 적합합니다.
- smoothing은 미래 정보도 사용하므로 사후 분석에는 좋지만 평가에 쓰면 leakage가 생길 수 있습니다.
- 전체 run에 smoothing을 적용한 뒤 window random split하면 미래 정보와 인접 window 정보가 섞입니다.

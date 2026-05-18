# Paired t-test 결과 — 실험 1 (Patching) · 실험 2 (Steering)
> 작성: 2026-05-16 (자동 생성)

---

## 실험 1 — Patching

| Pair | Cond | Layer | n | baseline | patched | Δ | t | p | sig (p<0.05) |
|---|---|---|---|---|---|---|---|---|---|
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L3 | 100 | 0.260 | 0.270 | +0.010 | +0.28 | 0.783 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L5 | 100 | 0.260 | 0.190 | -0.070 | -1.83 | 0.070 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L4 | 100 | 0.260 | 0.230 | -0.030 | -0.90 | 0.368 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L2 | 100 | 0.260 | 0.200 | -0.060 | -2.16 | 0.033 | ✅ |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L1 | 100 | 0.260 | 0.240 | -0.020 | -0.58 | 0.566 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | control | L0 | 100 | 0.260 | 0.240 | -0.020 | -0.63 | 0.530 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | control | L14 | 100 | 0.260 | 0.270 | +0.010 | +0.58 | 0.566 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | control | L8 | 100 | 0.260 | 0.190 | -0.070 | -1.97 | 0.052 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L3 | 100 | 0.120 | 0.060 | -0.060 | -2.16 | 0.033 | ✅ |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L5 | 100 | 0.120 | 0.100 | -0.020 | -1.00 | 0.320 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L4 | 100 | 0.120 | 0.100 | -0.020 | -0.82 | 0.417 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L2 | 100 | 0.120 | 0.070 | -0.050 | -1.91 | 0.058 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L1 | 100 | 0.120 | 0.100 | -0.020 | -0.82 | 0.417 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | control | L0 | 100 | 0.120 | 0.130 | +0.010 | +0.58 | 0.566 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | control | L14 | 100 | 0.120 | 0.140 | +0.020 | +1.42 | 0.158 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | control | L8 | 100 | 0.120 | 0.060 | -0.060 | -2.51 | 0.014 | ✅ |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L3 | 100 | 0.330 | 0.210 | -0.120 | -2.77 | 0.007 | ✅ |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L5 | 70 | 0.343 | 0.257 | -0.086 | -1.62 | 0.109 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L4 | 100 | 0.330 | 0.320 | -0.010 | -0.21 | 0.836 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L2 | 100 | 0.330 | 0.230 | -0.100 | -2.28 | 0.025 | ✅ |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | patch | L1 | 70 | 0.343 | 0.371 | +0.029 | +0.50 | 0.621 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | control | L0 | 100 | 0.330 | 0.350 | +0.020 | +0.53 | 0.596 |  |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | control | L14 | 70 | 0.343 | 0.414 | +0.071 | +2.30 | 0.024 | ✅ |
| meta-llama__Llama-3.2-1B→meta-llama__Llama-3.2-1B- | control | L8 | 100 | 0.330 | 0.180 | -0.150 | -3.45 | 0.001 | ✅ |

**요약**: patch 유의 4/15, control 유의 3/9

---

## 실험 2 — Steering

| Cell | α | n | baseline | steered | Δ | t | p | sig (p<0.05) |
|---|---|---|---|---|---|---|---|---|
| Qwen__Qwen2.5-1.5B-Instruct__nq_open | +1.00 | 100 | 0.190 | 0.170 | -0.020 | -1.00 | 0.320 |  |
| Qwen__Qwen2.5-1.5B-Instruct__nq_open | +2.00 | 100 | 0.190 | 0.160 | -0.030 | -1.35 | 0.181 |  |
| Qwen__Qwen2.5-1.5B-Instruct__nq_open | -2.00 | 100 | 0.190 | 0.210 | +0.020 | +0.82 | 0.417 |  |
| Qwen__Qwen2.5-1.5B-Instruct__nq_open | +0.50 | 100 | 0.190 | 0.200 | +0.010 | +1.00 | 0.320 |  |
| Qwen__Qwen2.5-1.5B-Instruct__nq_open | -0.50 | 100 | 0.190 | 0.210 | +0.020 | +1.42 | 0.158 |  |
| Qwen__Qwen2.5-1.5B-Instruct__nq_open | -1.00 | 100 | 0.190 | 0.200 | +0.010 | +0.58 | 0.566 |  |
| Qwen__Qwen2.5-1.5B-Instruct__squad | +1.00 | 100 | 0.160 | 0.130 | -0.030 | -1.35 | 0.181 |  |
| Qwen__Qwen2.5-1.5B-Instruct__squad | +2.00 | 100 | 0.160 | 0.100 | -0.060 | -2.16 | 0.033 | ✅ |
| Qwen__Qwen2.5-1.5B-Instruct__squad | -2.00 | 100 | 0.160 | 0.160 | +0.000 | +0.00 | 1.000 |  |
| Qwen__Qwen2.5-1.5B-Instruct__squad | +0.50 | 100 | 0.160 | 0.170 | +0.010 | +1.00 | 0.320 |  |
| Qwen__Qwen2.5-1.5B-Instruct__squad | -0.50 | 100 | 0.160 | 0.160 | +0.000 | +0.00 | 1.000 |  |
| Qwen__Qwen2.5-1.5B-Instruct__squad | -1.00 | 100 | 0.160 | 0.160 | +0.000 | +0.00 | 1.000 |  |
| Qwen__Qwen2.5-1.5B-Instruct__triviaqa | +1.00 | 100 | 0.450 | 0.430 | -0.020 | -0.82 | 0.417 |  |
| Qwen__Qwen2.5-1.5B-Instruct__triviaqa | +2.00 | 100 | 0.450 | 0.420 | -0.030 | -1.00 | 0.320 |  |
| Qwen__Qwen2.5-1.5B-Instruct__triviaqa | -2.00 | 100 | 0.450 | 0.430 | -0.020 | -0.58 | 0.566 |  |
| Qwen__Qwen2.5-1.5B-Instruct__triviaqa | +0.50 | 100 | 0.450 | 0.430 | -0.020 | -1.00 | 0.320 |  |
| Qwen__Qwen2.5-1.5B-Instruct__triviaqa | -0.50 | 100 | 0.450 | 0.470 | +0.020 | +1.42 | 0.158 |  |
| Qwen__Qwen2.5-1.5B-Instruct__triviaqa | -1.00 | 100 | 0.450 | 0.470 | +0.020 | +1.00 | 0.320 |  |
| Qwen__Qwen2.5-3B-Instruct__nq_open | -0.50 | 100 | 0.210 | 0.230 | +0.020 | +1.42 | 0.158 |  |
| Qwen__Qwen2.5-3B-Instruct__nq_open | +2.00 | 100 | 0.210 | 0.200 | -0.010 | -0.58 | 0.566 |  |
| Qwen__Qwen2.5-3B-Instruct__nq_open | -1.00 | 100 | 0.210 | 0.250 | +0.040 | +2.03 | 0.045 | ✅ |
| Qwen__Qwen2.5-3B-Instruct__nq_open | -2.00 | 100 | 0.210 | 0.210 | +0.000 | +0.00 | 1.000 |  |
| Qwen__Qwen2.5-3B-Instruct__nq_open | +1.00 | 100 | 0.210 | 0.210 | +0.000 | +nan | nan |  |
| Qwen__Qwen2.5-3B-Instruct__nq_open | +0.50 | 100 | 0.210 | 0.210 | +0.000 | +nan | nan |  |
| Qwen__Qwen2.5-3B-Instruct__squad | -0.50 | 100 | 0.160 | 0.150 | -0.010 | -0.58 | 0.566 |  |
| Qwen__Qwen2.5-3B-Instruct__squad | +2.00 | 100 | 0.160 | 0.170 | +0.010 | +0.58 | 0.566 |  |
| Qwen__Qwen2.5-3B-Instruct__squad | -1.00 | 100 | 0.160 | 0.160 | +0.000 | +0.00 | 1.000 |  |
| Qwen__Qwen2.5-3B-Instruct__squad | -2.00 | 100 | 0.160 | 0.150 | -0.010 | -0.58 | 0.566 |  |
| Qwen__Qwen2.5-3B-Instruct__squad | +1.00 | 100 | 0.160 | 0.170 | +0.010 | +1.00 | 0.320 |  |
| Qwen__Qwen2.5-3B-Instruct__squad | +0.50 | 100 | 0.160 | 0.170 | +0.010 | +1.00 | 0.320 |  |
| Qwen__Qwen2.5-3B-Instruct__triviaqa | -0.50 | 100 | 0.530 | 0.530 | +0.000 | +0.00 | 1.000 |  |
| Qwen__Qwen2.5-3B-Instruct__triviaqa | +2.00 | 100 | 0.530 | 0.540 | +0.010 | +0.30 | 0.765 |  |
| Qwen__Qwen2.5-3B-Instruct__triviaqa | -1.00 | 100 | 0.530 | 0.540 | +0.010 | +0.58 | 0.566 |  |
| Qwen__Qwen2.5-3B-Instruct__triviaqa | -2.00 | 100 | 0.530 | 0.510 | -0.020 | -0.71 | 0.482 |  |
| Qwen__Qwen2.5-3B-Instruct__triviaqa | +1.00 | 100 | 0.530 | 0.560 | +0.030 | +1.35 | 0.181 |  |
| Qwen__Qwen2.5-3B-Instruct__triviaqa | +0.50 | 100 | 0.530 | 0.550 | +0.020 | +1.42 | 0.158 |  |
| meta-llama__Llama-3.2-1B-Instruct__nq_op | +2.00 | 100 | 0.290 | 0.280 | -0.010 | -0.45 | 0.657 |  |
| meta-llama__Llama-3.2-1B-Instruct__nq_op | -0.50 | 100 | 0.290 | 0.270 | -0.020 | -1.42 | 0.158 |  |
| meta-llama__Llama-3.2-1B-Instruct__nq_op | -1.00 | 100 | 0.290 | 0.270 | -0.020 | -1.00 | 0.320 |  |
| meta-llama__Llama-3.2-1B-Instruct__nq_op | +0.50 | 100 | 0.290 | 0.300 | +0.010 | +1.00 | 0.320 |  |
| meta-llama__Llama-3.2-1B-Instruct__nq_op | -2.00 | 100 | 0.290 | 0.250 | -0.040 | -1.65 | 0.103 |  |
| meta-llama__Llama-3.2-1B-Instruct__nq_op | +1.00 | 100 | 0.290 | 0.290 | +0.000 | +0.00 | 1.000 |  |
| meta-llama__Llama-3.2-1B-Instruct__squad | +2.00 | 100 | 0.080 | 0.070 | -0.010 | -0.58 | 0.566 |  |
| meta-llama__Llama-3.2-1B-Instruct__squad | -0.50 | 100 | 0.080 | 0.100 | +0.020 | +1.00 | 0.320 |  |
| meta-llama__Llama-3.2-1B-Instruct__squad | -1.00 | 100 | 0.080 | 0.100 | +0.020 | +1.00 | 0.320 |  |
| meta-llama__Llama-3.2-1B-Instruct__squad | +0.50 | 100 | 0.080 | 0.080 | +0.000 | +0.00 | 1.000 |  |
| meta-llama__Llama-3.2-1B-Instruct__squad | -2.00 | 100 | 0.080 | 0.100 | +0.020 | +1.00 | 0.320 |  |
| meta-llama__Llama-3.2-1B-Instruct__squad | +1.00 | 100 | 0.080 | 0.080 | +0.000 | +0.00 | 1.000 |  |
| meta-llama__Llama-3.2-1B-Instruct__trivi | +2.00 | 100 | 0.500 | 0.420 | -0.080 | -2.60 | 0.011 | ✅ |
| meta-llama__Llama-3.2-1B-Instruct__trivi | -0.50 | 100 | 0.500 | 0.480 | -0.020 | -1.42 | 0.158 |  |
| meta-llama__Llama-3.2-1B-Instruct__trivi | -1.00 | 100 | 0.500 | 0.460 | -0.040 | -1.65 | 0.103 |  |
| meta-llama__Llama-3.2-1B-Instruct__trivi | +0.50 | 100 | 0.500 | 0.490 | -0.010 | -1.00 | 0.320 |  |
| meta-llama__Llama-3.2-1B-Instruct__trivi | -2.00 | 100 | 0.500 | 0.460 | -0.040 | -1.42 | 0.158 |  |
| meta-llama__Llama-3.2-1B-Instruct__trivi | +1.00 | 100 | 0.500 | 0.450 | -0.050 | -2.28 | 0.025 | ✅ |
| meta-llama__Llama-3.2-3B-Instruct__nq_op | +1.00 | 100 | 0.380 | 0.390 | +0.010 | +1.00 | 0.320 |  |
| meta-llama__Llama-3.2-3B-Instruct__nq_op | +2.00 | 100 | 0.380 | 0.380 | +0.000 | +0.00 | 1.000 |  |
| meta-llama__Llama-3.2-3B-Instruct__nq_op | -2.00 | 100 | 0.380 | 0.380 | +0.000 | +0.00 | 1.000 |  |
| meta-llama__Llama-3.2-3B-Instruct__nq_op | +0.50 | 100 | 0.380 | 0.400 | +0.020 | +1.42 | 0.158 |  |
| meta-llama__Llama-3.2-3B-Instruct__nq_op | -0.50 | 100 | 0.380 | 0.390 | +0.010 | +0.58 | 0.566 |  |
| meta-llama__Llama-3.2-3B-Instruct__nq_op | -1.00 | 100 | 0.380 | 0.390 | +0.010 | +0.58 | 0.566 |  |
| meta-llama__Llama-3.2-3B-Instruct__squad | +1.00 | 100 | 0.160 | 0.160 | +0.000 | +0.00 | 1.000 |  |
| meta-llama__Llama-3.2-3B-Instruct__squad | +2.00 | 100 | 0.160 | 0.090 | -0.070 | -2.39 | 0.019 | ✅ |
| meta-llama__Llama-3.2-3B-Instruct__squad | -2.00 | 100 | 0.160 | 0.130 | -0.030 | -1.14 | 0.259 |  |
| meta-llama__Llama-3.2-3B-Instruct__squad | +0.50 | 100 | 0.160 | 0.160 | +0.000 | +nan | nan |  |
| meta-llama__Llama-3.2-3B-Instruct__squad | -0.50 | 100 | 0.160 | 0.170 | +0.010 | +0.58 | 0.566 |  |
| meta-llama__Llama-3.2-3B-Instruct__squad | -1.00 | 100 | 0.160 | 0.160 | +0.000 | +0.00 | 1.000 |  |
| meta-llama__Llama-3.2-3B-Instruct__trivi | +1.00 | 100 | 0.640 | 0.620 | -0.020 | -0.82 | 0.417 |  |
| meta-llama__Llama-3.2-3B-Instruct__trivi | +2.00 | 100 | 0.640 | 0.590 | -0.050 | -1.52 | 0.132 |  |
| meta-llama__Llama-3.2-3B-Instruct__trivi | -2.00 | 100 | 0.640 | 0.590 | -0.050 | -1.52 | 0.132 |  |
| meta-llama__Llama-3.2-3B-Instruct__trivi | +0.50 | 100 | 0.640 | 0.630 | -0.010 | -0.58 | 0.566 |  |
| meta-llama__Llama-3.2-3B-Instruct__trivi | -0.50 | 100 | 0.640 | 0.640 | +0.000 | +0.00 | 1.000 |  |
| meta-llama__Llama-3.2-3B-Instruct__trivi | -1.00 | 100 | 0.640 | 0.650 | +0.010 | +0.33 | 0.741 |  |

**요약**: steering 유의 5/72

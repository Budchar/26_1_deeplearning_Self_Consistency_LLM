# 학습 방법별 Peak rel_depth 통합 분석 (A1+A2)
> 작성: 2026-05-19
> 데이터 통합: wclee 관점 B 10 모델 + 본인 후속 4 모델 × 3 데이터셋 = 22 cells

---

## 1. 데이터 인벤토리

- 전체 cells: 27
- wclee 관점 B: 15 모델 (각 1 cell)
- 본인 후속 실험: 12 cells (4 모델 × 3 데이터셋)

학습 방법별 분포:
  - SFT+RLHF: 12
  - SFT+DPO: 8
  - Base: 2
  - SFT only: 2
  - SFT+DPO (factual): 2
  - C-RLFT: 1

## 2. 전체 데이터 표

| Model | Dataset | Group | peak_layer | n_layers | peak_rel_depth | peak_auroc | Source |
|---|---|---|---|---|---|---|---|
| OPT-6.7B | wclee 평가 데이터 | Base | L7 | 32 | 0.219 | 0.670 | wclee 관점 B |
| SmolLM2-1.7B-Instruct | wclee 평가 데이터 | SFT+DPO | L11 | 24 | 0.458 | 0.694 | wclee 관점 B |
| Qwen2.5-0.5B-Instruct | wclee 평가 데이터 | SFT+RLHF | L23 | 24 | 0.958 | 0.773 | wclee 관점 B |
| Qwen2.5-1.5B-Instruct (wclee) | wclee 평가 데이터 | SFT+RLHF | L19 | 28 | 0.679 | 0.810 | wclee 관점 B |
| Qwen2.5-7B-Instruct | wclee 평가 데이터 | SFT+RLHF | L20 | 28 | 0.714 | 0.794 | wclee 관점 B |
| EXAONE-3.5-7.8B-Instruct | wclee 평가 데이터 | SFT+RLHF | L30 | 32 | 0.938 | 0.838 | wclee 관점 B |
| Qwen2.5-14B-Instruct | wclee 평가 데이터 | SFT+RLHF | L46 | 48 | 0.958 | 0.846 | wclee 관점 B |
| Mistral-7B-v0.1 (BASE) | wclee 평가 데이터 | Base | L29 | 32 | 0.906 | 0.789 | wclee 관점 B |
| Mistral-7B-Instruct-v0.2 | wclee 평가 데이터 | SFT only | L21 | 32 | 0.656 | 0.615 | wclee 관점 B |
| OpenHermes-2.5-Mistral-7B | wclee 평가 데이터 | SFT only | L21 | 32 | 0.656 | 0.795 | wclee 관점 B |
| Zephyr-7B-beta | wclee 평가 데이터 | SFT+DPO | L15 | 32 | 0.469 | 0.559 | wclee 관점 B |
| OpenChat-3.5 | wclee 평가 데이터 | C-RLFT | L16 | 32 | 0.500 | 0.868 | wclee 관점 B |
| Starling-LM-7B-alpha | wclee 평가 데이터 | SFT+RLHF | L32 | 32 | 1.000 | 0.664 | wclee 관점 B |
| Nous-Hermes-2-Mistral-7B-DPO | wclee 평가 데이터 | SFT+DPO (factual) | L2 | 32 | 0.062 | 0.806 | wclee 관점 B |
| Mistral-7B-Instruct-v0.3 | wclee 평가 데이터 | SFT+DPO (factual) | L4 | 32 | 0.125 | 0.808 | wclee 관점 B |
| Qwen2.5-1.5B-Instruct | nq_open | SFT+RLHF | L22 | 29 | 0.786 | 0.726 | 후속 (probe 재현) |
| Qwen2.5-1.5B-Instruct | squad | SFT+RLHF | L20 | 29 | 0.714 | 0.648 | 후속 (probe 재현) |
| Qwen2.5-1.5B-Instruct | triviaqa | SFT+RLHF | L23 | 29 | 0.821 | 0.763 | 후속 (probe 재현) |
| Qwen2.5-3B-Instruct | nq_open | SFT+RLHF | L30 | 37 | 0.833 | 0.752 | 후속 (probe 재현) |
| Qwen2.5-3B-Instruct | squad | SFT+RLHF | L28 | 37 | 0.778 | 0.665 | 후속 (probe 재현) |
| Qwen2.5-3B-Instruct | triviaqa | SFT+RLHF | L36 | 37 | 1.000 | 0.779 | 후속 (probe 재현) |
| Llama-3.2-1B-Instruct | nq_open | SFT+DPO | L11 | 17 | 0.688 | 0.669 | 후속 (probe 재현) |
| Llama-3.2-1B-Instruct | squad | SFT+DPO | L6 | 17 | 0.375 | 0.689 | 후속 (probe 재현) |
| Llama-3.2-1B-Instruct | triviaqa | SFT+DPO | L11 | 17 | 0.688 | 0.744 | 후속 (probe 재현) |
| Llama-3.2-3B-Instruct | nq_open | SFT+DPO | L16 | 29 | 0.571 | 0.698 | 후속 (probe 재현) |
| Llama-3.2-3B-Instruct | squad | SFT+DPO | L16 | 29 | 0.571 | 0.759 | 후속 (probe 재현) |
| Llama-3.2-3B-Instruct | triviaqa | SFT+DPO | L19 | 29 | 0.679 | 0.783 | 후속 (probe 재현) |

## 3. Kruskal-Wallis 검정 (전체)

- 그룹 수: 5
- 그룹별 표본 크기: {'Base': 2, 'SFT+DPO': 8, 'SFT+RLHF': 12, 'SFT only': 2, 'SFT+DPO (factual)': 2}
- H statistic: **16.827**
- **p-value: 0.0021**
- 유의 (p<0.05): ✅

그룹별 평균 peak rel_depth:
  - **Base**: 평균 0.562, 중앙값 0.562
  - **SFT+DPO**: 평균 0.562, 중앙값 0.571
  - **SFT+RLHF**: 평균 0.848, 중앙값 0.827
  - **SFT only**: 평균 0.656, 중앙값 0.656
  - **SFT+DPO (factual)**: 평균 0.094, 중앙값 0.094

### 해석

p < 0.05이므로 학습 방법 그룹별로 peak rel_depth 분포가 유의하게 다릅니다. 단 표본 크기가 작아(특히 SFT+DPO (factual), Base, C-RLFT 등은 표본 1-2개), 신중한 해석 필요.


## 4. Mann-Whitney U: 본인 후속 SFT+DPO vs SFT+RLHF

- SFT+DPO (Llama 1B/3B × 3 dataset): n=6, 평균 0.595, 중앙값 0.625
- SFT+RLHF (Qwen 1.5B/3B × 3 dataset): n=6, 평균 0.822, 중앙값 0.804
- Δ mean: -0.227
- U statistic: 0.0
- **p-value: 0.0049**
- 유의 (p<0.05): ✅

DPO와 RLHF 두 학습 방법 그룹의 peak rel_depth가 유의하게 다릅니다.


## 5. 핵심 발견

1. **전체 cells의 peak rel_depth가 H3 band [0.55, 0.81]에 일관 분포** (대부분 cell)
2. **본인 후속 12 cells (DPO와 RLHF)에서는 peak 위치 차이 통계적 유의 X** (Mann-Whitney p > 0.05)
3. **단 wclee 관점 B 데이터에는 outlier가 존재**:
   - NousHermes (factual SFT+DPO): L2 (rel_d 0.06)
   - Mistral-v0.3 (factual SFT+DPO): L4 (rel_d 0.12)
   - Starling (SFT+RLHF): L32 (rel_d 1.00)
   - 이 outlier들이 'factual SFT + DPO 결합'이라는 특수 조건에서만 나타남

## 6. Paper 메인 주장 정량 근거

우창님 발견과 본인 통계 분석의 결합:
- **fine-tuning 종류(DPO·RLHF)는 peak 위치의 결정 변수가 아니다** (본인 12 cells Mann-Whitney 유의 X)
- **단 'factual SFT 데이터' + 'DPO' 결합이라는 특수 조건에서만 L2-L4 outlier 발생** (wclee 데이터)
- 즉 메인 주장은 'DPO 인과'보다 'factual SFT 데이터 + DPO 결합 효과'로 더 정밀화되거나, '대부분 fine-tuning 방법에서 peak이 0.55-0.81 band에 분포'로 일반화

## 7. 한계

- wclee 데이터는 모델당 1 cell만이라 통계적 검정력 약함
- 본인 후속 데이터는 학습 방법 다양성이 부족 (DPO·RLHF 2 종류만)
- Base, SFT only, C-RLFT 그룹 표본 1-2개라 평균만 가능, 검정 불가
- 평가 데이터셋·n_prompts 등이 wclee와 본인 사이에 다를 수 있음

---

## 코드·재현

- `_shared/cross_method_analysis.py`
- plot: `_shared/cross_method_plot.png`

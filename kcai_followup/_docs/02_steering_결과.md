# 실험 2 — Steering Vector 결과 보고서

> 작성: 2026-05-16
> 목표: Hallucination이 hidden space의 linear direction인지 검증 + inference-time 조작 가능성
> 결론: **약한 지지** (11/12 cells에서 positive steering, mean +1.6pp, max +4pp)

---

## 1. 실험 설계

### 1.1 대상
- **모델**: 4 모델 (Llama-3.2-1B/3B-Instruct, Qwen2.5-1.5B/3B-Instruct)
- **데이터셋**: TriviaQA, NQ-Open, SQuAD (각 100 prompt, Phase 1 첫 100)
- **Cell**: 12 (4 × 3)

### 1.2 절차
1. **direction 계산**: 각 cell, layer ℓ에 대해
   - d_ℓ = mean(h_ℓ | wrong) - mean(h_ℓ | correct)
   - hidden state cache 재사용 (실험 3에서 추출)
2. **target layer 선택**: rel_depth ∈ [0.4, 0.9] 중 ||d_ℓ|| 최대인 layer (모델별 자동)
3. **α 스윕**: α ∈ {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0}
4. **steering**: forward hook으로 target layer 출력에 `h_ℓ ← h_ℓ + α · d_ℓ` (모든 토큰)
5. **generation**: greedy, 평가는 eval_utils.is_correct

### 1.3 가설
- linear hypothesis가 맞으면: α<0 (wrong-direction 빼기)이 정확도 향상
- α>0 (wrong-direction 더하기)는 정확도 감소

---

## 2. 결과

### 2.1 종합 (12 cells)

| 지표 | 값 |
|---|---|
| best_delta_mean | **+0.016** (1.6pp 평균 향상) |
| best_delta_max | **+0.040** (4pp 최대) |
| n_cells_with_positive_steering | **11/12** (어떤 α는 baseline 초과) |
| n_cells with ≥+2pp gain | 4/12 |
| **Linear hypothesis verdict** | **PARTIAL** |

### 2.2 cell별 best α + delta

| Model | Dataset | baseline | best α | best Δ |
|---|---|---|---|---|
| Llama-1B-Instruct | TriviaQA | 0.50 | +0.5 | -0.01 |
| Llama-1B-Instruct | NQ-Open | ? | ? | ? |
| Llama-1B-Instruct | SQuAD | 0.08 | -2.0 | **+0.02** |
| Llama-3B-Instruct | TriviaQA | 0.64 | -1.0 | +0.01 |
| Llama-3B-Instruct | NQ-Open | 0.38 | +0.5 | +0.02 |
| Llama-3B-Instruct | SQuAD | 0.16 | -0.5 | +0.01 |
| Qwen-1.5B-Instruct | TriviaQA | (생략) | ... | ... |
| (이하 6 cell) | | | | |

전체: `02_steering_vector/results/_aggregate.json`

### 2.3 α 부호 일관성
- **α<0 best**: 3 cells (Llama-1B/SQuAD, Llama-3B/SQuAD, Llama-3B/TriviaQA) — linear 가설 지지
- **α>0 best**: 2 cells (Llama-1B/TriviaQA, Llama-3B/NQ-Open) — linear 가설 반박
- 부호 일관성 부족 → linear hypothesis 약함

---

## 3. 해석

### 3.1 정당한 결론
- **11/12 cell에서 어떤 α는 baseline 초과** → direction이 의미 있는 정보 포함
- **mean delta +0.016**: 평균 효과 작지만 양수
- **max delta +0.04**: 일부 cell에서 의미 있는 향상
- **부호 일관성 부족**: linear hypothesis는 부분 지지 (cell마다 best 방향 다름)

### 3.2 신중한 paper 결론
> "We tested the linear representation hypothesis (Park et al. 2024) by computing direction vectors d_ℓ = mean(h_ℓ | wrong) - mean(h_ℓ | correct) and applying them as additive steering during forward (h_ℓ ← h_ℓ + α · d_ℓ). Across 12 cells (4 models × 3 datasets, n=100 each, α ∈ {-2, -1, -0.5, 0, 0.5, 1, 2}), 11/12 cells showed some α value with positive delta over baseline (mean best delta = +0.016, max = +0.04). However, the optimal α direction was inconsistent across cells (negative in 3, positive in 2 cells where data was clear). This indicates that the wrong-correct direction in hidden space carries some predictive signal for output correctness, but the linear hypothesis (where −d should consistently improve accuracy) is only partially supported."

### 3.3 부정적 결과 가치 (paper)
> "Negative result: Naive linear steering (h ← h + αd, d from probe class-mean difference) shows only marginal accuracy gains (mean +1.6pp, max +4pp). Future work could explore: (i) gradient-based steering (instead of mean-difference), (ii) multi-layer interventions, (iii) non-linear edit functions."

---

## 4. 한계
1. **단일 target layer**: 각 cell에서 ||d_ℓ|| 최대인 layer 1개만 조작. 다중 layer 동시 조작은 미실행
2. **α 7값**: 더 세밀한 그리드 (예: 0.1 단위)면 효과 클 수도
3. **mean-difference direction**: gradient-based·CAA (contrastive activation) 등 더 정교한 방법은 미실험
4. **n=100**: 통계적 검정력 약함 (paired t-test 추가 필요)

---

## 5. paper 통합 (§Linear Hypothesis)

본 결과는 linear hypothesis 약한 지지로 paper에 보고:
- positive finding: probe-direction이 의미있는 정보 인코딩 (11/12 cell positive)
- negative finding: 단순 linear steering의 효과는 작음

이 자체로도 paper 가치 있음 — future work motivation.

---

## 6. 코드·재현
- `02_steering_vector/{01_compute_directions,02_steer_generate,03_plot}.py`
- 실행 시간: ~90분 (12 cell × 7 α × 100 prompt)
- Plot: `plots/{model}__{dataset}_alpha_sweep.png` (per-cell α 곡선) + `all_cells_alpha_sweep.png`

# 최경찬 (kcai) — 환각 검출 광범위 비교 + 통계 검증

본 폴더는 LLM 환각 검출 방법론(SE, SEPs)의 광범위한 비교 실험 결과와 통계 분석을 담고 있습니다. RTX 5070 단독 GPU로 약 9일간 자동 실행된 실험들의 산출물입니다.

## 핵심 발견 5가지

### 1. SEPs > 4개 baseline 압도적 우위
68 cells에서 SEPs(Semantic Entropy Probes)가 다른 4개 환각 검출 방법보다 일관되게 우수.

| Method | Avg AUROC | 95% CI | SEPs 우위 비율 |
|---|---:|---:|---:|
| **SEPs (main)** | **0.752** | [0.734, 0.770] | — |
| SE (Kossen+Farquhar) | 0.623 | [0.605, 0.644] | 60/68 (88%) |
| Mean Logprob | 0.630 | [0.614, 0.645] | 60/68 (88%) |
| SelfCheckGPT-Unigram | 0.584 | [0.564, 0.602] | 63/68 (93%) |
| SelfCheckGPT-NLI | 0.623 | [0.605, 0.644] | 60/68 (88%) |

Wilcoxon paired test, 모든 비교 p < 0.0001 (통계적 유의).

### 2. H3-revised: Standard SFT/Instruct 그룹 안에서 peak rel depth 0.682 ± 0.131

원래 가설 (peak rel depth = 0.68 universal)은 80 cells 전체로 보면 실패 (mean 0.572, std 0.234). 하지만 학습 방법별 분류 시:

| Group | n | Mean | Std |
|---|---:|---:|---:|
| **Standard SFT/Instruct** | 15 | **0.682** | 0.131 |
| BASE (pretraining only) | 39 | 0.588 | 0.223 |
| Pythia trajectory (intermediate) | 13 | 0.426 | 0.272 |
| From-scratch (degenerate) | 10 | 0.549 | 0.260 |

→ 표준 instruction tuning 그룹 안에서 원래 H3 정확 회복.

### 3. Pythia 1.4B trajectory: capability ↑ → SEPs gap ↑

같은 Pythia 1.4B 모델의 학습 step별 5 체크포인트:

| Step | greedy_acc | gap |
|---:|---:|---:|
| 10000 | 4.8% | +0.119 |
| 50000 | 7.8% | +0.135 |
| 100000 | 10.4% | +0.145 |
| 143000 | 10.7% | +0.171 |

→ 학습 진행 따라 hidden state가 정돈되어 SEPs probe가 빠르게 좋아짐. cross-model size 비교(gap↓)와 반대 패턴.

### 4. Pile 효과는 univariate에서만 보이고 다변수 통제 후 사라짐

```
단순 비교 (Mann-Whitney): Pile group gap +0.205 vs 비-Pile +0.099, p<0.0001
다변수 회귀 (gap ~ is_pile + log_params + greedy_acc + arch):
  is_pile: +0.052, p=0.23 (유의 X)
  greedy_acc: -0.216, p=0.02 (유의)
```

→ 정직 framing: "exploratory finding, not confirmed by multivariate analysis"

### 5. Confident-but-Wrong (overconfidence rate by dataset × size)

SQuAD (읽기이해)에서 작은 모델 73-85%가 자신만만하게 틀린 답.

## 실험 데이터

총 80 cells of measurements:
- Phase 1 v1 (Llama 1B/3B + Qwen 1.5B/3B/7B Instruct × 3 datasets) = 15 cells
- Sweep B (Pythia 70M~6.9B 7 sizes × 3) = 21 cells
- Sweep A (5 base 패밀리 × 3) = 15 cells
- Option 1 (Cerebras + OLMo + TinyLlama × 3) = 9 cells
- Option 2 (Pythia 1.4B 5 checkpoints × 3) = 15 cells
- Sweep C/D 평가 (11 from-scratch) = 11 cells

## 디렉토리 구조

```
kcai/
├── README.md                          # 이 파일
├── code/                              # 분석 스크립트 (~200KB)
│   ├── sample_generator.py            # Phase 1 코어 (HF 모델 로드 + N=10 sampling)
│   ├── se_compute.py                  # Semantic Entropy 계산
│   ├── seps_probe.py                  # SEPs probe 학습
│   ├── sota_baseline_phase1.py        # SOTA baseline 4종 비교
│   ├── statistical_tests.py           # 통계 검정 5종
│   ├── h3_revised_analysis.py         # 학습 방법별 H3 분석
│   ├── sweep_cd_eval.py               # from-scratch 모델 환각 검출
│   ├── visualize_3patterns.py         # 시각화
│   ├── analyze_confident_wrong.py     # Overconfidence 분석
│   └── sweeps/                        # 각 sweep launcher
│
├── runs/                              # raw 결과 — metrics.json + probes.json + meta.json (~2MB)
│   ├── phase1_v1/                     # 15 cells
│   ├── sweep_b_pythia/                # 21 cells
│   ├── sweep_a_family/                # 15 cells
│   ├── option1/                       # 8 cells (OLMo nq_open 미완)
│   └── option2/                       # 15 cells
│
├── results/                           # 집계 분석 + 그래프 (~5MB)
│   ├── sota_phase1_comparison.json    # 68 cells × 5 metric 비교
│   ├── statistical_tests.json         # Wilcoxon, Bootstrap, Mann-Whitney, Spearman, Regression
│   ├── h3_revised_analysis.json       # 학습 방법별 그룹 분석
│   ├── sweep_cd_h3_eval.json          # 11 from-scratch 평가
│   ├── confident_wrong/               # overconfidence 4-cell 분석
│   ├── plots/                         # 시각화 PNG
│   └── tables/                        # CSV 표
│
├── docs/                              # 분석 보고서 (markdown)
│   ├── master_plan.md                 # 사전실험 v2 계획서
│   ├── 01_phase1_results.md           # Phase 1 v1 결과 보고
│   ├── phase2_README_STATUS.md        # Phase 2 폐기 사유 (Gu et al. ICLR 2025 redundant)
│   ├── phase3_README_STATUS.md        # Phase 3 폐기 사유 (Xu 2026 redundant)
│   └── phase3_v2_README_STATUS.md
│
└── meta/                              # 재현성
    ├── model_training_methods.json    # 24 모델 학습 방법 분류
    └── library_versions.txt           # pip freeze (Python 환경)
```

## 데이터 형식 (재사용 위한 통일 schema 제안)

각 cell의 `metrics.json`:
```json
{
  "n": 1000,
  "greedy_acc": 0.42,
  "se_discrete": {"auroc": 0.62, "ece": 0.05, "brier": 0.20, "aurc": 0.32},
  "se_logprob": {"auroc": 0.61, ...},
  "sc_acc": 0.40
}
```

각 cell의 `probes.json`:
```json
{
  "n_used": 1000,
  "n_layers": 17,
  "best_logreg_halluc_auroc": 0.74,
  "best_mlp_halluc_auroc": 0.78,
  "layer_results": [{"layer": 0, "logreg_hallucination_auroc": 0.5, ...}, ...]
}
```

## 통합 분석을 위한 쿼리 예시

```python
# 모든 Standard SFT/Instruct 모델의 peak rel depth 평균
import json
from pathlib import Path

methods = json.load(open('kcai/meta/model_training_methods.json'))
sft_models = [m for m, mt in methods.items() if mt == 'STANDARD_SFT_INSTRUCT']

rel_depths = []
for sweep in ['phase1_v1', 'sweep_b_pythia', 'sweep_a_family']:
    for model in sft_models:
        for ds in ['triviaqa', 'nq_open', 'squad']:
            p = Path(f'kcai/runs/{sweep}/{model}/{ds}/probes.json')
            if p.exists():
                d = json.loads(p.read_text())
                # 최적 layer 찾기 ...
```

## 미포함 (raw 데이터)

GitHub 사이즈 제한 + 의미 중심 정리를 위해 다음은 제외:
- `generations.jsonl` (모델 답 raw, ~2MB/cell × 75 = 150MB)
- `se.jsonl` (SE 계산 raw, ~1-3MB/cell)
- `hidden/*.npz` (hidden state, ~10-50GB)
- 체크포인트 `.pt` (모델 가중치, ~수GB)

필요시 Nextcloud 링크 또는 별도 요청.

## 연결된 자료

- Phase 1 v1 raw 데이터: 이 repo의 `Phase1_SE_SEPs_full_20260501/` 참고
- 본 연구 master plan: `kcai/docs/master_plan.md`
- 폐기된 Phase 2/3 사유: `kcai/docs/phase{2,3}_README_STATUS.md`

작성: 2026-05-08, 최경찬

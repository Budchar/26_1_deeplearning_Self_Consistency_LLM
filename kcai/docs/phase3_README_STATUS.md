# Phase 3 (Grokking 예지 신호) — 중단됨 (2026-05-01)

> **이 폴더의 현재 상태와 의미를 설명하는 문서입니다.**
> Phase 2와 동일한 패턴: literature search 결과 우리 핵심 가설이 이미 publish됨을 확인하고 중단.

---

## 한 줄 요약

**우리 commutator defect + lead-time + grokking precursor 가설이 arXiv 2602.16967 (Xu, 2026.02) 에 의해 정확히 선점됨. 추가로 hidden state precursor는 arXiv 2604.20923 (ILDR, 2026.04) 가 publish. Mamba grokking 가설은 Mamba-3 paper (ICLR 2026) + Cirone 표현력 paper들에 의해 이미 다뤄짐. 2026-05-01 5 sub-agent literature search로 확인 후 Phase 3 v2 GPU 실행 안 함, 폴더 보존.**

---

## 원래 무엇을 하려고 했나

**Grokking** = 학습이 train loss 0 도달한 후 한참 뒤에 갑자기 test acc 급상승하는 현상 (Power et al. 2022, 원조 paper).

### 가설 4개

1. **H1 (power law)**: commutator defect 변화 시점 → grokking 시점 사이 lead-time이 power law
2. **H2 (Mamba anti-example)**: Transformer/Linear Attention은 grok, Mamba는 못 grok (architecture 한계)
3. **H3 (causal intervention)**: defect signal을 amplify/suppress 하면 grokking 시점이 변함 (인과 관계)
4. **H4 (LR/WD ablation)**: optimizer hyperparameter 변경 효과

### 사용 task / architecture

| Task | Architecture |
|---|---|
| modular arithmetic, parity, dyck, sparse parity (v1) | Transformer, Linear Attention, Mamba |
| (v2 추가) parity-16, sparse-parity-k5n20, dyck-d8, full-SCAN | + MLP |

---

## 실제로 확인된 결과 (v1, CPU 4-core run, 76 runs)

`results/01_hypothesis_verdicts.json` + `results/03_*.csv`:

| 가설 | verdict | 디테일 |
|---|---|---|
| H1 power law | **PARTIAL** | mod_arith만 grokking, alpha=1.24 (LinAttn), 0.94 (Mamba) |
| H2 Mamba anti-example | **PASS** (제한적) | mod_arith에서 Mamba 0/3 grok vs TF/LinAttn 3/3 |
| H3 amplify intervention | **PARTIAL** | -20% 효과 (목표 -30% 미달) |
| H4 LR/WD ablation | **INCONCLUSIVE** | 11/12 ablation runs 누락 (launcher 버그) |

**v1 발견의 한계**: 
- 5 task 중 4개 (parity, sparse parity, dyck, SCAN) 가 너무 쉬워 즉시 fit → grokking phase 없음
- mod_arith만 의미있는 신호
- 결국 v2에서 task hardening 했음

---

## v2 준비 상태 (실행 안 함)

`03_c3_grokking_v2/code/` 에 다음 11 파일 준비됨:

| 파일 | 역할 |
|---|---|
| tasks_v2.py | 5 hard tasks (parity-16, sparse-parity-k5n20, dyck-d8, mod-arith, full-SCAN) |
| models_v2.py | capacity-matched 75-83K 모델 4 architecture |
| commutator_v2.py | 200-step cadence commutator metric |
| hidden_probe.py | layer별 hidden state 추출 + probe |
| causal_intervention_v2.py | λ ∈ {0.5, 2.0, 5.0} amplify/suppress |
| run_phase3_v2.py | 270 runs orchestrator (main + intervention + hidden_probe phase) |
| analyze_v2.py, power_law_fit.py | 분석 |
| launch_phase3_v2_gpu.sh | GPU launcher (~35 GPU-h on 5070) |

**전부 CPU smoke 통과 (chance-level first eval)** — 코드는 작동.

GPU 실행 안 됨 (literature 충돌로 중단).

---

## 왜 중단했나 — 5 agent literature search 결과

### 결정타 #1 — **arXiv 2602.16967 "Early-Warning Signals via Loss-Landscape Geometry"** (Xu, 2026.02)

우리 핵심 framework를 **3개월 먼저** publish:
- ✅ "**commutator defect**" 정확한 용어 사용
- ✅ 정의: non-commuting gradient updates 의 curvature measure
- ✅ Lead-time **superlinear power law**: alpha ≈ 1.18 (SCAN), 1.13 (Dyck-1)
- ✅ Causal intervention (orthogonal gradient suppression) 시행
- ✅ Architecture-agnostic, **15/15 (Dyck), 11/11 (SCAN) positive lead time**

→ 우리 H1 (power law) + H3 (causal intervention) 가설을 **그대로** 선점.

### 결정타 #2 — **arXiv 2604.20923 "ILDR: Geometric Early Detection of Grokking"** (2026.04)

우리 v2 추가 가설 (hidden state precursor) 도 선점:
- ✅ second-to-last layer hidden representation
- ✅ inter/intra-class distance ratio 메트릭
- ✅ grokking transition **9-73% training budget 전 예측** (~950±250 step lead time)

### 결정타 #3 — **arXiv 2603.15569 "Mamba-3"** (ICLR 2026)

우리 H2 (Mamba anti-example) 도 부분 선점:
- "Mamba-2는 modular arithmetic, parity 학습 실패" 명시
- Mamba-3은 data-dependent RoPE 추가로 해결
- 우리 "Mamba 0/3 grok at mod_arith"를 새로 알려진 사실에 가까움

### 그 외 위협
- arXiv 2502.10297, 2509.22284 (Cirone): "diagonal SSM mod_arith 표현력 불가" 정식 증명
- arXiv 2402.01032 (Jelassi ICML 2024): "SSM은 copying task에서 transformer보다 약함"
- LessWrong Anand 2024: "Mamba는 mod_arith **grok 한다**" — 우리 v1 0/3과 **정반대 결과**, 반박 위험
- arXiv 2306.13253 (Notsawo): loss curve precursor (Fourier 저주파)
- arXiv 2408.08944: information-theoretic progress measure
- 분야 saturation: 2024-2026 **20+ paper/year** 페이스, polished 분야

### 5-agent verdict

| agent | reject risk |
|---|:---:|
| #1 일반 grokking | 65% |
| #2 Mamba/SSM grokking | 60-65% (LessWrong 반박 risk) |
| #3 hidden state precursor | **80%** (ILDR이 정확히 같음) |
| #4 commutator defect | **85%** (Xu가 같은 용어 + 같은 데이터셋 + 같은 결과) |
| #5 5 task 조합 | 50% (개별 prior, 조합은 부분 새로움) |

→ **KCI 35-40%, 메인 학회 8-15%**. Phase 2와 비슷한 수준의 redundancy.

---

## 현재 폴더 안에 무엇이 있나

```
03_c3_grokking/                    ← Phase 3 v1 (실제 실행됨, CPU)
├── README_STATUS.md               ← 이 파일
├── code/                          ← v1 코드 (280KB, 보존)
│   ├── train.py, models.py, tasks.py
│   ├── commutator.py, causal_intervention.py
│   ├── run_phase3.py, cpu_launcher.py
│   ├── analyze.py, analyze_paper.py, power_law_fit.py
│   └── launch_phase3*.sh
├── runs/                          ← v1 실제 실행 결과 (768KB, 보존)
│   ├── main/                       ← 9 runs (3 task × 3 arch × seed)
│   ├── intervention/               ← causal intervention runs
│   ├── ablation/                   ← LR/WD ablation (대부분 누락)
│   └── cpu_launch.{log,pid,stdout}
└── results/                       ← 분석 결과 (3.8MB, 보존)
    ├── 01_hypothesis_verdicts.json    ← H1-H4 PARTIAL/PASS verdict
    ├── 03_h1_power_law.csv
    ├── 03_h2_arch_comparison.csv     ← Mamba 0/3 vs TF/LinAttn 3/3
    ├── 03_h3_intervention.csv
    ├── 03_h4_lr_wd.csv
    ├── plots/                         ← 발표용 그래프
    └── tables/

03_c3_grokking_v2/                 ← Phase 3 v2 (코드 준비만, GPU 실행 안 함)
├── README_STATUS.md               ← 이 파일 (symlink 또는 v1 참조)
├── code/                          ← v2 코드 (232KB, 보존)
│   ├── train_v2.py, models_v2.py, tasks_v2.py
│   ├── commutator_v2.py, causal_intervention_v2.py
│   ├── hidden_probe.py            ← 추가된 hidden state precursor 모듈
│   ├── run_phase3_v2.py
│   ├── analyze_v2.py, power_law_fit.py
│   ├── launch_phase3_v2_gpu.sh    ← GPU launcher (실행 안 함)
│   └── _smoke_hard_tasks.py       ← CPU smoke 통과 확인
├── runs/                          ← 비어 있음 (실행 안 함)
└── results/                       ← 비어 있음 (실행 안 함)
```

**전체 디스크 점유: ~5 MB** (작음, 정리 안 함).

---

## 보존된 결과의 활용 가능성

| 자료 | 활용 |
|---|---|
| v1 H1-H4 verdicts | "내가 직접 grokking 실험 돌려봤다" 학습 기록 |
| v1 main/ runs JSON | Mamba vs Transformer grokking 비교 baseline 데이터 |
| v1 plots/ tables/ | 발표 자료, 슬라이드용 |
| v2 코드 (CPU smoke pass) | 향후 pivot 시 재사용 가능 (예: hidden_probe.py 다른 metric 추가) |

**정직한 한계**:
- Xu et al. 2602.16967 reviewer가 알면 "이미 다 한 것" 비판. paper publishable X.
- v1 결과는 4 task 너무 쉬워 grokking 신호 거의 mod_arith만.
- v2는 GPU 실행 안 함 → 데이터 없음.

---

## 다시 시작하고 싶다면 (pivot 후)

만약 미래에 grokking 분야로 paper 쓰고 싶을 때:

1. **재현 목적**: v2 launcher 그대로 GPU 실행 (~35 GPU-h on 5070) → Xu 결과 재현
2. **paper 작성용 pivot 옵션** (literature search 다시 한 후):
   - **layer-depth resolved precursor timing**: Xu/ILDR 둘 다 single layer만, 우리는 layer별 lead-time 차이 분석 (40% novelty)
   - **multi-metric ensemble**: hidden state + commutator + spectral 3종 결합 (30% novelty)
   - **Group-theoretic conjugacy/centralizer metric**: commutator 너머 새 metric (60% novelty, 위험)
   - **anti-grokking 연결**: HTSR (arXiv:2506.04434) + WeightWatcher (arXiv:2602.02859) 와 cross-validation
3. **항상**: literature search 먼저 다시. saturated 분야라 매월 새 paper.

---

## 의사결정 기록

| 일자 | 결정 | 이유 |
|---|---|---|
| 2026-04-30 | Phase 3 v1 코드 작성 + CPU 4-core 실행 | grokking 예지 신호 paper-grade 후보 |
| 2026-04-30 | v1 결과 분석 — 4 task 너무 쉬움 발견 | 5 task 중 mod_arith만 의미있는 grokking |
| 2026-04-30 | v2 코드 준비 (task hardening) | parity-16, sparse-parity-k5n20, dyck-d8, full-SCAN 추가 |
| 2026-05-01 | v2 GPU 실행 미정 (Phase 1 보강 우선) | 시간/GPU 자원 우선순위 |
| 2026-05-01 22:33 | Phase 2 중단 + 사용자 우려: "Phase 3도 비슷한 risk?" | Gu et al. 사례로 검증 필요성 인지 |
| 2026-05-01 22:50 | **5 agent Phase 3 literature search 시작** | Phase 2 같은 실수 반복 안 함 |
| 2026-05-01 ~23:30 | **Xu 2602.16967 + ILDR 2604.20923 발견** | Phase 3 핵심 가설 모두 선점 |
| 2026-05-01 ~23:30 | **Phase 3 v2 GPU 실행 취소, 폴더 보존** | paper 가치 부족 확인 |

---

## 자매 폴더

- `01_se_seps/`: Phase 1 SE+SEPs — 팀 프로젝트 메인 (보강 sweep 진행 중)
- `02_c2_sinks/`: Phase 2 Attention Sinks — 중단 (Gu et al. ICLR 2025로 선점)
- `04_sweep_b_pythia/` ~ `07_sweep_d_width/`: Phase 1 보강 sweep
- `scripts/`: orchestrator, 분석, 알림 스크립트

---

## 결론

이 폴더는 **Phase 2와 동일 패턴 — "literature 사전 검증 안 한 결과"의 두 번째 기록**.

다행히 v2 GPU 실행 전에 발견 → ~35 GPU-h 손실 방지.
v1 CPU 실험 (~10h)은 sunk cost지만 학습 기록으로 의미.
모든 코드 보존 (5MB뿐) → 향후 pivot 시 재사용 가능.

**교훈 (Phase 2와 함께)**: 새 주제 시작 전 **반드시 5-agent literature search 먼저**. saturated 분야는 매월 새 paper로 풍경 바뀜.

작성 시각: 2026-05-01 23:35 KST  
작성자: Claude Code (사용자 최경찬 지시)

# Phase 3 v2 (개선판 Grokking) — GPU 실행 안 함 (2026-05-01)

> **상위 폴더 [`../03_c3_grokking/README_STATUS.md`](../03_c3_grokking/README_STATUS.md) 와 함께 읽어주세요.**

---

## 한 줄 요약

**v2 코드 준비 + CPU smoke 통과까지 완료. GPU launch 직전 literature search로 핵심 가설(commutator defect + hidden state precursor)이 arXiv 2602.16967 (Xu, 2026.02) + arXiv 2604.20923 (ILDR, 2026.04) 에 선점된 것 확인. ~35 GPU-h 손실 방지하기 위해 GPU 실행 취소.**

---

## v2가 v1과 무엇이 다른가

v1 (`../03_c3_grokking/`) 의 한계:
- 5 task 중 4개 (parity, sparse parity, dyck, SCAN) 가 너무 쉬워 즉시 fit → grokking 없음
- mod_arith만 의미있는 신호
- ablation 11/12 누락 (launcher 버그)

v2 개선:
- **Task hardening**:
  - parity 8-bit → **16-bit**
  - sparse parity k=3, n=10 → **k=5, n=20**
  - dyck depth=4 → **depth=8**
  - SCAN simple → **full-SCAN**
- **Capacity 통제**: 모델 75-83K 파라미터 매칭 (4 architecture 공정 비교)
- **Hidden state probe 추가** (`hidden_probe.py`) — v2 신규
- **Causal intervention 강화**: λ ∈ {0.5, 2.0, 5.0} amplify/suppress
- **270 runs 총 sweep** (3 architectures × 5 tasks × 3 seeds × phases)
- **GPU 학습 ~35 GPU-h on 5070**

CPU smoke 6 task 모두 chance-level first eval 통과 → 학습 시작 직전 단계까지 도달.

---

## 코드 파일

```
code/
├── _smoke_hard_tasks.py      ← CPU smoke (통과)
├── tasks_v2.py               ← 5 hardened tasks
├── models_v2.py              ← 4 architecture (TF/LinAttn/Mamba/MLP) capacity-matched
├── commutator_v2.py          ← 200-step cadence commutator metric
├── hidden_probe.py           ← 신규 hidden state precursor probe (이게 가장 redundant — ILDR과 겹침)
├── causal_intervention_v2.py ← λ amplify/suppress
├── run_phase3_v2.py          ← orchestrator (main + intervention + hidden_probe phase)
├── train_v2.py               ← per-run training
├── analyze_v2.py             ← 분석
├── power_law_fit.py          ← lead-time power law fit
├── launch_phase3_v2_gpu.sh   ← GPU launcher (실행 안 함)
└── launch_phase3_v2_cpu.sh   ← CPU smoke launcher
```

---

## runs/, results/ 비어 있음

GPU 실행 안 했으므로 비어 있는 것이 정상.

---

## 만약 미래에 실행하고 싶을 때 (pivot 후)

**바로 실행하면 안 됨**. literature search 먼저:

```bash
# 1. Literature 재검증 (수개월마다 풍경 바뀜)
# 5 sub-agent 병렬 search 다시 돌리기

# 2. pivot 결정 후 코드 수정 (예: layer-depth resolved precursor)

# 3. 그 다음 실행
nohup bash code/launch_phase3_v2_gpu.sh > /tmp/phase3_v2.log 2>&1 &
```

예상 시간: 5070에서 ~35 GPU-h. 단, 이미 publish된 논문과 차별점 명확하지 않으면 **시간 낭비**.

---

## 보존 이유

코드 5MB만 차지. 삭제할 가치 없고, 향후:
- hidden_probe.py 등 모듈은 다른 grokking 분석에 재사용 가능
- v1 → v2 진화 과정 학습 기록

작성 시각: 2026-05-01 23:35 KST  
작성자: Claude Code (사용자 최경찬 지시)

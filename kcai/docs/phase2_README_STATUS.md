# Phase 2 (C2 Attention Sinks Emergence) — 중단됨 (2026-05-01)

> **이 폴더의 현재 상태와 의미를 설명하는 문서입니다.**
> 미래에 이 폴더를 다시 보게 됐을 때 맥락을 잃지 않도록 작성.

---

## 한 줄 요약

**우리가 하려던 실험이 ICLR 2025 Spotlight (Gu et al.)에 이미 publish 됐음을 literature search로 확인해 2026-05-01 22:33 KST 학습 중단. 체크포인트 23GB 삭제, 학습 로그·metrics·코드는 보존 (1.5MB).**

---

## 원래 무엇을 하려고 했나

GPT-2 124M 모델을 4가지 attention 변형 각각 from-scratch 50K step 학습:

| variant | 특징 |
|---|---|
| **softmax** | 표준 (rows sum to 1, normalization 강제) |
| **sigmoid** | 각 entry 독립 (0,1), normalization 없음 |
| **softpick** | clamped softmax 변형 |
| **softplus** | 양의 무한대 가능, normalization 약함 |

학습 도중 **attention sink** (특정 token이 모든 query의 attention을 빨아들이는 현상)이 언제·어떻게 자발적으로 생기는지 sink_max metric으로 step별 추적.

**가설**: softmax의 normalization 제약이 sink의 원인이라면, softmax는 sink가 강하게 emerge하고 sigmoid/softpick/softplus는 약하게 또는 안 emerge할 것.

---

## 실제로 확인된 결과 (보존됨)

학습 진행 + 결과:

| variant | 학습 | 50K step sink_max | wall time |
|---|:---:|---:|---:|
| **softmax** | ✅ 완료 | **0.887** (강한 sink) | 3.8h |
| **sigmoid** | ✅ 완료 | **0.165** (약한 sink) | 3.8h |
| **softpick** | ⚠️ 중단 (step 8500/50000, 17%) | step 8500까지만 기록 | ~4h 소요 |
| **softplus** | ❌ 시작 안 함 | — | — |

**보존 위치**: `results/{variant}_full_metrics.jsonl` + `results/{variant}_full_train.log`

**핵심 정성적 결론**: softmax (0.887) vs sigmoid (0.165) 약 **5배 차이**. 가설 강하게 지지. 완성된 2 variant만으로도 정성적 메시지 명확.

---

## 왜 중단했나 — Literature 충돌 발견

2026-05-01 KCI 학술지 등재 가능성 평가 중, 5개 sub-agent 병렬 literature search로 다음 prior art들이 우리 핵심 발견을 이미 publish했음을 확인:

### 가장 결정적 — **Gu et al. ICLR 2025 Spotlight**
- **arXiv**: 2410.10781
- **제목**: "When Attention Sink Emerges in Language Models: An Empirical View"
- **이미 한 것**:
  - softmax vs sigmoid (no-norm) **from-scratch** 비교
  - **60M ~ 1B** 모델 (우리 124M 포함 범위)
  - **학습 step별 sink emergence 추적** (1k~2k step에서 emerge라고 보고)
  - "sigmoid attention without normalization → sink emerge 안 함" 결론
- 우리 plan과 **거의 정확히 일치**
- 우리 H100 옵션(1.3B 확장)도 Gu의 1B 범위 안

### 그 외 위협 paper
- **Xiao et al. ICLR 2024 (StreamingLLM, 2309.17453)**: sink 발견·활용
- **Bondarenko NeurIPS 2023 (2306.12929)**: softmax no-op 메커니즘
- **Cancedda ACL 2024 (2402.09221)**: spectral 관점 sink 메커니즘
- **Zuhri 2025 (2504.20966) "Softpick"**: 우리가 쓰던 softpick variant + 340M/1.8B sink rate 비교 이미 publish
- **Wu 2025 (2501.13428)**: GPT-2 + softplus attention 이미 publish
- **arXiv 2603.11487 (2026)**: "normalization → sink" 정식 수학 proof

### 5개 agent 종합 verdict
| Agent | reject risk |
|---|:---:|
| #1 종합 검색 | 70% |
| #2 메커니즘 (P1-P5 모두 publish됨) | **75-85%** |
| #3 softpick/softplus 사전연구 | "marginal novelty" |
| #4 워크샵/OpenReview | "established fact" |
| #5 KCI 한국 학술지 | 한국에선 5-15% (낮음) |

→ 메인 학회·국제 KCI 동급 등재 모두 위험. 한국 KCI에서는 reviewer가 Gu et al.을 모를 경우에만 accept 가능.

---

## 현재 폴더 안에 무엇이 있나

```
02_c2_sinks/
├── README_STATUS.md           ← 이 파일
├── code/                       ← 학습 코드 (148KB, 보존)
│   ├── train.py                ← Sweep C/D 에서도 재사용 (--depth/--width 인자 추가됨)
│   ├── model.py                ← GPT 클래스 + 4 attention variant 구현
│   ├── run_phase2.py           ← analyze 단계 (사용 안 함)
│   ├── plot_results.py         ← (사용 안 함)
│   ├── launch_full_train.sh    ← 4 variant 순차 학습 launcher (중단됨)
│   └── ...
├── results/                    ← 학습 로그·metrics (504KB, 보존)
│   ├── softmax_full_metrics.jsonl   ← 50K step 완전 (sink_max 0.887)
│   ├── softmax_full_train.log
│   ├── sigmoid_full_metrics.jsonl   ← 50K step 완전 (sink_max 0.165)
│   ├── sigmoid_full_train.log
│   ├── softpick_full_metrics.jsonl  ← 8500 step까지 (불완전)
│   ├── softpick_full_train.log
│   └── softmax_smoke_metrics.jsonl  ← 옛 smoke 테스트
├── runs/                       ← TensorBoard events (896KB, 보존)
│   ├── softmax_full/
│   ├── sigmoid_full/
│   ├── softpick_full/
│   └── softmax_smoke/
├── checkpoints/                ← ⚠️ 삭제됨 (2026-05-01, 23GB 회수)
│                                  복원하려면 launch_full_train.sh 재실행 (24h+ 학습)
├── main.pid                    ← 프로세스 추적 파일 (의미 없음)
└── runs_main.out               ← 옛 학습 stdout (대부분 무관)
```

**`code/train.py`는 Sweep C (깊이 단독 sweep) 와 Sweep D (너비 단독 sweep) 에서 재사용** 됩니다 (--depth/--width/--n-head/--run-name 인자 추가됨, backward compatible). 따라서 코드 폴더 삭제하면 Sweep C/D 망가짐.

---

## 보존된 결과의 활용 가능성

체크포인트는 삭제됐지만 **metrics.jsonl + train.log는 그대로**. 이 데이터로 아래는 가능:

1. **softmax vs sigmoid sink emergence 그래프** — 발표 자료, Phase 1 paper 부록 (만약 환각 detection과 sink 연결한다면)
2. **재현 증명** — "내가 직접 Gu et al. 결과 재현했다" 학습 기록
3. **단순 인용용 baseline** — Gu et al. 인용하면서 "우리 환경(5070, OpenWebText)에서도 같은 패턴 확인"

**불가능**:
- 모델 가중치 추가 분석 (probe 학습 등) — 가중치 없음
- 추가 학습 step (resume) — checkpoint 삭제로 from scratch 재시작 필요
- 새 variant 추가 비교 — softplus 학습이 안 됐고 다른 variant도 새로

---

## 다시 시작하고 싶다면

만약 미래에 다시 학습하고 싶을 때:

1. **재현이 목적** (Gu 결과 확인): `bash code/launch_full_train.sh` — 5070에서 약 24-50h
2. **paper 작성용 강화**: 다음 중 하나 — 아니면 시간 낭비
   - **한국어 데이터** 로 동일 실험 (사용자가 거부함, 5/1)
   - **Gu가 안 본 attention variant** (예: ReLU+bias, polynomial, Laplace)
   - **sink × downstream task** 인과 (한국어 reasoning 성능과 sink 강도 연결)
   - **multi-seed n=3** (Gu가 명시 안 함)
   - 위 중 어느 것도 **literature search 다시** 하고 시작할 것

---

## 의사결정 기록

| 일자 | 결정 | 이유 |
|---|---|---|
| 2026-04-30 | Phase 2 시작, softmax 학습 시작 | 환각 검출과 attention 메커니즘 연결 paper-grade 신호 기대 |
| 2026-05-01 13:57 | sigmoid 학습 시작 | softmax 끝 |
| 2026-05-01 17:46 | sigmoid 50K 완료, softpick 시작 | sink_max 0.165 확인 (5배 차이) |
| 2026-05-01 ~22:00 | KCI 등재 가능성 평가 시작 | 사용자 질문 "정말 가능한가?" |
| 2026-05-01 ~22:30 | 5 agent literature search | 사용자 우려: "이미 누가 했을 것 같음" |
| 2026-05-01 22:33 | **softpick 중단, orchestrator 중단** | Gu et al. ICLR 2025 Spotlight 발견. paper-grade novelty 부족 확정 |
| 2026-05-01 22:37 | **체크포인트 23GB 삭제** | 디스크 회수. 메타데이터 보존 |
| 2026-05-01 22:38 | Phase 1 보강으로 자원 재배치 | Sweep B/C/D/A 진행 (다른 영역) |

---

## 참고 — 자매 폴더

- `01_se_seps/`: Phase 1 (환각 검출 SE+SEPs) — 팀 프로젝트 메인
- `03_c3_grokking/` + `03_c3_grokking_v2/`: Phase 3 (Grokking 예지 신호) — novelty 검증 필요
- `04_sweep_b_pythia/` ~ `07_sweep_d_width/`: Phase 1 보강 sweep
- `scripts/`: orchestrator, 분석, 알림 스크립트

---

## 결론

이 폴더는 **"하지 말 걸"이 아니라 "literature 사전 검증 안 한 교훈"의 기록**입니다. softmax 50K + sigmoid 50K 학습은 Gu et al. 결과 재현 증거로 의미는 있고, 코드는 Sweep C/D에서 재사용되어 살아 있습니다. **다음 새 주제 시작 전에는 반드시 5-agent literature search 먼저** 라는 운영 원칙으로 이어집니다.

작성 시각: 2026-05-01 22:50 KST  
작성자: Claude Code (사용자 최경찬 지시)

# 사전실험 계획 v2 — 3주 확장판

작성일: 2026-05-01
대상 기간: 5/1 ~ 5/22 (22일)
GPU: 5070 단독 (12GB, sm_120)

이 문서는 중간에 끊겨도 이어서 할 수 있도록 모든 sweep · 의존성 · resume 포인트 · 진행 상태를 한 곳에 모아 놓은 마스터 계획서입니다. 새 세션을 열면 이 파일부터 읽고, 마지막 in_progress 항목부터 이어서 진행합니다.

---

## 0. 동기 (왜 v2를 추가하는가)

원안(Phase 1/2/3)은 paper-ready 신호 확보가 목표였고 이미 다음을 얻었음:

- Phase 1 H2 PASS (SEPs > SE gap +0.065, size-dependent r=-0.53 p=0.041)
- Phase 1 H3 PASS (peak rel depth 0.68 ± 0.12)
- Phase 2 softmax 50K 완료, sink_max 0.015 → 0.887 (60×) emergence 확인
- Phase 3 v1: Mamba 0/3 grok vs TF/LinAttn 3/3 (mod_arith)

팀원(이우창) 추가 제안:
1. 실험 데이터는 많을수록 좋음 → **모델 사이즈/패밀리 sweep 확대**
2. **Confidence는 높은데 wrong 출력하는 case 발굴**
3. **입력 파라미터 모두 저장** (재현성)

병행해 사용자가 지적한 confound 이슈:
- Phase 1은 Llama(16층)/Qwen(28/36층) 혼합 → architecture × depth × width × params 모두 같이 변함. 변인 분리 필요.

v2는 위 4가지를 한꺼번에 해결하는 확장 sweep + 분석 + 인프라 패치.

---

## 1. 변인 통제 설계 — 4개 sweep으로 직교 분해

| Sweep | 통제 변인 | 변동 변인 | 산출물 |
|---|---|---|---|
| **A 패밀리** | 파라미터 ~1.3-1.5B | architecture (5종) | 일반화 주장 (패밀리 의존성 X) |
| **B 스케일 (Pythia)** | architecture, 토크나이저, 데이터(Pile), 학습 레시피 | 파라미터 (70M~6.9B 7개) | 사이즈-효과의 cleanest test |
| **C 깊이 단독** | width=512, 데이터, 토크나이저, 학습 레시피 | depth ∈ {4,8,12,16,24,32} | H3 인과 검증 (peak rel depth) |
| **D 너비 단독 (선택)** | depth=12, 나머지 모두 | width ∈ {256,384,512,768,1024} | 보너스 |

A, B는 사전학습 모델 추론(N=10 sampling + hidden state 추출). C, D는 Phase 2 인프라 재활용해서 from-scratch 학습.

---

## 2. 전체 일정 (Gantt) — **순서 변경됨 (5/1 19시 기준)**

Phase 3 v2를 맨 뒤로 옮기고, Phase 2 직후 Phase 1 관련 sweep(B → C → A)을 먼저 진행. 단계마다 자동 이메일 발송.

```
주차    5/1  5/2  5/3  5/4  5/5  5/6  5/7  5/8  5/9  5/10  5/11
[Phase2 학습] ████ ░  (sigmoid → softpick → softplus 순차)
[A 백필]      ✓ DONE
[B' self-log] ✓ DONE
[B ConfWrong] ✓ DONE
[코드 준비 B/C/A] ✓ DONE
[Phase2 분석]       ░  (training 끝나자마자 자동 실행)
[Sweep B Pythia]      ████████ (~50h)
[Sweep C 깊이]               ███████████ (~60h)
[Sweep A 패밀리]                          █████████ (~40h)
[Phase3 v2]                                       ███████ (~35h)
[통합 보고서]                                              ██
```

자동화: `master_orchestrator.sh`가 Phase 2 종료 감지 → 위 순서대로 자동 launch + 단계별 파인만식 이메일.

---

## 3. 단계별 상세 (resume 포인트 명시)

### Step 1 — A: 메타데이터 백필 (오늘, CPU)
- **상태**: PENDING
- **실행 명령**: `python ~/experiments/dl_team_v2/scripts/backfill_meta.py`
- **입력**: 모든 `01_se_seps/runs/*/*/`, `02_c2_sinks/runs/*/`
- **출력**: 각 폴더에 `meta.json`
- **포함 필드**: launch_args, prompt_template 샘플, git_commit, library_versions (torch/transformers/numpy), model_revision (HF SHA), host (GPU/CUDA), seed (현재 null = 미고정 정직 기록), launch_script_path + sha256
- **Resume**: idempotent, 이미 meta.json 있으면 skip
- **GPU 영향**: 없음 (Phase 2와 병행)

### Step 2 — B: Confident-but-Wrong 분석 (오늘부터, CPU)
- **상태**: PENDING
- **실행 명령**: `python ~/experiments/dl_team_v2/scripts/analyze_confident_wrong.py --runs-root ~/experiments/dl_team_v2/01_se_seps/runs`
- **입력**: 5개 모델 × 3 데이터셋 = 15개 generations.jsonl
- **산출물**:
  - `4cell_table.csv`: (model, dataset, conf_metric) × (correct/wrong × high/low conf) 4셀
  - `ece.csv`: Expected Calibration Error per (model, dataset, metric)
  - `risk_coverage.csv`: confidence threshold 곡선
  - `disagreement.csv`: SE-low ∧ SEPs-high (또는 역) 케이스 카운트
  - `confident_wrong_examples.jsonl`: top-quartile conf × wrong 케이스 샘플 100개
  - `plots/`: reliability diagram, R-C curve, 4-cell heatmap
- **Confidence 정의 3종**: C_SE = -SE, C_logp = mean(sample_logprobs), C_SEPs = probe(hidden) score
- **Correctness**: exact_match + F1>0.5 (둘 다 기록)
- **Resume**: 모델별/데이터셋별로 partial 결과 디스크에 저장, 재실행 시 done check
- **GPU 영향**: 없음

### Step 3 — Phase 2 학습 완료 대기 (5/2 ~13시)
- **상태**: IN_PROGRESS (sigmoid step ~34880/50000, sps=1.45)
- **모니터링**: `tail -f ~/experiments/dl_team_v2/02_c2_sinks/results/sigmoid_full_train.log`
- **다음**: sigmoid 끝나면 softpick → softplus 자동 launch (`launch_full_train.sh` resume-safe)
- **Resume**: 정전/OOM 시 동일 launcher 재실행 (checkpoint 5K 마다)

### Step 4 — B' self-logging 패치 (5/2 11시 이후)
- **상태**: PENDING (Phase 2 학습 중에는 만지면 위험)
- **블로킹**: Phase 2 완료
- **변경 파일**:
  - `01_se_seps/code/sample_generator.py` (run() 함수 시작 시 config.json dump)
  - `02_c2_sinks/code/train.py` (main 시작 시)
  - `03_c3_grokking_v2/code/run_phase3_v2.py` (각 run 시작 시)
- **추가 인자**: `--seed` (Phase 3 v2는 seed ∈ {0,1,2} 3-seed 평균)
- **Resume**: 패치는 30분 작업, 끊기면 git status로 확인

### Step 5 — Phase 2 분석 + Phase 3 v2 launch (5/3)
- **상태**: PENDING
- **블로킹**: Phase 2 + B'
- **명령**:
  - `python ~/experiments/dl_team_v2/02_c2_sinks/code/plot_results.py`
  - `bash ~/experiments/dl_team_v2/03_c3_grokking_v2/code/launch_phase3_v2_gpu.sh` (~35h, 270 runs)
- **Resume**: launcher resume-safe

### Step 6 — Sweep B (Pythia 7개) 코드 준비 (5/3-5/6 백그라운드, CPU)
- **상태**: PENDING
- **실행 위치**: `~/experiments/dl_team_v2/04_sweep_b_pythia/code/`
- **재사용**: 01_se_seps의 sample_generator.py, se_compute.py, seps_probe.py, metrics.py
- **새 launcher**: `launch_pythia_sweep.sh`
- **모델**: EleutherAI/pythia-{70m,160m,410m,1b,1.4b,2.8b,6.9b}-deduped
- **6.9B**: 4-bit 강제, 나머지: fp16
- **데이터셋**: triviaqa, nq_open, squad (각 1000q)
- **준비 완료 기준**: smoke 테스트 (70m × triviaqa × 10q) 통과
- **GPU 영향**: 준비는 CPU, 실행은 Phase 3 v2 끝난 뒤

### Step 7 — Sweep B 실행 (5/7-5/10, ~50 GPU-h)
- **상태**: PENDING
- **블로킹**: Phase 3 v2 완료
- **명령**: `bash ~/experiments/dl_team_v2/04_sweep_b_pythia/code/launch_pythia_sweep.sh`
- **결과 위치**: `04_sweep_b_pythia/runs/pythia-{size}/{dataset}/`
- **Resume**: 기존 sample_generator 처럼 done id check
- **분석**: 끝난 후 `analyze_pythia.py` → 사이즈-효과 곡선

### Step 8 — Sweep C (깊이 단독) 코드 준비 (5/10-5/11, CPU)
- **상태**: PENDING
- **실행 위치**: `~/experiments/dl_team_v2/05_sweep_c_depth/code/`
- **재사용**: 02_c2_sinks의 model.py (softmax variant), train.py
- **변경**: width=512 고정, depth만 인자로
- **데이터**: OpenWebText 100M tokens (Phase 2 동일)
- **준비 완료 기준**: depth=4 smoke 1K step 통과

### Step 9 — Sweep C 실행 (5/12-5/15, ~60 GPU-h)
- **상태**: PENDING
- **블로킹**: Sweep B 완료
- **명령**: `bash ~/experiments/dl_team_v2/05_sweep_c_depth/code/launch_depth_sweep.sh`
- **결과 위치**: `05_sweep_c_depth/runs/depth_{N}/`
- **분석 후 추가**: 학습된 6개 모델로 환각 검출 inference (sample_generator 재사용) → H3 인과 검증

### Step 10 — Sweep A (패밀리 5종) 실행 (5/16-5/19, ~40 GPU-h)
- **상태**: PENDING
- **블로킹**: Sweep C 완료
- **모델**: Pythia-1.4B / Llama-3.2-1B / Qwen2.5-1.5B / OPT-1.3B / GPT-Neo-1.3B
- **명령**: `bash ~/experiments/dl_team_v2/06_sweep_a_family/code/launch_family_sweep.sh`

### Step 11 — Confident-Wrong 확장 분석 (5/19-5/20, CPU)
- **상태**: PENDING
- **블로킹**: Sweep A/B/C 모두 완료
- **내용**: Sweep A/B/C 결과로 Step 2 분석 재실행 → confident-wrong rate가 사이즈/패밀리/깊이에 따라 어떻게 변하나 그래프

### Step 12 — 통합 보고서 (5/20-5/22)
- **상태**: PENDING
- **출력 위치**: `~/Nextcloud/2. 계속관리/AI대학원/딥러닝/팀프로젝트/사전실험_결과/04_v2_확장_통합.md`
- **포함**:
  - 4개 sweep 그래프 (A/B/C/D)
  - Confident-wrong 4-cell + 사이즈 변화
  - Phase 1/2/3 v2 통합
  - 발표용 PPT 그래프 6장

---

## 4. 의존성 그래프 (DAG)

```
Step1 백필 ──┐
Step2 ConfWrong ──┐  (둘 다 CPU, Phase 2와 병렬)
                  │
Step3 Phase2 학습 ──→ Step4 B'패치 ──→ Step5 Phase3v2 ──→ Step6 SweepB준비
                                                              │
                                                              ↓
                                                         Step7 SweepB실행
                                                              │
                                                              ↓
                                                         Step8 SweepC준비 → Step9 SweepC실행
                                                                                    │
                                                                                    ↓
                                                                              Step10 SweepA실행
                                                                                    │
                                                                                    ↓
                                                                          Step11 ConfWrong 확장
                                                                                    │
                                                                                    ↓
                                                                          Step12 통합 보고서
```

병렬 가능:
- Step1+2 ↔ Step3 (CPU vs GPU)
- Step6 준비 ↔ Step5 (Phase 3 v2가 GPU 100%지만 Step6은 CPU 코드 작성)
- Step8 준비 ↔ Step7 (마찬가지)
- Step11 ↔ Step10 마지막 부분 (분석은 CPU)

---

## 5. Resume 가이드 (세션 끊겼을 때)

1. **이 파일 § 3을 위에서부터 읽고**, "IN_PROGRESS" 또는 "PENDING with 블로킹 해제" 항목 찾기
2. **GPU 점유 상태**: `nvidia-smi`, `ps aux | grep -E "train.py|sample_generator"` 로 학습 잡 확인
3. **Phase 2/3 학습 잡**: 살아 있으면 건드리지 말 것 (resume-safe checkpoint 마다 저장)
4. **CPU 분석 잡**: idempotent, 재실행 OK
5. **Task tracker**: `TaskList` (35-42번)
6. **각 sweep 결과 위치**:
   - Phase 1: `~/experiments/dl_team_v2/01_se_seps/runs/`
   - Phase 2: `~/experiments/dl_team_v2/02_c2_sinks/runs/`
   - Phase 3 v1: `~/experiments/dl_team_v2/03_c3_grokking/runs/`
   - Phase 3 v2: `~/experiments/dl_team_v2/03_c3_grokking_v2/runs/`
   - Sweep A/B/C: `~/experiments/dl_team_v2/{04,05,06}_*/runs/`

---

## 6. 진행 상태 로그

| 일자 | 단계 | 상태 변화 | 비고 |
|---|---|---|---|
| 5/1 | Step3 Phase2 sigmoid | step 34880/50000 (70%) | sps=1.45, GPU 98% |
| 5/1 | 계획서 v2 작성 | DONE | 본 문서 |
| 5/1 | Step1 메타 백필 | DONE | 21개 폴더 meta.json |
| 5/1 | Step2 ConfWrong 분석 | DONE | summary.csv + 15개 케이스 + 5개 plot |
| 5/1 | Step6 Sweep B 코드 준비 | DONE | run_phase1.py에 --out-root 추가 (BC) |
| 5/1 | Step8 Sweep C 코드 준비 | DONE | model.py + train.py에 --depth/--width/--n-head/--run-name 추가 (BC, Phase 2 비동기 영향 X) |
| 5/1 | Step4 B' 부분 패치 | DONE (부분) | sample_generator.py + run_phase3_v2.py 자기 로그. train.py도 Sweep C 패치하면서 BC하게 적용됨 |

(이 표는 각 단계 시작/완료마다 업데이트)

---

## 7. 위험 · 대비책

| 위험 | 대비 |
|---|---|
| Phase 2 정전 다시 발생 | launcher resume-safe, 5K 마다 checkpoint |
| Phase 3 v2 OOM | 모델 작아서 가능성 낮음, but `--bs` 줄이는 인자 있음 |
| Pythia-6.9B 4-bit 환경 깨짐 | bnb 설정은 sample_generator.py에 이미 있음 (max_memory + cpu offload) |
| 디스크 부족 | hidden state 22GB/모델 추정. 7개 모델 × 3 데이터셋 ≈ 460GB. /home/kcai/Downloads 확인 후 외장 또는 압축 |
| HF rate limit | HF_TOKEN 사용 (~/hf_token.txt 이미 있음) |
| 본 문서 분실 | Nextcloud sync 자동 |

---

## 8. 산출물 매핑 (어디로 쓸 것인가)

- 학회 paper: Sweep B (Pythia clean scaling) + Confident-wrong + H3 인과(Sweep C) → 메인 contribution
- Phase 2 sink emergence: 발표 메인 그래프
- Phase 3 v2 Mamba anti-example: discussion section
- Sweep A: 일반화 주장 (appendix)
- Phase 1 mixed-family 결과: motivation 섹션 (왜 controlled study가 필요한가)

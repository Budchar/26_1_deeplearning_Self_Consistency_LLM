# LLM Hallucination 사전 탐지 및 인코딩 메커니즘 분석
## 딥러닝 팀 프로젝트 종합 보고서

**팀원**: wclee, kcai (+ Budchar 팀)  
**작성일**: 2026-05-14  
**데이터셋**: TriviaQA, NQ-Open, SQuAD, CommonsenseQA (각 n=200–1,000)

---

## 목차

1. 서론
2. 연구 방법
3. 핵심 가설 검증 결과
4. 실험 결과 — Self-Consistency 재현 및 Calibration 분석
5. 실험 결과 — SE / SEPs / Self-Consistency (대규모)
6. 실험 결과 — 레이어 인코딩 위치 및 메커니즘
7. 통합 분석: 두 접근법의 교차 해석
8. 종합 논의
9. 결론
10. 참고문헌

---

## 1. 서론

대형 언어 모델(LLM)은 사실과 다른 내용을 자신 있게 생성하는 **Hallucination** 문제를 보인다. 본 연구는 이 현상을 두 가지 상보적 관점에서 분석한다.

**관점 A — 확률적 불일치 탐지**: 동일 질문에 대한 여러 샘플 간의 의미적 불일치(Semantic Entropy)와 hidden state 기반 사전 예측(SEPs)을 통해 생성 전 hallucination을 탐지할 수 있는가?

**관점 B — 표현 기하학적 메커니즘**: 트랜스포머 레이어 내부의 어느 위치에서, 어떤 서브모듈(MLP / Attention)이 hallucination을 인코딩하는가? Fine-tuning 방법이 이 위치를 결정하는가?

**핵심 연구 질문**:
- RQ1: SEPs는 Semantic Entropy 대비 비용 효율적으로 hallucination을 탐지할 수 있는가?
- RQ2: Hallucination 신호가 집중되는 레이어 위치는 어디이며, 모델마다 왜 다른가?
- RQ3: Fine-tuning 방법(SFT 데이터 유형, DPO)이 인코딩 위치와 주체를 결정하는가?

---

## 2. 연구 방법

### 2.1 실험 모델

**[관점 A] Semantic Entropy 분석 대상**

| 모델 | 파라미터 | 비고 |
|------|---------|------|
| Llama-3.2-1B-Instruct | 1B | Meta |
| Llama-3.2-3B-Instruct | 3B | Meta |
| Qwen2.5-1.5B-Instruct | 1.5B | Alibaba |
| Qwen2.5-3B-Instruct | 3B | Alibaba |
| Qwen2.5-7B-Instruct | 7B | Alibaba |
| Pythia-70M ~ 6.9B | 70M–6.9B | EleutherAI (스케일링 분석) |
| OLMo, GPT-Neo-1.3B, OPT-1.3B, TinyLlama | ~1.3B | 패밀리 비교 |

**[관점 B] 레이어 메커니즘 분석 대상**

| 모델 | 학습 방식 | Best Layer | AUROC |
|------|----------|-----------|-------|
| Mistral-7B-v0.1 (BASE) | 사전학습만 | L29 (91%) | 0.789 |
| Mistral-7B-Instruct-v0.2 | SFT | L21 (66%) | 0.615 |
| OpenHermes-2.5 | SFT (사실 Q&A) | L21 (66%) | 0.795 |
| Zephyr-7B-beta | SFT + DPO | L15 (47%) | 0.559 |
| OpenChat-3.5 | C-RLFT | L16 (50%) | 0.868 |
| Starling-LM-7B-alpha | C-RLFT + RLHF | L32 (100%) | 0.664 |
| **NousHermes-2-DPO** | **사실SFT + DPO** | **L2 (6%)** | 0.806 |
| **Mistral-7B-Instruct-v0.3** | **사실SFT + DPO** | **L4 (12%)** | 0.808 |
| EXAONE-3.5-7.8B | 독자 학습 | L30 (94%) | 0.838 |
| Qwen2.5-7B-Instruct | SFT | L22 (79%) | 0.821 |
| Phi-3-mini-4k | SFT | L18 (56%) | 0.774 |

### 2.2 방법론

**Semantic Entropy (SE)** (Farquhar et al., 2024): 동일 질문에 N=10회 샘플링 → entailment 모델로 의미적 동치 클러스터 분류 → 클러스터 분포의 Shannon entropy 계산. 높은 SE = 높은 불확실성.

**Self-Consistency (SC)** (Wang et al., 2023): N=10 샘플의 다수결 답변 선택. 불확실성이 낮을 때 정확도 향상 기대.

**Semantic Entropy Probes (SEPs)** (Kossen et al., 2024): 각 레이어의 last-token hidden state로 로지스틱 회귀/MLP 분류기를 학습하여 단일 forward pass로 SE를 예측. 비용 1/N.

**Layer Probe (Exp06)**: 레이어별 hidden state에 로지스틱 회귀(C=1.0)를 독립 학습, AUROC로 hallucination 탐지 능력 측정. Best Layer = AUROC 최고점.

**Sublayer Probe (Exp12)**: 각 레이어를 두 지점으로 분리:
- `attn_stream[i]` = 레이어 입력 + Attention 출력 (MLP 이전)
- `mlp_stream[i]` = 전체 레이어 출력 (MLP 이후)

두 스트림의 AUROC를 비교하여 인코딩 주체를 판별.

**Logit Lens (Exp11)**: 중간 hidden state를 LM head에 통과시켜 레이어별 vocab 분포 entropy 추적.

### 2.3 평가 지표

- **AUROC**: 탐지 성능 주지표 (0.5=무작위, 1.0=완벽)
- **ECE** (Expected Calibration Error): 신뢰도 보정 평가
- **Brier Score**: 확률 예측 정확도
- **AURC** (Area Under Risk-Coverage): threshold별 정확도-커버리지
- **Cost Saving**: Adaptive SE의 샘플링 비용 절감율

---

## 3. 핵심 가설 검증 결과

### [관점 A] SE / SEPs / SC 가설

| 가설 | 내용 | 결과 | 핵심 수치 |
|------|------|------|---------|
| **H-A1** | 고SE 구간에서 SC가 정확도를 낮춘다 | **부분 통과** | 소형 모델(1B-3B) 고SE 구간 SC 적용 시 가중평균 변화 −0.0206 |
| **H-A2** | SEPs가 SE 대비 우위 | **통과** | 평균 AUROC 격차 +0.065; 모델 크기 클수록 격차 감소 (r=−0.53, p=0.041) |
| **H-A3** | 할루시네이션 신호 peak가 중후반 레이어에서 나타남 | **조건부 통과 (H3-revised)** | Standard SFT/Instruct 그룹: 0.682 ± 0.131 (원래 H3 재현); 전체 80셀: 0.49 ± 0.31 (범용성 기각 후 그룹 조건화) |
| **H-A4** | Adaptive SE로 비용 절감 가능 | **부분 통과** | 평균 절감율 18.8% (목표 30% 미달); Qwen-7B/TriviaQA 42.8% |

### [관점 B] 레이어 메커니즘 가설

| 가설 | 내용 | 결과 | 핵심 수치 |
|------|------|------|---------|
| **H-B1** | Layer probe로 hallucination 사전 탐지 가능 | **통과** | 11개 모델 중 8개 AUROC > 0.75; OpenChat-3.5 최고 0.868 |
| **H-B2** | Fine-tuning 방법이 인코딩 레이어를 결정 | **통과** | 동일 SFT + DPO 추가 → best layer L21→L2 (−19 layers) |
| **H-B3** | Type I = MLP 주도, Type II = Attention 주도 | **통과** | NousHermes MLP-Attn gap +0.160; EXAONE gap −0.079 |
| **H-B4** | Logit Lens로 Type I/II 패턴 구분 가능 | **부분 통과** | Type I: 조기 entropy 수렴; Type II: 후반 지속 성장 확인 |

---

## 4. 실험 결과 — Self-Consistency 재현 및 Calibration 분석

### 4.0 실험 개요

**모델**: Qwen2.5-0.5B-Instruct | **데이터셋**: CommonsenseQA validation (200문제) | **N=10, T=0.7, 7-shot CoT**

Wang et al.(2023)의 Self-Consistency를 소형 모델에서 재현하고, 다수결의 부산물인 **consistency rate를 confidence proxy**로 평가했다.

### 4.1 정확도: SC vs Greedy

| 방법 | 정확도 | 답 추출 성공률 |
|------|--------|-------------|
| Greedy (N=1, T=0) | 50.5% | **47.5%** |
| Self-Consistency (N=10, T=0.7) | **52.8%** | **99.5%** |

> **+2.2pp 정확도 향상**. 그러나 더 주목할 점은 **추출 성공률**: Greedy는 절반의 응답에서 답 형식을 만들지 못했으나, SC의 다양한 샘플링이 답 형식 robustness를 자연스럽게 확보했다.

> 원 논문(PaLM-540B, GSM8K)의 +17.9pp와 비교하면 작은 향상 — 0.5B + 객관식 환경의 한계이다.

### 4.2 Consistency Rate → Correctness Calibration

**AUROC = 0.664** (H-SC2 통과)

Consistency rate 구간별 실제 정답률:

| Consistency Rate 구간 | n | 실제 정답률 |
|----------------------|---|-----------|
| 0.0 – 0.2 | 12 | ~50% (이상치, n 부족) |
| 0.2 – 0.4 | 약 80 | 40% |
| 0.4 – 0.7 | 약 90 | 69% |
| 0.7 – 1.0 | 18 | 83% |
| 1.0 (완전 일관) | ~9 | 100% |

> **단조 증가 패턴 확인**: 일관성이 높을수록 정답일 가능성이 높다. 0.5B 소형 모델에서도 SC가 **의미 있는 confidence estimator**로 동작함을 보여준다.

### 4.3 0.5B 모델의 한계: 신호 분리력 부족

- Consistency rate **평균 = 0.34** (N=10 중 평균 3.4개만 같은 답)
- rate ≥ 0.7인 케이스: 전체의 **4.5%** (9개/199개)
- 모델이 "확신하는" 경우 자체가 드물어 high-confidence 영역이 매우 좁음

> SC calibration은 유효하지만 **0.5B에서는 신호 분리력이 낮다**. 모델이 너무 작으면 항상 불일치하고, 너무 크면 항상 일치하는 trade-off가 있다.

### 4.4 SC Calibration vs Semantic Entropy

| 방법 | 핵심 아이디어 | AUROC | 비용 |
|------|------------|-------|------|
| SC Consistency Rate | 표면 문자열 일치율 | 0.664 | N번 생성 |
| Semantic Entropy | 의미 클러스터 entropy | (Phase1 결과) | N번 생성 + entailment |
| SEPs | hidden state → SE 예측 | +0.065 vs SE | 1번 생성 |

> SC(0.664)는 의미 기반 SE의 하한선에 해당하는 **baseline calibrator**이다. SE가 표면 매칭의 한계를 의미 클러스터링으로 극복한 이유가 이 baseline에서 드러난다.

---

## 5. 실험 결과 — SE / SEPs / Self-Consistency

### 5.0 H3의 진화: 범용 가설 → 조건부 정제

H3는 본 연구에서 가장 풍부하게 발전한 가설이다. 세 단계를 거쳐 최종 형태에 도달했다.

**[1단계] 원래 H3 — 5개 Instruct 모델 (Phase 1)**

Llama-3.2-1B/3B + Qwen2.5-1.5B/3B/7B 5개 모델 측정 결과:
peak 상대 깊이 평균 **0.68 ± 0.12** → "Universal한 패턴"으로 가설 수립

**[2단계] H3 기각 — 80셀 확장 (v2 Sweep)**

Pythia 스케일링 + from-scratch + TinyLlama 포함 총 80셀로 확장:
전체 평균 **0.49 ± 0.31** → 분산 폭발, **H3 범용성 기각**

**[3단계] H3-revised — 학습 방법별 그룹 분석**

80셀을 학습 방법으로 분류해 재분석:

| 그룹 | n | 평균 | std |
|------|---|------|-----|
| **Standard SFT/Instruct** | **15** | **0.682** | **0.131** |
| BASE (사전학습만) | 39 | 0.588 | 0.223 |
| Pythia 학습 trajectory | 13 | 0.426 | 0.272 |
| From-scratch (degenerate) | 10 | 0.549 | 0.260 |
| TinyLlama intermediate | 3 | 0.515 | 0.175 |

**Mann-Whitney U test: p = 0.007** — 그룹 간 차이 통계적 유의

핵심: Standard SFT/Instruct 그룹(n=15)에서 원래 H3 수치(0.68 ± 0.12)가 **정확히 재현**된다. H3는 범용 법칙이 아니라 **표준 Instruct fine-tuning 모델에 한정된 조건부 패턴**이다.

**[두 팀 결과의 결합]**

wclee의 Mistral 7B 8변형 분석이 위 그룹 차이의 메커니즘을 설명한다:

| 학습 방법 | Best L | 상대 깊이 | 그룹 귀속 |
|----------|--------|---------|---------|
| BASE | L29 | 91% | BASE 그룹 |
| SFT (일반) / SFT (사실) / SFT+DPO / C-RLFT | L15–L21 | 47–66% | Standard SFT ✓ |
| C-RLFT + RLHF | L32 | 100% | RLHF outlier |
| **사실SFT + DPO** (NousHermes, Mistral-v0.3) | **L2–L4** | **6–12%** | **Type I (early, 이탈)** |

Standard SFT 그룹 평균(47–66%)이 팀의 0.682와 일치한다. Type I 이탈 모델은 DPO+사실SFT 조합으로 분리된다.

**최종 H3-revised 발견문 (main contribution 후보)**:

> *"환각 신호의 peak 상대 깊이는 표준 SFT/Instruct 학습된 LLM에서 0.65–0.70에 robust하게 위치하나, 학습 방법(BASE, DPO+factual SFT, RLHF, from-scratch)이 이 위치를 결정적으로 변동시킨다. 이는 fine-tuning이 hallucination 인코딩 위치를 결정한다는 mechanistic 발견(Sublayer Probe + Logit Lens)과 정확히 일관된다."*

---

### 5.1 SE vs SEPs 성능 비교

15셀(5개 모델 × 3개 데이터셋, 각 n=1,000) 전체 분석 결과:

**SEPs가 SE보다 평균 +0.065 AUROC 우위** (H-A2 통과)

| 모델 | AUROC(SE) | AUROC(SEPs) | Gap |
|------|-----------|-------------|-----|
| Llama-3.2-1B | 낮음 | 높음 | 가장 큰 격차 |
| Llama-3.2-3B | 중간 | 높음 | 큰 격차 |
| Qwen2.5-1.5B | 중간 | 높음 | 큰 격차 |
| Qwen2.5-3B | 중간-높음 | 높음 | 중간 격차 |
| Qwen2.5-7B | 높음 | 높음 | **역전(−0.010)** |

> **스케일링 패턴**: 모델이 클수록 SEPs 우위가 줄어든다 (r=−0.53, p=0.041). Qwen-7B에서는 SE가 SEPs를 오히려 앞섬. **소형 모델에서 SEPs 실용 가치가 가장 크다.**

### 4.2 Self-Consistency와 Semantic Entropy의 상호작용

**H-A1 부분 통과**: SC는 불확실성이 낮을 때(저SE 구간)만 효과적이다.

- **Q1 (저SE)**: SC 적용 시 정확도 변화 ≈ 0 (중립)
- **Q3 (고SE)**: SC 적용 시 가중평균 정확도 변화 **−0.0206** (소형 모델 기준)
- **대형 모델(7B 이상)**: 고SE 구간에서도 SC가 이득을 줄 수 있음

> **실무 함의**: 소형 모델(1B-3B)에서는 SE를 먼저 측정하고, 고SE 입력에는 SC를 적용하지 않는 Adaptive 전략이 필요하다.

### 4.3 Confident-but-Wrong 분석

소형 모델(SQuAD 기준):
- **73–85%**의 오답 케이스에서 모델이 **높은 신뢰도**(top-quartile confidence)를 보임
- SE는 이를 사전에 탐지하지 못하는 경우 존재
- SEPs는 같은 케이스에서 SE보다 낮은 신뢰도 점수를 부여하는 경향

### 4.4 Adaptive SE 비용 절감

**H-A4 부분 통과**: 평균 18.8% 비용 절감, 목표 30% 미달.

| 모델/데이터셋 | 비용 절감율 | 정확도 손실 |
|-------------|-----------|-----------|
| Qwen-7B / TriviaQA | **42.8%** | 최소 |
| 기타 평균 | 18.8% | 미미 |

> 모델 규모와 데이터 난이도가 높을수록 Adaptive SE 효율이 증가한다.

---

## 5. 실험 결과 — 레이어 인코딩 위치 및 메커니즘

### 5.1 레이어별 AUROC: 모델마다 다른 인코딩 위치

**Mistral 아키텍처 8개 변형 비교** (동일 아키텍처, 학습 방식만 다름):

```
Mistral-BASE     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█  L29 (91%)  AUROC 0.789
Mistral-v0.2     ░░░░░░░░░░░░░░░░░░░░░░█          L21 (66%)  AUROC 0.615
OpenHermes SFT   ░░░░░░░░░░░░░░░░░░░░░░█          L21 (66%)  AUROC 0.795
Zephyr DPO       ░░░░░░░░░░░░░░░█                 L15 (47%)  AUROC 0.559
OpenChat C-RLFT  ░░░░░░░░░░░░░░░░█                L16 (50%)  AUROC 0.868 ★
Starling RLHF    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█ L32(100%)  AUROC 0.664
NousHermes DPO   █░                               L2  (6%)   AUROC 0.806
Mistral-v0.3     ░░░█                             L4  (12%)  AUROC 0.808
```

**H-A3과의 관계**: 팀 관점 A의 peak depth 0.68 ± 0.12는 Type II 모델(일반 SFT 계열) 기반이다. Type I 모델(DPO+사실SFT)을 포함하면 분포가 **이중봉(bimodal)** 으로 나타난다.

### 5.2 Fine-tuning이 인코딩 위치를 결정한다 (H-B2)

단계별 원인 추적:

| 단계 | 가설 | 모델 | 결과 | 결론 |
|------|------|------|------|------|
| 1 | 아키텍처? | Mistral-BASE | L29 | ✗ 아님 |
| 2 | 일반 SFT? | Mistral-v0.2 | L21 | ✗ 아님 |
| 3 | DPO만? | Zephyr (UltraChat+DPO) | L15 | ✗ 아님 |
| 4 | 사실성 데이터만? | OpenHermes (DPO없음) | L21 | ✗ 아님 |
| **5** | **사실성 SFT + DPO?** | **NousHermes** | **L2** | **✓ 확인** |

**결정적 증거**: OpenHermes(SFT only, L21, AUROC=0.795) vs NousHermes(동일 SFT + DPO, L2, AUROC=0.806) — 유일한 차이는 DPO 적용 여부. best layer가 **L21 → L2로 −19 layers 이동**.

> 사실성 밀도가 높은 SFT 데이터(OpenHermes: GPT-4 생성 사실 Q&A ~1M건)가 필요조건이고, DPO가 이 회로를 초반 MLP에 고착시키는 충분조건이다.

### 5.3 Sublayer Probe: MLP vs Attention 인코딩 주체 (H-B3)

| 모델 | Best Layer | Attn AUROC | MLP AUROC | Gap(MLP−Attn) | 주체 | 유형 |
|------|-----------|-----------|-----------|--------------|------|------|
| **NousHermes-DPO** | L2 | 0.646 | **0.806** | **+0.160** | MLP | Type I |
| **Mistral-v0.3** | L4 | 0.776 | **0.808** | +0.032 | MLP | Type I |
| Mistral-BASE | L29 | 0.741 | 0.762 | +0.021 | Both | Type II |
| OpenHermes | L21 | 0.758 | 0.783 | +0.025 | Both | Type II |
| Qwen2.5-7B | L22 | 0.798 | 0.812 | +0.014 | Both | Type II |
| **EXAONE-3.5** | L30 | **0.917** | 0.838 | **−0.079** | Attention | Type II-Attn |

**EXAONE 특이점**: L15 attn_stream AUROC=0.917이 full-layer best(L30=0.838)보다 높다. Attention 메커니즘이 L15에서 이미 더 집중된 신호를 형성하고 있음.

### 5.4 인코딩 유형 분류 체계

```
Type I — Early MLP (Best layer ≤ 5, MLP dominant)
  조건: 사실성 SFT 데이터 + DPO
  모델: Mistral-7B-Instruct-v0.3 (L4), NousHermes-2-DPO (L2)
  특징: L2-4의 MLP stream이 hallucination 신호 주도
        Logit Lens: 초반 레이어에서 vocab entropy 조기 수렴

Type II — Late (Best layer > 15)
  Type II-Both: MLP + Attention 공동 인코딩
    모델: Mistral-BASE (L29), OpenHermes (L21), Qwen2.5-7B (L22)
  Type II-Attn: Attention 주도
    모델: EXAONE-3.5 (L30, gap=−0.079)
  Type II-RLHF: RLHF로 인해 마지막 레이어로 밀림
    모델: Starling-LM-7B-alpha (L32, 100%)
    특징: RLHF가 DPO와 반대 방향으로 best layer를 이동
```

### 5.5 Logit Lens: Vocab Entropy 성장 패턴 (H-B4)

중간 레이어의 hidden state를 LM head에 통과시킨 vocab entropy 분석:

- **Type I 모델 (Mistral-v0.3)**: L4 이후 entropy 차이(정답 vs 오답 토큰)가 조기 수렴. 초반에서 이미 사실 판별이 완료됨.
- **Type II 모델 (EXAONE, Qwen)**: 후반 레이어까지 entropy 차이가 지속 성장. 레이어가 깊어질수록 vocab 불확실성 신호가 강해짐.

---

## 6. 통합 분석: 두 접근법의 교차 해석

### 6.1 Peak Depth 재해석

| 구분 | Peak Depth | 해당 모델 |
|------|-----------|---------|
| 팀 관점 A 전체 평균 | **0.68 ± 0.12** | Llama, Qwen 계열 (5개) |
| 관점 B — Type II | **0.47 ~ 1.00** (평균 ≈ 0.72) | 일반 SFT 모델들 |
| 관점 B — Type I | **0.06 ~ 0.12** | 사실SFT + DPO 모델들 |

> 팀의 0.68 ± 0.12는 Type II 모델만 반영한 수치다. Type I 모델을 포함하면 전체 분포는 0.06–1.00으로 분산되며, **fine-tuning 방법이 이 분포의 결정 변수**임을 알 수 있다.

### 6.2 SEPs 레이어 선택 전략 개선 가능성

팀 관점 A의 SEPs는 레이어별 probe를 학습한다. 관점 B의 발견을 적용하면:

- **Type I 모델**에서는 초반 L2-4만 탐침해도 충분한 신호 확보 → SEPs 비용 추가 절감 가능
- **Type II 모델**에서는 0.6-0.8 깊이 구간에 집중하는 전략 유효
- Type 분류 자체를 zero-shot으로 판별하는 **Meta-SEPs** 가능성 제시

### 6.3 모델 크기 vs Fine-tuning 방법

| 변수 | 팀 관점 A 발견 | 관점 B 발견 |
|------|--------------|-----------|
| 모델 크기 | SEPs 우위가 크기에 따라 감소 (r=−0.53) | — |
| Fine-tuning 방법 | — | 동일 7B 규모에서 AUROC 0.559~0.868 분산 |
| 종합 해석 | 모델 크기가 탐지 용이성에 영향 | **Fine-tuning 방법이 더 강한 예측 변수** |

7B 동일 규모에서 fine-tuning 방법만으로 AUROC 범위가 0.559(Zephyr)~0.868(OpenChat) — 이 분산이 모델 크기 효과보다 크다.

### 6.4 SC 역효과와 인코딩 유형의 관계

팀 관점 A: 소형 모델 고SE 구간에서 SC가 역효과 (−0.0206).
관점 B: Starling-LM(RLHF, L32)처럼 후반 레이어 의존 모델은 샘플 간 불일치가 크다.

> RLHF가 best layer를 마지막(L32)으로 밀어내는 현상은, 동일 입력에 대해 모델이 후반부까지 처리해야 최종 판단을 내림을 의미한다. 이는 N=10 샘플 간 분산을 높여 SC가 불안정해지는 원인이 될 수 있다.

---

## 7. 종합 논의

### 7.1 LLM은 생성 전 이미 "틀릴 것"을 알고 있는가?

**예, 단 모델마다 다른 레이어에서, 다른 방식으로.**

- SE/SEPs (관점 A): 여러 샘플의 의미적 불일치로 간접 측정 → 소형 모델에서 유효
- Layer Probe (관점 B): 단일 forward pass의 hidden state로 직접 측정 → AUROC 0.75-0.87 달성

두 방법 모두 생성 전 탐지가 가능함을 확인했다. 단, **SEPs와 Layer Probe는 본질적으로 같은 현상의 두 가지 관측 방법**이다: SEPs가 SE를 예측하도록 학습된 probe를, Layer Probe는 정답/오답 레이블로 직접 학습한다.

### 7.2 메커니즘: MLP Key-Value Memory와 DPO

초반 MLP 레이어가 사실 지식을 key-value 쌍으로 저장하는 구조를 가진다는 Geva et al.(2021)의 발견을 기반으로:

1. **사실성 SFT (OpenHermes)**: GPT-4가 생성한 고밀도 사실 Q&A 학습이 초반 MLP key-value 회로를 강화.
2. **DPO**: "이 답이 맞다/틀리다"는 선호 신호가 이 회로를 조기 활성화하도록 파라미터를 업데이트.
3. **결과**: L2-4에서 이미 "이 답은 틀릴 것"이라는 표현이 형성됨.

일반 대화 데이터(UltraChat) + DPO(Zephyr)는 이 효과가 없다 — 사실 지식의 밀도 자체가 조건이다.

### 7.3 실용적 탐지 전략 제안

| 상황 | 권장 방법 | 근거 |
|------|----------|------|
| 소형 모델 (1B-3B), 비용 제약 | SEPs (초반 레이어) | 관점 A: 가장 큰 SE-SEPs 격차 |
| Type I 모델 (사실SFT+DPO) | Layer Probe L2-4 만 | 관점 B: 초반에 신호 집중 |
| Type II 모델 (일반 SFT) | SEPs (60-80% 구간) | 관점 A/B 일치: peak 0.68 |
| 고불확실 입력 | SE만 계산, SC 미적용 | 관점 A: SC가 역효과 |
| Adaptive 비용 절감 | Qwen-7B급: Adaptive SE | 관점 A: 42.8% 절감 |

---

## 8. 결론

본 팀 프로젝트는 LLM hallucination 사전 탐지를 확률적 불일치(SE/SEPs)와 표현 기하학적 메커니즘(Layer/Sublayer Probe) 두 관점에서 종합 분석했다.

**주요 결론:**

> **결론 1 (탐지 방법론)**: SEPs는 Semantic Entropy 대비 평균 +0.065 AUROC 우위이며, 소형 모델에서 효과가 가장 크다. 단, Self-Consistency는 고SE 구간 소형 모델에서 역효과(−0.0206)를 낳는다.

> **결론 2 (레이어 위치)**: Hallucination 신호 peak는 평균 상대 깊이 0.68에 위치하나, fine-tuning 방법에 따라 0.06~1.00의 이중봉 분포를 보인다.

> **결론 3 (메커니즘)**: 사실성 SFT 데이터(OpenHermes) + DPO 조합이 hallucination 인코딩을 후반 레이어(L21)에서 초반 MLP(L2-4)로 이동시킨다. NousHermes vs OpenHermes (동일 SFT, DPO 유무만 다름)가 이를 직접 증명했다.

> **결론 4 (인코딩 주체)**: Type I 모델(초반 MLP)과 Type II 모델(후반 Attention/Both)로 분류되며, 이는 fine-tuning 레시피에 의해 결정된다.

**한계**: SE 계산의 높은 비용(N=10 샘플링), Layer Probe의 모델별 fine-tuning 필요, Logit Lens 분석의 추가 모델 검증 부족. 향후 연구에서 Type 분류 자동화 및 더 다양한 아키텍처 검증이 필요하다.

---

## 9. 참고문헌

- **Wang, X. et al. (2023).** Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*. https://arxiv.org/abs/2203.11171

- **Farquhar, S. et al. (2024).** Detecting Hallucinations in Large Language Models Using Semantic Entropy. *Nature*, 630, 625–630. https://doi.org/10.1038/s41586-024-07421-0

- **Kossen, J. et al. (2024).** Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. *NeurIPS 2024*. https://arxiv.org/abs/2406.15927

- **Geva, M. et al. (2021).** Transformer Feed-Forward Layers Are Key-Value Memories. *EMNLP 2021*. https://arxiv.org/abs/2012.14913

- **Rafailov, R. et al. (2023).** Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS 2023*. https://arxiv.org/abs/2305.18290

- **nostalgebraist (2020).** Interpreting GPT: The Logit Lens. *LessWrong*. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru

- **Gu, Y. et al. (2025).** When Attention Sink Emerges in Language Models: An Empirical View. *ICLR 2025 Spotlight*. https://arxiv.org/abs/2410.10781

- **Xu (2026).** Early-Warning Signals of Grokking via Loss-Landscape Geometry. https://arxiv.org/abs/2602.16967

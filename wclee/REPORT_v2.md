# LLM Hallucination 사전 탐지 및 메커니즘 분석
## 딥러닝 텀프로젝트 보고서 v2

---

## 1. 서론

대형 언어 모델(LLM)은 사실에 기반하지 않은 내용을 그럴듯하게 생성하는 **hallucination** 문제를 가진다. 특히 모델이 틀린 답을 높은 확신으로 제시하는 경우, 사용자가 이를 의심 없이 수용할 위험이 있다. 본 연구는 세 가지 핵심 질문을 탐구한다.

- **RQ1.** 출력이 완전히 생성되기 전, 토큰 생성 과정에서 hallucination을 탐지할 수 있는가?
- **RQ2.** Hallucination 정보는 모델 내부의 어느 레이어에 인코딩되며, 이는 파라미터 수·아키텍처와 어떤 관계인가?
- **RQ3.** 아키텍처마다 탐지 난이도가 다른 근본 원인은 무엇이며, 파인튜닝 방법은 인코딩 위치를 어떻게 바꾸는가?

로컬 LLM 11개 이상 모델(총 파라미터 규모: 0.36B–14B)에 대해 TriviaQA / MMLU / NaturalQuestions 데이터셋으로 실험을 수행하였다. 사전 탐지 방법론 5종, 레이어 프로브, 스케일링 법칙 분석, 크로스 아키텍처 비교, 다중 데이터셋 일반화, 프로브 기반 개입, Logit Lens 분석, Sublayer Probe, Mistral 패밀리 8개 변형 비교까지 총 12가지 실험을 진행하였다.

---

## 2. 실험 설정

### 2.1 모델

| 모델 | 파라미터 | 레이어 | 계열 |
|------|----------|--------|------|
| SmolLM2-360M-Instruct | 0.36B | 32 | SmolLM (HuggingFaceTB) |
| Qwen2.5-0.5B-Instruct | 0.5B | 24 | Qwen (Alibaba) |
| SmolLM2-1.7B-Instruct | 1.7B | 24 | SmolLM |
| Qwen2.5-1.5B-Instruct | 1.5B | 28 | Qwen |
| Qwen2.5-3B-Instruct | 3.0B | 36 | Qwen |
| OPT-6.7B | 6.7B | 32 | OPT (Meta) |
| Qwen2.5-7B-Instruct | 7.0B | 28 | Qwen |
| Mistral-7B-Instruct-v0.2/v0.3 | 7.0B | 32 | Mistral |
| Falcon-7B-Instruct | 7.0B | 32 | Falcon (TII) |
| EXAONE-3.5-7.8B-Instruct | 7.8B | 32 | EXAONE (LG AI) |
| Qwen2.5-14B-Instruct | 14.0B | 48 | Qwen |

**Mistral 패밀리 확장 실험 (Exp12 추가)**

| 모델 | 학습 방법 | 파라미터 |
|------|----------|----------|
| Mistral-7B-v0.1 (BASE) | 없음 | 7.0B |
| Mistral-7B-Instruct-v0.2 | SFT | 7.0B |
| OpenHermes-2.5-Mistral-7B | SFT (OpenHermes) | 7.0B |
| Zephyr-7B-beta | SFT (UltraChat) + DPO | 7.0B |
| OpenChat-3.5 | C-RLFT | 7.0B |
| Starling-LM-7B-alpha | C-RLFT + RLHF | 7.0B |
| NousHermes-2-Mistral-7B-DPO | SFT (OpenHermes) + DPO | 7.0B |
| Mistral-7B-Instruct-v0.3 | SFT + DPO | 7.0B |

### 2.2 데이터셋 및 평가

| 데이터셋 | 태스크 | 특성 |
|----------|--------|------|
| **TriviaQA** (n=150–300 per model) | 사실 지식 기반 개방형 QA | 주요 실험 데이터셋 |
| **MMLU** (n=200) | 57개 주제 다지선다 | 정확도 극단적으로 높음 → 탐지 역전 현상 |
| **NaturalQuestions** (n=200) | 사실 기반 개방형 QA | TriviaQA 대비 낮은 정확도 |

정답 판정: 모델 출력 내 정답 문자열 포함 여부 (case-insensitive). 주요 지표: AUROC.

### 2.3 실험 개요

| # | 실험 | 방법 |
|---|------|------|
| 01 | Token Entropy | 토큰 생성 시 logit 분포 엔트로피 |
| 02 | Semantic Entropy | 의미 클러스터링 엔트로피 (Kuhn et al., 2023) |
| 03 | Self-Consistency | 반복 샘플링 일관성 |
| 04 | Calibration (ECE) | Expected Calibration Error |
| 05 | Verbalized Confidence | 모델 자기 확신도 언어화 |
| 06 | Layer Probe | 레이어별 hidden state → correctness 분류기 |
| 07 | Scaling Analysis | Qwen 0.5B→14B 스케일링 |
| 08 | Cross-Arch Comparison | 다중 아키텍처 비교 (11모델) |
| 09 | Multi-Dataset | TriviaQA / MMLU / NaturalQuestions 일반화 |
| 10 | Intervention | Layer probe 기반 재샘플링 (부정 결과) |
| 11 | Hidden State Geometry | Logit Lens + 표현 분리 분석 |
| **12** | **Sublayer Probe + Mistral Family** | **MLP vs Attention 서브레이어, 8개 변형 비교** |

---

## 3. RQ1 — 사전 탐지 방법론 비교

### 3.1 EXAONE-3.5-7.8B 기준 전체 방법 비교 (TriviaQA, n=300)

| 방법 | AUROC | 비고 |
|------|-------|------|
| **N-Clusters (Semantic Entropy)** | **0.839** | 최고 |
| Layer Probe (L30/32) | 0.838 | Architecture-agnostic |
| Token Entropy (max) | 0.798 | |
| Semantic Entropy (raw) | 0.783 | |
| Verbalized Confidence | 0.752 | |
| Self-Consistency | 0.661 | 최저 |

Semantic Entropy는 모델이 동일 질문에 대해 생성하는 답변들을 의미 클러스터로 묶어 그 수를 불확실성 지표로 사용한다. 정답 시 평균 2.1개, 오답 시 5.3개의 클러스터로 뚜렷한 차이가 관찰되었다.

### 3.2 아키텍처별 탐지 성능 비교 (Exp08)

| 모델 | 정확도 | AUROC (Entropy) | AUROC (N-Clust) | Entropy Gap |
|------|--------|-----------------|-----------------|-------------|
| SmolLM2-360M | 0.320 | 0.706 | 0.665 | +0.313 |
| SmolLM2-1.7B | 0.575 | 0.775 | 0.734 | +0.301 |
| OPT-6.7B | 0.440 | 0.515 ⚠️ | 0.488 ⚠️ | +0.022 |
| Mistral-7B-v0.3 | **0.820** | 0.594 ⚠️ | 0.639 | +0.087 |
| Qwen2.5-7B | 0.675 | 0.765 | 0.775 | +0.261 |
| EXAONE-3.5-7.8B | 0.610 | 0.787 | **0.797** | **+0.374** |

**핵심 발견 ①**: Mistral-7B-v0.3이 가장 높은 정확도(82%)임에도 Entropy AUROC=0.594로 거의 랜덤 수준. OPT-6.7B(instruction tuning 없음)도 AUROC≈0.5. 두 모델 모두 엔트로피 기반 탐지가 구조적으로 실패한다.

### 3.3 파라미터 스케일과 탐지 난이도 (Exp07, Qwen2.5)

| 모델 | 정확도 | Entropy AUROC | Entropy Gap |
|------|--------|---------------|-------------|
| Qwen2.5-0.5B | 0.390 | 0.773 | +0.541 |
| Qwen2.5-1.5B | 0.485 | 0.797 | +0.461 |
| Qwen2.5-3B | 0.575 | 0.762 | +0.367 |
| Qwen2.5-7B | 0.675 | 0.765 | +0.261 |
| Qwen2.5-14B | **0.760** | 0.726 | +0.163 |

**핵심 발견 ②**: 정확도↑ vs Entropy Gap↓의 강한 반비례 관계 (r = −0.86). 크고 정확한 모델일수록 정답/오답 간 토큰 엔트로피 차이가 줄어들어 탐지가 어려워진다.

### 3.4 다중 데이터셋 일반화 (Exp09)

| 모델 | 데이터셋 | 정확도 | AUROC (Entropy) | AUROC (N-Clust) |
|------|----------|--------|-----------------|-----------------|
| EXAONE | TriviaQA | 0.610 | 0.787 | 0.775 |
| EXAONE | NaturalQ | 0.295 | 0.670 | 0.611 |
| EXAONE | **MMLU** | **0.975** | **0.311** ⚠️ | **0.387** ⚠️ |
| Qwen-7B | TriviaQA | 0.675 | 0.765 | 0.759 |
| Qwen-7B | NaturalQ | 0.355 | 0.692 | 0.608 |

**핵심 발견 ③**: MMLU에서 정확도=97.5% → 오답이 5개뿐 → AUROC 0.5 미만(랜덤보다 나쁨). 엔트로피 탐지는 오답 샘플이 충분히 존재해야 유효하다.

---

## 4. RQ2 — 레이어별 Hallucination 인코딩 (Exp06)

### 4.1 모델별 최적 Layer Probe 위치

각 레이어의 last-token hidden state에 로지스틱 회귀 분류기를 학습해 AUROC를 측정하였다.

| 모델 | 파라미터 | 최적 레이어 | 상대 깊이 | Probe AUROC |
|------|----------|-----------|----------|-------------|
| Mistral-7B-v0.3 | 7.0B | L4/32 | **12%** ← 이상치 | 0.808 |
| OPT-6.7B | 6.7B | L7/32 | 22% | 0.670 |
| SmolLM2-1.7B | 1.7B | L11/24 | 46% | 0.694 |
| Qwen2.5-1.5B | 1.5B | L19/28 | 68% | 0.810 |
| Qwen2.5-7B | 7.0B | L20/28 | 71% | 0.794 |
| EXAONE-3.5-7.8B | 7.8B | L30/32 | 94% | 0.838 |
| Qwen2.5-0.5B | 0.5B | L23/24 | 96% | 0.773 |
| Qwen2.5-14B | 14.0B | L46/48 | 96% | **0.846** |

**핵심 발견 ④**: Layer Probe는 엔트로피 탐지가 실패하는 Mistral(AUROC=0.808)과 OPT에서도 유효하다. 최적 레이어 위치는 아키텍처마다 크게 다르며, Qwen 계열은 90% 이상 깊이에서 최대를 보인다.

### 4.2 Mistral 이상치: 왜 L4인가?

Mistral-7B-v0.3의 최적 레이어가 L4(12%)라는 것은 다른 모든 모델과 비교해 극단적으로 얕다. 이를 규명하기 위해 동일 Mistral-7B 아키텍처의 8개 변형을 테스트하였다 (Exp12, 다음 섹션).

---

## 5. RQ3 — 인코딩 메커니즘 분석

### 5.1 Logit Lens 분석 (Exp11)

Logit Lens는 각 중간 레이어의 hidden state를 최종 LM head로 투영해 어느 레이어에서 어휘 불확실성이 형성되는지 추적하는 기법이다.

| 모델 | Best Probe Layer | Logit Lens Gap @ Best L | Peak Gap | 인코딩 유형 |
|------|-----------------|------------------------|----------|------------|
| Mistral-7B | L4 (12%) | **−0.002 ≈ 0** | +0.391 @ L28 | **Type I** |
| Qwen2.5-7B | L20 (71%) | +1.090 | +1.804 @ L26 | **Type II-b** |
| EXAONE-7.8B | L30 (94%) | **+1.489 (최대)** | +1.489 @ L30 | **Type II-a** |

**세 가지 인코딩 유형 정의**

| 유형 | 모델 | 메커니즘 |
|------|------|---------|
| **Type I** | Mistral, OPT | 초기 레이어에서 기하학적 분리. 어휘 불확실성으로 이어지지 않음 → 엔트로피 탐지 실패 |
| **Type II-a** | EXAONE | 기하학적 분리와 어휘 불확실성이 동일 레이어(L30)에서 동시 최대 |
| **Type II-b** | Qwen-7B | 기하학적 분리(L20) 후에도 어휘 불확실성이 계속 증가(L26까지) |

**핵심 발견 ⑤**: 엔트로피 탐지의 실패는 "모델이 confident하기 때문"이 아니라, **hallucination 인코딩 형태 자체가 아키텍처마다 다르기 때문**이다. Type I 모델에서는 정보가 기하학적으로 존재하나 어휘 분포에 반영되지 않는다.

### 5.2 Mistral 패밀리 비교 — 파인튜닝이 인코딩 위치를 결정 (Exp12A)

동일한 Mistral-7B 아키텍처에 서로 다른 학습을 적용한 8개 변형 모델에 대해 Layer Probe를 수행하였다.

| 모델 | 학습 방법 | 최적 레이어 | 깊이 | AUROC | L2 AUROC | L4 AUROC |
|------|----------|-----------|------|-------|----------|----------|
| Mistral-7B-v0.1 (BASE) | 없음 | L29/32 | 91% | 0.789 | 0.484 | 0.422 |
| Mistral-7B-Instruct-v0.2 | SFT | L21/32 | 66% | 0.615 | 0.491 | 0.429 |
| OpenHermes-2.5 | SFT (OpenHermes) | L21/32 | 66% | 0.795 | 0.602 | 0.590 |
| Zephyr-7B-beta | SFT (UltraChat) + DPO | L15/32 | 47% | 0.559 | 0.466 | 0.497 |
| OpenChat-3.5 | C-RLFT | L16/32 | 50% | **0.868** | 0.611 | 0.542 |
| Starling-LM-7B-alpha | C-RLFT + RLHF | L32/32 | 100% | 0.664 | 0.552 | 0.496 |
| **NousHermes-2-DPO** | **SFT (OpenHermes) + DPO** | **L2/32** | **6%** | **0.806** | **0.806** | 0.729 |
| **Mistral-7B-Instruct-v0.3** | SFT + DPO | **L4/32** | **12%** | **0.808** | 0.712 | **0.808** |

**핵심 발견 ⑥ — DPO + 사실성 데이터 = 초기 MLP 인코딩**

- OpenHermes(SFT only) → L21 (66%) → **DPO 추가(NousHermes) → L2 (6%)**
- 동일 SFT 데이터에 DPO만 추가했을 때 최적 레이어가 15배 앞으로 이동
- Zephyr도 DPO를 사용하지만 SFT 데이터가 달라(UltraChat) L15(47%)에 그침
- **Starling(RLHF)**: L32(100%) — RLHF는 오히려 인코딩을 마지막 레이어로 밀어냄

파인튜닝 방법에 따른 인코딩 깊이 스펙트럼:

```
RLHF: 100% → SFT/BASE: 66-91% → C-RLFT: 50% → DPO(타 데이터): 47%
→ DPO(OpenHermes): 6-12% ← Type I (Mistral-v0.3 & NousHermes)
```

**DPO 메커니즘 해석**: DPO는 모델이 "알고 있는 사실"과 "모르는 사실"을 명시적으로 구분하도록 학습시킨다. 사실성 Q&A 데이터(OpenHermes)와 결합되면, 초기 MLP 레이어(L1–4)에 **사실 지식 존재 여부를 게이팅하는 회로**가 형성된다. 이는 Geva et al.(2021)의 MLP key-value memory 가설과 일치한다 — 정보가 Attention을 통해 문맥을 파악하기 전에 이미 L2에서 판단이 이루어진다.

### 5.3 Sublayer Probe — MLP vs Attention 기여 (Exp12B)

각 레이어 내부를 두 구간으로 분리해 probe하였다.

- `attn_stream[i]` = layer_input[i] + attention_output[i] (MLP 이전)
- `mlp_stream[i]` = full layer output[i] (MLP 이후)

| 모델 | Exp06 최적 L | attn@best | mlp@best | gap | 주도 | 글로벌 최대 (attn) | 글로벌 최대 (mlp) |
|------|------------|-----------|----------|-----|------|-----------------|----------------|
| Mistral-7B-v0.3 | L4 | 0.768 | 0.736 | −0.032 | Attn | L3: 0.768 | **L3: 0.808** |
| NousHermes-2-DPO | L2 | 0.722 | 0.708 | −0.014 | Both | L19: 0.785 | **L1: 0.806** |
| OpenHermes-2.5 | L21 | 0.801 | 0.764 | −0.037 | Attn | L21: 0.801 | L20: 0.795 |
| Zephyr-7B-beta | L15 | 0.509 | 0.559 | +0.050 | MLP | L0: 0.565 | L14: 0.559 |
| OpenChat-3.5 | L16 | 0.785 | 0.764 | −0.021 | Attn | **L15: 0.882** | L15: 0.868 |
| Starling-LM-7B | — | — | — | — | — | L30: 0.664 | L31: 0.672 |
| Mistral-7B-BASE | L29 | 0.609 | 0.590 | −0.019 | Both | L28: 0.770 | L28: 0.789 |
| EXAONE-3.5-7.8B | L30 | 0.838 | 0.810 | −0.028 | Attn | **L15: 0.917** | L15: 0.875 |

**핵심 발견 ⑦**:

- **Mistral-v0.3 / NousHermes-DPO**: MLP stream이 L1–L3에서 AUROC=0.806–0.808. 초기 MLP가 hallucination 게이팅을 주도. DPO 시그널이 early MLP에 사실성 분류 회로를 형성.
- **EXAONE**: Attention stream 글로벌 최대 L15=0.917로 Exp06 최고치(L30=0.838)를 크게 초과. Type II-a 모델에서 Attention이 hallucination 신호를 주도적으로 인코딩하고 MLP는 신호를 약화.
- **OpenChat-3.5**: Attention L15=0.882 — C-RLFT가 Attention 기반의 강한 중간층 인코딩을 형성. 8개 변형 중 가장 높은 Exp06 AUROC(0.868).
- **Starling(RLHF)**: 최후반부(L30–31) 전체에 신호 분산 — RLHF의 "최종 출력 품질" 보상 구조가 인코딩을 후반부로 밀어냄.

학습 방법과 Sublayer 역할의 관계:

| 학습 방법 | 인코딩 레이어 | 주도 서브레이어 | 해석 |
|----------|------------|--------------|------|
| DPO + 사실성 데이터 | L1–4 (초기) | MLP | 사실 게이팅 회로 |
| C-RLFT | L15–16 (중간) | Attention | 대화 문맥 통합 |
| SFT only | L20–29 (후반) | Attention/MLP 공동 | 자연적 축적 |
| RLHF | L30–32 (최후) | 분산 | 최종 출력 최적화 |

---

## 6. 개입 실험 (Exp10)

### 6.1 실험 설계

Layer probe로 hallucination 위험(1 − P(correct)) > 0.6인 샘플을 탐지한 뒤 최대 3회 재샘플링하여 정확도 향상 여부를 측정하였다.

### 6.2 결과 (EXAONE-3.5-7.8B, n=75)

| 조건 | 정확도 | AUROC | 개입 횟수 |
|------|--------|-------|----------|
| Baseline | 0.560 | 0.665 | — |
| Layer Probe 개입 | 0.560 | 0.665 | 31/75 (41.3%) |
| **향상** | **+0.000** | **+0.000** | — |

**핵심 발견 ⑧**: 정확도 향상이 전혀 없었다. Layer probe가 hallucination을 올바르게 탐지해도, 동일 질문에 재샘플링하면 모델이 해당 사실 지식을 갖추지 못했다면 계속 같은 오답을 생성한다. **"탐지 가능 ≠ 교정 가능"** — 재샘플링으로는 모델의 지식 한계를 극복할 수 없으며, RAG 등 외부 지식 연동이 필요하다.

---

## 7. 종합 논의

### 7.1 사전 탐지 방법 비교

| 특성 | Token Entropy | Semantic Entropy | Layer Probe |
|------|--------------|------------------|-------------|
| 오버헤드 | 없음 | 중간 (n회 샘플링) | 중간 |
| 아키텍처 범용성 | 낮음 | 낮음 | **높음** |
| 최고 AUROC | 0.798 | **0.839** | 0.838 |
| 훈련 데이터 필요 | 불필요 | 불필요 | **필요** |
| 개입 효과 | N/A | N/A | **없음** |

### 7.2 연구 한계

- **정답 판정**: 문자열 매칭으로 paraphrase 오답 처리 가능
- **MMLU 탐지 역전**: 정확도 > 90% 시 구조적 실패
- **Falcon 비호환**: DynamicCache 반환으로 logit 측정 전면 실패
- **OPT 형식 미준수**: Base model의 출력 형식 불일치로 AUROC≈0.5
- **DPO 가설 한계**: Mistral-v0.3의 실제 학습 세부 사항 미공개 — NousHermes 결과는 간접 증거

### 7.3 주요 기여

1. **탐지 방법론 비교**: 12가지 실험으로 방법론별 성능·한계 체계적 정리
2. **파라미터-탐지 트레이드오프**: 정확도 vs 엔트로피 갭 r=−0.86 실증
3. **Type I/II 인코딩 유형 발견**: Logit Lens로 아키텍처별 hallucination 인코딩 메커니즘 분류
4. **DPO → 초기 MLP 인코딩 발견**: 8개 변형 비교로 DPO+사실성 데이터가 L2 초기 게이팅 회로를 형성함을 입증
5. **EXAONE Attention 주도 발견**: L15 Attention stream AUROC=0.917로 전체 layer probe 최고치 초과
6. **개입 실험 부정 결과**: "탐지 가능 ≠ 교정 가능" 실증적 확인

---

## 8. 결론

본 연구는 11개 이상의 로컬 LLM에 대해 12가지 실험을 통해 hallucination 사전 탐지와 메커니즘 분석을 수행하였다.

**RQ1**: N-Clusters 기반 Semantic Entropy가 AUROC 0.839로 최고 탐지 성능. 단, 엔트로피 방법은 아키텍처(Mistral, OPT)와 도메인 정확도(MMLU)에 크게 의존한다. Layer Probe는 모든 모델에서 범용적으로 유효하나, 탐지된 hallucination을 재샘플링으로 교정하는 것은 효과가 없다.

**RQ2**: 최적 probe 레이어는 아키텍처마다 크게 다르다 — Mistral-v0.3은 L4(12%), EXAONE은 L30(94%), Qwen-14B는 L46(96%). 이 차이는 단순한 아키텍처 설계가 아닌, 학습 방법에 의해 결정된다.

**RQ3**: Logit Lens(Exp11)로 Type I(Mistral — 기하학적 조기 결정, 어휘 불확실성 억제)과 Type II(EXAONE/Qwen — 어휘 불확실성 성장)를 분류하였다. Sublayer Probe(Exp12)와 8개 변형 실험으로 **DPO + OpenHermes 계열 사실성 데이터 조합이 초기 MLP(L1–4)에 hallucination 게이팅 회로를 형성**함을 입증하였다. NousHermes(SFT+DPO)에서 OpenHermes(SFT only)와 동일 데이터, DPO만 추가 시 best layer가 L21→L2로 이동한다는 직접적 증거를 확보하였다.

---

## 참고문헌

- Kuhn, L., et al. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *Nature*, 616, 1–7.
- Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.
- Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. *arXiv:2207.05221*.
- Xiong, M., et al. (2024). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. *ICLR 2024*.
- Ji, Z., et al. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*, 55(12), 1–38.
- Geva, M., et al. (2021). Transformer Feed-Forward Layers Are Key-Value Memories. *EMNLP 2021*.
- nostalgebraist (2020). Interpreting GPT: the logit lens. *LessWrong*.

---

## 부록: 그림 목록

| 파일 | 내용 |
|------|------|
| MAIN_scaling_layer_analysis.png | Qwen 스케일링 + 레이어 AUROC 종합 |
| MAIN_cross_model_comparison.png | 크로스 아키텍처 비교 (11모델) |
| MAIN_hidden_geometry.png | Hidden state 기하학 + Logit Lens (Exp11) |
| MAIN_geometry_overlay.png | 3개 모델 Logit Lens 오버레이 |
| MAIN_sublayer_probe.png | Sublayer Probe per-model (Exp12) |
| MAIN_sublayer_overlay.png | Sublayer Probe 오버레이 |
| MAIN_mistral_family_exp06.png | Mistral 8개 변형 Layer Probe 비교 |
| MAIN_mistral_family_exp12.png | Mistral 8개 변형 MLP-Attn gap 비교 |

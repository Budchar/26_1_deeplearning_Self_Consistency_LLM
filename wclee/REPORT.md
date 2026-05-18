# LLM Hallucination 사전 탐지 및 Overconfidence 분석
## 딥러닝 텀프로젝트 보고서

---

## 1. 서론

대형 언어 모델(LLM)은 사실에 기반하지 않은 내용을 그럴듯하게 생성하는 **hallucination** 문제를 가진다. 특히 모델이 틀린 답을 높은 확신으로 제시하는 경우, 사용자가 이를 의심 없이 수용할 위험이 있다. 본 연구는 두 가지 핵심 질문을 탐구한다.

- **RQ1.** 출력이 완전히 생성되기 전, 토큰 생성 과정에서 hallucination을 탐지할 수 있는가?
- **RQ2.** Hallucination 정보는 모델 내부의 어느 레이어에 인코딩되며, 이는 파라미터 수·아키텍처와 어떤 관계인가?

로컬 LLM 10개 이상 모델(총 파라미터 규모: 0.36B–14B, 7개 아키텍처 계열)에 대해 TriviaQA / MMLU / NaturalQuestions 데이터셋으로 실험을 수행하였으며, 사전 탐지 방법론 5종, 레이어 프로브, 스케일링 법칙 분석, 크로스 아키텍처 비교, 다중 데이터셋 일반화, 프로브 기반 개입까지 총 10가지 실험을 진행하였다.

---

## 2. 실험 설정

### 2.1 모델

| 모델 | 파라미터 | 레이어 수 | 아키텍처 계열 |
|------|----------|-----------|--------------|
| SmolLM2-360M-Instruct | 0.36B | 32 | SmolLM (HuggingFaceTB) |
| Qwen2.5-0.5B-Instruct | 0.5B | 24 | Qwen (Alibaba) |
| SmolLM2-1.7B-Instruct | 1.7B | 24 | SmolLM |
| Qwen2.5-1.5B-Instruct | 1.5B | 28 | Qwen |
| Qwen2.5-3B-Instruct | 3.0B | 36 | Qwen |
| OPT-6.7B | 6.7B | 32 | OPT (Meta) |
| Qwen2.5-7B-Instruct | 7.0B | 28 | Qwen |
| Mistral-7B-Instruct-v0.2/v0.3 | 7.0B | 32 | Mistral |
| Falcon-7B-Instruct | 7.0B | 32 | Falcon (TII) |
| EXAONE-3.5-7.8B-Instruct | 7.8B | 32 | EXAONE (LG AI Research) |
| Qwen2.5-14B-Instruct | 14.0B | 48 | Qwen |

### 2.2 데이터셋 및 평가

| 데이터셋 | 태스크 | 특성 |
|----------|--------|------|
| **TriviaQA** (n=200–300 per model) | 사실 지식 기반 개방형 QA | 주요 실험 데이터셋 |
| **MMLU** (n=200) | 57개 주제 다지선다 (MC) | 정확도가 매우 높아 탐지 역전 현상 발생 |
| **NaturalQuestions** (n=200) | 사실 기반 개방형 QA | TriviaQA 대비 낮은 정확도 경향 |

- 정답 판정: 모델 출력 내 정답 문자열 포함 여부 (case-insensitive)
- 주요 지표: AUROC (높을수록 hallucination 탐지력 우수), ECE (낮을수록 calibration 우수)

### 2.3 방법론 개요

| 실험 번호 | 방법 | 핵심 아이디어 |
|----------|------|--------------|
| Exp01 | **Token Entropy** | 각 토큰 생성 시 logit distribution 엔트로피 측정 |
| Exp02 | **Semantic Entropy** | 의미 동등 답변을 클러스터링 후 클러스터 수준 엔트로피 계산 (Kuhn et al., 2023) |
| Exp03 | **Self-Consistency** | 동일 질문 반복 샘플링, 일관성을 confidence proxy로 사용 (Wang et al., 2023) |
| Exp04 | **Calibration (ECE)** | 모델 confidence vs. 실제 정확도 비교 |
| Exp05 | **Verbalized Confidence** | 모델이 직접 자신의 확신도(%)를 언어로 표현하도록 유도 (Kadavath et al., 2022) |
| Exp06 | **Layer Probe** | 각 레이어의 hidden state에 로지스틱 회귀 probe를 훈련, 레이어별 hallucination 인코딩 정도 측정 |
| Exp07 | **Scaling Analysis** | Qwen 0.5B→14B 스케일링에 따른 탐지 지표 변화 |
| Exp08 | **Cross-arch Comparison** | 7개 아키텍처에 걸친 ~7B 크기 모델 비교 (12개 모델) |
| Exp09 | **Multi-Dataset** | TriviaQA / MMLU / NaturalQuestions 세 데이터셋 일반화 |
| Exp10 | **Intervention** | Layer probe 기반 실시간 재샘플링으로 정확도 개선 시도 |

---

## 3. 결과: RQ1 — 사전 탐지 방법론 비교

### 3.1 EXAONE-3.5-7.8B 심층 분석 (TriviaQA, n=300)

EXAONE 모델에 대해 5가지 사전 탐지 방법론을 전수 비교하였다. 모델 정확도는 **58.0%**였다.

**표 1. 방법론별 Hallucination 탐지 AUROC**

| 방법 | AUROC | 비고 |
|------|-------|------|
| **N-Clusters (Semantic Entropy)** | **0.839** | 최고 성능 |
| Token Entropy (max) | 0.798 | |
| Token Entropy (mean) | 0.793 | |
| Semantic Entropy (raw) | 0.783 | |
| Self-Consistency (embedding) | 0.772 | |
| Verbalized Confidence | 0.752 | |
| Self-Consistency (majority vote) | 0.661 | 최저 |

**핵심 발견 ①**: 가장 효과적인 방법은 **N-Clusters** (AUROC 0.839)였다. 정답을 생성한 경우 평균 **2.1개**의 의미 클러스터가 형성된 반면, 오답 생성 시 **5.3개**로 크게 증가하였다. 즉, 모델이 정답을 모를수록 다양하고 일관성 없는 답변을 생성한다는 것이 정량적으로 확인된다.

**토큰 엔트로피 분석**: 정답 생성 시 평균 엔트로피 **0.216**, 오답 시 **0.498**로, 오답 시 토큰 분포가 현저히 더 불균일하다 (gap = 0.282).

### 3.2 Overconfidence 분석

**표 2. Calibration 및 Overconfidence 결과 (EXAONE)**

| 지표 | 값 | 해석 |
|------|-----|------|
| ECE (log-prob confidence) | 0.205 | calibration 불량 |
| ECE (verbalized confidence) | 0.278 | calibration 더욱 불량 |
| Verbalized conf. (정답) | 0.936 | 모델이 맞출 때 94% 확신 |
| Verbalized conf. (오답) | 0.789 | 모델이 **틀릴 때도 79% 확신** |
| Overconfident 비율 | 2.4% | conf ≥ 0.8 이면서 틀린 경우 |

**핵심 발견 ②**: EXAONE은 오답을 생성할 때에도 평균 79%의 확신을 언어로 표현하며, ECE가 0.28에 달해 심각한 calibration 문제를 보인다. Reliability diagram 분석 결과, 저확신 구간(0–0.1)에서도 정확도가 36%에 달해 log-prob 기반 confidence가 실제 성능을 제대로 반영하지 못하고 있다.

---

## 4. 결과: RQ1 확장 — 파라미터 스케일과 탐지 난이도

### 4.1 Qwen 스케일링 법칙 (Exp07)

Qwen2.5 패밀리 5개 모델(0.5B→14B)을 동일 조건에서 비교하였다.

**표 3. Qwen 스케일링: 파라미터 vs. 탐지 성능**

| 모델 | 파라미터 | 정확도 | AUROC (Token Ent.) | AUROC (N-Clusters) | Entropy Gap |
|------|----------|--------|---------------------|---------------------|-------------|
| Qwen2.5-0.5B | 0.5B | 0.275 | 0.674 | 0.668 | +0.319 |
| Qwen2.5-1.5B | 1.5B | 0.415 | 0.720 | 0.720 | +0.353 |
| Qwen2.5-3B | 3.0B | 0.570 | 0.742 | 0.675 | +0.213 |
| Qwen2.5-7B | 7.0B | 0.675 | 0.765 | 0.775 | +0.261 |
| Qwen2.5-14B | 14.0B | 0.825 | 0.729 | 0.710 | +0.108 |

**핵심 발견 ③**: **정확도와 Entropy Gap 사이에 강한 음의 상관관계(r = −0.86)** 가 존재한다. 즉, 모델이 클수록 hallucination을 덜 하지만, 오답 시에도 엔트로피가 낮아져 탐지가 더 어려워진다. 14B 모델의 엔트로피 갭(0.108)은 0.5B(0.319)의 1/3에 불과하다. **크고 정확한 모델일수록 엔트로피 기반 탐지가 취약해지는 트레이드오프**가 존재한다.

### 4.2 크로스 아키텍처 비교 (Exp08 Extended — 10개 모델)

TriviaQA 200개 샘플로 소형 모델부터 7B+ 모델까지 전 규모 비교를 실시하였다.

**표 4. 전 아키텍처 크로스 비교 (파라미터 순)**

| 모델 | 파라미터 | 정확도 | AUROC (Entropy) | AUROC (N-Clust) | Entropy Gap |
|------|----------|--------|-----------------|-----------------|-------------|
| SmolLM2-360M | 0.36B | 0.320 | 0.706 | 0.665 | — |
| SmolLM2-1.7B | 1.7B | 0.575 | 0.775 | 0.734 | — |
| OPT-6.7B | 6.7B | 0.440 | 0.515 | 0.488 ⚠️ | — |
| Mistral-7B-v0.3 | 7.0B | **0.820** | 0.594 ⚠️ | 0.639 | +0.087 |
| Qwen2.5-7B | 7.0B | 0.675 | 0.765 | 0.775 | +0.261 |
| EXAONE-3.5-7.8B | 7.8B | 0.610 | 0.787 | **0.797** | **+0.374** |
| Falcon-7B | 7.0B | — | — | — | 비호환 |

> ⚠️ **Falcon-7B**: `trust_remote_code=True` 조건에서 내부적으로 DynamicCache 객체가 `output.scores`로 반환되어 모든 logit 기반 지표 계산 불가. Entropy AUROC, N-Clusters 모두 측정 실패. 로그 기반 지표가 의존하는 표준 텐서 인터페이스를 지원하지 않는 아키텍처 비호환 사례.

> ⚠️ **OPT-6.7B**: Instruction tuning 없는 base 모델로, 개방형 QA에서 답변 형식을 따르지 않아 AUROC가 0.5에 가까운 무작위 수준. Mistral과 유사한 패턴이나 원인이 다름 (Mistral: 초기 레이어 결정 → 이후 엔트로피 평탄화; OPT: 형식 미준수 → 정답 판정 불가).

**핵심 발견 ④**: **SmolLM2 계열(0.36B, 1.7B)은 유사 파라미터 Qwen 대비 낮은 정확도지만 엔트로피 기반 AUROC는 비슷하거나 높다** (SmolLM2-1.7B: AUROC 0.775). 즉, "탐지 용이성"은 파라미터 수보다 아키텍처와 훈련 방식에 더 강하게 의존한다.

---

## 5. 결과: RQ1 확장 — 다중 데이터셋 일반화 (Exp09)

TriviaQA에 최적화된 탐지 지표가 다른 도메인에서도 일관되게 작동하는지 검증하였다.

**표 5. EXAONE-3.5-7.8B: 데이터셋별 탐지 성능**

| 데이터셋 | 정확도 | AUROC (Entropy) | AUROC (N-Clust) | Entropy Gap |
|----------|--------|-----------------|-----------------|-------------|
| TriviaQA | 0.610 | 0.787 | 0.775 | +0.374 |
| NaturalQuestions | 0.295 | 0.670 | 0.611 | +0.181 |
| MMLU | **0.975** | **0.311** ⚠️ | **0.387** ⚠️ | −0.143 |

**표 6. Qwen2.5-7B: 데이터셋별 탐지 성능**

| 데이터셋 | 정확도 | AUROC (Entropy) | AUROC (N-Clust) | Entropy Gap |
|----------|--------|-----------------|-----------------|-------------|
| TriviaQA | 0.675 | 0.765 | 0.759 | +0.261 |
| NaturalQuestions | 0.355 | 0.692 | 0.608 | +0.193 |
| MMLU | **0.895** | **0.580** | **0.545** | −0.023 |

**핵심 발견 ⑤: MMLU에서 탐지 역전 현상 (AUROC < 0.5)**

EXAONE은 MMLU에서 97.5%의 정확도를 보이며, 이 경우 오답 샘플이 극소수(5개/200)에 불과하다. 이 소수의 오답이 특히 "어려운 경계 문제"인 반면, 다수 정답에는 쉬운 문제가 포함되어 있어 **오답의 엔트로피가 오히려 정답보다 낮아지는 역전 현상**이 발생한다. Entropy gap = −0.143으로 AUROC가 0.5 미만(0.311)을 기록, 랜덤 추측보다 나쁜 탐지 결과를 낸다.

**결론**: 엔트로피 기반 탐지법은 모델이 충분히 많은 오류를 범하는 도메인(정확도 40–70%)에서 효과적이며, 정확도가 극단적으로 높은 쉬운 데이터셋에서는 구조적으로 실패한다. TriviaQA와 NaturalQuestions에서는 두 모델 모두 일관된 탐지력을 보여 개방형 QA 도메인에서의 일반화 가능성을 확인하였다.

---

## 6. 결과: RQ2 — 레이어별 Hallucination 인코딩 분석

### 6.1 레이어 프로브 방법론

각 레이어의 마지막 토큰 hidden state에 로지스틱 회귀(Logistic Regression with StandardScaler)를 훈련하여, 어느 레이어가 정답 여부를 가장 잘 예측하는지 AUROC로 측정하였다. 학습 데이터 150개, 테스트 비율 20%.

### 6.2 전체 결과

**표 7. 모델별 최적 프로브 레이어 및 AUROC**

| 모델 | 파라미터 | 레이어 수 | 최적 레이어 | 깊이(%) | 남은 레이어 수 | Probe AUROC | 정확도 |
|------|----------|-----------|------------|---------|----------------|-------------|--------|
| **Mistral-7B-v0.3** | 7.0B | 32 | **L4** | **12%** ← 이상치 | 28 | 0.808 | 0.840 |
| OPT-6.7B | 6.7B | 32 | L7 | 22% | 25 | 0.670 | 0.473 |
| SmolLM2-1.7B | 1.7B | 24 | L11 | 46% | 13 | 0.694 | 0.587 |
| Qwen2.5-1.5B | 1.5B | 28 | L19 | 68% | 9 | **0.810** | 0.447 |
| Qwen2.5-3B | 3.0B | 36 | L25 | 69% | 11 | 0.766 | 0.567 |
| Qwen2.5-7B | 7.0B | 28 | L20 | 71% | 8 | 0.794 | 0.637 |
| EXAONE-3.5-7.8B | 7.8B | 32 | L30 | 94% | 2 | 0.838 | 0.590 |
| Qwen2.5-0.5B | 0.5B | 24 | L23 | 96% | 1 | 0.773 | 0.293 |
| Qwen2.5-14B | 14.0B | 48 | L46 | 96% | 2 | **0.846** | 0.787 |

### 6.3 주요 발견: 아키텍처별 인코딩 위치의 극명한 차이

**[1] Mistral·OPT: 초기 레이어 인코딩 (12–22% 깊이)**

Mistral-7B의 hallucination 정보는 **레이어 4(12%)**에서 최대로 추출된다. OPT-6.7B도 유사하게 **레이어 7(22%)**에서 최적 프로브를 보인다. 두 모델 모두 엔트로피 기반 AUROC가 0.5 수준에 불과하지만, layer probe AUROC는 Mistral 0.808, OPT 0.670으로 의미 있는 탐지력을 갖는다. 이는 이른 레이어에서 결정된 확신이 이후 레이어에서 균일한 엔트로피로 표출되어 entropy 기반 탐지를 무력화하지만, 내부 표현 자체에는 정보가 보존됨을 의미한다. Mistral이 instruction tuning된 모델인 데 반해 OPT는 base 모델임에도 같은 패턴을 보인다는 점은 이 초기 인코딩 현상이 훈련 방식보다 아키텍처 구조에 더 깊이 기인함을 시사한다.

**[2] Qwen 중간 크기 (1.5B–7B): 68–71% 깊이**

Qwen 1.5B, 3B, 7B 모델은 모두 전체 레이어의 약 68–71% 지점에서 최적 프로브를 보인다. 출력 직전 8–11개 레이어는 hallucination 정보를 추가하지 않으며, 어휘 매핑(vocabulary projection)에 특화된 "출력 정제 구간"으로 볼 수 있다.

**[3] 극단 크기 (0.5B, 14B)와 EXAONE: 94–96% 깊이**

매우 작은 모델(0.5B)과 큰 모델(14B), 그리고 EXAONE은 모두 마지막 1–2개 레이어에서 hallucination 정보가 집중된다. 소형 모델은 표현 용량 부족으로 마지막까지 처리가 지속되고, 대형 모델은 레이어 수가 많아 절대적인 "남은 레이어 수"(2개)가 동일하더라도 정규화 깊이가 96%로 높게 나타난다.

**그림 1** (MAIN_scaling_layer_analysis.png): 레이어 깊이별 Probe AUROC 곡선, Qwen 스케일링, 최적 레이어 vs. 파라미터 버블 차트를 포함한 종합 시각화.

### 6.4 핵심 발견 ⑥: Layer Probe의 범용성

**표 8. Entropy AUROC vs. Layer Probe AUROC 비교 (~7B 모델)**

| 모델 | Entropy AUROC | Layer Probe AUROC | 향상폭 |
|------|---------------|-------------------|--------|
| Mistral-7B | 0.594 | **0.808** | +0.214 |
| Qwen-7B | 0.765 | **0.794** | +0.029 |
| EXAONE-7.8B | 0.787 | **0.838** | +0.051 |

엔트로피 기반 탐지가 사실상 실패하는 Mistral에서도, 레이어 프로브는 AUROC 0.808을 달성하였다. **레이어 프로브는 모델이 엔트로피로 불확실성을 표현하는 방식과 무관하게 내부 표현에서 직접 hallucination 정보를 추출하므로 더 범용적인 탐지 방법임을 실험적으로 확인하였다.**

---

## 7. 결과: Exp11 — Logit Lens 분석: "왜" 아키텍처마다 탐지 난이도가 다른가

### 7.1 Logit Lens 방법론

Logit Lens (nostalgebraist, 2020) 기법은 각 중간 레이어의 hidden state를 최종 layer norm + LM head(출력 선형 레이어)로 직접 투영하여, **그 레이어에서 모델이 어떤 어휘 분포를 "생각"하고 있는지** 추정한다. 각 레이어의 어휘 분포 엔트로피를 구하면, 어느 시점에 정답/오답 간 불확실성 차이가 발생하는지 추적할 수 있다.

### 7.2 Mistral vs EXAONE: 완전히 다른 두 가지 인코딩 방식

**표 9. Logit Lens 어휘 엔트로피 gap 비교 (wrong − correct)**

| 모델 | L0 (입력 직후) | Best Probe Layer | 마지막 레이어 | Gap 최대치 | Peak Layer |
|------|---------------|-----------------|--------------|-----------|------------|
| Mistral-7B | +0.000 | L4: **−0.002** ← 거의 0 | +0.136 | +0.391 | L28 |
| Qwen2.5-7B | +0.000 | L20: **+1.090** | +0.375 | **+1.804** | L26 (93%) |
| EXAONE-7.8B | +0.000 | L30: **+1.489** ← 최대 | +0.626 | +1.489 | L30 (94%) |

**[Type I: Mistral·OPT — 기하학적 조기 결정, 어휘 불확실성 억제]**

Mistral L4에서 layer probe AUROC=0.808 → 정답/오답이 hidden state 벡터 공간에서 분리되어 있음. 그러나 logit lens 엔트로피 gap = **−0.002 ≈ 0** → 두 군집 모두 어휘 분포 수준에서는 동등하게 불확실하다.

해석: Mistral은 레이어 4에서 정답 여부를 **벡터의 기하학적 위치**로 인코딩한다. 이 기하학적 신호는 이후 28개 레이어를 거치며 어휘 분포로 "번역"되는 과정에서 희석된다. 결과적으로 최종 출력의 엔트로피(정답 0.437, 오답 0.573, gap 0.136)는 탐지 신호로 사용하기 부족하다. **기하학은 분리되어 있지만, 어휘 불확실성은 같은 것이다.**

**[Type II: EXAONE/Qwen-7B — 어휘 불확실성이 레이어를 거치며 성장]**

EXAONE에서 logit lens gap은 L0=0에서 시작하여 레이어가 깊어질수록 단조 증가하고, **L30(94% 깊이)에서 +1.489로 최대**가 된다. 정답(1.567)과 오답(3.056)의 어휘 엔트로피가 거의 2배 차이다.

Qwen2.5-7B도 동일한 Type II 패턴을 보인다. Gap은 L20(best probe layer, 71% 깊이)에서 이미 +1.090에 달하며, **L26(93% 깊이)에서 +1.804로 피크**에 도달한다. 흥미롭게도 피크가 best probe layer(L20) 이후에 나타난다 — 기하학적 분리(L20)가 확립된 뒤에도 어휘 불확실성이 계속 증가하는 양상이다.

해석: EXAONE과 Qwen-7B 모두 네트워크 전반에 걸쳐 어휘 수준의 불확실성을 점진적으로 형성한다. 토큰 엔트로피 기반 탐지가 효과적인 이유는, 기하학적 분리와 어휘 불확실성이 함께 성장하기 때문이다. **어휘 불확실성 자체가 hallucination 신호다.**

### 7.3 세 가지 인코딩 유형의 정의

| 유형 | 모델 예시 | Probe AUROC | Entropy AUROC | 메커니즘 |
|------|----------|-------------|---------------|---------|
| **Type I: 기하학적 조기 결정** | Mistral, OPT | 높음 (0.67–0.81) | 낮음 (0.51–0.59) | 초기 레이어(12–22%)에서 표현 공간을 분리하지만, 어휘 불확실성으로 이어지지 않음 |
| **Type II-a: 어휘 불확실성과 기하학의 동시 성장** | EXAONE | 높음 (0.84) | 높음 (0.79) | 어휘 엔트로피 gap이 best probe layer에서 정확히 최대 (gap=+1.489@L30) |
| **Type II-b: 기하학 선행, 어휘 불확실성 후행** | Qwen2.5-7B | 높음 (0.79) | 높음 (0.77) | 기하학적 분리(L20, 71%)가 확립된 뒤 어휘 gap이 L26(93%)까지 계속 증가 (peak=+1.804) |

**핵심 발견 ⑦**: 엔트로피 기반 탐지의 실패는 단순히 "모델이 confident하기 때문"이 아니라, **hallucination 정보의 인코딩 형태 자체가 다르기 때문**이다. Type I 모델에서는 정보가 기하학적으로 존재하나 어휘 분포에 반영되지 않으며, Type II 모델에서는 기하학과 어휘 불확실성이 동시에(또는 순차적으로) 성장한다. Qwen-7B(Type II-b)의 패턴은 기하학과 어휘 불확실성이 반드시 같은 레이어에서 최대화될 필요가 없음을 보여준다.

**그림 2** (MAIN_geometry_overlay.png, MAIN_hidden_geometry.png): Layer probe AUROC 곡선, 표현 분리 곡선(cosine similarity), Logit Lens 어휘 엔트로피 gap 곡선의 3-panel 비교 (3개 모델 오버레이 및 개별 4-row 패널).

---

## 8. 결과: Exp12 — Sublayer Probe & Mistral 패밀리 비교

### 8.1 동기: Mistral-v0.3의 L4 이상치는 왜 발생하는가?

Exp11의 Type I 분류에서 Mistral-7B-Instruct-v0.3의 L4(12%) 조기 인코딩이 이상치임을 확인하였다. 이 현상이 **아키텍처 때문인지, 특정 파인튜닝 때문인지**를 규명하기 위해 두 가지 추가 실험을 진행하였다.

**Exp12A (Mistral 패밀리 Exp06 반복)**: 동일 Mistral-7B 아키텍처에 서로 다른 학습을 적용한 5개 변형 모델에 대해 Layer Probe(Exp06)를 수행하였다.

**Exp12B (Sublayer Probe)**: 각 트랜스포머 레이어 내부를 두 구간으로 분리하여 probe를 수행하였다.

- `attn_stream[i]` = layer_input[i] + attention_output[i]  (MLP 이전 잔차 스트림)
- `mlp_stream[i]`  = full layer output[i]                  (MLP 이후 잔차 스트림)

이를 통해 각 레이어에서 Attention 서브레이어와 MLP 서브레이어 중 어느 쪽이 hallucination 신호를 주도하는지 측정하였다.

### 8.2 Mistral 패밀리 Layer Probe 비교 (Exp12A) — 8개 변형 모델

**표 10. Mistral-7B 기반 8개 변형 모델 Layer Probe 결과 (TriviaQA, n=150)**

| 모델 | 학습 방법 | 최적 레이어 | 깊이 | AUROC | L2 | L4 |
|------|----------|-----------|------|-------|----|----|
| Mistral-7B-v0.1 (BASE) | 없음 | L29/32 | **91%** | 0.789 | 0.484 | 0.422 |
| Mistral-7B-Instruct-v0.2 | SFT | L21/32 | 66% | 0.615 | 0.491 | 0.429 |
| OpenHermes-2.5 | SFT (OpenHermes) | L21/32 | 66% | 0.795 | 0.602 | 0.590 |
| Zephyr-7B-beta | SFT (UltraChat) + DPO | L15/32 | 47% | 0.559 | 0.466 | 0.497 |
| OpenChat-3.5 | C-RLFT | L16/32 | 50% | **0.868** | 0.611 | 0.542 |
| Starling-LM-7B-alpha | C-RLFT + RLHF | L32/32 | **100%** | 0.664 | 0.552 | 0.496 |
| **NousHermes-2-Mistral-7B-DPO** | **SFT (OpenHermes) + DPO** | **L2/32** | **6%** | **0.806** | **0.806** | 0.729 |
| **Mistral-7B-Instruct-v0.3** | SFT + DPO | **L4/32** | **12%** | **0.808** | 0.712 | **0.808** |

> **핵심 발견 ⑧ (확장)**: L1–4 초기 인코딩은 **"OpenHermes 계열 SFT 데이터 + DPO"** 조합에서만 나타난다.
> - OpenHermes(SFT only) → L21 (66%) → **DPO 추가(NousHermes)** → **L2 (6%)** : 동일 데이터, DPO만 추가했을 때 15배 이상 앞으로 이동
> - Zephyr(다른 SFT 데이터 + DPO) → L15 (47%): DPO는 있지만 SFT 데이터가 달라 효과 약함
> - Starling(C-RLFT + RLHF) → **L32 (100%)**: RLHF는 오히려 인코딩을 가장 늦은 레이어로 밀어냄

파인튜닝 방법에 따른 인코딩 깊이 패턴:
- **RLHF** (Starling): 100% — 최후방
- **SFT only / BASE**: 66–91% — 후반부 (Type II)
- **C-RLFT** (OpenChat): 50% — 중간부
- **SFT(다른 데이터) + DPO** (Zephyr): 47%
- **SFT(OpenHermes) + DPO** (NousHermes, v0.3): **6–12%** — Type I 초기 인코딩

### 8.3 Sublayer Probe 결과 (Exp12B): MLP vs Attention 기여

**표 11. Sublayer Probe — Exp06 최적 레이어에서의 MLP vs Attention AUROC (8개 변형 + EXAONE)**

| 모델 | Exp06 최적 L | attn@best | mlp@best | gap | 주도 | 글로벌 최대 (attn) | 글로벌 최대 (mlp) |
|------|------------|-----------|----------|-----|------|-----------------|----------------|
| Mistral-7B-v0.3 | L4 | 0.768 | 0.736 | −0.032 | Attn | L3: 0.768 | **L3: 0.808** |
| NousHermes-2-DPO | L2 | 0.722 | 0.708 | −0.014 | Both | L19: 0.785 | **L1: 0.806** |
| OpenHermes-2.5 | L21 | 0.801 | 0.764 | −0.037 | Attn | L21: 0.801 | L20: 0.795 |
| Zephyr-7B-beta | L15 | 0.509 | 0.559 | +0.050 | MLP | L0: 0.565 | L14: 0.559 |
| OpenChat-3.5 | L16 | 0.785 | 0.764 | −0.021 | Attn | **L15: 0.882** | L15: 0.868 |
| Starling-LM-7B | L32 (OOB) | — | — | — | — | L30: 0.664 | L31: 0.672 |
| Mistral-7B-v0.2 | L21 | 0.559 | 0.559 | 0.000 | Both | L19: 0.609 | L20: 0.615 |
| Mistral-7B-BASE | L29 | 0.609 | 0.590 | −0.019 | Both | L28: 0.770 | L28: 0.789 |
| EXAONE-3.5-7.8B | L30 | 0.838 | 0.810 | −0.028 | Attn | **L15: 0.917** | L15: 0.875 |

전체 레이어를 탐색하면 더 극적인 패턴이 드러난다:

- **Mistral-7B-v0.3 / NousHermes-DPO**: MLP stream이 L1–L3에서 AUROC=0.806–0.808로 글로벌 최대. **초기 MLP가 hallucination 인코딩을 주도**. Geva et al.(2021) MLP key-value memory 가설과 일치 — DPO 시그널이 초기 MLP 레이어에 사실성 게이트 회로를 형성한 것으로 해석.
- **EXAONE-3.5-7.8B**: Attention stream 글로벌 최대 **L15=0.917**, Exp06 best(L30=0.838)를 크게 초과. MLP는 신호를 오히려 약화. Type II-a 모델에서 Attention이 hallucination을 주도적으로 인코딩.
- **OpenChat-3.5**: Attention L15=0.882로 최대 — C-RLFT 방식이 Attention 기반의 강한 중간층 인코딩을 형성. 가장 높은 Exp06 AUROC(0.868).
- **Starling-LM-7B**: Exp06 best_layer=32 (마지막 레이어), Exp12에서도 L30–31이 최대 — RLHF가 인코딩을 최후반부로 밀어냄. OpenChat보다 AUROC가 낮고 인코딩이 더 분산됨.

> **핵심 발견 ⑨ (확장)**: 학습 방법이 Sublayer 수준의 역할까지 결정한다.
>
> - **DPO + 사실성 SFT 데이터** → 초기 MLP(L1–4) 회로 형성 → Type I 인코딩
> - **C-RLFT** → 중간층 Attention(L15–16) 주도
> - **RLHF** → 후반부 전체에 신호 분산, 마지막 레이어 최대
> - **SFT only** → 중후반부 Attention/MLP 공동 인코딩

**그림 3** (MAIN_mistral_family_exp06.png, MAIN_mistral_family_exp12.png): Mistral 패밀리 8개 모델의 Layer Probe AUROC 곡선 비교 및 MLP-Attention gap 비교.

---

## 9. 결과: Exp10 — Layer Probe 기반 개입 실험

### 9.1 실험 설계

1. 학습: TriviaQA 75개 샘플(전체의 50%)로 Layer probe 훈련
2. 테스트: 나머지 75개 샘플에서 1차 답변 생성
3. 개입 조건: Hallucination 확률(1 − P(correct)) > 0.6이면 최대 3회 재샘플링
4. 선택: 가장 낮은 hallucination 확률의 후보 답변 채택

### 9.2 결과 (EXAONE-3.5-7.8B)

| 조건 | 정확도 | AUROC | 개입 횟수 |
|------|--------|-------|----------|
| Baseline (개입 없음) | 0.560 | 0.665 | — |
| Layer Probe 개입 | 0.560 | 0.665 | 31/75 (41.3%) |
| **정확도 향상** | **+0.000** | **+0.000** | — |

> Probe 학습 AUROC: 0.760, 최적 레이어: L15/32

### 9.3 해석: 중요한 부정적 결과 (Negative Finding)

개입 실험에서 **정확도 향상이 전혀 관찰되지 않았다.** 이는 다음의 근본적 한계를 드러낸다:

- **재샘플링의 한계**: Layer probe는 hallucination 위험이 높은 질문을 올바르게 식별하지만, 동일 질문에 재샘플링해도 모델이 해당 사실 지식을 갖추고 있지 않다면 계속 같은 오답을 생성한다.
- **Probe vs. 지식의 분리**: Probe가 탐지하는 불확실성 신호와 실제 정답 생성 능력은 별개다. 불확실성이 높다고 판단되더라도 재시도로 정답이 나오지 않는다.
- **실용적 함의**: Layer probe 기반 실시간 개입 전략은 모델의 지식 한계를 극복하지 못한다. 효과적인 개입을 위해서는 RAG(Retrieval-Augmented Generation), 외부 지식 베이스 연동, 또는 앙상블 모델 전환 등이 필요하다.

이 부정적 결과는 그 자체로 중요한 연구 기여다: **"탐지 가능 = 교정 가능"이 아님을 실험적으로 입증하였다.**

---

## 10. 종합 논의

### 10.1 사전 탐지 방법 비교 요약

| 특성 | Token Entropy | Semantic Entropy (N-Clust) | Layer Probe |
|------|--------------|----------------------------|-------------|
| 추론 시간 오버헤드 | 없음 | 중간 (n회 샘플링) | 중간 (forward pass 추가) |
| 아키텍처 범용성 | 낮음 (Mistral·OPT에 실패) | 낮음 | **높음 (모든 모델 유효)** |
| 데이터셋 범용성 | 중간 (MMLU에서 실패) | 중간 | 중간 |
| 최고 AUROC (EXAONE) | 0.798 | **0.839** | 0.838 |
| 훈련 데이터 필요 | 불필요 | 불필요 | **필요 (레이블된 샘플)** |
| 개입 효과 | N/A | N/A | **없음** (부정 결과) |

### 10.2 연구 한계

- **정답 판정**: 문자열 매칭 방식으로 파라프레이즈된 정답을 오답 처리할 가능성 있음
- **MMLU 탐지 역전**: 정확도 > 90%인 데이터셋에서 엔트로피 기반 탐지가 구조적으로 실패
- **아키텍처 비호환**: Falcon-7B (`trust_remote_code=True`)의 DynamicCache 반환으로 logit 기반 지표 측정 전면 실패
- **OPT 형식 미준수**: Instruction-tuned 모델 대비 기반 모델(OPT-6.7B)의 형식적 불일치로 탐지 AUROC ≈ 0.5
- **개입 실험 한계**: 재샘플링 기반 개입이 모델 지식 한계를 극복하지 못함

### 10.3 주요 기여

1. **파라미터-탐지 트레이드오프의 정량적 확인**: 정확도와 엔트로피 갭 사이 r = −0.86의 강한 반비례 관계를 5개 모델에서 실증
2. **Mistral 이상치 발견 및 기원 규명**: 동일 크기 모델 중 가장 높은 정확도를 가진 Mistral-v0.3이 엔트로피 탐지에 완전히 실패하며, Sublayer Probe(Exp12)로 L3–4의 MLP 주도 조기 인코딩이 v0.3 학습 특이적임을 확인 (BASE/v0.2/Zephyr/OpenHermes는 모두 중~후반부 레이어)
3. **아키텍처별 최적 프로브 위치 지도 작성**: 11개 이상 모델에 대해 최적 hallucination 탐지 레이어를 실험적으로 특정
4. **Sublayer 수준 기여도 분석**: MLP vs Attention stream probe로 모델별 주도 서브레이어를 분리 — EXAONE은 L15 Attention이 AUROC=0.917로 Exp06 전체 최고치(L30=0.838)를 초과
5. **다중 데이터셋 일반화 검증**: 탐지법이 TriviaQA/NQ에서 일관되나 MMLU처럼 정확도가 극단적으로 높은 도메인에서 역전 현상을 발견
6. **개입 실험의 부정 결과 보고**: "탐지 가능 ≠ 교정 가능"의 실증적 확인

---

## 11. 결론

본 연구는 11개 이상의 로컬 LLM에 대해 12가지 실험을 통해 hallucination 사전 탐지, 레이어 수준 분석, 아키텍처 메커니즘 비교를 수행하였다.

**RQ1에 대하여**: N-Clusters 기반 Semantic Entropy가 AUROC 0.839로 최고 탐지 성능을 보였으며, 정답 시 2.1개 vs 오답 시 5.3개의 의미 클러스터 수 차이가 주요 탐지 신호로 작동하였다. 단, 엔트로피 기반 방법은 ① 아키텍처(Mistral, OPT처럼 훈련 방식에 따라 실패), ② 도메인 정확도(MMLU처럼 정확도가 극단적으로 높을 때 역전)에 크게 의존한다.

**RQ2에 대하여**: 레이어 프로브는 모든 모델에서 유효하며(AUROC 0.77–0.85), 특히 엔트로피 탐지가 실패하는 Mistral에서도 0.808을 달성하였다. 최적 레이어 위치는 모델마다 크게 다르며—Mistral-v0.3은 레이어 4(12%), 중간 Qwen은 68–71%, EXAONE과 대형 Qwen은 94–96%—이는 아키텍처별 정보 처리 방식의 근본적 차이를 반영한다.

**RQ3에 대하여 (Exp11–12)**: Mistral-v0.3의 L4 조기 인코딩이 아키텍처가 아닌 v0.3 특이적 학습에 기인함을 5개 변형 모델 비교로 실증하였다. Sublayer Probe는 EXAONE의 Attention stream이 L15에서 AUROC=0.917(Exp06 최고치 0.838 초과)에 달함을 보여주며, Mistral-v0.3의 조기 MLP 인코딩과 대비되는 두 가지 극단적 메커니즘을 확인하였다.

실용적 함의: **엔트로피 기반 탐지는 계산 비용이 낮지만 모델·도메인 의존적**이며, **레이어 프로브는 레이블된 훈련 데이터가 필요하지만 범용적이고 강인하다**. 실시간 hallucination 필터링이 필요한 시스템에서는 두 방법을 앙상블하되, 대상 모델에 대한 레이어 프로브 사전 훈련이 권장된다. 단, 탐지된 hallucination을 단순 재샘플링으로 교정하는 전략은 효과가 없으며, RAG 등 외부 지식 연동이 필수적임을 본 연구의 개입 실험이 확인하였다.

---

## 참고문헌

- Kuhn, L., et al. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *Nature*, 616, 1–7.
- Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.
- Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. *arXiv:2207.05221*.
- Xiong, M., et al. (2024). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. *ICLR 2024*.
- Ji, Z., et al. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*, 55(12), 1–38.

---

## 부록

### A. 생성된 결과 파일

```
results/raw/
├── 01_token_entropy_exaone_triviaqa_*.jsonl      # 토큰 엔트로피 (300샘플)
├── 02_semantic_entropy_exaone_triviaqa_*.jsonl   # 시맨틱 엔트로피
├── 03_self_consistency_exaone_triviaqa_*.jsonl   # 자기일관성
├── 04_calibration_exaone_triviaqa_*.jsonl        # calibration 원시 데이터
├── 04_calibration_summary_*.json                 # ECE 요약
├── 05_verbalized_confidence_*.jsonl              # 언어화된 확신도
├── 06_layer_probe_{model}_triviaqa_*.json        # 레이어 프로브 (7개 모델)
├── 07_scaling_qwen_triviaqa_*.json               # 스케일링 분석
├── 08_cross_extended_*.json                      # 크로스 아키텍처 (10개 모델)
├── 09_multi_dataset_{model}_*.json               # 다중 데이터셋 (EXAONE, Qwen7B)
└── 10_intervention_exaone_*.json                 # 개입 실험 (부정 결과)

results/figures/
├── MAIN_scaling_layer_analysis.png              # [메인] 레이어+스케일링 종합 (3패널)
├── MAIN_cross_model_comparison.png              # [메인] 크로스 아키텍처 4패널
├── SUMMARY_all_methods.png                      # 방법론 AUROC 비교 + ROC 곡선
├── SUMMARY_calibration.png                      # Reliability Diagram + 분포
└── SUMMARY_semantic_entropy.png                 # 시맨틱 엔트로피 심층 분석
```

### B. 실험 재현 방법

```bash
cd hallucination_detect

# 단일 모델 (EXAONE) 전체 방법론 재현
bash scripts/run_all.sh exaone

# 개별 실험
python experiments/01_token_entropy/run.py --model exaone --n_samples 300
python experiments/06_layer_probe/run.py   --model mistral --n_samples 150
python experiments/07_model_scaling/run.py --n_samples 200
python experiments/08_cross_model/run_extended.py --n_samples 200

# Exp09: 다중 데이터셋 일반화
python experiments/09_multi_dataset/run.py --model qwen_7b --n_samples 200 \
    --datasets triviaqa mmlu naturalquestions

# Exp10: Layer probe 개입 실험
python experiments/10_intervention/run.py --model exaone --n_samples 150 \
    --threshold 0.6 --max_retries 3

# 시각화
python scripts/plot_scaling_layer.py
python scripts/plot_cross_model.py
python scripts/final_summary.py
```

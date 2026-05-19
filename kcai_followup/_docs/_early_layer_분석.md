# 초반 Layer 활성화 분석 (2026-05-19)

> 우창님 발견("SFT만 한 모델도 초반 layer hallucination 활성화")에 대한 본인 4 모델 데이터 검증.
> 입력: `04_layer_probe_replication/results/*_probe.json` 12 cells

---

## 분석 질문

1. 본인 4 모델 (Llama 1B/3B, Qwen 1.5B/3B)에서도 초반 layer (rel_depth ≤ 0.25, ≈ L0-L5)에 의미 있는 hallucination 신호가 보이나?
2. DPO·RLHF 학습 방법별로 초반 layer 활성화에 차이가 있나?
3. peak이 후반에 있어도 초반 layer도 부수 신호를 가지나?

---

## 결과 표

| 모델 | 데이터셋 | peak layer | peak rel_d | peak AUROC | **초반 max AUROC** | mid max | late max |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | nq_open | L22 | 0.79 | 0.726 | **0.660** | 0.704 | 0.726 |
| Qwen2.5-1.5B-Instruct | squad | L20 | 0.71 | 0.648 | **0.571** | 0.592 | 0.648 |
| Qwen2.5-1.5B-Instruct | triviaqa | L23 | 0.82 | 0.763 | **0.669** | 0.715 | 0.763 |
| Qwen2.5-3B-Instruct | nq_open | L30 | 0.83 | 0.752 | **0.614** | 0.675 | 0.752 |
| Qwen2.5-3B-Instruct | squad | L28 | 0.78 | 0.665 | **0.558** | 0.650 | 0.665 |
| Qwen2.5-3B-Instruct | triviaqa | L36 | 1.00 | 0.779 | **0.635** | 0.694 | 0.779 |
| Llama-3.2-1B-Instruct | nq_open | L11 | 0.69 | 0.669 | **0.558** | 0.630 | 0.669 |
| Llama-3.2-1B-Instruct | squad | L6 | 0.38 | 0.689 | **0.651** | 0.689 | 0.675 |
| Llama-3.2-1B-Instruct | triviaqa | L11 | 0.69 | 0.744 | **0.676** | 0.742 | 0.744 |
| Llama-3.2-3B-Instruct | nq_open | L16 | 0.57 | 0.698 | **0.592** | 0.682 | 0.698 |
| Llama-3.2-3B-Instruct | squad | L16 | 0.57 | 0.759 | **0.686** | 0.751 | 0.759 |
| Llama-3.2-3B-Instruct | triviaqa | L19 | 0.68 | 0.783 | **0.687** | 0.778 | 0.783 |

---

## 종합 통계

| 지표 | 값 |
|---|---|
| 초반 max AUROC 평균 | **0.630** |
| 중앙값 | 0.643 |
| 표준편차 | 0.048 |
| 범위 | 0.558 ~ 0.687 |
| AUROC > 0.65 인 cells | **6/12 (50%)** |
| AUROC > 0.70 인 cells | **0/12 (0%)** |

---

## 학습 방법별 비교

| 모델군 | 학습 방법 | 초반 max AUROC 평균 | > 0.65 인 cells |
|---|---|---|---|
| Llama-3.2-1B-Instruct | SFT + DPO (Meta) | 0.628 | 2/3 |
| Llama-3.2-3B-Instruct | SFT + DPO (Meta) | 0.655 | 2/3 |
| Qwen2.5-1.5B-Instruct | SFT + RLHF (Alibaba) | 0.633 | 2/3 |
| Qwen2.5-3B-Instruct | SFT + RLHF (Alibaba) | 0.602 | 0/3 |

**DPO와 RLHF 평균 차이 약 0.02** (0.628-0.655 vs 0.602-0.633). 학습 방법 차이가 작음.

---

## 핵심 발견

### 1. 초반 layer에 약한 신호는 있음
- 12 cells 모두 초반 layer AUROC > 0.55 (chance 0.5 위)
- 절반은 > 0.65 (의미 있는 신호)
- 단 > 0.70 도달 cells 없음

### 2. peak은 여전히 후반
- 12 cells 중 11개의 peak이 후반 (rel_depth > 0.55)
- 예외 1: Llama-3.2-1B / SQuAD (L6, rel_d 0.38)
- 즉 본인 4 모델은 "초반 약한 신호 + 후반 강한 peak" 패턴

### 3. 학습 방법(DPO vs RLHF)에 따른 초반 차이 미미
- Llama (SFT+DPO) 평균 0.628-0.655
- Qwen (SFT+RLHF) 평균 0.602-0.633
- 차이 약 0.02 → fine-tuning 종류가 초반 layer 활성화의 결정 변수 아님

---

## 우창님 발견과 본인 데이터 비교

| 측면 | 우창님 관찰 | 본인 4 모델 |
|---|---|---|
| 초반 layer hallucination 활성화 | **있음** (Mistral-7B SFT only 포함) | **약하게 있음** (AUROC 0.6-0.7) |
| peak이 초반에 있는지 | "초반에 발생" 표현 (모호) | **peak은 후반** (rel_d 0.71 평균) |
| DPO 효과 | "DPO 없어도 발생" → DPO 가설 약화 | 본인 데이터에서도 DPO·RLHF 차이 약함 |

**해석**:
- 우창님 발견을 부분적으로 확인: 초반 layer에 신호가 있음
- 단 강도는 약함 (peak 아님)
- "DPO만의 효과"라는 단일 인과 주장은 본인 데이터에서도 약함

NousHermes peak L2 (관점 B) 결과가 본인 4 모델과 다른 이유:
- 가능성 1: 진짜 DPO 효과 (가설 부분 유지)
- 가능성 2: NousHermes 특수 SFT 데이터 효과 (가설 약화)
- 우창님 발견(SFT만 한 Mistral-7B도 초반)은 **가능성 2 지지**

---

## Paper 메인 주장 재정의

### v1 (기존)
"DPO가 hallucination 인코딩 위치를 L21에서 L2로 이동시킨다"
→ **약화**. 우창님 발견 + 본인 4 모델 초반 활성화 패턴

### v2 (수정)
"**0.68 peak depth는 fine-tuning 방법(SFT/DPO/RLHF)에 무관하게 hallucination 인코딩의 의미 결정 경계로 작동**"

- 실험 3 Fisher 92% (12 cells, Llama 1B/3B + Qwen 1.5B/3B) — peak 0.65 평균
- 학습 방법 다양 (DPO + RLHF 혼합)인데 peak 일관
- 우창님 관찰(SFT only도 초반 활성화)은 "peak이 어디인지"보다 "전 layer에 신호 분포"라는 다른 측면

### Paper 통합 위치
- **메인**: §Peak Depth Interpretation
- 보조: §Cross-method Comparison (학습 방법별 초반 layer 활성화 표)
- 약화: §Causal Analysis (DPO 인과는 관찰 수준)

---

## 한계

1. **본인 4 모델 모두 fine-tune 된 Instruct**: SFT only 모델 (예: OpenHermes-2.5-Mistral-7B)이 본인 분석에 없음. 우창님 관찰의 직접 검증 X
2. **n=1000 prompts**: 통계 검정력 약함
3. **probe = sklearn LogisticRegression**: 다른 probe (MLP, kernel SVM)에서는 다를 수 있음

---

## 결론

우창님 발견과 본인 데이터는 일관: **fine-tuning 종류가 초반 layer 활성화의 강도를 크게 좌우하지 않음**. paper 메인 주장을 "DPO 인과"에서 "fine-tuning 무관 일반 패턴"으로 이동하는 게 데이터에 더 정직함.

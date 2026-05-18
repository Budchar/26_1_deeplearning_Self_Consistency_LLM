# Layer Mechanism Analysis: Fine-tuning이 Hallucination 인코딩 위치를 결정한다

**기여자**: wclee  
**날짜**: 2026-05-14  
**데이터셋**: TriviaQA  
**분석 대상**: Mistral-7B 아키텍처 8개 변형 + EXAONE-3.5, Qwen2.5-7B, Phi-3-mini (총 11개)

---

## 개요

팀 Phase1에서 확인된 *"hallucination 신호 peak 상대 깊이 0.68 ± 0.12"* 발견에 이어,
**왜 모델마다 인코딩 위치가 다른가**를 fine-tuning 방법 관점에서 분석한다.

핵심 질문: *"같은 아키텍처, 같은 SFT 데이터인데 DPO 하나가 best layer를 L21 → L2로 바꿀 수 있는가?"*

---

## 핵심 가설 결과

**H1 (레이어 탐침 탐지 가능성)**: **PASS**
> 11개 모델 중 8개에서 AUROC > 0.75 달성. 팀 SEPs 결과(μ=0.767)와 일관됨.
> OpenChat-3.5 AUROC=0.868이 전체 최고 (TriviaQA 기준).

**H2 (Fine-tuning이 인코딩 위치를 결정)**: **PASS**
> OpenHermes(SFT only, L21) vs NousHermes(동일 SFT + DPO, L2): best layer가 −19 layers 이동.
> DPO 적용 여부가 유일한 차이. 일반 대화 데이터 + DPO(Zephyr)는 이 효과 없음.

**H3 (초반 MLP vs 후반 Attention 인코딩 주체 구분)**: **PASS**
> Type I 모델(DPO+사실SFT): MLP stream이 주도 (gap > 0).
> Type II 모델(나머지): Attention stream이 주도하거나 동등 (EXAONE gap=−0.079).

**H4 (Logit Lens vocab entropy 패턴 구분)**: **PASS** *(2026-05-14 업그레이드)*
> Exp11 Logit Lens 결과 4개 모델 전수 확인. late_gap_avg(후반 50% entropy gap 평균) 기준 분류 100% 일치.
>
> - Type I (Mistral-v0.3): late_gap = −0.297 → **entropy 조기 수렴** (후반에서 wrong-correct 차 역전)
> - Type II (Mistral-BASE, EXAONE, Qwen-7B): late_gap = +0.244 ~ +0.781 → **후반까지 지속 성장**

---

## 주요 발견

- **Fine-tuning이 아키텍처보다 중요**: Mistral-BASE는 L29(90%), 동일 아키텍처의 Mistral-v0.3은 L4(12%). 아키텍처는 원인이 아님.

- **"사실성 SFT + DPO" 조합이 결정적**:
  - Zephyr (일반대화 + DPO) → L15: DPO만으로는 부족
  - OpenHermes (사실SFT, DPO없음) → L21: 사실성 데이터만으로는 부족
  - NousHermes (사실SFT + DPO) → **L2**: 두 조건 동시에 필요

- **RLHF는 반대 방향**: Starling-LM-7B(C-RLFT + RLHF)는 best layer가 L32(100%)로 마지막 레이어 — 팀 발견 H4("SC가 고불확실 구간에서 정확도를 낮춤")와 맥락이 닿음.

- **EXAONE Attention 이상**: L15 attn_stream AUROC=0.917이 full-layer L30 AUROC=0.838보다 높음. Attention 메커니즘이 후반에서 더 집중된 신호를 만듦.

- **팀 peak depth 0.68 재해석**: Type I 모델(early MLP) 제외 시 Type II 모델의 평균 깊이 ≈ 0.72, 팀 발견과 일관. 즉 팀의 0.68 ± 0.12 분포는 Type II 기반이며, Type I 포함 시 **이중봉 분포** 나타남.

---

## Mistral 아키텍처 패밀리 상세 결과

| 모델 | Fine-tuning | SFT 데이터 | DPO | Best L | 깊이(%) | AUROC |
|------|------------|-----------|-----|--------|---------|-------|
| Mistral-7B-v0.1 (BASE) | 없음 | — | ✗ | L29 | 91% | 0.789 |
| Mistral-7B-Instruct-v0.2 | SFT | 일반 instruction | ✗ | L21 | 66% | 0.615 |
| OpenHermes-2.5 | SFT | OpenHermes (사실 Q&A) | ✗ | L21 | 66% | 0.795 |
| Zephyr-7B-beta | SFT+DPO | UltraChat (일반 대화) | ✓ | L15 | 47% | 0.559 |
| OpenChat-3.5 | C-RLFT | GPT-4 대화 | ✗ | L16 | 50% | **0.868** |
| Starling-LM-7B-alpha | C-RLFT+RLHF | GPT-4 + Nectar | ✗ | L32 | 100% | 0.664 |
| **NousHermes-2 (DPO)** | SFT+DPO | **OpenHermes (사실 Q&A)** | **✓** | **L2** | **6%** | 0.806 |
| **Mistral-7B-Instruct-v0.3** | SFT+DPO | OpenHermes 계열 추정 | **✓** | **L4** | **12%** | 0.808 |

> **결정적 대조**: OpenHermes(L21, 0.795) vs NousHermes(L2, 0.806) — 유일한 차이는 DPO 적용 여부.

---

## Sublayer Probe 결과 (Exp12)

각 레이어를 두 지점으로 분리:
- **attn_stream[i]** = layer_input[i] + attn_output[i] (MLP 이전)
- **mlp_stream[i]** = full layer output (MLP 이후)

| 모델 | Best L | Attn AUROC | MLP AUROC | Gap (MLP−Attn) | 주체 |
|------|--------|-----------|-----------|----------------|------|
| Mistral-v0.3 | L4 | 0.776 | 0.808 | +0.032 | MLP |
| NousHermes-DPO | L2 | 0.646 | 0.806 | **+0.160** | MLP |
| Mistral-BASE | L29 | 0.741 | 0.762 | +0.021 | Both |
| OpenHermes | L21 | 0.758 | 0.783 | +0.025 | Both |
| EXAONE-3.5 | L30 | **0.917** | 0.838 | −0.079 | **Attention** |
| Qwen2.5-7B | L22 | 0.798 | 0.812 | +0.014 | Both |

---

## 인코딩 유형 분류 (Type I / II)

```
Type I — Early MLP (L ≤ 5, MLP dominant)
  ├── Mistral-7B-Instruct-v0.3  (L4,  12%)
  └── NousHermes-2-DPO          (L2,   6%)
  공통: OpenHermes 사실성 SFT + DPO

Type II — Late (L > 15)
  ├── Attention dominant: EXAONE-3.5    (L30, 94%)
  ├── Both:              Mistral-BASE   (L29, 91%)
  │                      OpenHermes     (L21, 66%)
  │                      Qwen2.5-7B     (L22, 79%)
  └── MLP (late):        Phi-3-mini     (L18, 56%)
  └── (outlier) RLHF:    Starling-7B    (L32, 100%)
```

---

## 메커니즘 해석

**왜 사실성 SFT + DPO가 초반 MLP 인코딩을 만드는가?**

1. **MLP Key-Value Memory** (Geva et al., 2021): 초반 MLP 레이어는 사실 지식을 key-value 쌍으로 저장하는 구조를 가짐.

2. **SFT 역할**: OpenHermes의 고밀도 사실 Q&A 학습이 이 key-value 회로를 강화.

3. **DPO 역할**: "이 답이 맞다/틀리다" 선호 신호가 초반 MLP의 사실 판별 회로를 조기 활성화하도록 파라미터 업데이트.

결과: 모델이 토큰을 생성하기 전, L2–4에서 이미 "이 답은 틀릴 것"이라는 신호가 형성됨.

---

## 팀 Phase1 발견과의 관계

| 팀 Phase1 발견 | 본 분석 보완 |
|--------------|------------|
| SEPs가 SE보다 +0.065 AUROC 향상 | Type I 모델에서는 초반 레이어 탐침만으로도 충분한 신호 확보 가능 → SEPs의 레이어 선택 전략에 활용 |
| Peak depth 0.68 ± 0.12 | Type I(early MLP) / Type II(late) 이중 분포로 재해석. DPO+사실SFT 모델은 이 패턴 외부 |
| 소형 모델에서 SEPs 효과 큼 (r=−0.53) | Fine-tuning 방법이 모델 크기보다 더 강한 예측 변수일 가능성. 7B 동일 규모에서 AUROC 0.559~0.868 분산 |
| SC가 고불확실 구간에서 역효과 | Starling(RLHF, L32)처럼 후반 레이어 의존 모델에서 SC의 불확실성 증폭 효과 큰 것으로 해석 가능 |

---

## Logit Lens Entropy 패턴 분석 (Exp11 — H4 검증)

**지표 정의**: `late_gap_avg` = 후반 50% 레이어에서의 `mean_entropy(wrong) − mean_entropy(correct)` 평균  
→ 양수가 클수록 wrong 답변의 entropy가 correct보다 크게 성장 = Type II 패턴  
→ 음수 전환 = early converge = Type I 패턴

| 모델 | Type | early\_gap\_avg | late\_gap\_avg | 패턴 |
| --- | --- | --- | --- | --- |
| Mistral-7B-v0.3 (DPO+사실SFT) | **Type I** | +0.003 | **−0.297** | L4 이후 역전 → 조기 수렴 |
| Mistral-7B-v0.1 (BASE) | Type II | +0.002 | +0.244 | 후반까지 지속 성장 |
| EXAONE-3.5-7.8B | Type II | −0.050 | +0.781 | 후반까지 지속 성장 |
| Qwen2.5-7B | Type II | −0.000 | +0.771 | 후반까지 지속 성장 |

**결론**: 4개 모델 전수에서 `late_gap_avg`의 음수 여부만으로 Type I/II 분류가 100% 일치.  
DPO+사실SFT가 hallucination 결정을 초반 레이어로 당기는 것이 Logit Lens에서도 관찰됨.

---

## 시각화

- `plots/MAIN_mistral_family_exp06.png` — Mistral 8변형 Layer Probe AUROC 곡선
- `plots/MAIN_mistral_family_exp12.png` — MLP vs Attn Gap 비교 (8변형)
- `plots/MAIN_sublayer_probe.png` — 모델별 Sublayer Probe 상세
- `plots/MAIN_sublayer_overlay.png` — 전체 모델 MLP-Attn gap 오버레이

---

## 참고문헌

- Geva, M. et al. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP 2021.
- Farquhar, S. et al. (2024). *Detecting Hallucinations in Large Language Models Using Semantic Entropy*. Nature.
- Kossen, J. et al. (2024). *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs*. arXiv:2406.15927.
- nostalgebraist (2020). *Interpreting GPT: The Logit Lens*. LessWrong.
- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023.

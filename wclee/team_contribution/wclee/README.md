# wclee — Layer Mechanism Analysis

**기여 범위**: Hallucination 인코딩 *위치*와 *주체*를 fine-tuning 방법론과 연결하는 메커니즘 분석

---

## 핵심 발견 (한 줄)

> **사실성 SFT 데이터(OpenHermes) + DPO 조합이 hallucination 인코딩을 후반 레이어(L21)에서 초반 MLP(L2–4)로 이동시킨다.**

---

## 실험 구성

| 실험 | 내용 | 결과 |
|------|------|------|
| Exp06 Layer Probe | 레이어별 hidden state → 로지스틱 회귀 AUROC | Best layer 위치가 모델마다 상이 |
| Exp11 Logit Lens | 중간 hidden state → LM head → vocab entropy | Type I/II 모델 간 entropy 성장 패턴 차이 |
| Exp12 Sublayer Probe | 각 레이어를 Attn stream / MLP stream으로 분리 탐침 | Type I=MLP 주도, Type II=Attention 주도 |

**분석 모델**: Mistral-7B 8개 변형 + EXAONE-3.5 + Qwen2.5-7B + Phi-3-mini (총 11개)

---

## 디렉토리 구조

```
wclee/
├── README.md                          # 이 파일
├── code/
│   ├── sublayer_probe_run.py          # Exp12 실행 스크립트 (forward hook 기반)
│   ├── plot_sublayer.py               # Sublayer probe 시각화
│   └── plot_mistral_family.py         # Mistral 패밀리 비교 시각화
├── docs/
│   └── 02_layer_mechanism_결과.md    # 메인 결과 보고서
├── results/
│   ├── hypothesis_verdicts.json       # H1–H4 가설 검증 결과
│   ├── mistral_family_summary.json    # Mistral 8변형 Layer Probe 요약
│   └── sublayer_probe_summary.json    # Sublayer Probe (MLP vs Attn) 요약
└── plots/
    ├── MAIN_mistral_family_exp06.png  # Mistral 패밀리 Layer AUROC 곡선
    ├── MAIN_mistral_family_exp12.png  # MLP vs Attn gap (Mistral 패밀리)
    ├── MAIN_sublayer_probe.png        # 모델별 Sublayer Probe 상세
    └── MAIN_sublayer_overlay.png      # 전체 모델 MLP-Attn gap 오버레이
```

---

## 팀 발견과 연결

| 팀 Phase1 (SE/SEPs) | 본 분석 (Layer Mechanism) | 시사점 |
|--------------------|-----------------------|--------|
| SEPs peak depth 0.68 ± 0.12 | DPO+사실SFT 모델은 0.06–0.12, 나머지는 0.47–1.00 | 이중봉 분포; Type 분류 필요 |
| SEPs가 SE보다 +0.065 AUROC | Type I 모델은 초반 레이어만 탐침해도 충분 | SEPs 레이어 선택 전략 최적화 가능 |
| 소형 모델에서 SEPs 효과 큼 | 7B 동규모에서 fine-tuning 방법에 따라 AUROC 0.56–0.87 분산 | 모델 크기 외 fine-tuning이 추가 예측 변수 |

---

## 재현 방법

```bash
# 1. Sublayer Probe 실행 (예: Mistral-v0.3)
python code/sublayer_probe_run.py \
    --model mistral \
    --dataset triviaqa \
    --split validation

# 2. 시각화
python code/plot_mistral_family.py
python code/plot_sublayer.py
```

**요구 환경**: Python 3.10+, PyTorch 2.x, transformers 4.40+, scikit-learn, matplotlib  
**GPU**: A100 40GB (7B 모델 기준 레이어별 forward pass × n_samples)

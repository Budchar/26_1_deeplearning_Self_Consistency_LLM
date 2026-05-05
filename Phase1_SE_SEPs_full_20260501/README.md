# Phase 1 — Semantic Entropy + SEPs 사전실험 (재현·확장 패키지)

## 무엇이 들어있나

```
Phase1_SE_SEPs_full_*.zip
├── code/                        # 실행 코드 (Python)
│   ├── run_phase1.py            # main entry
│   ├── data_loader.py           # TriviaQA / NQ / SQuAD 로더
│   ├── sample_generator.py      # 모델 로드 + N=10 sampling + hidden state
│   ├── se_compute.py            # Semantic Entropy (entailment clustering)
│   ├── seps_probe.py            # SEPs (linear/MLP probe on hidden states)
│   ├── adaptive_se.py           # Cost-aware Adaptive SE 알고리즘
│   ├── metrics.py               # AUROC, ECE, Brier, AURC, Wilcoxon
│   ├── analyze_phase1.py        # 결과 통합 + 시각화
│   ├── launch_full_sweep.sh     # 5 모델 × 3 데이터 launcher
│   └── README.md                # 코드 사용 가이드
│
├── runs/                        # 셀별 raw 결과 (hidden states 제외)
│   └── {model}/{dataset}/
│       ├── metrics.json         # 셀 정량 (SE/SEPs AUROC, ECE, Brier)
│       ├── probes.json          # layer-wise probe AUROC
│       ├── adaptive.json        # Cost-aware Adaptive 결과
│       ├── generations.jsonl    # 모델이 생성한 답변 N=10 (재현용)
│       └── se.jsonl             # 질문별 SE 측정값
│
├── results/                     # 통합 결과
│   ├── sweep_summary.json       # 15 셀 종합
│   ├── 01_hypothesis_verdicts.json  # H1~H4 PASS/FAIL 판정
│   └── smoke_test_summary.json
│
├── 01_SE_SEPs_결과.md          # paper-ready 보고서
├── plots/01_*.png               # 시각화 4개 (600 DPI)
│   ├── 01_stratified_sc_se.png  (H1)
│   ├── 01_prehoc_vs_posthoc.png (H2)
│   ├── 01_layer_emergence_heatmap.png (H3)
│   └── 01_adaptive_cost_accuracy.png (H4)
├── tables/01_*.csv              # 정량 표 13개
│
└── _사전실험_계획.md           # 가설·실험 설계 원본
```

## 제외된 것

- **모델 가중치** (Qwen2.5, Llama-3.2): HuggingFace에서 자동 다운로드. 약 30GB
- **Hidden state .npz** (15셀 × 838MB ≈ 12.6GB): SEPs probe 학습 후 빈번히 안 쓰여서 제외. 필요 시 `sample_generator.py`로 재생성 가능

## 환경 셋업

```bash
# venv 생성 (Python 3.12+)
python3 -m venv .venv
source .venv/bin/activate

# PyTorch (Blackwell sm_120 GPU 시 nightly 필수, 그 외 stable)
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128

# 라이브러리
pip install transformers datasets accelerate bitsandbytes \
  evaluate scikit-learn matplotlib seaborn tqdm einops sentencepiece nltk

# HuggingFace 로그인 (Llama-3.2 license 동의 필요)
huggingface-cli login --token <YOUR_TOKEN>
```

## 재현 (전체 sweep)

```bash
cd code/
bash launch_full_sweep.sh
```

5070 단독 기준 ~16시간 (5 모델 × 3 데이터셋).

## 부분 재실행 (한 셀만)

```bash
python3 run_phase1.py \
  --models Qwen/Qwen2.5-1.5B-Instruct \
  --datasets triviaqa \
  --n 1000 --n-samples 10
```

## 분석/시각화 재생성

```bash
python3 analyze_phase1.py
```

## 확장 아이디어 (후속 연구로)

1. **Multi-seed (n≥3)**: 현재 single seed → 통계적 신뢰도 보강
2. **모델 family 다양화**: Phi-3.5, Mistral, Gemma 추가
3. **다른 task**: Reasoning (GSM8K), open-ended generation
4. **더 큰 모델**: 14B, 32B (4-bit + multi-GPU)
5. **Layer 동적 선택**: Best-2-layer 전략 vs full-layer

## 결과 핵심 (재인용용)

- H2 PASS: SEPs vs SE gap **+0.065** 평균, **모델 클수록 축소** (Pearson r = -0.53)
- H3 PASS: 환각 정보 emerge peak at **relative depth 0.68 ± 0.12** (layer 수 무관)
- H1 PARTIAL: SC × SE 보완성 단조 패턴 ✓ but greedy ceiling
- H4 PARTIAL: Adaptive SE **18.8% 비용 절감, ΔAUROC -0.008**

## Citation 정보

- Semantic Entropy: Farquhar et al., Nature 2024
- SEPs: Kossen et al., arXiv:2406.15927 (Oxford OATML)

## 문의

- 본인 이메일/카톡으로

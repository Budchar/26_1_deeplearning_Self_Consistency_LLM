# kcai_followup — 후속 메커니즘 실험 4건

> 작성: 2026-05-16 ~ 2026-05-18
> 자원: RTX 5070 12GB
> 분석 단위: 4 모델 × 3 데이터셋 = 12 cells (Qwen 7B는 OOM으로 제외)

## 목표

종합보고서의 핵심 가설(H3-revised, peak depth 0.682) 보강 + 인과 증거 확보.
"관찰 보고서"에서 "메커니즘 논문"으로 격상 시도.

## 4 실험 + 보강 요약

| 실험 | 평가 | 핵심 결과 | paper 섹션 |
|---|---|---|---|
| 3. Multi-metric trajectory | SUPPORT | Fisher 91.7% in H3 band | §Peak Depth Interpretation |
| 3+ Logit·Attn·Answer | 부분 보강 | Attn entropy 42% in band | 위 섹션 강화 |
| 4. Layer probe 재현 | STRONG REPRO | 10/12 cells \|Δ\|<0.05 vs Phase 1 | §Reproducibility |
| 1a. Patching Llama 1B | 부분 지지 | L3 Δ=-0.12 (TriviaQA) | §Causal Analysis |
| 1b. Patching Mistral 7B | 부분 지지 | L3 Δ=+0.04 (TriviaQA) | §Causal Analysis |
| 2. Steering vector | 약한 지지 | 11/12 positive, +1.6pp mean | §Linear Hypothesis |
| paired t-test | 통계 검증 | patch sig 4/24, steering sig 5/72 | §Statistical Validity |

자세한 내용은 `_docs/_통합보고서.md` 참조.

## 디렉토리

```
kcai_followup/
├── 01_activation_patching/      실험 1 코드·결과·plot
├── 02_steering_vector/          실험 2
├── 03_multi_metric_trajectory/  실험 3 + 보강
├── 04_layer_probe_replication/  실험 4
├── _shared/                     공용 util (model loader·eval·resumable·prompt format)
└── _docs/                       보고서 markdown 11개
```

## 코드 실행 환경

- venv: `/home/kcai/experiments/dl_team_v2/shared/.venv` (Phase 1 환경 재사용)
- torch 2.12 + CUDA 12.8 (RTX 5070 sm_120)
- transformers 5.7.0, peft, trl, bitsandbytes 0.49.2
- sklearn, einops, numpy, pandas, matplotlib

## 제외된 자료 (.gitignore)

- `_data/hidden_states/` (1.3 GB hidden state cache, 재추출 가능)
- `_data/logit_attn/` (logit·attention raw)
- `01_activation_patching/results/*/cache_src/` (patching MLP cache)

재실행 시 `python 01_extract_hidden.py --models all --datasets all` 등으로 재생성.

## 한계

- Qwen 7B 미포함 (5070 12GB OOM)
- Mistral 7B patching은 TriviaQA만 (transformers/bnb 4-bit 호환 버그)
- 실험 1·2 n=100 (통계 검정력 약함)
- wclee 원본 코드 미확보 → sklearn 자체 재구현

## 다음 단계

- 실험 5 (동일 base SFT vs SFT+DPO 통제 학습) → 우창님 A100에 부탁
- paper 작성 시 본 보고서 11개 → 4 paper section 통합

## 보고서 문서 (_docs/)

- `_통합보고서.md` ⭐ 시작점
- `_후속실험_계획서.md`
- `03_trajectory_결과.md`
- `03_trajectory_extras_결과.md`
- `04_probe_재현_결과.md`
- `01_patching_결과.md`
- `01_patching_mistral_결과.md`
- `02_steering_결과.md`
- `_paired_ttest_결과.md`

# Phase 1 — Semantic Entropy + SEPs

## Layout

```
code/
  data_loader.py      TriviaQA / NQ-Open / SQuAD validation loaders (1k each, cached)
  sample_generator.py Load LLM, N=10 sampling + greedy + last-prompt-token hidden states
  se_compute.py       Bidirectional-NLI clustering -> SE (discrete + logprob), SC majority
  seps_probe.py       Per-layer LogReg + MLP probes on hidden states (-> SEPs AUROC table)
  adaptive_se.py      Cost-aware Adaptive SE (H4): truncate samples by early SE estimate
  metrics.py          AUROC / ECE / Brier / AURC / Wilcoxon / stratified-by-quartile
  run_phase1.py       Sweep runner ((model, dataset) -> generation -> SE -> probes -> adaptive)
  launch_full_sweep.sh  Full 5-model x 3-dataset sweep with time budget
```

## Quick smoke test

```bash
source ~/experiments/dl_team_v2/shared/.venv/bin/activate
cd ~/experiments/dl_team_v2/01_se_seps/code
python run_phase1.py \
  --models Qwen/Qwen2.5-1.5B-Instruct \
  --datasets triviaqa \
  --n 50 --n-samples 10 --limit 50 \
  --summary-name smoke.json
```

Outputs land at `01_se_seps/runs/<model_safe>/<dataset>/`:
- `generations.jsonl` (greedy + N samples + per-sample avg logprob)
- `hidden/<id>.npz` (last-prompt-token hidden states, fp16)
- `se.jsonl` (cluster_ids + SE + correctness + SC pred)
- `probes.json` (per-layer SEPs probe AUROCs)
- `adaptive.json` (Adaptive SE summary + per-question)
- `metrics.json` (aggregate AUROC/ECE/Brier/AURC + Wilcoxon + stratified)

## Full sweep

```bash
bash ~/experiments/dl_team_v2/01_se_seps/code/launch_full_sweep.sh
```

Estimated total wall time on 5070 alone: ~16 h (run overnight). Resume-safe.

## Hypotheses tested

| ID | Implemented in              | Output to inspect |
|----|------------------------------|-------------------|
| H1 | metrics.stratified_acc + wilcoxon_paired | metrics.json -> stratified_acc, wilcoxon_sc_vs_greedy |
| H2 | seps_probe.run_probes + se AUROC by model | probes.json -> best_*_auroc vs metrics.json -> se_discrete.auroc |
| H3 | seps_probe.run_probes layer_results       | probes.json -> layer_results (AUROC per layer) |
| H4 | adaptive_se.run                            | adaptive.json -> summary.cost_save_frac, auroc_delta |

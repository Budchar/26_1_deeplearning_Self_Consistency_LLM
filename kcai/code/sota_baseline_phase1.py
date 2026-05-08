"""SOTA baseline 1단계 — Mean logprob + SelfCheckGPT-Unigram + SelfCheckGPT-NLI.

기존 generations.jsonl + se.jsonl 데이터 재활용. 추가 GPU 0.

각 cell (model, dataset)마다 5가지 metric AUROC 비교:
  - SE (Semantic Entropy, baseline 기존)
  - SEPs (probe, 우리 main)
  - Mean Logprob
  - SelfCheckGPT-Unigram
  - SelfCheckGPT-NLI

출력: results/sota_phase1_comparison.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 기존 metrics 재활용
import sys
sys.path.insert(0, "/home/kcai/experiments/dl_team_v2/01_se_seps/code")
from metrics import auroc

# 모든 sweep results 위치
SWEEP_RUNS = {
    "phase1_v1": "/home/kcai/experiments/dl_team_v2/01_se_seps/runs",
    "sweep_b": "/home/kcai/experiments/dl_team_v2/04_sweep_b_pythia/runs",
    "sweep_a": "/home/kcai/experiments/dl_team_v2/06_sweep_a_family/runs",
    "option1": "/home/kcai/experiments/dl_team_v2/08_option1_pile_check/runs",
    "option2": "/home/kcai/experiments/dl_team_v2/09_option2_pythia_trajectory/runs",
}


def selfcheck_unigram(samples: List[str]) -> float:
    """SelfCheckGPT-Unigram score: average pairwise unigram overlap.
    Higher score = more consistent answers = lower hallucination probability.
    Returns 1 - overlap (so higher = more hallucination, matches our convention).
    """
    if len(samples) < 2:
        return 0.5
    # tokenize each sample to unigrams (lowercase, simple split)
    grams = [set(s.lower().split()) for s in samples]
    overlaps = []
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            if not grams[i] or not grams[j]:
                overlaps.append(0.0)
                continue
            inter = len(grams[i] & grams[j])
            union = len(grams[i] | grams[j])
            overlaps.append(inter / union if union > 0 else 0.0)
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    # 환각 score = 1 - overlap (낮은 일관성 = 높은 환각 가능성)
    return 1.0 - avg_overlap


def selfcheck_nli_from_clusters(cluster_ids: List[int]) -> float:
    """SelfCheckGPT-NLI proxy via existing cluster_ids.
    More distinct clusters = less consistent = higher hallucination.
    Normalized: (n_clusters - 1) / (N - 1).
    """
    if not cluster_ids:
        return 0.5
    n_clusters = len(set(cluster_ids))
    n = len(cluster_ids)
    if n <= 1:
        return 0.0
    return (n_clusters - 1) / (n - 1)


def mean_logprob_score(sample_logprobs: List[float]) -> float:
    """Negative mean logprob — higher = more hallucination.
    LLM 자신감 낮으면 (logprob 낮으면) 환각 가능성 ↑.
    -inf 값은 -100으로 클립.
    """
    if not sample_logprobs:
        return 0.5
    finite = [v if (not math.isinf(v) and not math.isnan(v)) else -100.0 for v in sample_logprobs]
    mean_lp = sum(finite) / len(finite)
    return -mean_lp  # negate so higher = more uncertain = more hallucination


def compute_metrics_for_cell(generations_path: Path, se_path: Path,
                             metrics_path: Path, probes_path: Path) -> Optional[Dict]:
    """Compute all 5 metrics for one cell. Returns dict with AUROCs."""
    if not all(p.exists() for p in [generations_path, se_path, metrics_path, probes_path]):
        return None

    # Load data
    gens = {}
    with open(generations_path) as f:
        for line in f:
            r = json.loads(line)
            gens[r["id"]] = r

    se_recs = {}
    with open(se_path) as f:
        for line in f:
            r = json.loads(line)
            se_recs[r["id"]] = r

    metrics_existing = json.loads(metrics_path.read_text())
    probes = json.loads(probes_path.read_text())

    # Compute new baseline scores
    ids = sorted(set(gens) & set(se_recs))
    if not ids:
        return None

    correct = []
    se_scores = []
    seps_scores = []  # SEPs not stored per-question, use overall best layer AUROC
    logprob_scores = []
    unigram_scores = []
    nli_scores = []

    for qid in ids:
        gen_r = gens[qid]
        se_r = se_recs[qid]
        c = se_r.get("greedy_correct", 0)
        correct.append(int(c))
        # SE (already computed)
        se_scores.append(float(se_r.get("se_discrete", 0.0)))
        # Mean logprob
        lps = gen_r.get("sample_logprobs", [])
        logprob_scores.append(mean_logprob_score(lps))
        # SelfCheckGPT-Unigram
        samples = gen_r.get("samples", [])
        unigram_scores.append(selfcheck_unigram(samples))
        # SelfCheckGPT-NLI proxy
        cluster_ids = se_r.get("cluster_ids", [])
        nli_scores.append(selfcheck_nli_from_clusters(cluster_ids))

    # AUROC (1=hallucination, 0=correct)
    halluc = [1 - c for c in correct]
    n = len(correct)
    if sum(halluc) < 5 or sum(correct) < 5:
        # too imbalanced
        return {
            "n": n,
            "greedy_acc": sum(correct) / n,
            "skipped": "too_imbalanced",
        }

    return {
        "n": n,
        "greedy_acc": sum(correct) / n,
        "se_discrete_auroc": auroc(se_scores, halluc),
        "seps_best_auroc": probes.get("best_logreg_halluc_auroc")
                          or probes.get("best_logreg_hallucination_auroc"),
        "mean_logprob_auroc": auroc(logprob_scores, halluc),
        "selfcheck_unigram_auroc": auroc(unigram_scores, halluc),
        "selfcheck_nli_auroc": auroc(nli_scores, halluc),
    }


def main():
    out_root = Path("/home/kcai/experiments/dl_team_v2/results")
    out_root.mkdir(exist_ok=True)

    all_results = {}
    for sweep_name, runs_dir_str in SWEEP_RUNS.items():
        runs_dir = Path(runs_dir_str)
        if not runs_dir.exists():
            continue
        all_results[sweep_name] = {}
        for model_dir in sorted(runs_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for ds_dir in sorted(model_dir.iterdir()):
                if not ds_dir.is_dir() or ds_dir.name == "hidden":
                    continue
                ds_name = ds_dir.name
                # for option2, model_dir IS the step (step1000), ds_dir is dataset
                gen = ds_dir / "generations.jsonl"
                se = ds_dir / "se.jsonl"
                metrics_p = ds_dir / "metrics.json"
                probes_p = ds_dir / "probes.json"
                cell_key = f"{model_name}/{ds_name}"
                m = compute_metrics_for_cell(gen, se, metrics_p, probes_p)
                if m is None:
                    continue
                all_results[sweep_name][cell_key] = m
                if "skipped" not in m:
                    print(f"[{sweep_name}] {cell_key}: SE={m['se_discrete_auroc']:.3f} "
                          f"SEPs={m['seps_best_auroc']:.3f} "
                          f"LogP={m['mean_logprob_auroc']:.3f} "
                          f"Unigram={m['selfcheck_unigram_auroc']:.3f} "
                          f"NLI={m['selfcheck_nli_auroc']:.3f}")

    out_path = out_root / "sota_phase1_comparison.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))

    # Summary
    print("\n=== Summary (모든 sweep 평균 AUROC) ===")
    methods = ["se_discrete_auroc", "seps_best_auroc", "mean_logprob_auroc",
               "selfcheck_unigram_auroc", "selfcheck_nli_auroc"]
    for method in methods:
        vals = []
        for sweep, cells in all_results.items():
            for cell, m in cells.items():
                if "skipped" in m:
                    continue
                v = m.get(method)
                if v is not None and not math.isnan(v):
                    vals.append(v)
        if vals:
            print(f"  {method:<28}: avg={sum(vals)/len(vals):.3f}  n={len(vals)}")

    # Pairwise comparisons
    print("\n=== SEPs vs each baseline (cells where SEPs > baseline) ===")
    pairs = [
        ("seps_best_auroc", "se_discrete_auroc"),
        ("seps_best_auroc", "mean_logprob_auroc"),
        ("seps_best_auroc", "selfcheck_unigram_auroc"),
        ("seps_best_auroc", "selfcheck_nli_auroc"),
    ]
    for a, b in pairs:
        n_total = 0
        n_a_wins = 0
        for sweep, cells in all_results.items():
            for cell, m in cells.items():
                if "skipped" in m:
                    continue
                va = m.get(a)
                vb = m.get(b)
                if va is None or vb is None or math.isnan(va) or math.isnan(vb):
                    continue
                n_total += 1
                if va > vb:
                    n_a_wins += 1
        print(f"  SEPs > {b:<28}: {n_a_wins}/{n_total} ({100*n_a_wins/max(n_total,1):.0f}%)")

    print(f"\n[done] {out_path}")


if __name__ == "__main__":
    main()

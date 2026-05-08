"""Statistical tests 5종 — paper용 통계 검정.

1. Wilcoxon paired test (SEPs vs SE)
2. Bootstrap CI for AUROC
3. Mann-Whitney U test (Pile vs non-Pile group)
4. Spearman correlation (size vs gap)
5. Linear regression with controls (gap ~ data + arch + size + capability)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 결과 모음 (sota_phase1 결과 + 모든 sweep cells)
RESULTS_PATH = Path("/home/kcai/experiments/dl_team_v2/results/sota_phase1_comparison.json")
OUT_PATH = Path("/home/kcai/experiments/dl_team_v2/results/statistical_tests.json")


# 모델 메타 정보 (수동 매핑 — 회귀 분석용)
MODEL_META = {
    # phase1_v1 (Instruct)
    "meta-llama__Llama-3.2-1B-Instruct": {"params_b": 1.24, "is_pile": 0, "arch": "llama", "is_instruct": 1},
    "meta-llama__Llama-3.2-3B-Instruct": {"params_b": 3.21, "is_pile": 0, "arch": "llama", "is_instruct": 1},
    "Qwen__Qwen2.5-1.5B-Instruct": {"params_b": 1.54, "is_pile": 0, "arch": "qwen", "is_instruct": 1},
    "Qwen__Qwen2.5-3B-Instruct": {"params_b": 3.09, "is_pile": 0, "arch": "qwen", "is_instruct": 1},
    "Qwen__Qwen2.5-7B-Instruct": {"params_b": 7.62, "is_pile": 0, "arch": "qwen", "is_instruct": 1},
    # sweep_b (Pythia)
    "EleutherAI__pythia-70m-deduped": {"params_b": 0.07, "is_pile": 1, "arch": "gpt-neox", "is_instruct": 0},
    "EleutherAI__pythia-160m-deduped": {"params_b": 0.16, "is_pile": 1, "arch": "gpt-neox", "is_instruct": 0},
    "EleutherAI__pythia-410m-deduped": {"params_b": 0.41, "is_pile": 1, "arch": "gpt-neox", "is_instruct": 0},
    "EleutherAI__pythia-1b-deduped": {"params_b": 1.0, "is_pile": 1, "arch": "gpt-neox", "is_instruct": 0},
    "EleutherAI__pythia-1.4b-deduped": {"params_b": 1.4, "is_pile": 1, "arch": "gpt-neox", "is_instruct": 0},
    "EleutherAI__pythia-2.8b-deduped": {"params_b": 2.8, "is_pile": 1, "arch": "gpt-neox", "is_instruct": 0},
    "EleutherAI__pythia-6.9b-deduped": {"params_b": 6.9, "is_pile": 1, "arch": "gpt-neox", "is_instruct": 0},
    # sweep_a (5 base)
    "meta-llama__Llama-3.2-1B": {"params_b": 1.24, "is_pile": 0, "arch": "llama", "is_instruct": 0},
    "Qwen__Qwen2.5-1.5B": {"params_b": 1.54, "is_pile": 0, "arch": "qwen", "is_instruct": 0},
    "facebook__opt-1.3b": {"params_b": 1.3, "is_pile": 0, "arch": "opt", "is_instruct": 0},  # OPT는 Pile 일부 포함이지만 단순화
    "EleutherAI__gpt-neo-1.3B": {"params_b": 1.3, "is_pile": 1, "arch": "gpt-neo", "is_instruct": 0},
    # option1
    "cerebras__Cerebras-GPT-1.3B": {"params_b": 1.3, "is_pile": 1, "arch": "gpt3-style", "is_instruct": 0},
    "allenai__OLMo-1B-hf": {"params_b": 1.18, "is_pile": 0, "arch": "olmo", "is_instruct": 0},
    "TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T": {"params_b": 1.1, "is_pile": 0, "arch": "llama", "is_instruct": 0},
    # option2 (Pythia trajectory)
    "step1000": {"params_b": 1.4, "is_pile": 1, "arch": "gpt-neox", "is_instruct": 0},
    "step10000": {"params_b": 1.4, "is_pile": 1, "arch": "gpt-neox", "is_instruct": 0},
    "step50000": {"params_b": 1.4, "is_pile": 1, "arch": "gpt-neox", "is_instruct": 0},
    "step100000": {"params_b": 1.4, "is_pile": 1, "arch": "gpt-neox", "is_instruct": 0},
    "step143000": {"params_b": 1.4, "is_pile": 1, "arch": "gpt-neox", "is_instruct": 0},
}


def load_all_cells() -> List[Dict]:
    """sota_phase1_comparison.json + 모델 메타 합쳐서 셀 리스트 반환."""
    if not RESULTS_PATH.exists():
        return []
    data = json.loads(RESULTS_PATH.read_text())
    rows = []
    for sweep_name, cells in data.items():
        for cell_key, m in cells.items():
            if "skipped" in m:
                continue
            model_name, ds_name = cell_key.split("/", 1)
            meta = MODEL_META.get(model_name)
            if not meta:
                continue
            seps = m.get("seps_best_auroc")
            se = m.get("se_discrete_auroc")
            if seps is None or se is None or math.isnan(seps) or math.isnan(se):
                continue
            row = {
                "sweep": sweep_name,
                "model": model_name,
                "ds": ds_name,
                "seps": seps,
                "se": se,
                "gap": seps - se,
                "logprob": m.get("mean_logprob_auroc", float("nan")),
                "unigram": m.get("selfcheck_unigram_auroc", float("nan")),
                "nli": m.get("selfcheck_nli_auroc", float("nan")),
                "greedy_acc": m.get("greedy_acc", 0),
                **meta,
            }
            rows.append(row)
    return rows


def test_wilcoxon(rows: List[Dict], a: str, b: str) -> Dict:
    """Wilcoxon paired (a vs b across cells)."""
    from scipy.stats import wilcoxon
    pairs = [(r[a], r[b]) for r in rows if not math.isnan(r[a]) and not math.isnan(r[b])]
    if len(pairs) < 5:
        return {"n": len(pairs), "skipped": "too few pairs"}
    a_vals = [p[0] for p in pairs]
    b_vals = [p[1] for p in pairs]
    diffs = [p[0] - p[1] for p in pairs]
    if all(d == 0 for d in diffs):
        return {"n": len(pairs), "skipped": "all zero diff"}
    stat, p = wilcoxon(a_vals, b_vals, alternative="greater")
    return {
        "n": len(pairs),
        "a_mean": float(np.mean(a_vals)),
        "b_mean": float(np.mean(b_vals)),
        "diff_mean": float(np.mean(diffs)),
        "wilcoxon_stat": float(stat),
        "p_value_one_sided_greater": float(p),
        "a_wins": sum(1 for d in diffs if d > 0),
    }


def bootstrap_ci_auroc(rows: List[Dict], col: str, n_bootstrap: int = 1000,
                       seed: int = 0) -> Dict:
    """Bootstrap 95% CI for mean of col across cells."""
    vals = [r[col] for r in rows if not math.isnan(r[col])]
    if not vals:
        return {"skipped": "no values"}
    rng = np.random.default_rng(seed)
    arr = np.array(vals)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    return {
        "n": len(vals),
        "mean": float(np.mean(arr)),
        "ci_95_low": float(np.percentile(boot_means, 2.5)),
        "ci_95_high": float(np.percentile(boot_means, 97.5)),
    }


def test_mannwhitney(rows: List[Dict], group_col: str, value_col: str) -> Dict:
    """Mann-Whitney U test: group_col == 1 vs 0 on value_col."""
    from scipy.stats import mannwhitneyu
    g1 = [r[value_col] for r in rows if r.get(group_col) == 1 and not math.isnan(r[value_col])]
    g0 = [r[value_col] for r in rows if r.get(group_col) == 0 and not math.isnan(r[value_col])]
    if len(g1) < 3 or len(g0) < 3:
        return {"n_group1": len(g1), "n_group0": len(g0), "skipped": "too few"}
    stat, p = mannwhitneyu(g1, g0, alternative="greater")
    return {
        "n_group1": len(g1),
        "n_group0": len(g0),
        "mean_group1": float(np.mean(g1)),
        "mean_group0": float(np.mean(g0)),
        "diff": float(np.mean(g1) - np.mean(g0)),
        "u_stat": float(stat),
        "p_value_one_sided_greater": float(p),
    }


def test_spearman(rows: List[Dict], x_col: str, y_col: str) -> Dict:
    """Spearman correlation."""
    from scipy.stats import spearmanr
    pairs = [(r[x_col], r[y_col]) for r in rows
             if not math.isnan(r[x_col]) and not math.isnan(r[y_col])]
    if len(pairs) < 5:
        return {"n": len(pairs), "skipped": "too few"}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rho, p = spearmanr(xs, ys)
    return {
        "n": len(pairs),
        "rho": float(rho),
        "p_value": float(p),
    }


def regression_with_controls(rows: List[Dict], y_col: str = "gap") -> Dict:
    """gap ~ is_pile + log(params) + greedy_acc + arch_dummies."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"skipped": "statsmodels not available"}
    # build df
    valid = [r for r in rows if not math.isnan(r[y_col])]
    if len(valid) < 10:
        return {"skipped": f"too few ({len(valid)})"}
    y = np.array([r[y_col] for r in valid])
    X_dict = {
        "is_pile": [r["is_pile"] for r in valid],
        "log_params": [math.log10(r["params_b"]) for r in valid],
        "greedy_acc": [r["greedy_acc"] for r in valid],
        "is_instruct": [r["is_instruct"] for r in valid],
    }
    # arch dummies (one-hot, drop reference)
    archs = sorted(set(r["arch"] for r in valid))
    ref_arch = archs[0]
    for arch in archs[1:]:
        X_dict[f"arch_{arch}"] = [1 if r["arch"] == arch else 0 for r in valid]
    X = sm.add_constant(np.column_stack(list(X_dict.values())))
    feature_names = ["const"] + list(X_dict.keys())
    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        return {"skipped": f"OLS failed: {e}"}
    return {
        "n": len(valid),
        "r_squared": float(model.rsquared),
        "coefficients": {name: {"coef": float(model.params[i]),
                                "p_value": float(model.pvalues[i])}
                         for i, name in enumerate(feature_names)},
        "ref_arch": ref_arch,
    }


def main():
    rows = load_all_cells()
    print(f"Loaded {len(rows)} cells")
    if not rows:
        print("No data. Run sota_baseline_phase1.py first.")
        return

    out = {"n_cells": len(rows)}

    # 1. Wilcoxon — SEPs vs each baseline
    print("\n=== Wilcoxon paired tests (SEPs > X) ===")
    out["wilcoxon"] = {}
    for baseline in ["se", "logprob", "unigram", "nli"]:
        result = test_wilcoxon(rows, "seps", baseline)
        out["wilcoxon"][f"seps_vs_{baseline}"] = result
        if "p_value_one_sided_greater" in result:
            print(f"  SEPs > {baseline}: a_mean={result['a_mean']:.3f} b_mean={result['b_mean']:.3f} "
                  f"diff={result['diff_mean']:+.3f} p={result['p_value_one_sided_greater']:.4f} "
                  f"({result['a_wins']}/{result['n']} cells)")

    # 2. Bootstrap CI for each method
    print("\n=== Bootstrap 95% CI ===")
    out["bootstrap_ci"] = {}
    for method in ["seps", "se", "logprob", "unigram", "nli"]:
        result = bootstrap_ci_auroc(rows, method)
        out["bootstrap_ci"][method] = result
        if "mean" in result:
            print(f"  {method:<10}: mean={result['mean']:.3f} CI95=[{result['ci_95_low']:.3f}, "
                  f"{result['ci_95_high']:.3f}]")

    # 3. Mann-Whitney — Pile vs non-Pile gap
    print("\n=== Mann-Whitney: Pile group gap > non-Pile group gap ===")
    result = test_mannwhitney(rows, "is_pile", "gap")
    out["mannwhitney_pile"] = result
    if "p_value_one_sided_greater" in result:
        print(f"  Pile group: mean gap = {result['mean_group1']:+.3f} (n={result['n_group1']})")
        print(f"  Non-Pile  : mean gap = {result['mean_group0']:+.3f} (n={result['n_group0']})")
        print(f"  diff = {result['diff']:+.3f}, p = {result['p_value_one_sided_greater']:.4f}")

    # 4. Spearman: params vs gap
    print("\n=== Spearman correlation (size vs gap) ===")
    result = test_spearman(rows, "params_b", "gap")
    out["spearman_size_gap"] = result
    if "rho" in result:
        print(f"  rho = {result['rho']:+.3f}, p = {result['p_value']:.4f} (n={result['n']})")

    # 5. Regression with controls
    print("\n=== Linear Regression (gap ~ is_pile + log_params + greedy_acc + arch) ===")
    result = regression_with_controls(rows, "gap")
    out["regression"] = result
    if "coefficients" in result:
        print(f"  R^2 = {result['r_squared']:.3f}, n = {result['n']}")
        for name, vals in result["coefficients"].items():
            sig = "***" if vals["p_value"] < 0.001 else "**" if vals["p_value"] < 0.01 \
                  else "*" if vals["p_value"] < 0.05 else ""
            print(f"  {name:<20}: coef={vals['coef']:+.4f} p={vals['p_value']:.4f} {sig}")

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[done] {OUT_PATH}")


if __name__ == "__main__":
    main()

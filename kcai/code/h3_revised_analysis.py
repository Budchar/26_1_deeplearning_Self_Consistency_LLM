"""H3-revised: 학습 방법별 peak rel depth 분석.

기존 모든 cells을 학습 방법으로 분류해 그룹별 peak rel depth 측정.

분류:
  - STANDARD_SFT_INSTRUCT: Llama/Qwen-Instruct
  - BASE: Pythia/Llama/Qwen/OPT/GPT-Neo/Cerebras/OLMo/TinyLlama base
  - PYTHIA_TRAJECTORY: Pythia 1.4B 학습 step별
  - FROM_SCRATCH: Sweep C/D 작은 모델

결과: 그룹별 peak rel depth 평균 ± std + Kruskal-Wallis test.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# 모든 sweep 위치
SWEEPS = {
    "phase1_v1": "/home/kcai/experiments/dl_team_v2/01_se_seps/runs",
    "sweep_b": "/home/kcai/experiments/dl_team_v2/04_sweep_b_pythia/runs",
    "sweep_a": "/home/kcai/experiments/dl_team_v2/06_sweep_a_family/runs",
    "option1": "/home/kcai/experiments/dl_team_v2/08_option1_pile_check/runs",
    "option2": "/home/kcai/experiments/dl_team_v2/09_option2_pythia_trajectory/runs",
}


# 학습 방법 분류
TRAINING_METHOD = {
    # phase1_v1 — 모두 Instruct (SFT)
    "meta-llama__Llama-3.2-1B-Instruct": "STANDARD_SFT_INSTRUCT",
    "meta-llama__Llama-3.2-3B-Instruct": "STANDARD_SFT_INSTRUCT",
    "Qwen__Qwen2.5-1.5B-Instruct": "STANDARD_SFT_INSTRUCT",
    "Qwen__Qwen2.5-3B-Instruct": "STANDARD_SFT_INSTRUCT",
    "Qwen__Qwen2.5-7B-Instruct": "STANDARD_SFT_INSTRUCT",
    # sweep_b — Pythia 모두 BASE
    "EleutherAI__pythia-70m-deduped": "BASE",
    "EleutherAI__pythia-160m-deduped": "BASE",
    "EleutherAI__pythia-410m-deduped": "BASE",
    "EleutherAI__pythia-1b-deduped": "BASE",
    "EleutherAI__pythia-1.4b-deduped": "BASE",
    "EleutherAI__pythia-2.8b-deduped": "BASE",
    "EleutherAI__pythia-6.9b-deduped": "BASE",
    # sweep_a — 5 base
    "meta-llama__Llama-3.2-1B": "BASE",
    "Qwen__Qwen2.5-1.5B": "BASE",
    "facebook__opt-1.3b": "BASE",
    "EleutherAI__gpt-neo-1.3B": "BASE",
    # option1
    "cerebras__Cerebras-GPT-1.3B": "BASE",
    "allenai__OLMo-1B-hf": "BASE",
    "TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T": "BASE_INTERMEDIATE",
    # option2 — Pythia 1.4B 학습 step별 (intermediate base)
    "step1000": "PYTHIA_TRAJECTORY",
    "step10000": "PYTHIA_TRAJECTORY",
    "step50000": "PYTHIA_TRAJECTORY",
    "step100000": "PYTHIA_TRAJECTORY",
    "step143000": "PYTHIA_TRAJECTORY",
}


def get_peak_rel_depth(probes_json_path: Path) -> Tuple[float, float, int]:
    """probes.json에서 best layer + total layers + AUROC 추출.
    Returns: (rel_depth, peak_auroc, n_layers)
    """
    p = json.loads(probes_json_path.read_text())
    layers = p.get("layer_results", [])
    if not layers:
        return float("nan"), float("nan"), 0
    best_idx = -1
    best_auroc = -1
    for i, l in enumerate(layers):
        a = l.get("logreg_hallucination_auroc") or 0
        if isinstance(a, (int, float)) and a > best_auroc:
            best_auroc = a
            best_idx = i
    n_layers = p.get("n_layers", len(layers))
    if best_idx < 0 or n_layers <= 1:
        return float("nan"), float("nan"), n_layers
    rel = best_idx / max(1, n_layers - 1)
    return rel, float(best_auroc), n_layers


def collect_cells():
    """모든 sweep cells 수집."""
    rows = []
    for sweep_name, runs_dir in SWEEPS.items():
        runs_path = Path(runs_dir)
        if not runs_path.exists():
            continue
        for model_dir in sorted(runs_path.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            method = TRAINING_METHOD.get(model_name)
            if not method:
                continue
            for ds_dir in sorted(model_dir.iterdir()):
                if not ds_dir.is_dir() or ds_dir.name == "hidden":
                    continue
                probes_p = ds_dir / "probes.json"
                metrics_p = ds_dir / "metrics.json"
                if not probes_p.exists() or not metrics_p.exists():
                    continue
                rel, auroc, n_layers = get_peak_rel_depth(probes_p)
                if math.isnan(rel):
                    continue
                m = json.loads(metrics_p.read_text())
                rows.append({
                    "sweep": sweep_name,
                    "model": model_name,
                    "ds": ds_dir.name,
                    "training_method": method,
                    "rel_depth": rel,
                    "peak_auroc": auroc,
                    "n_layers": n_layers,
                    "greedy_acc": m.get("greedy_acc", 0),
                })
    # Add Sweep C/D from sweep_cd_h3_eval.json
    cd_path = Path("/home/kcai/experiments/dl_team_v2/05_sweep_c_depth/results/sweep_cd_h3_eval.json")
    if cd_path.exists():
        cd_data = json.loads(cd_path.read_text())
        for r in cd_data.get("results", []):
            if "error" in r:
                continue
            rel = r.get("peak_rel_depth")
            auroc = r.get("peak_auroc")
            if rel is None or math.isnan(rel) or auroc is None or math.isnan(auroc):
                continue
            ckpt = Path(r["ckpt"])
            model_name = ckpt.parent.name
            rows.append({
                "sweep": "sweep_cd",
                "model": model_name,
                "ds": "triviaqa",
                "training_method": "FROM_SCRATCH",
                "rel_depth": rel,
                "peak_auroc": auroc,
                "n_layers": r.get("n_layer_probe_levels", 0),
                "greedy_acc": r.get("greedy_acc", 0),
            })
    return rows


def main():
    rows = collect_cells()
    print(f"Total cells: {len(rows)}")
    print()

    # 그룹별 통계
    groups = {}
    for r in rows:
        m = r["training_method"]
        groups.setdefault(m, []).append(r)

    print(f"{'group':<24} {'n':>4} {'mean rel_depth':>15} {'std':>8} {'min':>6} {'max':>6}  {'mean greedy':>12}")
    print('-' * 90)
    summary = {}
    for method in sorted(groups, key=lambda k: -len(groups[k])):
        gr = groups[method]
        rels = np.array([r["rel_depth"] for r in gr])
        greedies = np.array([r["greedy_acc"] for r in gr])
        summary[method] = {
            "n": len(gr),
            "mean_rel_depth": float(rels.mean()),
            "std_rel_depth": float(rels.std()),
            "min_rel_depth": float(rels.min()),
            "max_rel_depth": float(rels.max()),
            "mean_greedy_acc": float(greedies.mean()),
        }
        print(f"{method:<24} {len(gr):>4} {rels.mean():>15.3f} {rels.std():>8.3f} "
              f"{rels.min():>6.3f} {rels.max():>6.3f}  {greedies.mean():>12.3f}")

    # H3 universal 검정 vs H3-revised
    print()
    print("=== H3 (peak rel depth = 0.68 universal) 검정 ===")
    all_rels = np.array([r["rel_depth"] for r in rows])
    print(f"  전체 (n={len(all_rels)}): mean={all_rels.mean():.3f}, std={all_rels.std():.3f}")
    print(f"  H3 (0.68 ± 0.12)와 호환? {'YES' if abs(all_rels.mean() - 0.68) < 0.05 and all_rels.std() < 0.20 else 'NO'}")

    print()
    print("=== H3-revised (그룹별 0.65-0.70) 검정 ===")
    for method, s in summary.items():
        in_range = 0.55 <= s["mean_rel_depth"] <= 0.80 and s["std_rel_depth"] < 0.15
        verdict = "✓ H3 호환" if in_range else "✗ outlier 그룹"
        print(f"  {method:<24}: mean={s['mean_rel_depth']:.3f}±{s['std_rel_depth']:.3f}  {verdict}")

    # Kruskal-Wallis test (그룹간 차이 유의?)
    try:
        from scipy.stats import kruskal
        group_data = [np.array([r["rel_depth"] for r in g]) for g in groups.values()]
        if len(group_data) >= 2 and all(len(g) >= 2 for g in group_data):
            stat, p = kruskal(*group_data)
            print()
            print(f"=== Kruskal-Wallis test (그룹간 차이) ===")
            print(f"  H_stat = {stat:.3f}, p = {p:.4f}")
            print(f"  → {'유의함 (그룹별 다름)' if p < 0.05 else '유의 X'}")
    except Exception as e:
        print(f"Kruskal-Wallis fail: {e}")

    # Pairwise Mann-Whitney
    try:
        from scipy.stats import mannwhitneyu
        print()
        print(f"=== Pairwise Mann-Whitney (그룹별 비교) ===")
        method_list = list(groups.keys())
        for i in range(len(method_list)):
            for j in range(i + 1, len(method_list)):
                a = method_list[i]
                b = method_list[j]
                ga = [r["rel_depth"] for r in groups[a]]
                gb = [r["rel_depth"] for r in groups[b]]
                if len(ga) < 3 or len(gb) < 3:
                    continue
                stat, p = mannwhitneyu(ga, gb, alternative="two-sided")
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                if p < 0.1:
                    print(f"  {a} vs {b}: p = {p:.4f} {sig}")
    except Exception as e:
        print(f"Mann-Whitney fail: {e}")

    # Save
    out = {
        "n_cells": len(rows),
        "all_mean_rel_depth": float(all_rels.mean()),
        "all_std_rel_depth": float(all_rels.std()),
        "groups": summary,
        "raw_cells": rows,
    }
    out_path = Path("/home/kcai/experiments/dl_team_v2/results/h3_revised_analysis.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[done] {out_path}")


if __name__ == "__main__":
    main()

"""실험 2 Step 3: α vs accuracy plot per cell + cross-cell aggregate.

입력: results/{model}__{dataset}/_summary.json
출력: plots/{model}__{dataset}_alpha_sweep.png
       plots/all_cells_alpha_sweep.png
       results/_aggregate.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "_shared"))

from paths import EXP2_STEERING

RESULTS = EXP2_STEERING / "results"
PLOTS = EXP2_STEERING / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def plot_one(summary: dict, out_path: Path):
    ar = summary["alpha_results"]
    alphas = sorted([float(k.replace("alpha_", "")) for k in ar.keys()])
    accs = [ar[f"alpha_{a:+.2f}"]["acc"] for a in alphas]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alphas, accs, "-o", color="steelblue", lw=2, markersize=8)
    baseline_acc = ar.get("alpha_+0.00", {}).get("acc", None)
    if baseline_acc is not None:
        ax.axhline(baseline_acc, color="gray", linestyle="--", alpha=0.6, label=f"baseline (α=0) acc={baseline_acc:.3f}")
    ax.set_xlabel("α (negative: subtract wrong-direction)")
    ax.set_ylabel("accuracy")
    title = f"{summary['model']} / {summary['dataset']} (L{summary['target_layer']} rel_d={summary['target_rel_depth']:.2f}, n={summary['n_prompts']})"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_all(summaries: list[dict], out_path: Path):
    n = len(summaries)
    if n == 0:
        return
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharey=True)
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, s in zip(axes, summaries):
        ar = s["alpha_results"]
        alphas = sorted([float(k.replace("alpha_", "")) for k in ar.keys()])
        accs = [ar[f"alpha_{a:+.2f}"]["acc"] for a in alphas]
        ax.plot(alphas, accs, "-o", color="steelblue", lw=1.5, markersize=5)
        baseline_acc = ar.get("alpha_+0.00", {}).get("acc", None)
        if baseline_acc is not None:
            ax.axhline(baseline_acc, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{s['model'].split('/')[-1].replace('-Instruct','')} / {s['dataset']}", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Steering vector α sweep across cells", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def aggregate(summaries: list[dict]) -> dict:
    """모든 cell에서 best α·평균 효과."""
    rows = []
    for s in summaries:
        ar = s["alpha_results"]
        baseline = ar.get("alpha_+0.00", {}).get("acc", float("nan"))
        best_alpha = None
        best_delta = -float("inf")
        for k, v in ar.items():
            alpha = float(k.replace("alpha_", ""))
            delta = v["acc"] - baseline
            if delta > best_delta and alpha != 0:
                best_delta = delta
                best_alpha = alpha
        rows.append({
            "model": s["model"], "dataset": s["dataset"],
            "baseline_acc": baseline, "best_alpha": best_alpha, "best_delta": best_delta,
        })

    deltas = [r["best_delta"] for r in rows]
    return {
        "n_cells": len(rows),
        "per_cell": rows,
        "best_delta_mean": float(np.mean(deltas)) if deltas else float("nan"),
        "best_delta_max": float(np.max(deltas)) if deltas else float("nan"),
        "n_cells_with_positive_steering": sum(1 for d in deltas if d > 0),
        "linear_hypothesis_verdict": _verdict(deltas),
    }


def _verdict(deltas: list[float]) -> str:
    n = len(deltas)
    if n == 0:
        return "no data"
    n_pos = sum(1 for d in deltas if d > 0.02)  # +2pp threshold
    if n_pos / n >= 0.5:
        return f"SUPPORT linear hypothesis ({n_pos}/{n} cells show ≥+2pp gain via steering)"
    if n_pos / n >= 0.25:
        return f"PARTIAL ({n_pos}/{n} cells)"
    return f"WEAK ({n_pos}/{n} cells)"


def main():
    files = sorted(RESULTS.glob("*/_summary.json"))
    summaries = [json.loads(f.read_text()) for f in files]
    print(f"Found {len(summaries)} cell summaries")
    for s in summaries:
        out_path = PLOTS / f"{s['model'].replace('/', '__')}__{s['dataset']}_alpha_sweep.png"
        plot_one(s, out_path)

    if not summaries:
        return
    plot_all(summaries, PLOTS / "all_cells_alpha_sweep.png")
    agg = aggregate(summaries)
    (RESULTS / "_aggregate.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""실험 4 Step 2: probe AUROC layer 곡선 + peak distribution + Phase 1 비교.

입력: results/{model}__{dataset}_probe.json
출력:
  plots/{model}__{dataset}_probe.png
  plots/all_cells_probe_grid.png
  plots/peak_layer_distribution.png
  results/_summary.json (Phase 1 best_logreg_halluc_auroc와 비교 가능하면)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "_shared"))

from paths import EXP4_PROBE, PHASE1

RESULTS = EXP4_PROBE / "results"
PLOTS = EXP4_PROBE / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def plot_one(cell_metric: dict, out_path: Path):
    lr = cell_metric["layer_results"]
    rel_d = [r["rel_depth"] for r in lr]
    auroc = [r["auroc_mean"] for r in lr]
    auroc_std = [r["auroc_std"] for r in lr]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(rel_d, auroc, yerr=auroc_std, fmt="-o", color="steelblue", markersize=4, lw=1.5, capsize=2)
    pl = cell_metric["peak_layer"]
    pa = cell_metric["peak_auroc"]
    pd = cell_metric["peak_rel_depth"]
    ax.scatter([pd], [pa], color="red", s=120, zorder=5, marker="*", label=f"peak L{pl} rel_d={pd:.2f}, AUROC={pa:.3f}")
    ax.axvspan(0.682 - 0.131, 0.682 + 0.131, color="green", alpha=0.10, label="H3-revised band")
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("relative depth")
    ax.set_ylabel("Layer Probe AUROC (5-fold CV)")
    ax.set_title(f"{cell_metric['model']} / {cell_metric['dataset']} (n={cell_metric['n_prompts']}, acc={cell_metric['n_correct'] / cell_metric['n_prompts'] * 100:.1f}%)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_all_grid(all_metrics: list[dict], out_path: Path):
    n = len(all_metrics)
    if n == 0:
        return
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True)
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, m in zip(axes, all_metrics):
        lr = m["layer_results"]
        rel_d = [r["rel_depth"] for r in lr]
        auroc = [r["auroc_mean"] for r in lr]
        ax.plot(rel_d, auroc, "-", color="steelblue", lw=1.3)
        ax.scatter([m["peak_rel_depth"]], [m["peak_auroc"]], color="red", s=40, zorder=5, marker="*")
        ax.axvspan(0.682 - 0.131, 0.682 + 0.131, color="green", alpha=0.10)
        ax.axhline(0.5, color="gray", linestyle=":", lw=0.7)
        title = m["model"].split("/")[-1].replace("-Instruct", "") + " / " + m["dataset"]
        ax.set_title(f"{title}\npeak rel_d={m['peak_rel_depth']:.2f}, AUROC={m['peak_auroc']:.3f}", fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Layer Probe AUROC across {n} cells (5-fold CV · green band = H3-revised 0.682 ± 0.131)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_peak_distribution(all_metrics: list[dict], out_path: Path) -> dict:
    peaks = [m["peak_rel_depth"] for m in all_metrics]
    aurocs = [m["peak_auroc"] for m in all_metrics]
    n = len(peaks)
    if n == 0:
        return {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax1, ax2 = axes
    ax1.hist(peaks, bins=10, range=(0, 1), color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvspan(0.682 - 0.131, 0.682 + 0.131, color="green", alpha=0.15, label="H3-revised band")
    ax1.axvline(0.682, color="green", linestyle="--", label="H3-revised mean 0.682")
    ax1.set_xlabel("peak relative depth"); ax1.set_ylabel("# cells")
    ax1.set_title("Peak rel_depth distribution"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.scatter(peaks, aurocs, c="coral", edgecolor="black", s=80)
    for i, m in enumerate(all_metrics):
        label = m["model"].split("/")[-1].replace("-Instruct", "")[:14] + "/" + m["dataset"][:6]
        ax2.annotate(label, (peaks[i], aurocs[i]), fontsize=6, alpha=0.7)
    ax2.axvspan(0.682 - 0.131, 0.682 + 0.131, color="green", alpha=0.15)
    ax2.set_xlabel("peak rel_depth"); ax2.set_ylabel("peak AUROC")
    ax2.set_title("Peak AUROC vs depth"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    arr = np.array(peaks)
    in_band = ((arr >= 0.682 - 0.131) & (arr <= 0.682 + 0.131)).mean()
    return {
        "n_cells": n,
        "peak_rel_depth_mean": float(arr.mean()),
        "peak_rel_depth_median": float(np.median(arr)),
        "peak_rel_depth_std": float(arr.std()),
        "in_h3_band_rate": float(in_band),
        "peak_auroc_mean": float(np.mean(aurocs)),
        "peak_auroc_min": float(np.min(aurocs)),
        "peak_auroc_max": float(np.max(aurocs)),
    }


def compare_with_phase1(all_metrics: list[dict]) -> list[dict]:
    """Phase 1의 probes.json (best_logreg_halluc_auroc 등)과 비교."""
    out = []
    for m in all_metrics:
        p1_path = PHASE1 / m["model"].replace("/", "__") / m["dataset"] / "probes.json"
        if not p1_path.exists():
            continue
        p1 = json.loads(p1_path.read_text())
        out.append({
            "model": m["model"],
            "dataset": m["dataset"],
            "ours_peak_auroc": m["peak_auroc"],
            "ours_peak_rel_depth": m["peak_rel_depth"],
            "phase1_best_logreg_halluc_auroc": p1.get("best_logreg_halluc_auroc"),
            "phase1_best_mlp_halluc_auroc": p1.get("best_mlp_halluc_auroc"),
            "ours_minus_phase1_logreg": (m["peak_auroc"] - p1.get("best_logreg_halluc_auroc", 0)) if p1.get("best_logreg_halluc_auroc") is not None else None,
        })
    return out


def main():
    files = sorted(RESULTS.glob("*_probe.json"))
    print(f"Found {len(files)} probe results")
    all_metrics = [json.loads(f.read_text()) for f in files]
    for m in all_metrics:
        out_path = PLOTS / f"{m['model'].replace('/', '__')}__{m['dataset']}_probe.png"
        plot_one(m, out_path)

    if not all_metrics:
        return

    grid_path = PLOTS / "all_cells_probe_grid.png"
    plot_all_grid(all_metrics, grid_path)
    print(f"grid → {grid_path}")

    peak_path = PLOTS / "peak_layer_distribution.png"
    summary_stats = plot_peak_distribution(all_metrics, peak_path)
    print(f"peak → {peak_path}")

    comparison = compare_with_phase1(all_metrics)
    summary = {
        "n_cells": len(all_metrics),
        "peak_stats": summary_stats,
        "phase1_comparison": comparison,
    }
    (RESULTS / "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()

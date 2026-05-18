"""실험 3 Step 3: trajectory plot. cell별 + 통합 비교.

입력: results/{model}__{dataset}_metrics.json
출력:
  plots/{model}__{dataset}_trajectory.png   — 5 panel single cell
  plots/all_cells_grid.png                  — 모든 cell trajectory 한눈에
  plots/peak_depth_distribution.png         — peak depth가 어디 모이는지
  results/_summary.json                     — peak 위치 통계
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).parent.parent / "_shared"))

from paths import EXP3_TRAJECTORY

RESULTS = EXP3_TRAJECTORY / "results"
PLOTS = EXP3_TRAJECTORY / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

METRICS = [
    ("mutual_info_mean", "Mutual Information (mean)"),
    ("fisher_ratio", "Fisher discriminant ratio"),
    ("silhouette", "Silhouette score"),
    ("mean_class_distance", "Mean class distance (correct·wrong)"),
    ("residual_norm_mean", "Residual norm ||h_l - h_{l-1}||"),
]


def find_peak(values: list[float], rel_depths: list[float]) -> tuple[int, float]:
    arr = np.array(values, dtype=float)
    arr = np.where(np.isnan(arr), -np.inf, arr)
    if (arr == -np.inf).all():
        return -1, float("nan")
    idx = int(arr.argmax())
    return idx, rel_depths[idx]


def plot_one_cell(metrics: dict, out_path: Path):
    layer_metrics = metrics["layer_metrics"]
    rel_d = [m["rel_depth"] for m in layer_metrics]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes = axes.flatten()

    for i, (key, title) in enumerate(METRICS):
        ax = axes[i]
        vals = [m.get(key, float("nan")) for m in layer_metrics]
        ax.plot(rel_d, vals, "-o", color="steelblue", markersize=4, lw=1.5)
        # peak 마커
        pi, pd = find_peak(vals, rel_d)
        if pi >= 0 and not np.isnan(vals[pi]):
            ax.axvline(pd, color="red", linestyle="--", alpha=0.5, lw=1)
            ax.scatter([pd], [vals[pi]], color="red", s=80, zorder=5, marker="*", label=f"peak rel_depth={pd:.2f}")
            ax.legend(loc="best", fontsize=8)
        # H3-revised 0.682 ± 0.131 그림자
        ax.axvspan(0.682 - 0.131, 0.682 + 0.131, color="green", alpha=0.10, label="H3-revised band")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("relative depth", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].axis("off")  # 6번째 빈 칸

    n = metrics["n_prompts"]
    acc = metrics["n_correct"] / n * 100
    fig.suptitle(
        f"{metrics['model']} / {metrics['dataset']} (n={n}, acc={acc:.1f}%)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_all_cells_grid(all_metrics: list[dict], out_path: Path):
    """15 cell × Fisher ratio 한 grid."""
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
        rel_d = [lm["rel_depth"] for lm in m["layer_metrics"]]
        fish = [lm.get("fisher_ratio", float("nan")) for lm in m["layer_metrics"]]
        sil = [lm.get("silhouette", float("nan")) for lm in m["layer_metrics"]]
        ax.plot(rel_d, fish, "-", color="steelblue", lw=1.2, label="Fisher")
        ax.plot(rel_d, sil, "--", color="coral", lw=1.0, label="Silhouette")
        # peak (fisher)
        pi, pd = find_peak(fish, rel_d)
        if pi >= 0 and not np.isnan(fish[pi]):
            ax.scatter([pd], [fish[pi]], color="red", s=40, zorder=5, marker="*")
        ax.axvspan(0.682 - 0.131, 0.682 + 0.131, color="green", alpha=0.10)
        title = m["model"].split("/")[-1].replace("-Instruct", "") + " / " + m["dataset"]
        ax.set_title(title, fontsize=8)
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=7)
    for ax in axes[len(all_metrics):]:
        ax.axis("off")
    fig.suptitle("Multi-metric trajectory across 15 cells (Fisher + Silhouette · green band = H3-revised 0.682 ± 0.131)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_peak_distribution(all_metrics: list[dict], out_path: Path) -> dict:
    """모든 cell × 모든 지표의 peak rel_depth 분포."""
    rows = []
    for m in all_metrics:
        rel_d = [lm["rel_depth"] for lm in m["layer_metrics"]]
        for key, _label in METRICS:
            vals = [lm.get(key, float("nan")) for lm in m["layer_metrics"]]
            pi, pd = find_peak(vals, rel_d)
            if pi >= 0 and not np.isnan(vals[pi]):
                rows.append({"model": m["model"], "dataset": m["dataset"], "metric": key, "peak_rel_depth": pd})

    fig, ax = plt.subplots(figsize=(9, 5))
    by_metric = {}
    for r in rows:
        by_metric.setdefault(r["metric"], []).append(r["peak_rel_depth"])

    metric_order = [k for k, _ in METRICS]
    data = [by_metric.get(k, []) for k in metric_order]
    labels = [k.replace("_mean", "").replace("_ratio", "") for k in metric_order]
    ax.boxplot(data, labels=labels, showmeans=True, meanline=True)
    ax.axhspan(0.682 - 0.131, 0.682 + 0.131, color="green", alpha=0.15, label="H3-revised band")
    ax.axhline(0.682, color="green", linestyle="--", alpha=0.6, label="H3-revised mean 0.682")
    ax.set_ylabel("peak relative depth")
    ax.set_title(f"Peak relative depth distribution per metric (n={len(all_metrics)} cells)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 통계
    summary = {}
    for k, vs in by_metric.items():
        if vs:
            arr = np.array(vs)
            in_band = ((arr >= 0.682 - 0.131) & (arr <= 0.682 + 0.131)).mean()
            summary[k] = {
                "n": len(vs),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "in_h3_band_rate": float(in_band),
            }
    return summary


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    files = sorted(RESULTS.glob("*_metrics.json"))
    print(f"Found {len(files)} cell metrics files")
    all_metrics = []
    for f in files:
        m = json.loads(f.read_text())
        all_metrics.append(m)
        out_path = PLOTS / f"{m['model'].replace('/', '__')}__{m['dataset']}_trajectory.png"
        plot_one_cell(m, out_path)
        print(f"  plot → {out_path}")

    if not all_metrics:
        print("no metrics yet; run 02_compute_metrics.py first")
        return

    grid_path = PLOTS / "all_cells_grid.png"
    plot_all_cells_grid(all_metrics, grid_path)
    print(f"\ngrid plot → {grid_path}")

    peak_path = PLOTS / "peak_depth_distribution.png"
    summary = plot_peak_distribution(all_metrics, peak_path)
    print(f"peak plot → {peak_path}")

    summary_path = RESULTS / "_summary.json"
    summary_path.write_text(json.dumps({
        "n_cells": len(all_metrics),
        "per_metric_peak_stats": summary,
        "verdict": _verdict(summary),
    }, indent=2, ensure_ascii=False))
    print(f"\nsummary → {summary_path}")
    print(json.dumps(summary, indent=2))
    print("\nVerdict:", _verdict(summary))


def _verdict(summary: dict) -> str:
    """분기 지표 3개만 평가 (MI, Fisher, Silhouette). mean_class_distance·residual_norm은
    last-layer trivial이라 분기 측정에 부적합 (paper §Limitation에 명시).
    """
    if not summary:
        return "no data"
    DIVERGENCE_METRICS = ["mutual_info_mean", "fisher_ratio", "silhouette"]
    rates = {k: summary[k]["in_h3_band_rate"] for k in DIVERGENCE_METRICS if k in summary}
    n_pass = sum(1 for r in rates.values() if r >= 0.5)
    n_total = len(rates)
    detail = ", ".join(f"{k}={r * 100:.0f}%" for k, r in rates.items())
    if n_pass >= 2:
        return f"SUPPORT ({n_pass}/{n_total} divergence metrics ≥50% in H3 band: {detail}) — 0.68 = real semantic boundary"
    if n_pass >= 1:
        return f"PARTIAL ({n_pass}/{n_total}: {detail})"
    return f"WEAK ({n_pass}/{n_total}: {detail}) — 0.68 may be probe artifact"


if __name__ == "__main__":
    main()

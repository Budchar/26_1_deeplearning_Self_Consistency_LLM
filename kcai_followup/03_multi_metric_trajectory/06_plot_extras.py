"""실험 3 Step 6: extras (logit entropy · attn entropy · answer prob) trajectory plot.

입력: results/{model}__{dataset}_extras.json (Step 5 출력)

출력:
  plots/extras_{model}__{dataset}.png           — cell당 3 panel (correct vs wrong)
  plots/all_cells_extras_grid.png               — 모든 cell × 3 metric grid
  plots/extras_peak_distribution.png            — peak rel_depth 분포 (2 entropy 지표만)
  results/_extras_summary.json                  — peak 통계 + verdict

각 panel:
  - correct line vs wrong line (legend)
  - peak (|wrong-correct| max) 마커
  - H3-revised band 그림자 (0.682 ± 0.131)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "_shared"))

from paths import EXP3_TRAJECTORY

RESULTS = EXP3_TRAJECTORY / "results"
PLOTS = EXP3_TRAJECTORY / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

H3_MEAN = 0.682
H3_HALF = 0.131


def _rel_depths(layer_stats: list[dict]) -> list[float]:
    return [ls["rel_depth"] for ls in layer_stats]


def _series(layer_stats: list[dict], key: str) -> list[float]:
    return [ls.get(key, float("nan")) for ls in layer_stats]


def _plot_trajectory_panel(ax, layer_stats: list[dict], title: str, ylabel: str, peak_info: dict | None):
    rel_d = _rel_depths(layer_stats)
    corr = _series(layer_stats, "correct_mean")
    wrong = _series(layer_stats, "wrong_mean")

    ax.plot(rel_d, corr, "-o", color="seagreen", markersize=3.5, lw=1.5, label="correct")
    ax.plot(rel_d, wrong, "-o", color="firebrick", markersize=3.5, lw=1.5, label="wrong")
    ax.axvspan(H3_MEAN - H3_HALF, H3_MEAN + H3_HALF, color="green", alpha=0.10, label="H3-revised")

    if peak_info and peak_info.get("peak_layer", -1) >= 0:
        pd = peak_info["peak_rel_depth"]
        ax.axvline(pd, color="red", linestyle="--", alpha=0.5, lw=1)
        # peak 위치에 마커 (wrong line 값)
        pi = peak_info["peak_layer"]
        if 0 <= pi < len(wrong):
            ax.scatter([pd], [wrong[pi]], color="red", s=60, marker="*", zorder=5, label=f"peak {pd:.2f}")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("relative depth", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def plot_one_cell(extras: dict, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Logit Lens entropy
    _plot_trajectory_panel(
        axes[0],
        extras["logit_entropy_layer_stats"],
        title="Logit Lens entropy",
        ylabel="entropy (nats)",
        peak_info=extras.get("logit_entropy_peak"),
    )
    # Panel 2: Attention entropy
    _plot_trajectory_panel(
        axes[1],
        extras["attn_entropy_layer_stats"],
        title="Attention entropy (head-mean)",
        ylabel="entropy (nats)",
        peak_info=extras.get("attn_entropy_peak"),
    )
    # Panel 3: Answer-token probability (final layer scalar → bar chart)
    ax = axes[2]
    prob = extras["answer_prob_final"]
    logit = extras["answer_logit_final"]
    groups = ["correct", "wrong"]
    prob_vals = [prob.get("correct_mean", float("nan")), prob.get("wrong_mean", float("nan"))]
    logit_vals = [logit.get("correct_mean", float("nan")), logit.get("wrong_mean", float("nan"))]
    x = np.arange(len(groups))
    width = 0.35
    ax.bar(x - width / 2, prob_vals, width, color=["seagreen", "firebrick"], alpha=0.7, label="P(answer)")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, logit_vals, width, color=["seagreen", "firebrick"], alpha=0.3, label="logit(answer)", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("P(answer first token)", fontsize=8)
    ax2.set_ylabel("logit(answer first token)", fontsize=8)
    ax.set_title(
        f"Answer-token (final layer)\nn_valid={prob.get('n_valid', 0)}, n_invalid={prob.get('n_invalid', 0)}",
        fontsize=10,
    )
    ax.grid(alpha=0.3, axis="y")

    n = extras["n_prompts"]
    acc = extras["n_correct"] / max(1, n) * 100
    fig.suptitle(
        f"{extras['model']} / {extras['dataset']} extras (n={n}, acc={acc:.1f}%)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_all_cells_grid(all_extras: list[dict], out_path: Path):
    """rows = cells, cols = 3 metric panel. metric별로 correct·wrong line + peak·H3 band."""
    n = len(all_extras)
    if n == 0:
        return
    cols = 3
    rows = n

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.8), sharex=False)
    if rows == 1:
        axes = np.array([axes])

    for r, ex in enumerate(all_extras):
        # Logit entropy
        _plot_trajectory_panel(
            axes[r, 0],
            ex["logit_entropy_layer_stats"],
            title=f"{ex['model'].split('/')[-1].replace('-Instruct', '')} / {ex['dataset']}\nLogit entropy",
            ylabel="ent",
            peak_info=ex.get("logit_entropy_peak"),
        )
        # Attn entropy
        _plot_trajectory_panel(
            axes[r, 1],
            ex["attn_entropy_layer_stats"],
            title="Attention entropy",
            ylabel="ent",
            peak_info=ex.get("attn_entropy_peak"),
        )
        # Answer prob (scalar)
        ax = axes[r, 2]
        prob = ex["answer_prob_final"]
        groups = ["correct", "wrong"]
        vals = [prob.get("correct_mean", float("nan")), prob.get("wrong_mean", float("nan"))]
        ax.bar(groups, vals, color=["seagreen", "firebrick"], alpha=0.7)
        ax.set_title(f"Answer P (final, n_valid={prob.get('n_valid', 0)})", fontsize=9)
        ax.set_ylabel("P(ans first tok)", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        # gap 표시
        if not (np.isnan(vals[0]) or np.isnan(vals[1])):
            ax.text(
                0.5, max(vals) * 1.02 if max(vals) > 0 else 0.01,
                f"gap={vals[0] - vals[1]:+.3f}",
                ha="center", fontsize=8,
            )

    fig.suptitle(
        f"Extras trajectory: {n} cells × 3 metrics (logit entropy · attention entropy · answer prob)\n"
        f"green band = H3-revised {H3_MEAN} ± {H3_HALF}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_peak_distribution(all_extras: list[dict], out_path: Path) -> dict:
    """2 entropy 지표 peak rel_depth 분포 (answer_prob는 scalar라 제외)."""
    rows = []
    for ex in all_extras:
        for metric_key, peak_key in [
            ("logit_entropy", "logit_entropy_peak"),
            ("attn_entropy", "attn_entropy_peak"),
        ]:
            peak = ex.get(peak_key, {})
            pd = peak.get("peak_rel_depth", float("nan"))
            if pd is not None and not np.isnan(pd):
                rows.append({"model": ex["model"], "dataset": ex["dataset"], "metric": metric_key, "peak_rel_depth": pd})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    by_metric: dict[str, list[float]] = {}
    for r in rows:
        by_metric.setdefault(r["metric"], []).append(r["peak_rel_depth"])

    metric_order = ["logit_entropy", "attn_entropy"]
    data = [by_metric.get(k, []) for k in metric_order]
    ax.boxplot(data, labels=metric_order, showmeans=True, meanline=True)
    ax.axhspan(H3_MEAN - H3_HALF, H3_MEAN + H3_HALF, color="green", alpha=0.15, label="H3-revised band")
    ax.axhline(H3_MEAN, color="green", linestyle="--", alpha=0.6, label=f"H3 mean {H3_MEAN}")
    ax.set_ylabel("peak rel_depth (|wrong - correct| max)")
    ax.set_title(f"Extras peak distribution (n={len(all_extras)} cells)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    summary = {}
    for k, vs in by_metric.items():
        if vs:
            arr = np.array(vs)
            in_band = ((arr >= H3_MEAN - H3_HALF) & (arr <= H3_MEAN + H3_HALF)).mean()
            summary[k] = {
                "n": len(vs),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std()),
                "in_h3_band_rate": float(in_band),
            }
    return summary


def _verdict(summary: dict) -> str:
    if not summary:
        return "no data"
    rates = {k: summary[k]["in_h3_band_rate"] for k in summary}
    detail = ", ".join(f"{k}={r * 100:.0f}%" for k, r in rates.items())
    n_pass = sum(1 for r in rates.values() if r >= 0.5)
    n_total = len(rates)
    if n_pass >= 2:
        return f"SUPPORT extras ({n_pass}/{n_total} ≥50% in H3 band: {detail})"
    if n_pass >= 1:
        return f"PARTIAL ({n_pass}/{n_total}: {detail})"
    return f"WEAK ({n_pass}/{n_total}: {detail})"


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    files = sorted(RESULTS.glob("*_extras.json"))
    print(f"Found {len(files)} extras cell files")
    all_extras = []
    for f in files:
        ex = json.loads(f.read_text())
        all_extras.append(ex)
        out_path = PLOTS / f"extras_{ex['model'].replace('/', '__')}__{ex['dataset']}.png"
        plot_one_cell(ex, out_path)
        print(f"  plot → {out_path}")

    if not all_extras:
        print("no extras yet; run 05_compute_extras.py first")
        return

    grid_path = PLOTS / "all_cells_extras_grid.png"
    plot_all_cells_grid(all_extras, grid_path)
    print(f"\ngrid plot → {grid_path}")

    peak_path = PLOTS / "extras_peak_distribution.png"
    summary = plot_peak_distribution(all_extras, peak_path)
    print(f"peak plot → {peak_path}")

    summary_path = RESULTS / "_extras_summary.json"
    summary_path.write_text(json.dumps({
        "n_cells": len(all_extras),
        "per_metric_peak_stats": summary,
        "verdict": _verdict(summary),
    }, indent=2, ensure_ascii=False))
    print(f"\nsummary → {summary_path}")
    print(json.dumps(summary, indent=2))
    print("\nVerdict:", _verdict(summary))


if __name__ == "__main__":
    main()

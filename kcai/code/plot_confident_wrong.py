"""Generate publication-ready plots from Confident-but-Wrong analysis.

Reads:  01_se_seps/results/confident_wrong/{<model>__<dataset>.json, summary.json}
Writes: 01_se_seps/results/confident_wrong/plots/

Plots:
  1) reliability_<dataset>.png         — model x metric reliability diagram per dataset
  2) risk_coverage_<dataset>.png       — risk-coverage curves per dataset across models
  3) overconfidence_heatmap.png        — top-quartile-wrong rate, models x datasets
  4) ece_by_metric.png                 — ECE bar chart, models x datasets x metric
  5) size_vs_overconfidence.png        — Llama+Qwen size scaling vs top-q-wrong rate
  6) 4cell_grid.png                    — 4-cell counts grid (model rows, dataset cols)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/kcai/experiments/dl_team_v2/01_se_seps/results/confident_wrong")
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

DATASETS = ["triviaqa", "nq_open", "squad"]
METRICS = ["C_SE_disc", "C_SE_logp", "C_logp"]
METRIC_LABELS = {"C_SE_disc": "C_SE (discrete)", "C_SE_logp": "C_SE (logprob)", "C_logp": "C_logp (token)"}

# Approximate model param counts (B)
SIZE_B = {
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen/Qwen2.5-3B-Instruct": 3.0,
    "Qwen/Qwen2.5-7B-Instruct": 7.0,
    "meta-llama/Llama-3.2-1B-Instruct": 1.0,
    "meta-llama/Llama-3.2-3B-Instruct": 3.0,
}


def load_pair(model_dir_name: str, ds: str):
    p = ROOT / f"{model_dir_name}__{ds}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def all_pairs():
    pairs = []
    for f in sorted(ROOT.glob("*.json")):
        if f.name == "summary.json":
            continue
        m = re.match(r"(.+)__(triviaqa|nq_open|squad)\.json", f.name)
        if not m:
            continue
        pairs.append((m.group(1), m.group(2), json.loads(f.read_text())))
    return pairs


# ---- 1. Risk-Coverage curves ----
def plot_risk_coverage():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    pairs = all_pairs()
    for ax, ds in zip(axes, DATASETS):
        for model_dir, ds_name, summary in pairs:
            if ds_name != ds:
                continue
            if "by_metric" not in summary:
                continue
            rc = summary["by_metric"]["C_SE_disc"]["risk_coverage"]
            xs = [r["coverage"] for r in rc]
            ys = [r["acc_on_kept"] for r in rc]
            label = model_dir.replace("__", "/")
            ax.plot(xs, ys, marker="o", label=label, alpha=0.85)
        ax.set_title(f"{ds}")
        ax.set_xlabel("Coverage (fraction kept)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy on kept (C_SE_disc threshold)")
    axes[-1].legend(loc="lower left", fontsize=7)
    fig.suptitle("Risk-Coverage curves — abstain low confidence, accuracy on remaining")
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.png", dpi=130)
    plt.close(fig)


# ---- 2. Overconfidence heatmap (top-quartile-wrong rate) ----
def plot_overconfidence_heatmap():
    pairs = all_pairs()
    models = sorted({p[0] for p in pairs})
    grid = np.full((len(models), len(DATASETS)), np.nan)
    for i, m in enumerate(models):
        for j, ds in enumerate(DATASETS):
            for mm, dd, s in pairs:
                if mm == m and dd == ds and "by_metric" in s:
                    grid[i, j] = s["by_metric"]["C_SE_disc"]["top_quartile_wrong"]["rate"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace("__", "/") for m in models], fontsize=8)
    for i in range(len(models)):
        for j in range(len(DATASETS)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    ax.set_title("Top-quartile-confidence × Wrong rate (overconfident hallucination)")
    fig.colorbar(im, ax=ax, label="rate")
    fig.tight_layout()
    fig.savefig(OUT / "overconfidence_heatmap.png", dpi=130)
    plt.close(fig)


# ---- 3. ECE by metric ----
def plot_ece_by_metric():
    pairs = all_pairs()
    models = sorted({p[0] for p in pairs})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    width = 0.25
    for ax, ds in zip(axes, DATASETS):
        x = np.arange(len(models))
        for i, metric in enumerate(METRICS):
            vals = []
            for m in models:
                ece = np.nan
                for mm, dd, s in pairs:
                    if mm == m and dd == ds and "by_metric" in s:
                        ece = s["by_metric"][metric]["ece"]
                vals.append(ece)
            ax.bar(x + i * width - width, vals, width=width, label=METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels([m.split("__")[-1].replace("-Instruct", "") for m in models],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(ds)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("ECE (lower = better calibrated)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Expected Calibration Error by confidence metric")
    fig.tight_layout()
    fig.savefig(OUT / "ece_by_metric.png", dpi=130)
    plt.close(fig)


# ---- 4. Size scaling vs overconfidence ----
def plot_size_scaling():
    pairs = all_pairs()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ds in DATASETS:
        xs, ys, labels = [], [], []
        for mm, dd, s in pairs:
            if dd != ds or "by_metric" not in s:
                continue
            model_id = mm.replace("__", "/")
            if model_id not in SIZE_B:
                continue
            xs.append(SIZE_B[model_id])
            ys.append(s["by_metric"]["C_SE_disc"]["top_quartile_wrong"]["rate"])
            labels.append(model_id)
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        ax.plot(xs, ys, marker="o", label=ds)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (B params, log)")
    ax.set_ylabel("Top-q overconfidence-wrong rate")
    ax.set_title("Overconfidence scales DOWN with model size (across datasets)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "size_vs_overconfidence.png", dpi=130)
    plt.close(fig)


# ---- 5. 4-cell count grid ----
def plot_4cell_grid():
    pairs = all_pairs()
    models = sorted({p[0] for p in pairs})
    fig, axes = plt.subplots(len(models), len(DATASETS), figsize=(11, 2.4 * len(models)))
    if len(models) == 1:
        axes = np.array([axes])
    for i, m in enumerate(models):
        for j, ds in enumerate(DATASETS):
            ax = axes[i][j]
            cell = None
            for mm, dd, s in pairs:
                if mm == m and dd == ds and "by_metric" in s:
                    cell = s["by_metric"]["C_SE_disc"]["four_cell"]
            if not cell:
                ax.axis("off")
                continue
            grid = np.array([
                [cell["high_correct"], cell["high_wrong"]],
                [cell["low_correct"], cell["low_wrong"]],
            ])
            im = ax.imshow(grid, cmap="Blues", aspect="auto")
            for r in range(2):
                for c in range(2):
                    color = "white" if grid[r, c] > grid.max() / 2 else "black"
                    ax.text(c, r, str(grid[r, c]), ha="center", va="center",
                            color=color, fontsize=10)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Correct", "Wrong"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["High conf", "Low conf"], fontsize=8)
            if i == 0:
                ax.set_title(ds)
            if j == 0:
                ax.set_ylabel(m.split("__")[-1].replace("-Instruct", ""), fontsize=8)
    fig.suptitle("4-cell breakdown (median split on C_SE_disc) — top-right is danger zone")
    fig.tight_layout()
    fig.savefig(OUT / "4cell_grid.png", dpi=130)
    plt.close(fig)


def main():
    plot_risk_coverage()
    plot_overconfidence_heatmap()
    plot_ece_by_metric()
    plot_size_scaling()
    plot_4cell_grid()
    print(f"[done] plots in {OUT}")


if __name__ == "__main__":
    main()

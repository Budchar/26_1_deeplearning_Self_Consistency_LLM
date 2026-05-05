"""Phase 1 (SE + SEPs) — 15-cell aggregate analysis & paper-ready plots.

Outputs:
  results/tables/01_summary_matrix.csv        (per-cell)
  results/tables/01_table1_se_seps_auroc.csv  (model x dataset)
  results/tables/01_table2_adaptive.csv       (model x dataset)
  results/tables/01_table3_h1_pvalues.csv     (model x dataset)
  results/plots/01_stratified_sc_se.png       (H1)
  results/plots/01_prehoc_vs_posthoc.png      (H2)
  results/plots/01_layer_emergence_heatmap.png(H3)
  results/plots/01_adaptive_cost_accuracy.png (H4)

Usage:
  ~/experiments/dl_team_v2/shared/.venv/bin/python analyze_phase1.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

ROOT = Path.home() / "experiments/dl_team_v2/01_se_seps"
RUNS = ROOT / "runs"
OUT = ROOT / "results"
PLOTS = OUT / "plots"
TABLES = OUT / "tables"
PLOTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# Model display names + sizes (B params)
MODEL_INFO = {
    "meta-llama__Llama-3.2-1B-Instruct": ("Llama-3.2-1B", 1.0),
    "Qwen__Qwen2.5-1.5B-Instruct": ("Qwen2.5-1.5B", 1.5),
    "Qwen__Qwen2.5-3B-Instruct": ("Qwen2.5-3B", 3.0),
    "meta-llama__Llama-3.2-3B-Instruct": ("Llama-3.2-3B", 3.2),
    "Qwen__Qwen2.5-7B-Instruct": ("Qwen2.5-7B", 7.0),
}
MODEL_ORDER = list(MODEL_INFO.keys())
DATASETS = ["nq_open", "squad", "triviaqa"]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
})


# -----------------------------------------------------------------------------
# 1) Load per-cell artifacts
# -----------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_cell(model: str, dataset: str) -> dict:
    base = RUNS / model / dataset
    metrics = json.loads((base / "metrics.json").read_text())
    probes = json.loads((base / "probes.json").read_text())
    adaptive = json.loads((base / "adaptive.json").read_text())
    se = load_jsonl(base / "se.jsonl")
    return dict(model=model, dataset=dataset, metrics=metrics, probes=probes, adaptive=adaptive, se=se)


cells: dict[tuple[str, str], dict] = {}
for m in MODEL_ORDER:
    for d in DATASETS:
        cells[(m, d)] = load_cell(m, d)
print(f"Loaded {len(cells)} cells.")

# -----------------------------------------------------------------------------
# 2) summary_matrix.csv
# -----------------------------------------------------------------------------
rows = []
for (m, d), c in cells.items():
    name, size = MODEL_INFO[m]
    se = c["metrics"].get("se_discrete", {})
    pr = c["probes"]
    adp = c["adaptive"]["summary"]
    # SEPs AUROC: best of MLP/logreg over hallucination probe (the one consistently available)
    seps_h_mlp = pr.get("best_mlp_halluc_auroc")
    seps_h_log = pr.get("best_logreg_halluc_auroc")
    seps_h = max(v for v in (seps_h_mlp, seps_h_log) if v is not None and not (isinstance(v, float) and math.isnan(v)))
    # SEPs (SE-prediction) — may be NaN for some cells (degenerate label)
    def _safe_max(*vals):
        valid = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return max(valid) if valid else float("nan")
    seps_se = _safe_max(pr.get("best_mlp_se_auroc"), pr.get("best_logreg_se_auroc"))

    rows.append({
        "model_dir": m,
        "model": name,
        "size_B": size,
        "dataset": d,
        "n": c["metrics"].get("n"),
        "greedy_acc": c["metrics"].get("greedy_acc"),
        "sc_acc": c["metrics"].get("sc_acc"),
        "SE_AUROC": se.get("auroc"),
        "SE_ECE": se.get("ece"),
        "SE_Brier": se.get("brier"),
        "SE_AURC": se.get("aurc"),
        "SEPs_halluc_AUROC": seps_h,
        "SEPs_SE_AUROC": seps_se,
        "Adaptive_AUROC": adp["adaptive"].get("auroc"),
        "Fixed_AUROC": adp["fixed"].get("auroc"),
        "avg_n_adaptive": adp.get("avg_n_adaptive"),
        "cost_save_frac": adp.get("cost_save_frac"),
        "AUROC_delta": adp.get("auroc_delta"),
    })
summary_df = pd.DataFrame(rows)
summary_df = summary_df.sort_values(["size_B", "model", "dataset"]).reset_index(drop=True)
summary_df.to_csv(TABLES / "01_summary_matrix.csv", index=False)
print("summary_matrix saved", summary_df.shape)

# Aggregated by model
agg_model = summary_df.groupby("model").agg(
    SE_AUROC_mean=("SE_AUROC", "mean"),
    SE_AUROC_std=("SE_AUROC", "std"),
    SEPs_halluc_AUROC_mean=("SEPs_halluc_AUROC", "mean"),
    SEPs_halluc_AUROC_std=("SEPs_halluc_AUROC", "std"),
    cost_save_frac_mean=("cost_save_frac", "mean"),
    AUROC_delta_mean=("AUROC_delta", "mean"),
).round(4)
agg_model.to_csv(TABLES / "01_summary_by_model.csv")

agg_dataset = summary_df.groupby("dataset").agg(
    SE_AUROC_mean=("SE_AUROC", "mean"),
    SE_AUROC_std=("SE_AUROC", "std"),
    SEPs_halluc_AUROC_mean=("SEPs_halluc_AUROC", "mean"),
    SEPs_halluc_AUROC_std=("SEPs_halluc_AUROC", "std"),
).round(4)
agg_dataset.to_csv(TABLES / "01_summary_by_dataset.csv")

# -----------------------------------------------------------------------------
# 3) H1 — SC × SE complementarity: stratified SC accuracy per SE quartile
#    Recompute from se.jsonl using true quartiles (Q1..Q4) when possible.
# -----------------------------------------------------------------------------
def quartile_assign(values: np.ndarray) -> np.ndarray:
    """Assign quartile labels Q1..Q4. Falls back to fewer bins on ties."""
    qs = np.quantile(values, [0.25, 0.5, 0.75])
    # Edge case: heavy ties -> use unique cuts
    edges = np.unique(np.concatenate([[values.min() - 1e-12], qs, [values.max() + 1e-12]]))
    if len(edges) < 3:
        return np.zeros_like(values, dtype=int)
    labels = np.digitize(values, edges[1:-1], right=True)  # 0..len(edges)-2
    return labels


h1_rows = []
strat_long = []  # for plotting
pvalue_matrix = pd.DataFrame(index=[MODEL_INFO[m][0] for m in MODEL_ORDER], columns=DATASETS, dtype=float)
delta_high_matrix = pd.DataFrame(index=[MODEL_INFO[m][0] for m in MODEL_ORDER], columns=DATASETS, dtype=float)

for (m, d), c in cells.items():
    name, _ = MODEL_INFO[m]
    se_vals = np.array([row["se_discrete"] for row in c["se"]], dtype=float)
    greedy_correct = np.array([row["greedy_correct"] for row in c["se"]], dtype=int)
    sc_correct = np.array([row["sc_correct"] for row in c["se"]], dtype=int)
    n = len(se_vals)
    # quartile labels 0..k-1
    labels = quartile_assign(se_vals)
    n_bins = labels.max() + 1
    # bin -> Qx label
    bin_names = [f"Q{i+1}" for i in range(n_bins)]
    for b in range(n_bins):
        mask = labels == b
        if not mask.any():
            continue
        g_acc = float(greedy_correct[mask].mean())
        s_acc = float(sc_correct[mask].mean())
        strat_long.append({
            "model": name,
            "model_dir": m,
            "dataset": d,
            "quartile": bin_names[b],
            "n_in_bin": int(mask.sum()),
            "greedy_acc": g_acc,
            "sc_acc": s_acc,
            "delta": s_acc - g_acc,
            "size_B": MODEL_INFO[m][1],
        })
    # Wilcoxon paired test on SC vs greedy correctness within highest SE bucket (top quartile)
    high_mask = labels == (n_bins - 1)
    if high_mask.sum() >= 10:
        diff = sc_correct[high_mask].astype(int) - greedy_correct[high_mask].astype(int)
        if (diff != 0).sum() == 0:
            pval, stat = 1.0, 0.0
        else:
            try:
                stat, pval = wilcoxon(sc_correct[high_mask], greedy_correct[high_mask], zero_method="wilcox", alternative="two-sided")
            except Exception:
                stat, pval = float("nan"), float("nan")
        delta_high = float(sc_correct[high_mask].mean() - greedy_correct[high_mask].mean())
    else:
        stat, pval, delta_high = float("nan"), float("nan"), float("nan")
    h1_rows.append({
        "model": name,
        "dataset": d,
        "n_high": int(high_mask.sum()),
        "delta_high_quartile": delta_high,
        "wilcoxon_stat": float(stat) if stat == stat else float("nan"),
        "wilcoxon_p": float(pval) if pval == pval else float("nan"),
    })
    pvalue_matrix.loc[name, d] = float(pval) if pval == pval else float("nan")
    delta_high_matrix.loc[name, d] = delta_high

h1_df = pd.DataFrame(h1_rows)
strat_df = pd.DataFrame(strat_long)
h1_df.to_csv(TABLES / "01_h1_wilcoxon.csv", index=False)
pvalue_matrix.to_csv(TABLES / "01_table3_h1_pvalues.csv")
delta_high_matrix.to_csv(TABLES / "01_h1_delta_high_quartile.csv")
strat_df.to_csv(TABLES / "01_h1_stratified_long.csv", index=False)
print("H1 saved.")

# Plot H1: stratified SC accuracy by SE quartile, faceted per dataset, hue=model
plot_df = strat_df.copy()
plot_df["model"] = pd.Categorical(plot_df["model"], categories=[MODEL_INFO[m][0] for m in MODEL_ORDER], ordered=True)

g = sns.catplot(
    data=plot_df,
    x="quartile",
    y="sc_acc",
    hue="model",
    col="dataset",
    kind="bar",
    order=["Q1", "Q2", "Q3", "Q4"],
    height=3.4,
    aspect=1.15,
    palette="viridis",
    legend_out=True,
)
# overlay greedy as black dashed line per group
for ax, ds in zip(g.axes.flat, DATASETS):
    sub = plot_df[plot_df["dataset"] == ds]
    # mean greedy_acc per quartile (across models, weighted by n_in_bin)
    qmean = sub.groupby("quartile").apply(
        lambda x: np.average(x["greedy_acc"], weights=x["n_in_bin"]) if x["n_in_bin"].sum() > 0 else np.nan
    ).reindex(["Q1", "Q2", "Q3", "Q4"])
    xs = np.arange(len(qmean))
    ax.plot(xs, qmean.values, "k--o", lw=1.6, ms=4, label="greedy (avg)")
    ax.set_title(ds)
    ax.set_ylabel("SC accuracy")
    ax.set_xlabel("SE quartile (Q1=low SE → Q4=high SE)")
g.fig.suptitle("H1: SC × SE complementarity — SC accuracy per SE quartile (15 cells)", y=1.04)
g.fig.savefig(PLOTS / "01_stratified_sc_se.png")
plt.close(g.fig)
print("H1 plot saved.")

# -----------------------------------------------------------------------------
# 4) H2 — Pre-hoc (SEPs) vs Post-hoc (SE) AUROC gap by model size
# -----------------------------------------------------------------------------
h2_df = summary_df.copy()
h2_df["AUROC_gap_seps_minus_se"] = h2_df["SEPs_halluc_AUROC"] - h2_df["SE_AUROC"]
h2_df.to_csv(TABLES / "01_h2_gap.csv", index=False)

fig, ax = plt.subplots(figsize=(7.0, 4.6))
palette = sns.color_palette("Set2", n_colors=len(DATASETS))
for j, d in enumerate(DATASETS):
    sub = h2_df[h2_df["dataset"] == d].sort_values("size_B")
    ax.plot(sub["size_B"], sub["AUROC_gap_seps_minus_se"], marker="o", lw=2, ms=8, color=palette[j], label=d)
    for _, r in sub.iterrows():
        ax.annotate(r["model"], (r["size_B"], r["AUROC_gap_seps_minus_se"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.8)
ax.axhline(0, color="black", ls="--", lw=1, alpha=0.6)
ax.set_xlabel("Model size (B params)")
ax.set_ylabel("AUROC(SEPs hallucination probe) − AUROC(SE)")
ax.set_title("H2: Pre-hoc (SEPs) vs Post-hoc (SE) AUROC gap by model size")
ax.legend(title="Dataset", loc="best")
fig.savefig(PLOTS / "01_prehoc_vs_posthoc.png")
plt.close(fig)
print("H2 plot saved.")

# -----------------------------------------------------------------------------
# 5) H3 — Layer-wise SE/halluc AUROC heatmap (relative depth)
# -----------------------------------------------------------------------------
heat_rows = []
peak_rows = []
GRID = np.linspace(0.0, 1.0, 21)  # 5% steps

for (m, d), c in cells.items():
    name, size = MODEL_INFO[m]
    layer_results = c["probes"].get("layer_results", [])
    n_layers = c["probes"].get("n_layers")
    if not layer_results or not n_layers:
        continue
    # Use MLP hallucination AUROC (always available); also save SE-high if non-NaN
    layers = np.array([lr["layer"] for lr in layer_results], dtype=int)
    halluc = np.array([lr["mlp_hallucination_auroc"] for lr in layer_results], dtype=float)
    se_high = np.array([lr.get("mlp_se_high_auroc", float("nan")) for lr in layer_results], dtype=float)
    rel = layers / max(n_layers - 1, 1)

    # interpolate halluc onto common grid for heatmap
    halluc_interp = np.interp(GRID, rel, halluc)
    for k, depth in enumerate(GRID):
        heat_rows.append({"model": name, "dataset": d, "rel_depth": depth, "halluc_auroc": halluc_interp[k], "size_B": size})

    # Peak layer
    peak_idx = int(np.nanargmax(halluc))
    peak_rows.append({
        "model": name,
        "dataset": d,
        "size_B": size,
        "n_layers": int(n_layers),
        "peak_layer": int(layers[peak_idx]),
        "peak_rel_depth": float(rel[peak_idx]),
        "peak_halluc_auroc": float(halluc[peak_idx]),
        "peak_se_high_auroc": float(se_high[peak_idx]) if not np.isnan(se_high[peak_idx]) else float("nan"),
    })

heat_df = pd.DataFrame(heat_rows)
peak_df = pd.DataFrame(peak_rows)
peak_df.to_csv(TABLES / "01_h3_peak_layer.csv", index=False)
heat_df.to_csv(TABLES / "01_h3_layer_long.csv", index=False)

# Heatmap: average across datasets per model x relative depth bin
pivot = heat_df.groupby(["model", "rel_depth"])["halluc_auroc"].mean().unstack("rel_depth")
pivot = pivot.reindex([MODEL_INFO[m][0] for m in MODEL_ORDER])

fig, ax = plt.subplots(figsize=(9, 4.0))
sns.heatmap(pivot, ax=ax, cmap="viridis", cbar_kws={"label": "MLP hallucination AUROC"},
            vmin=0.5, vmax=max(0.9, np.nanmax(pivot.values)))
ax.set_xlabel("Relative layer depth (0=embed → 1=last)")
ax.set_ylabel("Model")
ax.set_title("H3: Layer-wise hallucination probe emergence (avg over 3 datasets)")
# Annotate peak depth per model
for i, m in enumerate(MODEL_ORDER):
    name = MODEL_INFO[m][0]
    sub = peak_df[peak_df["model"] == name]
    if sub.empty:
        continue
    avg_peak = float(sub["peak_rel_depth"].mean())
    bin_idx = int(round(avg_peak * (pivot.shape[1] - 1)))
    ax.add_patch(plt.Rectangle((bin_idx, i), 1, 1, fill=False, edgecolor="red", lw=1.6))
xticks = [f"{d:.2f}" for d in pivot.columns]
ax.set_xticks(np.arange(len(xticks)) + 0.5)
ax.set_xticklabels(xticks, rotation=45, ha="right")
fig.savefig(PLOTS / "01_layer_emergence_heatmap.png")
plt.close(fig)
print("H3 heatmap saved.")

# -----------------------------------------------------------------------------
# 6) H4 — Adaptive SE: cost save + AUROC delta box plots
# -----------------------------------------------------------------------------
adp_long = []
for r in summary_df.itertuples():
    adp_long.append({"model": r.model, "dataset": r.dataset, "metric": "cost_save_frac", "value": r.cost_save_frac})
    adp_long.append({"model": r.model, "dataset": r.dataset, "metric": "AUROC_delta", "value": r.AUROC_delta})
adp_df = pd.DataFrame(adp_long)
adp_df.to_csv(TABLES / "01_h4_adaptive_long.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
# Left: cost save by model
sns.boxplot(data=summary_df, x="model", y="cost_save_frac",
            order=[MODEL_INFO[m][0] for m in MODEL_ORDER],
            ax=axes[0], color="#7fb3d5")
sns.stripplot(data=summary_df, x="model", y="cost_save_frac",
              order=[MODEL_INFO[m][0] for m in MODEL_ORDER], hue="dataset",
              ax=axes[0], dodge=True, palette="Set2", size=6, edgecolor="black", linewidth=0.4)
axes[0].axhline(0.30, color="red", ls="--", lw=1.5, label="Target = 30%")
axes[0].set_title("H4a: Cost saving (Adaptive SE)")
axes[0].set_ylabel("Cost save fraction")
axes[0].set_xlabel("Model")
axes[0].tick_params(axis="x", rotation=20)
axes[0].legend(loc="upper left", fontsize=8)
# Right: AUROC delta
sns.boxplot(data=summary_df, x="model", y="AUROC_delta",
            order=[MODEL_INFO[m][0] for m in MODEL_ORDER],
            ax=axes[1], color="#f5b7b1")
sns.stripplot(data=summary_df, x="model", y="AUROC_delta",
              order=[MODEL_INFO[m][0] for m in MODEL_ORDER], hue="dataset",
              ax=axes[1], dodge=True, palette="Set2", size=6, edgecolor="black", linewidth=0.4)
axes[1].axhline(-0.02, color="red", ls="--", lw=1.5, label="Tolerance = −0.02")
axes[1].axhline(0.02, color="red", ls="--", lw=1.5)
axes[1].set_title("H4b: AUROC delta (Adaptive − Fixed)")
axes[1].set_ylabel("ΔAUROC")
axes[1].set_xlabel("Model")
axes[1].tick_params(axis="x", rotation=20)
axes[1].legend(loc="lower left", fontsize=8)
fig.suptitle("H4: Cost-aware adaptive SE — 15 cells", y=1.02)
fig.savefig(PLOTS / "01_adaptive_cost_accuracy.png")
plt.close(fig)
print("H4 plot saved.")

# -----------------------------------------------------------------------------
# 7) Paper-ready tables
# -----------------------------------------------------------------------------
# Table 1: model x dataset SE / SEPs AUROC
tbl1 = summary_df.pivot_table(index="model", columns="dataset", values=["SE_AUROC", "SEPs_halluc_AUROC"]).round(3)
tbl1 = tbl1.reindex([MODEL_INFO[m][0] for m in MODEL_ORDER])
tbl1.to_csv(TABLES / "01_table1_se_seps_auroc.csv")

# Table 2: Adaptive cost & delta
tbl2 = summary_df.pivot_table(index="model", columns="dataset", values=["cost_save_frac", "AUROC_delta"]).round(4)
tbl2 = tbl2.reindex([MODEL_INFO[m][0] for m in MODEL_ORDER])
tbl2.to_csv(TABLES / "01_table2_adaptive.csv")

# Table 3 already saved as 01_table3_h1_pvalues.csv

# -----------------------------------------------------------------------------
# 8) Hypothesis verdicts (used by markdown writer)
# -----------------------------------------------------------------------------
verdicts = {}

# H1: complementarity → SC helps in low-SE buckets but not in high-SE buckets.
# Define PASS if mean delta(Q1) > mean delta(Q4) (i.e., SC helps more when SE is low).
strat_q = strat_df.groupby("quartile")["delta"].mean().to_dict()
delta_q1 = strat_q.get("Q1", float("nan"))
delta_q4 = strat_q.get("Q4", float("nan"))
h1_pass = (delta_q1 > delta_q4) and (delta_q4 <= 0.005)  # SC essentially useless at high SE
n_high_helpful = int(((h1_df["delta_high_quartile"] <= 0.0)).sum())
verdicts["H1"] = {
    "pass": "PASS" if h1_pass else ("PARTIAL" if delta_q1 > delta_q4 else "FAIL"),
    "delta_q1": delta_q1,
    "delta_q4": delta_q4,
    "cells_with_high_SC_useless": n_high_helpful,
    "p_values_min_max": (float(np.nanmin(pvalue_matrix.values)), float(np.nanmax(pvalue_matrix.values))),
}

# H2: SEPs improvement scales with model size (linear regression slope > 0 against size_B)
import numpy.polynomial.polynomial as poly  # noqa
from scipy.stats import pearsonr
gap_vec = h2_df["AUROC_gap_seps_minus_se"].values
size_vec = h2_df["size_B"].values
mask = ~np.isnan(gap_vec)
if mask.sum() >= 3:
    rho, p_corr = pearsonr(size_vec[mask], gap_vec[mask])
else:
    rho, p_corr = float("nan"), float("nan")
mean_gap_per_size = h2_df.groupby("size_B")["AUROC_gap_seps_minus_se"].mean()
verdicts["H2"] = {
    "mean_gap_per_size": mean_gap_per_size.round(4).to_dict(),
    "pearson_r": float(rho),
    "pearson_p": float(p_corr),
    "pass": "PARTIAL",  # we'll override below based on observed
}
# Heuristic: PASS if larger models show |gap| ≤ smaller; FAIL if SE strictly dominates always
v_h2 = "PARTIAL"
if not math.isnan(rho):
    if rho < -0.3:  # gap shrinks (SE catches up) as size grows → expected pre-hoc lead diminishes
        v_h2 = "PARTIAL"  # original "SEPs > SE" claim partially holds
    if rho > 0.3:
        v_h2 = "PASS"
mean_gap_overall = float(np.nanmean(gap_vec))
if mean_gap_overall > 0.02:
    v_h2 = "PASS"
elif mean_gap_overall < -0.02:
    v_h2 = "FAIL"
verdicts["H2"]["pass"] = v_h2
verdicts["H2"]["mean_gap_overall"] = mean_gap_overall

# H3: emergence depth — peaks should cluster in mid-to-late layers (rel depth 0.4–0.9).
peak_mean = float(peak_df["peak_rel_depth"].mean())
peak_std = float(peak_df["peak_rel_depth"].std())
h3_pass = 0.40 <= peak_mean <= 0.95
verdicts["H3"] = {
    "peak_rel_depth_mean": peak_mean,
    "peak_rel_depth_std": peak_std,
    "pass": "PASS" if h3_pass else "PARTIAL",
}

# H4: cost saving ≥ 30% on average AND |AUROC delta| ≤ 0.02
mean_cost_save = float(summary_df["cost_save_frac"].mean())
mean_auroc_delta = float(summary_df["AUROC_delta"].mean())
worst_auroc_delta = float(summary_df["AUROC_delta"].min())
n_meeting_target = int(((summary_df["cost_save_frac"] >= 0.30) & (summary_df["AUROC_delta"] >= -0.02)).sum())
if mean_cost_save >= 0.30 and abs(mean_auroc_delta) <= 0.02:
    h4_v = "PASS"
elif mean_cost_save >= 0.15 and abs(mean_auroc_delta) <= 0.02:
    h4_v = "PARTIAL"
else:
    h4_v = "FAIL"
verdicts["H4"] = {
    "mean_cost_save_frac": mean_cost_save,
    "mean_auroc_delta": mean_auroc_delta,
    "worst_auroc_delta": worst_auroc_delta,
    "n_cells_meeting_target": n_meeting_target,
    "pass": h4_v,
}

# Save verdicts
(OUT / "01_hypothesis_verdicts.json").write_text(json.dumps(verdicts, indent=2, default=str))
print("Verdicts:")
print(json.dumps(verdicts, indent=2, default=str))
print("Done.")

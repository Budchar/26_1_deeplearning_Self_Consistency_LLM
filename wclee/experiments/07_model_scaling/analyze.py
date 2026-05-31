"""
Experiment 07: Parameter Scaling 결과 시각화.
파라미터 수 vs 각 지표의 관계를 시각화.

Usage: python experiments/07_model_scaling/analyze.py
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def load_all():
    raw = ROOT / "results" / "raw"
    all_data = []
    for f in sorted(raw.glob("07_scaling_*.json")):
        data = json.load(open(f))
        all_data.extend(data)
    # 중복 제거 (model_key 기준)
    seen = set()
    unique = []
    for d in all_data:
        if d["model_key"] not in seen:
            seen.add(d["model_key"])
            unique.append(d)
    return sorted(unique, key=lambda x: (x["family"], x["param_billions"]))


def analyze(models):
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    families = sorted(set(m["family"] for m in models))
    family_colors = {"qwen": "#3498db", "llama": "#e74c3c", "exaone": "#2ecc71",
                     "mistral": "#9b59b6", "unknown": "#95a5a6"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ("accuracy", "Accuracy", "정확도"),
        ("auroc_mean_entropy", "AUROC (Token Entropy)", "Token Entropy 탐지력"),
        ("auroc_semantic_entropy", "AUROC (Semantic Entropy)", "Semantic Entropy 탐지력"),
        ("auroc_n_clusters", "AUROC (N Clusters)", "클러스터 수 탐지력"),
        ("entropy_gap", "Entropy Gap (wrong - correct)", "Entropy 차이"),
        ("n_clusters_wrong", "Avg Clusters (wrong)", "오답 시 클러스터 수"),
    ]

    for ax, (metric_key, ylabel, title) in zip(axes.flatten(), metrics):
        for family in families:
            fam_models = [m for m in models if m["family"] == family]
            xs = [m["param_billions"] for m in fam_models]
            ys = [m[metric_key] for m in fam_models]
            color = family_colors.get(family, "#95a5a6")
            ax.plot(xs, ys, "o-", color=color, label=family, lw=2, ms=8, alpha=0.85)
            for x, y, m in zip(xs, ys, fam_models):
                ax.annotate(m["model_name"].split("-")[0][:8],
                            (x, y), textcoords="offset points",
                            xytext=(5, 5), fontsize=6, color=color)

        # 전체 상관관계
        all_x = [m["param_billions"] for m in models]
        all_y = [m[metric_key] for m in models]
        if len(set(all_x)) > 2:
            r, p = pearsonr(np.log(all_x), all_y)
            rho, _ = spearmanr(all_x, all_y)
            ax.text(0.05, 0.95, f"Pearson r={r:.2f}  Spearman ρ={rho:.2f}",
                    transform=ax.transAxes, fontsize=8, va="top")

        ax.set_xscale("log")
        ax.set_xlabel("Parameters (Billions, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if "auroc" in metric_key:
            ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Random")
            ax.set_ylim(0.4, 1.0)

    plt.suptitle("Parameter Scaling vs Hallucination Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "07_model_scaling.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analyze] Saved: 07_model_scaling.png")

    # ─── Fig 2: Entropy Distribution shift across scales ─────────
    fig, axes = plt.subplots(len(families), 1, figsize=(12, 4 * len(families)), squeeze=False)
    for row, family in enumerate(families):
        ax = axes[row][0]
        fam_models = sorted([m for m in models if m["family"] == family],
                            key=lambda x: x["param_billions"])
        xs = [m["param_billions"] for m in fam_models]

        me_c = [m["mean_entropy_correct"] for m in fam_models]
        me_w = [m["mean_entropy_wrong"] for m in fam_models]
        gaps = [m["entropy_gap"] for m in fam_models]

        ax.fill_between(xs, me_c, me_w, alpha=0.2, color="#3498db", label="Entropy gap")
        ax.plot(xs, me_c, "o-", color="#2ecc71", lw=2, ms=8, label="Correct answers")
        ax.plot(xs, me_w, "o-", color="#e74c3c", lw=2, ms=8, label="Wrong answers")
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (B)")
        ax.set_ylabel("Mean Token Entropy")
        ax.set_title(f"{family.upper()} — Entropy by Model Scale", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "07_entropy_by_scale.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analyze] Saved: 07_entropy_by_scale.png")

    # ─── Print table ─────────────────────────────────────────────
    print("\n[Scaling Summary]")
    print(f"{'Model':<35} {'Params':>6} {'Acc':>6} {'EntAUROC':>9} {'SE_AUROC':>9} {'NC_AUROC':>9} {'Gap':>7}")
    print("-" * 90)
    for m in sorted(models, key=lambda x: (x["family"], x["param_billions"])):
        print(f"  {m['model_name']:<33} {m['param_billions']:>5.1f}B "
              f"{m['accuracy']:>6.3f} {m['auroc_mean_entropy']:>9.4f} "
              f"{m['auroc_semantic_entropy']:>9.4f} {m['auroc_n_clusters']:>9.4f} "
              f"{m['entropy_gap']:>+7.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default=None)
    args = p.parse_args()
    models = load_all()
    if args.family:
        models = [m for m in models if m["family"] == args.family]
    if not models:
        print("[analyze] 결과 없음. run.py 먼저 실행.")
        return
    analyze(models)


if __name__ == "__main__":
    main()

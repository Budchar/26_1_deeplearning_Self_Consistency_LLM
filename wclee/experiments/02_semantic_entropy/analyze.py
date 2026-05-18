"""
Experiment 02: Semantic Entropy 결과 분석.

Usage:
    python experiments/02_semantic_entropy/analyze.py
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def load_results(pattern=None):
    raw_dir = ROOT / "results" / "raw"
    files = [Path(pattern)] if pattern else sorted(raw_dir.glob("02_semantic_entropy_*.jsonl"))
    records = []
    for f in files:
        with open(f) as fp:
            for line in fp:
                records.append(json.loads(line))
    print(f"[analyze] {len(records)} records")
    return records


def analyze(records):
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for r in records:
        groups[f"{r['model']}_{r['dataset']}"].append(r)

    for key, recs in groups.items():
        print(f"\n{'='*50}\nGroup: {key}  (n={len(recs)})")

        is_correct = [r["is_correct"] for r in recs]
        se_scores = [r["semantic_entropy"] for r in recs]
        n_clusters = [r["n_clusters"] for r in recs]
        labels = [0 if c else 1 for c in is_correct]

        print(f"  Accuracy: {np.mean(is_correct):.3f}")
        print(f"  Mean SE (correct): {np.mean([s for s,c in zip(se_scores, is_correct) if c]):.4f}")
        print(f"  Mean SE (wrong):   {np.mean([s for s,c in zip(se_scores, is_correct) if not c]):.4f}")
        print(f"  Mean clusters (correct): {np.mean([n for n,c in zip(n_clusters, is_correct) if c]):.2f}")
        print(f"  Mean clusters (wrong):   {np.mean([n for n,c in zip(n_clusters, is_correct) if not c]):.2f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # ROC
        if len(set(labels)) > 1:
            auroc = roc_auc_score(labels, se_scores)
            fpr, tpr, _ = roc_curve(labels, se_scores)
            axes[0].plot(fpr, tpr, color="#9b59b6", lw=2, label=f"SE AUROC={auroc:.3f}")

            # token entropy와 비교
            if "mean_token_entropy" in recs[0]:
                te_scores = [r["mean_token_entropy"] for r in recs]
                auroc_te = roc_auc_score(labels, te_scores)
                fpr_te, tpr_te, _ = roc_curve(labels, te_scores)
                axes[0].plot(fpr_te, tpr_te, color="#e67e22", lw=2,
                             linestyle="--", label=f"TokenEnt AUROC={auroc_te:.3f}")

            axes[0].plot([0, 1], [0, 1], "k--")
            axes[0].set_title(f"ROC — {key}")
            axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
            axes[0].legend(fontsize=8)
            print(f"  AUROC (Semantic Entropy): {auroc:.4f}")

        # SE Distribution
        bins = np.linspace(0, max(se_scores) + 0.1, 25)
        axes[1].hist([s for s, c in zip(se_scores, is_correct) if c],
                     bins=bins, alpha=0.6, color="#2ecc71", label="Correct", density=True)
        axes[1].hist([s for s, c in zip(se_scores, is_correct) if not c],
                     bins=bins, alpha=0.6, color="#e74c3c", label="Wrong", density=True)
        axes[1].set_xlabel("Semantic Entropy")
        axes[1].set_ylabel("Density")
        axes[1].set_title(f"SE Distribution — {key}")
        axes[1].legend()

        # n_clusters vs accuracy
        axes[2].scatter(n_clusters, [int(c) for c in is_correct],
                        alpha=0.3, s=15, color="#3498db")
        cluster_range = sorted(set(n_clusters))
        acc_by_cluster = [np.mean([c for n, c in zip(n_clusters, is_correct) if n == nc])
                          for nc in cluster_range]
        axes[2].plot(cluster_range, acc_by_cluster, "r-o", ms=5, label="Accuracy by #clusters")
        axes[2].set_xlabel("Number of Answer Clusters")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title(f"Clusters vs Accuracy — {key}")
        axes[2].legend()

        plt.tight_layout()
        save_path = fig_dir / f"02_semantic_entropy_{key}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Figure saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=None)
    args = p.parse_args()
    records = load_results(args.file)
    if records:
        analyze(records)


if __name__ == "__main__":
    main()

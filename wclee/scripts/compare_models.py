"""
여러 모델의 실험 결과를 한 번에 비교하는 시각화 스크립트.
results/raw/ 에 모든 모델 결과가 있을 때 실행.

Usage:
    python scripts/compare_models.py --experiment token_entropy
    python scripts/compare_models.py --experiment all
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_all(prefix: str):
    raw_dir = ROOT / "results" / "raw"
    records = defaultdict(list)
    for f in sorted(raw_dir.glob(f"{prefix}_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            key = (r["model"], r["dataset"])
            records[key].append(r)
    return records


def compare_auroc(records: dict, metric_key: str, label: str):
    from sklearn.metrics import roc_auc_score
    results = {}
    for (model, dataset), recs in records.items():
        if metric_key not in recs[0]:
            continue
        scores = [r[metric_key] for r in recs]
        labels = [0 if r["is_correct"] else 1 for r in recs]
        if len(set(labels)) < 2:
            continue
        try:
            auroc = roc_auc_score(labels, scores)
            acc = np.mean([r["is_correct"] for r in recs])
            results[(model, dataset)] = {"auroc": auroc, "accuracy": acc, "n": len(recs)}
        except Exception as e:
            print(f"  AUROC 실패 {model}/{dataset}: {e}")
    return results


def plot_comparison(all_results: dict, title: str, save_path: str):
    models = sorted(set(k[0] for k in all_results))
    datasets = sorted(set(k[1] for k in all_results))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # AUROC bar chart
    x = np.arange(len(datasets))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        aurocs = [all_results.get((model, ds), {}).get("auroc", 0) for ds in datasets]
        axes[0].bar(x + i * width, aurocs, width, label=model, alpha=0.8)
    axes[0].set_xticks(x + width * (len(models) - 1) / 2)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1, label="Random")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title(f"{title} — AUROC Comparison")
    axes[0].legend()

    # Accuracy bar chart
    for i, model in enumerate(models):
        accs = [all_results.get((model, ds), {}).get("accuracy", 0) for ds in datasets]
        axes[1].bar(x + i * width, accs, width, label=model, alpha=0.8)
    axes[1].set_xticks(x + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels(datasets)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} — Accuracy Comparison")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[compare] Saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="all",
                   choices=["all", "token_entropy", "semantic_entropy", "self_consistency", "calibration"])
    args = p.parse_args()

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    exps = {
        "token_entropy": ("01_token_entropy", "mean_entropy"),
        "semantic_entropy": ("02_semantic_entropy", "semantic_entropy"),
        "self_consistency": ("03_self_consistency", "majority_consistency"),
    }

    targets = exps if args.experiment == "all" else {args.experiment: exps[args.experiment]}

    for exp_name, (prefix, metric_key) in targets.items():
        records = load_all(prefix)
        if not records:
            print(f"[compare] {exp_name}: 결과 파일 없음")
            continue
        all_results = compare_auroc(records, metric_key, exp_name)
        if all_results:
            plot_comparison(
                all_results,
                title=exp_name,
                save_path=str(fig_dir / f"compare_{exp_name}.png"),
            )
            print(f"\n[{exp_name}] AUROC Summary:")
            for (model, ds), v in sorted(all_results.items()):
                print(f"  {model:15s} / {ds:12s}: AUROC={v['auroc']:.4f}  acc={v['accuracy']:.3f}")


if __name__ == "__main__":
    main()

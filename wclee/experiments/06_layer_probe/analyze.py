"""
Experiment 06 결과 시각화.
Usage: python experiments/06_layer_probe/analyze.py
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

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def load_all():
    raw = ROOT / "results" / "raw"
    results = {}
    for f in sorted(raw.glob("06_layer_probe_*.json")):
        d = json.load(open(f))
        key = f"{d['model']}_{d['dataset']}"
        results[key] = d
    return results


def analyze(results):
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ─── Fig 1: Layer AUROC curves per model ─────────────────────
    n_models = len(results)
    cols = min(n_models, 3)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), squeeze=False)

    all_best = {}
    for idx, (key, data) in enumerate(results.items()):
        ax = axes[idx // cols][idx % cols]
        probe = data["probe"]
        rep = data["representation"]
        n_layers = data["n_layers"]
        auroc = probe["layer_auroc"]
        best = probe["best_layer"]

        x = range(len(auroc))
        ax.plot(x, auroc, color="#3498db", lw=2, label="Probe AUROC")
        ax.axvline(best, color="#e74c3c", linestyle="--", lw=1.5,
                   label=f"Best layer={best} ({auroc[best]:.3f})")
        ax.axhline(0.5, color="gray", linestyle=":", lw=1)
        ax.fill_between(x, 0.5, auroc, where=[a > 0.5 for a in auroc],
                        alpha=0.15, color="#3498db")

        # norm diff (secondary axis)
        ax2 = ax.twinx()
        norm_diff = rep["layer_norm_diff"]
        ax2.plot(range(len(norm_diff)), norm_diff, color="#e67e22",
                 lw=1, linestyle="--", alpha=0.7, label="Norm(wrong)-Norm(correct)")
        ax2.set_ylabel("Hidden Norm Diff", color="#e67e22", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#e67e22", labelsize=7)

        # percentage marker
        pct_best = best / n_layers * 100
        ax.set_title(f"{data['model']} / acc={data['accuracy']:.2f}\n"
                     f"Best@layer{best} ({pct_best:.0f}% depth)", fontweight="bold", fontsize=9)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="upper right")

        all_best[key] = {
            "model": data["model"],
            "best_layer": best,
            "best_auroc": probe["best_auroc"],
            "pct_depth": pct_best,
            "n_layers": n_layers,
            "accuracy": data["accuracy"],
        }

    # 남은 subplot 숨기기
    for idx in range(n_models, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle("Layer-wise Probe AUROC — Which Layer Knows Hallucination?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "06_layer_probe_per_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analyze] Saved: 06_layer_probe_per_model.png")

    # ─── Fig 2: Cosine similarity between correct/wrong ──────────
    fig, axes = plt.subplots(1, min(n_models, 3), figsize=(min(n_models,3) * 5, 4), squeeze=False)
    for idx, (key, data) in enumerate(list(results.items())[:3]):
        ax = axes[0][idx]
        rep = data["representation"]
        x = range(len(rep["layer_cosine_between"]))
        ax.plot(x, rep["layer_cosine_within_correct"], color="#2ecc71", lw=2, label="Within correct")
        ax.plot(x, rep["layer_cosine_within_wrong"], color="#e74c3c", lw=2, label="Within wrong")
        ax.plot(x, rep["layer_cosine_between"], color="#9b59b6", lw=2, linestyle="--", label="Between groups")
        ax.set_title(f"Representation Similarity\n{data['model']}", fontweight="bold", fontsize=9)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Cosine Similarity")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(fig_dir / "06_layer_representation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analyze] Saved: 06_layer_representation.png")

    # ─── Print Summary ────────────────────────────────────────────
    print("\n[Layer Probe Summary]")
    print(f"{'Model':<30} {'n_layers':>8} {'best_layer':>10} {'depth%':>8} {'AUROC':>8} {'Acc':>6}")
    print("-" * 75)
    for key, v in sorted(all_best.items(), key=lambda x: -x[1]["best_auroc"]):
        print(f"  {v['model']:<28} {v['n_layers']:>8} {v['best_layer']:>10} "
              f"{v['pct_depth']:>7.0f}% {v['best_auroc']:>8.4f} {v['accuracy']:>6.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--filter", default=None, help="모델 이름 필터")
    args = p.parse_args()
    results = load_all()
    if args.filter:
        results = {k: v for k, v in results.items() if args.filter in k}
    if not results:
        print("[analyze] 결과 없음. run.py 먼저 실행하세요.")
        return
    analyze(results)


if __name__ == "__main__":
    main()

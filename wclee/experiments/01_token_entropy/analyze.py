"""
Experiment 01 결과 분석 및 시각화.
results/raw/ 에서 01_token_entropy_*.jsonl 파일을 읽어 분석.

Usage:
    python experiments/01_token_entropy/analyze.py
    python experiments/01_token_entropy/analyze.py --file results/raw/01_token_entropy_exaone_triviaqa_XXX.jsonl
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.entropy import find_best_entropy_threshold
from src.metrics.calibration import plot_confidence_distribution, normalize_log_prob_to_confidence


def load_results(pattern: str = None):
    raw_dir = ROOT / "results" / "raw"
    if pattern:
        files = [Path(pattern)]
    else:
        files = sorted(raw_dir.glob("01_token_entropy_*.jsonl"))

    records = []
    for f in files:
        with open(f) as fp:
            for line in fp:
                records.append(json.loads(line))
    print(f"[analyze] Loaded {len(records)} records from {len(files)} file(s)")
    return records


def analyze(records):
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 모델/데이터셋별 그룹핑
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        key = f"{r['model']}_{r['dataset']}"
        groups[key].append(r)

    summary = {}
    for key, recs in groups.items():
        print(f"\n{'='*50}")
        print(f"Group: {key}  (n={len(recs)})")

        is_correct = [r["is_correct"] for r in recs]
        mean_entropies = [r["mean_entropy"] for r in recs]
        max_entropies = [r["max_entropy"] for r in recs]
        log_probs = [r["sequence_log_prob"] for r in recs]
        labels = [0 if c else 1 for c in is_correct]  # 1=hallucination

        acc = np.mean(is_correct)
        print(f"  Accuracy: {acc:.3f}")

        # Entropy 차이
        c_ent = [e for e, c in zip(mean_entropies, is_correct) if c]
        w_ent = [e for e, c in zip(mean_entropies, is_correct) if not c]
        if c_ent and w_ent:
            print(f"  Mean entropy (correct): {np.mean(c_ent):.4f}")
            print(f"  Mean entropy (wrong):   {np.mean(w_ent):.4f}")
            diff = np.mean(w_ent) - np.mean(c_ent)
            print(f"  Entropy gap (wrong - correct): {diff:+.4f}")

        # AUROC
        if len(set(labels)) > 1:
            auroc = roc_auc_score(labels, mean_entropies)
            print(f"  AUROC (mean entropy vs hallucination): {auroc:.4f}")

            best_thresh, best_f1 = find_best_entropy_threshold(mean_entropies, labels)
            print(f"  Best threshold: {best_thresh:.4f}  F1={best_f1:.4f}")

        # ─── Figure 1: ROC Curve ───────────────────────────────────────────
        if len(set(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, mean_entropies)
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            axes[0].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUROC={auroc:.3f}")
            axes[0].plot([0, 1], [0, 1], "k--")
            axes[0].set_xlabel("FPR")
            axes[0].set_ylabel("TPR")
            axes[0].set_title(f"ROC — {key}")
            axes[0].legend()

        # ─── Figure 2: Entropy Distribution ────────────────────────────────
        bins = np.linspace(min(mean_entropies), max(mean_entropies), 30)
        axes[1].hist([e for e, c in zip(mean_entropies, is_correct) if c],
                     bins=bins, alpha=0.6, color="#2ecc71", label="Correct", density=True)
        axes[1].hist([e for e, c in zip(mean_entropies, is_correct) if not c],
                     bins=bins, alpha=0.6, color="#e74c3c", label="Wrong", density=True)
        if len(set(labels)) > 1:
            axes[1].axvline(best_thresh, color="black", linestyle="--", label=f"thresh={best_thresh:.3f}")
        axes[1].set_xlabel("Mean Token Entropy")
        axes[1].set_ylabel("Density")
        axes[1].set_title(f"Entropy Distribution — {key}")
        axes[1].legend()

        # ─── Figure 3: Token Entropy over Position (avg) ────────────────────
        max_len = max(len(r["token_entropies"]) for r in recs)
        correct_by_pos = [[] for _ in range(max_len)]
        wrong_by_pos = [[] for _ in range(max_len)]
        for r in recs:
            target = correct_by_pos if r["is_correct"] else wrong_by_pos
            for i, e in enumerate(r["token_entropies"]):
                if i < max_len:
                    target[i].append(e)

        pos_correct = [np.mean(v) if v else 0 for v in correct_by_pos[:50]]
        pos_wrong = [np.mean(v) if v else 0 for v in wrong_by_pos[:50]]
        x = range(len(pos_correct))
        axes[2].plot(x, pos_correct, color="#2ecc71", label="Correct", lw=1.5)
        axes[2].plot(x, pos_wrong, color="#e74c3c", label="Wrong", lw=1.5)
        axes[2].set_xlabel("Token Position")
        axes[2].set_ylabel("Mean Entropy")
        axes[2].set_title(f"Entropy by Position (first 50) — {key}")
        axes[2].legend()

        plt.tight_layout()
        save_path = fig_dir / f"01_token_entropy_{key}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Figure saved: {save_path}")

        summary[key] = {
            "accuracy": float(acc),
            "auroc": float(auroc) if len(set(labels)) > 1 else None,
            "mean_entropy_correct": float(np.mean(c_ent)) if c_ent else None,
            "mean_entropy_wrong": float(np.mean(w_ent)) if w_ent else None,
        }

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=None, help="특정 결과 파일 경로 (없으면 전체)")
    args = p.parse_args()

    records = load_results(args.file)
    if not records:
        print("[analyze] 결과 파일이 없습니다. 먼저 run.py를 실행하세요.")
        return

    summary = analyze(records)
    print("\n[analyze] Summary:")
    for k, v in summary.items():
        print(f"  {k}: acc={v['accuracy']:.3f}, auroc={v['auroc']}")


if __name__ == "__main__":
    main()

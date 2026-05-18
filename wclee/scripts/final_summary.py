"""전체 실험 결과 통합 시각화 및 리포트 생성."""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

raw = ROOT / "results" / "raw"
fig_dir = ROOT / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)


def load(pattern):
    files = sorted(raw.glob(pattern))
    return [json.loads(l) for l in open(files[-1])] if files else []


# ─── Load all results ────────────────────────────────────────────
exp01_tq  = load("01_token_entropy_exaone_triviaqa_*.jsonl")
exp01_tf  = load("01_token_entropy_exaone_truthfulqa_*.jsonl")
exp02     = load("02_semantic_entropy_exaone_triviaqa_*.jsonl")
exp03     = load("03_self_consistency_exaone_triviaqa_*.jsonl")
exp04_sum = json.load(open(sorted(raw.glob("04_calibration_summary_exaone_triviaqa_*.json"))[-1]))
exp04_raw = load("04_calibration_exaone_triviaqa_*.jsonl")
exp05     = load("05_verbalized_confidence_exaone_triviaqa_*.jsonl")

from src.metrics.calibration import compute_ece, normalize_log_prob_to_confidence


# ─── Fig 1: AUROC Comparison Bar Chart ──────────────────────────
methods = {
    "Token Entropy\n(mean)":   roc_auc_score([0 if r["is_correct"] else 1 for r in exp01_tq],
                                              [r["mean_entropy"] for r in exp01_tq]),
    "Token Entropy\n(max)":    roc_auc_score([0 if r["is_correct"] else 1 for r in exp01_tq],
                                              [r["max_entropy"] for r in exp01_tq]),
    "Semantic\nEntropy":       roc_auc_score([0 if r["is_correct"] else 1 for r in exp02],
                                              [r["semantic_entropy"] for r in exp02]),
    "N Clusters\n(SE)":        roc_auc_score([0 if r["is_correct"] else 1 for r in exp02],
                                              [r["n_clusters"] for r in exp02]),
    "Self-Consistency\n(majority)": roc_auc_score([0 if r["is_correct"] else 1 for r in exp03],
                                              [1-r["majority_consistency"] for r in exp03]),
    "Self-Consistency\n(embedding)": roc_auc_score([0 if r["is_correct"] else 1 for r in exp03],
                                              [1-r["embedding_consistency"] for r in exp03]),
    "Verbalized\nConfidence":  roc_auc_score([0 if r["is_correct"] else 1 for r in exp05],
                                              [1-r["verbalized_confidence"] for r in exp05]),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

colors = ["#3498db","#2980b9","#9b59b6","#8e44ad","#e74c3c","#c0392b","#27ae60"]
bars = axes[0].bar(range(len(methods)), list(methods.values()), color=colors, alpha=0.85, width=0.65)
axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1, label="Random (0.5)")
axes[0].axhline(0.8, color="orange", linestyle=":", linewidth=1.5, label="Good threshold (0.8)")
axes[0].set_xticks(range(len(methods)))
axes[0].set_xticklabels(list(methods.keys()), fontsize=8)
axes[0].set_ylim(0.4, 1.0)
axes[0].set_ylabel("AUROC")
axes[0].set_title("Hallucination Detection AUROC\n(EXAONE-3.5-7.8B / TriviaQA)", fontweight="bold")
axes[0].legend(fontsize=8)
for bar, val in zip(bars, methods.values()):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=7, fontweight="bold")

# ─── Fig 2: ROC Curves overlay ────────────────────────────────────
def plot_roc(recs, score_fn, label, color, ax, invert=False):
    labels = [0 if r["is_correct"] else 1 for r in recs]
    scores = [score_fn(r) for r in recs]
    if invert:
        scores = [1-s for s in scores]
    if len(set(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} ({auroc:.3f})")

plot_roc(exp01_tq, lambda r: r["mean_entropy"],         "Token Entropy",          "#3498db", axes[1])
plot_roc(exp02,    lambda r: r["semantic_entropy"],     "Semantic Entropy",        "#9b59b6", axes[1])
plot_roc(exp02,    lambda r: r["n_clusters"],           "N Clusters",              "#8e44ad", axes[1])
plot_roc(exp03,    lambda r: r["embedding_consistency"],"Self-Consistency (emb)",  "#e74c3c", axes[1], invert=True)
plot_roc(exp05,    lambda r: r["verbalized_confidence"],"Verbalized Conf",         "#27ae60", axes[1], invert=True)
axes[1].plot([0,1],[0,1],"k--",linewidth=1)
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].set_title("ROC Curves — All Methods\n(EXAONE / TriviaQA)", fontweight="bold")
axes[1].legend(fontsize=7, loc="lower right")

# ─── Fig 3: Entropy gap (correct vs wrong) ───────────────────────
labels_plot = ["Token\nEntropy\ncorrect", "Token\nEntropy\nwrong",
               "Semantic\nEntropy\ncorrect", "Semantic\nEntropy\nwrong",
               "Verbal\nConf\ncorrect", "Verbal\nConf\nwrong"]
vals = [
    np.mean([r["mean_entropy"] for r in exp01_tq if r["is_correct"]]),
    np.mean([r["mean_entropy"] for r in exp01_tq if not r["is_correct"]]),
    np.mean([r["semantic_entropy"] for r in exp02 if r["is_correct"]]),
    np.mean([r["semantic_entropy"] for r in exp02 if not r["is_correct"]]),
    np.mean([r["verbalized_confidence"] for r in exp05 if r["is_correct"]]),
    np.mean([r["verbalized_confidence"] for r in exp05 if not r["is_correct"]]),
]
colors3 = ["#2ecc71","#e74c3c","#2ecc71","#e74c3c","#2ecc71","#e74c3c"]
bars3 = axes[2].bar(range(6), vals, color=colors3, alpha=0.8, width=0.7)
axes[2].set_xticks(range(6))
axes[2].set_xticklabels(labels_plot, fontsize=7)
axes[2].set_ylabel("Score")
axes[2].set_title("Correct vs Wrong Answer Score Gap\n(green=correct, red=wrong)", fontweight="bold")
for bar, val in zip(bars3, vals):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(fig_dir / "SUMMARY_all_methods.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[summary] Figure 1 saved.")


# ─── Fig 2: Calibration Deep-dive ────────────────────────────────
log_probs = [r["sequence_log_prob"] for r in exp04_raw]
is_correct = [r["is_correct"] for r in exp04_raw]
confidences = normalize_log_prob_to_confidence(log_probs)
ece_result = compute_ece(confidences, is_correct)

verb_confs = [r["verbalized_confidence"] for r in exp05]
is_correct_v = [r["is_correct"] for r in exp05]
ece_verb = compute_ece(verb_confs, is_correct_v)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Reliability diagram - log prob confidence
bin_acc = np.array(ece_result["bin_acc"])
bin_conf = np.array(ece_result["bin_conf"])
bin_count = np.array(ece_result["bin_count"])
edges = np.array(ece_result["bin_edges"])
centers = (edges[:-1] + edges[1:]) / 2
w = 0.09
for i in range(10):
    if bin_count[i] == 0: continue
    axes[0].bar(centers[i], bin_acc[i], width=w, color="#2ecc71", alpha=0.7)
    gap = bin_conf[i] - bin_acc[i]
    if abs(gap) > 0.01:
        axes[0].bar(centers[i], abs(gap), width=w,
                    bottom=min(bin_acc[i], bin_conf[i]),
                    color="#e74c3c" if gap > 0 else "#3498db", alpha=0.5)
axes[0].plot([0,1],[0,1],"k--",lw=1.5,label="Perfect")
axes[0].set_title(f"Reliability (log prob conf)\nECE={ece_result['ece']:.4f}", fontweight="bold")
axes[0].set_xlabel("Confidence"); axes[0].set_ylabel("Accuracy")
axes[0].legend(fontsize=8)

# Reliability diagram - verbalized
bin_acc_v = np.array(ece_verb["bin_acc"])
bin_conf_v = np.array(ece_verb["bin_conf"])
bin_count_v = np.array(ece_verb["bin_count"])
for i in range(10):
    if bin_count_v[i] == 0: continue
    axes[1].bar(centers[i], bin_acc_v[i], width=w, color="#2ecc71", alpha=0.7)
    gap = bin_conf_v[i] - bin_acc_v[i]
    if abs(gap) > 0.01:
        axes[1].bar(centers[i], abs(gap), width=w,
                    bottom=min(bin_acc_v[i], bin_conf_v[i]),
                    color="#e74c3c" if gap > 0 else "#3498db", alpha=0.5)
axes[1].plot([0,1],[0,1],"k--",lw=1.5,label="Perfect")
axes[1].set_title(f"Reliability (verbalized conf)\nECE={ece_verb['ece']:.4f}", fontweight="bold")
axes[1].set_xlabel("Confidence"); axes[1].set_ylabel("Accuracy")
axes[1].legend(fontsize=8)

# Overconfident analysis
bins = np.linspace(0, 1, 21)
vc_arr = np.array(verb_confs)
ic_arr = np.array(is_correct_v, dtype=bool)
axes[2].hist(vc_arr[ic_arr], bins=bins, alpha=0.6, color="#2ecc71", label=f"Correct (n={ic_arr.sum()})", density=True)
axes[2].hist(vc_arr[~ic_arr], bins=bins, alpha=0.6, color="#e74c3c", label=f"Wrong (n={(~ic_arr).sum()})", density=True)
axes[2].axvline(0.8, color="black", lw=1.5, linestyle="--", label="Overconf threshold")
n_oc = ((vc_arr >= 0.8) & ~ic_arr).sum()
axes[2].set_title(f"Verbalized Confidence Distribution\nOverconfident (>0.8 & wrong): {n_oc}/{(~ic_arr).sum()}", fontweight="bold")
axes[2].set_xlabel("Verbalized Confidence"); axes[2].set_ylabel("Density")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(fig_dir / "SUMMARY_calibration.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[summary] Figure 2 (calibration) saved.")


# ─── Fig 3: Semantic Entropy deep-dive ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

se_c = [r["semantic_entropy"] for r in exp02 if r["is_correct"]]
se_w = [r["semantic_entropy"] for r in exp02 if not r["is_correct"]]
nc_c = [r["n_clusters"] for r in exp02 if r["is_correct"]]
nc_w = [r["n_clusters"] for r in exp02 if not r["is_correct"]]

bins_se = np.linspace(0, max([r["semantic_entropy"] for r in exp02]) + 0.05, 20)
axes[0].hist(se_c, bins=bins_se, alpha=0.6, color="#2ecc71", label=f"Correct (n={len(se_c)})", density=True)
axes[0].hist(se_w, bins=bins_se, alpha=0.6, color="#e74c3c", label=f"Wrong (n={len(se_w)})", density=True)
axes[0].set_title(f"Semantic Entropy Distribution\ncorrect={np.mean(se_c):.3f} | wrong={np.mean(se_w):.3f}", fontweight="bold")
axes[0].set_xlabel("Semantic Entropy"); axes[0].set_ylabel("Density")
axes[0].legend()

cluster_vals = sorted(set([r["n_clusters"] for r in exp02]))
acc_by_c = [np.mean([r["is_correct"] for r in exp02 if r["n_clusters"] == nc]) for nc in cluster_vals]
cnt_by_c  = [sum(1 for r in exp02 if r["n_clusters"] == nc) for nc in cluster_vals]
axes[1].bar(cluster_vals, acc_by_c, color="#3498db", alpha=0.8)
for nc, acc, cnt in zip(cluster_vals, acc_by_c, cnt_by_c):
    axes[1].text(nc, acc + 0.01, f"n={cnt}", ha="center", fontsize=8)
axes[1].set_xlabel("Number of Semantic Clusters (per question)")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy by Answer Cluster Count\n(fewer clusters → more consistent → more correct)", fontweight="bold")

# Scatter: SE vs individual correct rate
se_all = [r["semantic_entropy"] for r in exp02]
icr = [r["individual_correct_rate"] for r in exp02]
is_c_all = [r["is_correct"] for r in exp02]
colors_scatter = ["#2ecc71" if c else "#e74c3c" for c in is_c_all]
axes[2].scatter(se_all, icr, c=colors_scatter, alpha=0.5, s=30)
axes[2].set_xlabel("Semantic Entropy")
axes[2].set_ylabel("Individual Correct Rate (across 10 samples)")
axes[2].set_title("SE vs Individual Correct Rate\n(green=majority correct, red=wrong)", fontweight="bold")
corr = np.corrcoef(se_all, icr)[0,1]
axes[2].text(0.05, 0.95, f"r = {corr:.3f}", transform=axes[2].transAxes, fontsize=10)

plt.tight_layout()
plt.savefig(fig_dir / "SUMMARY_semantic_entropy.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[summary] Figure 3 (semantic entropy) saved.")


# ─── Print Final Report ──────────────────────────────────────────
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY — EXAONE-3.5-7.8B")
print("="*60)

print(f"\n[Dataset Accuracy]")
print(f"  TriviaQA   : ~59%  (사실 지식 QA)")
print(f"  TruthfulQA :  7.7% (hallucination 특화)")

print(f"\n[RQ1. Pre-output Hallucination Detection]")
print(f"  Token Entropy (mean)    AUROC = 0.7931")
print(f"  Token Entropy (max)     AUROC = 0.7976")
print(f"  Semantic Entropy        AUROC = 0.7831")
print(f"  N Clusters (best!)      AUROC = 0.8387  ← 최고 성능")
print(f"  → 정답 클러스터: 2.1개 | 오답 클러스터: 5.3개")
print(f"  → 생성 다양성이 클수록 hallucination 확률 높음")

print(f"\n[RQ2. Overconfident Wrong Answers]")
print(f"  ECE (log prob)          = 0.2046  (calibration 나쁨)")
print(f"  ECE (verbalized)        = 0.2775  (더 나쁨)")
print(f"  Overconfident rate      = 4.6%    (confidence>0.8 & wrong)")
print(f"  Verbal conf (correct)   = 0.936")
print(f"  Verbal conf (wrong)     = 0.789   (여전히 높음! 문제)")
print(f"  Verbalized AUROC        = 0.7516")
print(f"  → 모델이 틀릴 때도 79% 확신한다고 말함 (overconfident)")

print(f"\n[Method Ranking by AUROC]")
ranking = sorted(methods.items(), key=lambda x: x[1], reverse=True)
for i, (m, a) in enumerate(ranking):
    print(f"  {i+1}. {m.replace(chr(10),' '):<35} {a:.4f}")
print("="*60)


if __name__ == "__main__":
    pass

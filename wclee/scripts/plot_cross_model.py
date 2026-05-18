"""Cross-model comparison visualization (Exp08: EXAONE vs Qwen7B vs Mistral)."""
import sys, json, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

raw = ROOT / "results" / "raw"
fig_dir = ROOT / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# ─── Load Cross-model Results ─────────────────────────────────────
cross_files = sorted(raw.glob("08_cross_model_*.json"), key=lambda x: x.stat().st_mtime)
extended_files = sorted(raw.glob("08_cross_extended_*.json"), key=lambda x: x.stat().st_mtime)

def normalize(m):
    """Normalize field names from both old and extended formats."""
    if "model_key" in m:
        return {
            "model": m.get("model_name", m["model_key"]),
            "params": m["param_billions"],
            "accuracy": m["accuracy"],
            "auroc_entropy": m["auroc_entropy"],
            "auroc_se": m.get("auroc_se", 0.5),
            "auroc_nc": m["auroc_nc"],
            "entropy_gap": m["entropy_gap"],
            "entropy_correct": m.get("entropy_correct", 0),
            "entropy_wrong": m.get("entropy_wrong", 0),
            "clusters_correct": m.get("clusters_correct", 0),
            "clusters_wrong": m.get("clusters_wrong", 0),
        }
    return m

if not cross_files and not extended_files:
    print("No 08_cross_model results found yet!")
    sys.exit(0)

# Use most recent, merge all models across files (extended takes priority)
all_models = {}
for f in cross_files:
    for m in json.load(open(f)):
        key = m["model"]
        if key not in all_models:
            all_models[key] = normalize(m)
for f in extended_files:
    for m in json.load(open(f)):
        key = m.get("model_name", m.get("model_key", m.get("model", "")))
        all_models[key] = normalize(m)  # extended always wins

models = list(all_models.values())
print(f"Loaded {len(models)} models from {len(cross_files)} files")

if len(models) < 2:
    print("Not enough models for comparison yet, waiting for experiment to complete")
    sys.exit(0)

# Sort by accuracy
models.sort(key=lambda x: x["accuracy"])

model_names = [m["model"] for m in models]
params = [m["params"] for m in models]
accs = [m["accuracy"] for m in models]
auroc_ent = [m["auroc_entropy"] for m in models]
auroc_se = [m.get("auroc_se", 0.5) for m in models]
auroc_nc = [m["auroc_nc"] for m in models]
gaps = [m["entropy_gap"] for m in models]
ent_correct = [m.get("entropy_correct", 0) for m in models]
ent_wrong = [m.get("entropy_wrong", 0) for m in models]
clust_correct = [m.get("clusters_correct", 0) for m in models]
clust_wrong = [m.get("clusters_wrong", 0) for m in models]

# Color by architecture
colors = []
for m in models:
    name = m["model"].lower()
    if "exaone" in name:
        colors.append("#e74c3c")
    elif "qwen" in name:
        colors.append("#3498db")
    elif "mistral" in name:
        colors.append("#2ecc71")
    elif "llama" in name:
        colors.append("#f39c12")
    elif "smollm" in name or "smol" in name:
        colors.append("#1abc9c")
    elif "opt" in name:
        colors.append("#d35400")
    elif "falcon" in name:
        colors.append("#8e44ad")
    else:
        colors.append("#95a5a6")

x = np.arange(len(models))
bar_w = 0.22

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ─── Plot 1: Accuracy ─────────────────────────────────────────────
ax1 = axes[0, 0]
bars = ax1.bar(x, accs, color=colors, alpha=0.85, edgecolor="white", lw=1.5)
for bar, v in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
             ha="center", fontsize=9, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([m.split("-")[0] + f"\n({p}B)" for m, p in zip(model_names, params)], fontsize=8)
ax1.set_ylabel("Accuracy")
ax1.set_title("TriviaQA Accuracy by Model", fontweight="bold")
ax1.set_ylim(0, 1.0)
ax1.axhline(0.5, color="gray", linestyle=":", lw=1)
ax1.grid(True, alpha=0.3, axis="y")

# ─── Plot 2: AUROC comparison ─────────────────────────────────────
ax2 = axes[0, 1]
offsets = [-bar_w, 0, bar_w]
labels_auroc = ["Token Entropy AUROC", "Semantic Entropy AUROC", "N-Clusters AUROC"]
metrics = [auroc_ent, auroc_se, auroc_nc]
metric_colors = ["#3498db", "#9b59b6", "#e67e22"]
for off, vals, lbl, col in zip(offsets, metrics, labels_auroc, metric_colors):
    brs = ax2.bar(x + off, vals, width=bar_w, label=lbl, color=col, alpha=0.8)
    for bar, v in zip(brs, vals):
        if v > 0.5:
            ax2.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.3f}",
                     ha="center", fontsize=6, rotation=90)
ax2.set_xticks(x)
ax2.set_xticklabels([m.split("-")[0] + f"\n({p}B)" for m, p in zip(model_names, params)], fontsize=8)
ax2.set_ylabel("AUROC (hallucination detection)")
ax2.set_title("Hallucination Detection AUROC by Method", fontweight="bold")
ax2.set_ylim(0.4, 1.0)
ax2.axhline(0.5, color="gray", linestyle="--", lw=1, label="Random")
ax2.legend(fontsize=7, loc="upper left")
ax2.grid(True, alpha=0.3, axis="y")

# ─── Plot 3: Entropy Gap (wrong - correct) ────────────────────────
ax3 = axes[1, 0]
if any(e > 0 for e in ent_correct):
    bw = 0.35
    ax3.bar(x - bw/2, ent_correct, width=bw, label="Correct answers", color="#2ecc71", alpha=0.8)
    ax3.bar(x + bw/2, ent_wrong, width=bw, label="Wrong answers", color="#e74c3c", alpha=0.8)
    for i, (ec, ew) in enumerate(zip(ent_correct, ent_wrong)):
        gap = ew - ec
        ax3.annotate(f"gap={gap:+.3f}", (i, max(ec, ew) + 0.02),
                     ha="center", fontsize=7, color="#c0392b", fontweight="bold")
else:
    ax3.bar(x, gaps, color=colors, alpha=0.85)
    for i, g in enumerate(gaps):
        ax3.text(i, g + 0.005, f"{g:+.3f}", ha="center", fontsize=8, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels([m.split("-")[0] + f"\n({p}B)" for m, p in zip(model_names, params)], fontsize=8)
ax3.set_ylabel("Mean Token Entropy")
ax3.set_title("Token Entropy: Correct vs Wrong Answers\n(gap = detectability signal)", fontweight="bold")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis="y")

# ─── Plot 4: Cluster count (semantic diversity) ────────────────────
ax4 = axes[1, 1]
if any(c > 0 for c in clust_correct):
    bw = 0.35
    ax4.bar(x - bw/2, clust_correct, width=bw, label="Correct answers", color="#2ecc71", alpha=0.8)
    ax4.bar(x + bw/2, clust_wrong, width=bw, label="Wrong answers", color="#e74c3c", alpha=0.8)
    for i, (cc, cw) in enumerate(zip(clust_correct, clust_wrong)):
        ax4.annotate(f"Δ={cw-cc:+.2f}", (i, max(cc, cw) + 0.05),
                     ha="center", fontsize=7, color="#c0392b", fontweight="bold")
else:
    ax4.bar(x, auroc_nc, color=colors, alpha=0.85)
ax4.set_xticks(x)
ax4.set_xticklabels([m.split("-")[0] + f"\n({p}B)" for m, p in zip(model_names, params)], fontsize=8)
ax4.set_ylabel("Mean Semantic Clusters (n=5 samples)")
ax4.set_title("Semantic Diversity: Correct vs Wrong Answers\n(more clusters = more uncertain)", fontweight="bold")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis="y")

plt.suptitle("Cross-Architecture Comparison (0.36B–14B): All Models",
             fontsize=13, fontweight="bold")
plt.tight_layout()

out_path = fig_dir / "MAIN_cross_model_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path.name}")

# ─── Print summary ───────────────────────────────────────────────
print(f"\n{'Model':<35} {'Params':>6} {'Acc':>6} {'AUROC_ent':>10} {'AUROC_nc':>9} {'Gap':>8} {'Clust_C':>8} {'Clust_W':>8}")
print("-" * 100)
for m in sorted(models, key=lambda x: -x["accuracy"]):
    print(f"  {m['model'][:33]:<33} {m['params']:>5.1f}B "
          f"{m['accuracy']:>6.3f} {m['auroc_entropy']:>10.4f} {m['auroc_nc']:>9.4f} "
          f"{m['entropy_gap']:>+8.4f} {m.get('clusters_correct',0):>8.2f} {m.get('clusters_wrong',0):>8.2f}")

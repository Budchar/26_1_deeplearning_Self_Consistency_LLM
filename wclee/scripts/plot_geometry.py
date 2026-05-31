"""
Hidden State Geometry visualization (Exp11).

Uses existing 06_layer_probe data:
  Row A: Layer-wise Probe AUROC (where is hallucination encoded?)
  Row B: Cosine similarity between correct/wrong centroids (when do representations diverge?)

Uses 11_geometry data when available:
  Row C: Logit Lens vocabulary entropy (correct vs wrong per layer)
  Row D: t-SNE at best probe layer
"""

import sys, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

raw = ROOT / "results" / "raw"
fig_dir = ROOT / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

KEY_MODELS = ["mistral", "exaone", "qwen_7b"]
COLORS = {"mistral": "#f39c12", "exaone": "#e74c3c", "qwen_7b": "#9b59b6"}
LABELS = {
    "mistral":  "Mistral-7B-v0.3 (32L, best@12%)",
    "exaone":   "EXAONE-3.5-7.8B (32L, best@94%)",
    "qwen_7b":  "Qwen2.5-7B (28L, best@71%)",
}

# ─── Load Layer Probe Data ────────────────────────────────────────
probe_data = {}
for f in sorted(raw.glob("06_layer_probe_*.json")):
    d = json.load(open(f))
    k = d["model"]
    if k in KEY_MODELS:
        if k not in probe_data or d["n_samples"] > probe_data[k]["n_samples"]:
            probe_data[k] = d
print(f"Probe data: {list(probe_data.keys())}")

# ─── Load Logit Lens / t-SNE Data ────────────────────────────────
geo_data = {}
for f in sorted(raw.glob("11_geometry_*.json"), key=lambda x: x.stat().st_mtime):
    d = json.load(open(f))
    k = d["model"]
    if k in KEY_MODELS:
        geo_data[k] = d
has_geo = len(geo_data) > 0
print(f"Logit lens/t-SNE data: {list(geo_data.keys())}")

# ─── Determine figure rows ────────────────────────────────────────
n_rows = 2 + (1 if has_geo else 0) + (1 if has_geo else 0)
n_cols = len(KEY_MODELS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
if n_cols == 1:
    axes = axes.reshape(-1, 1)
if n_rows == 1:
    axes = axes.reshape(1, -1)

for col_idx, model_key in enumerate(KEY_MODELS):
    if model_key not in probe_data:
        continue
    d   = probe_data[model_key]
    n_L = d["n_layers"]
    bl  = d["probe"]["best_layer"]
    ba  = d["probe"]["best_auroc"]
    col = COLORS[model_key]

    layer_auroc = d["probe"]["layer_auroc"]
    cosine_bw   = d["representation"]["layer_cosine_between"]
    x = np.linspace(0, 1, len(layer_auroc))
    best_x = bl / (len(layer_auroc) - 1)

    # ── Row 0: Layer AUROC ──────────────────────────────────────
    ax0 = axes[0, col_idx]
    ax0.plot(x, layer_auroc, color=col, lw=2.5)
    ax0.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax0.axvline(best_x, color=col, ls=":", lw=1.5, alpha=0.7)
    ax0.scatter([best_x], [ba], color=col, s=120, zorder=5)
    ax0.fill_between(x, 0.5, layer_auroc,
                     where=[v > 0.5 for v in layer_auroc], alpha=0.1, color=col)
    ax0.annotate(
        f"Best: L{bl}/{n_L} ({best_x*100:.0f}%)\nAUROC={ba:.3f}",
        (best_x, ba), textcoords="offset points", xytext=(6, -20),
        fontsize=8, color=col, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=col, lw=1),
    )
    ax0.set_xlim(0, 1); ax0.set_ylim(0.4, 1.0)
    ax0.set_ylabel("Probe AUROC" if col_idx == 0 else "")
    ax0.set_xlabel("Normalized Layer Depth")
    ax0.set_title(f"{LABELS[model_key]}\n(A) Layer-wise Probe AUROC",
                  fontweight="bold", fontsize=10)
    ax0.grid(True, alpha=0.3)

    # ── Row 1: Cosine Separation ────────────────────────────────
    ax1 = axes[1, col_idx]
    ax1.plot(x, cosine_bw, color=col, lw=2.5, alpha=0.9,
             label="cos(correct, wrong) centroids")
    ax1.axvline(best_x, color=col, ls=":", lw=1.5, alpha=0.6,
                label=f"Best probe L{bl} ({best_x*100:.0f}%)")
    ax1.fill_between(x, np.array(cosine_bw), 1.0, alpha=0.08, color=col)
    diffs = np.diff(cosine_bw)
    drop_l = int(np.argmin(diffs))
    drop_x = drop_l / (len(layer_auroc) - 1)
    ax1.annotate(
        f"Sharpest drop\nL{drop_l} ({drop_x*100:.0f}%)",
        (drop_x, cosine_bw[drop_l]),
        textcoords="offset points", xytext=(6, 10),
        fontsize=7.5, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )
    ax1.set_xlim(0, 1); ax1.set_ylim(0.5, 1.05)
    ax1.set_ylabel("cos(correct centroid, wrong centroid)" if col_idx == 0 else "")
    ax1.set_xlabel("Normalized Layer Depth")
    ax1.set_title("(B) Representation Separation Curve\n(1.0=identical, lower=separated)",
                  fontweight="bold", fontsize=10)
    ax1.legend(fontsize=7, loc="lower left")
    ax1.grid(True, alpha=0.3)

    # ── Row 2: Logit Lens entropy ───────────────────────────────
    if has_geo and model_key in geo_data:
        gd = geo_data[model_key]
        mc = np.array(gd["logit_lens_mean_correct"])
        mw = np.array(gd["logit_lens_mean_wrong"])
        sc = np.array(gd["logit_lens_std_correct"])
        sw = np.array(gd["logit_lens_std_wrong"])
        if len(mc) > 0 and len(mw) > 0:
            xg = np.linspace(0, 1, len(mc))
            ax2 = axes[2, col_idx]
            ax2.plot(xg, mc, color="#2ecc71", lw=2, label="Correct answers")
            ax2.plot(xg, mw, color="#e74c3c", lw=2, label="Wrong answers")
            ax2.fill_between(xg, mc - sc/2, mc + sc/2, alpha=0.15, color="#2ecc71")
            ax2.fill_between(xg, mw - sw/2, mw + sw/2, alpha=0.15, color="#e74c3c")
            ax2.axvline(best_x, color=col, ls=":", lw=1.5, alpha=0.6)
            gap_at_best = mw[bl] - mc[bl] if bl < len(mw) else 0
            gap_at_last = mw[-1] - mc[-1]
            ax2.annotate(
                f"gap@L{bl}={gap_at_best:+.3f}",
                (best_x, max(mc[bl], mw[bl])),
                textcoords="offset points", xytext=(4, 6),
                fontsize=8, color=col,
            )
            ax2.set_xlim(0, 1)
            ax2.set_ylabel("Vocab Distribution Entropy" if col_idx == 0 else "")
            ax2.set_xlabel("Normalized Layer Depth")
            ax2.set_title(
                f"(C) Logit Lens: Vocab Entropy per Layer\n"
                f"(gap@last={gap_at_last:+.3f})",
                fontweight="bold", fontsize=10,
            )
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

    # ── Row 3: t-SNE ────────────────────────────────────────────
    if has_geo and model_key in geo_data and n_rows >= 4:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        gd  = geo_data[model_key]
        bl2 = gd["best_probe_layer"]
        key = str(bl2)
        recs = gd["records"]
        hs_list = [r["key_hidden_states"][key]
                   for r in recs if key in r["key_hidden_states"]]
        lab_list = [r["is_correct"]
                    for r in recs if key in r["key_hidden_states"]]
        if len(hs_list) >= 10:
            X = np.array(hs_list)
            y = np.array(lab_list)
            X_pca = PCA(n_components=min(50, X.shape[1])).fit_transform(X)
            X_2d  = TSNE(n_components=2, perplexity=30,
                         random_state=42).fit_transform(X_pca)
            ax3 = axes[3, col_idx]
            cm = y == 1; wm = y == 0
            ax3.scatter(X_2d[cm, 0], X_2d[cm, 1], c="#2ecc71",
                        s=22, alpha=0.75, label=f"Correct (n={cm.sum()})")
            ax3.scatter(X_2d[wm, 0], X_2d[wm, 1], c="#e74c3c",
                        s=22, alpha=0.75, label=f"Wrong (n={wm.sum()})")
            ax3.set_title(
                f"(D) t-SNE @ L{bl2}/{n_L} ({bl2/n_L*100:.0f}%)\nProbe AUROC={ba:.3f}",
                fontweight="bold", fontsize=10,
            )
            ax3.legend(fontsize=8)
            ax3.set_xticks([]); ax3.set_yticks([])
            ax3.grid(True, alpha=0.2)

plt.suptitle(
    "Hidden State Geometry: How Hallucination Is Encoded by Architecture\n"
    "(Mistral: early-layer decision  |  Qwen-7B: mid-layer  |  EXAONE: late-layer)",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
out_path = fig_dir / "MAIN_hidden_geometry.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")

# ─── Overlay figure: all models on 1 plot ────────────────────────
n_overlay_rows = 2 + (1 if has_geo else 0)
fig2, axes2 = plt.subplots(1, n_overlay_rows, figsize=(7 * n_overlay_rows, 5))
if n_overlay_rows == 1:
    axes2 = [axes2]

ax_a, ax_b = axes2[0], axes2[1]
ax_c = axes2[2] if n_overlay_rows >= 3 else None

for model_key in KEY_MODELS:
    if model_key not in probe_data:
        continue
    d   = probe_data[model_key]
    col = COLORS[model_key]
    lbl = LABELS[model_key]
    n   = d["n_layers"]
    bl  = d["probe"]["best_layer"]
    la  = d["probe"]["layer_auroc"]
    cb  = d["representation"]["layer_cosine_between"]
    x   = np.linspace(0, 1, len(la))
    bx  = bl / (len(la) - 1)

    ax_a.plot(x, la, color=col, lw=2, alpha=0.85, label=lbl)
    ax_a.scatter([bx], [la[bl]], color=col, s=80, zorder=5)
    ax_a.annotate(f"L{bl}/{n}\n({bx*100:.0f}%)", (bx, la[bl]),
                  textcoords="offset points", xytext=(4, 5),
                  fontsize=7.5, color=col)

    ax_b.plot(x, cb, color=col, lw=2, alpha=0.85, label=lbl)
    ax_b.axvline(bx, color=col, ls=":", lw=1.2, alpha=0.5)

    if ax_c and model_key in geo_data:
        gd = geo_data[model_key]
        mc = np.array(gd["logit_lens_mean_correct"])
        mw = np.array(gd["logit_lens_mean_wrong"])
        if len(mc) > 0:
            xg = np.linspace(0, 1, len(mc))
            ax_c.plot(xg, mw - mc, color=col, lw=2, alpha=0.85,
                      label=f"{lbl.split('(')[0].strip()}")
            ax_c.axvline(bx, color=col, ls=":", lw=1, alpha=0.4)

ax_a.axhline(0.5, color="gray", ls="--", lw=1, label="Random")
ax_a.set_xlabel("Normalized Layer Depth (0=input, 1=output)")
ax_a.set_ylabel("Probe AUROC")
ax_a.set_title("(A) Where Is Hallucination Encoded?\nLayer-wise Probe AUROC",
               fontweight="bold")
ax_a.set_ylim(0.4, 1.0)
ax_a.legend(fontsize=8)
ax_a.grid(True, alpha=0.3)

ax_b.set_xlabel("Normalized Layer Depth (0=input, 1=output)")
ax_b.set_ylabel("cos(correct centroid, wrong centroid)")
ax_b.set_title("(B) When Do Representations Diverge?\nCosine Similarity Between Correct/Wrong Centroids",
               fontweight="bold")
ax_b.set_ylim(0.5, 1.05)
ax_b.legend(fontsize=8)
ax_b.grid(True, alpha=0.3)

if ax_c:
    ax_c.axhline(0, color="gray", ls="--", lw=1, alpha=0.5)
    ax_c.set_xlabel("Normalized Layer Depth (0=input, 1=output)")
    ax_c.set_ylabel("Vocab Entropy Gap (wrong − correct)")
    ax_c.set_title("(C) How Is Uncertainty Expressed?\nLogit Lens Vocab Entropy Difference",
                   fontweight="bold")
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3)

plt.suptitle(
    "Hallucination Encoding Mechanisms: Mistral vs EXAONE vs Qwen-7B\n"
    "Three complementary views of how architecture shapes hallucination detection difficulty",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
overlay_path = fig_dir / "MAIN_geometry_overlay.png"
plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {overlay_path}")

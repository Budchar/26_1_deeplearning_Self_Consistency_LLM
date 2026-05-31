"""
Exp12 Sublayer Probe 시각화: MLP vs Attention stream per model.

각 레이어에서 attn_stream AUROC vs mlp_stream AUROC 비교.
MLP - Attn gap이 양수 → MLP가 hallucination 인코딩 주도
MLP - Attn gap이 음수 → Attention이 주도
"""

import sys, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

raw     = ROOT / "results" / "raw"
fig_dir = ROOT / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

COLORS  = {
    "mistral":          "#e74c3c",
    "mistral_base":     "#95a5a6",
    "mistral_v02":      "#f39c12",
    "zephyr_7b":        "#27ae60",
    "openhermes":       "#8e44ad",
    "nous_hermes_dpo":  "#d35400",
    "openchat_35":      "#16a085",
    "starling_7b":      "#2471a3",
    "exaone":           "#2980b9",
    "qwen_7b":          "#1abc9c",
    "phi3_mini":        "#c0392b",
}
LABELS  = {
    "mistral":          "Mistral-7B-v0.3 (ANOMALY)",
    "mistral_base":     "Mistral-7B-v0.1 (BASE)",
    "mistral_v02":      "Mistral-7B-v0.2 (SFT)",
    "zephyr_7b":        "Zephyr-7B (SFT+DPO)",
    "openhermes":       "OpenHermes-2.5 (SFT)",
    "nous_hermes_dpo":  "NousHermes-2 (SFT+DPO)",
    "openchat_35":      "OpenChat-3.5 (C-RLFT)",
    "starling_7b":      "Starling-LM-7B (RLHF)",
    "exaone":           "EXAONE-3.5-7.8B",
    "qwen_7b":          "Qwen2.5-7B",
    "phi3_mini":        "Phi-3-mini-4k",
}

# ── Load ───────────────────────────────────────────────────────
data = {}
for f in sorted(raw.glob("12_sublayer_*.json"), key=lambda x: x.stat().st_mtime):
    d = json.load(open(f))
    k = d["model"]
    data[k] = d   # 최신 파일이 덮어쓰기

if not data:
    print("No 12_sublayer_*.json found. Run experiments/12_sublayer_probe/run.py first.")
    import sys; sys.exit(0)

models_present = [k for k in COLORS if k in data]
print(f"Models: {models_present}")

n_models = len(models_present)
fig, axes = plt.subplots(2, max(n_models, 1), figsize=(6 * n_models, 10))
if n_models == 1:
    axes = axes.reshape(-1, 1)

for col, mkey in enumerate(models_present):
    d   = data[mkey]
    col_c = COLORS.get(mkey, "#95a5a6")
    lbl   = LABELS.get(mkey, mkey)
    n_L   = d["n_layers"]
    bl    = d.get("exp06_best_layer")

    attn_a = d["attn_stream_auroc"]
    mlp_a  = d["mlp_stream_auroc"]
    x      = np.linspace(0, 1, n_L)

    # ── Row 0: Attn vs MLP AUROC ──────────────────────────────
    ax0 = axes[0, col]
    ax0.plot(x, attn_a, color="#2980b9", lw=2.2, label="Attn stream")
    ax0.plot(x, mlp_a,  color=col_c,    lw=2.2, label="MLP stream", ls="--")
    ax0.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
    if bl is not None and bl < n_L:
        bx = bl / (n_L - 1)
        ax0.axvline(bx, color="black", ls=":", lw=1.2, alpha=0.5,
                    label=f"Exp06 best L{bl} ({bx*100:.0f}%)")
        ax0.annotate(
            f"L{bl}: attn={attn_a[bl]:.3f}\nmlp={mlp_a[bl]:.3f}",
            (bx, max(attn_a[bl], mlp_a[bl])),
            textcoords="offset points", xytext=(5, 5),
            fontsize=8, color="black",
        )
    ax0.set_xlim(0, 1); ax0.set_ylim(0.4, 1.0)
    ax0.set_xlabel("Normalized Layer Depth")
    ax0.set_ylabel("Probe AUROC" if col == 0 else "")
    ax0.set_title(f"{lbl}\n(A) Attn vs MLP Stream AUROC", fontweight="bold", fontsize=9)
    ax0.legend(fontsize=7)
    ax0.grid(True, alpha=0.3)

    # ── Row 1: MLP - Attn gap ────────────────────────────────
    ax1 = axes[1, col]
    gap = np.array(mlp_a) - np.array(attn_a)
    pos = np.maximum(gap, 0)
    neg = np.minimum(gap, 0)
    ax1.bar(x, pos, width=1/n_L, color=col_c, alpha=0.7,  label="MLP dominant")
    ax1.bar(x, neg, width=1/n_L, color="#2980b9", alpha=0.7, label="Attn dominant")
    ax1.axhline(0, color="black", lw=1)
    if bl is not None and bl < n_L:
        bx = bl / (n_L - 1)
        ax1.axvline(bx, color="black", ls=":", lw=1.2, alpha=0.7)
        ax1.annotate(
            f"L{bl}: {gap[bl]:+.3f}",
            (bx, gap[bl]),
            textcoords="offset points", xytext=(4, 4),
            fontsize=8, color="black", fontweight="bold",
        )
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Normalized Layer Depth")
    ax1.set_ylabel("MLP AUROC − Attn AUROC" if col == 0 else "")
    ax1.set_title("(B) MLP vs Attn Dominance\n(+= MLP leads, -= Attention leads)",
                  fontweight="bold", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

plt.suptitle(
    "Exp12: Sublayer Probe — Where Does Hallucination Encoding Come From?\n"
    "Attn stream (before MLP) vs MLP stream (after MLP) at each layer",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
out = fig_dir / "MAIN_sublayer_probe.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Overlay: all models MLP-Attn gap ──────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
ax_a, ax_b = axes2

for mkey in models_present:
    d   = data[mkey]
    n_L = d["n_layers"]
    x   = np.linspace(0, 1, n_L)
    attn_a = d["attn_stream_auroc"]
    mlp_a  = d["mlp_stream_auroc"]
    col_c  = COLORS.get(mkey, "#95a5a6")
    lbl    = LABELS.get(mkey, mkey)
    bl     = d.get("exp06_best_layer")

    ax_a.plot(x, mlp_a,  color=col_c, lw=2, label=lbl)
    ax_a.plot(x, attn_a, color=col_c, lw=1.5, ls="--", alpha=0.6)
    if bl is not None and bl < n_L:
        bx = bl / (n_L - 1)
        ax_a.scatter([bx], [mlp_a[bl]], color=col_c, s=60, zorder=5)

    gap = np.array(mlp_a) - np.array(attn_a)
    ax_b.plot(x, gap, color=col_c, lw=2, label=lbl)
    if bl is not None and bl < n_L:
        bx = bl / (n_L - 1)
        ax_b.scatter([bx], [gap[bl]], color=col_c, s=60, zorder=5)
        ax_b.annotate(f"{gap[bl]:+.3f}", (bx, gap[bl]),
                      textcoords="offset points", xytext=(4, 3),
                      fontsize=8, color=col_c)

ax_a.axhline(0.5, color="gray", ls="--", lw=1)
ax_a.set_xlabel("Normalized Layer Depth")
ax_a.set_ylabel("Probe AUROC")
ax_a.set_title("MLP stream AUROC (solid) vs Attn stream (dashed)\nBubble = Exp06 best layer",
               fontweight="bold")
ax_a.set_ylim(0.4, 1.0)
ax_a.legend(fontsize=8)
ax_a.grid(True, alpha=0.3)

ax_b.axhline(0, color="gray", ls="--", lw=1)
ax_b.set_xlabel("Normalized Layer Depth")
ax_b.set_ylabel("MLP AUROC − Attn AUROC")
ax_b.set_title("MLP vs Attention Dominance per Layer\n(+= MLP drives encoding, −= Attention drives)",
               fontweight="bold")
ax_b.legend(fontsize=8)
ax_b.grid(True, alpha=0.3)

plt.suptitle("Sublayer Probe: Who Encodes Hallucination — MLP or Attention?",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out2 = fig_dir / "MAIN_sublayer_overlay.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ── Summary table ─────────────────────────────────────────────
print(f"\n{'Model':<25} {'n_L':>4} {'Exp06L':>6} {'attn@exp06':>10} {'mlp@exp06':>10} {'gap':>8} {'Driver':>10}")
print("-" * 80)
for mkey in sorted(models_present):
    d   = data[mkey]
    n_L = d["n_layers"]
    bl  = d.get("exp06_best_layer", 0)
    aa  = d["attn_stream_auroc"][bl] if bl < n_L else 0
    ma  = d["mlp_stream_auroc"][bl] if bl < n_L else 0
    gap = ma - aa
    drv = "MLP" if gap > 0.02 else ("Attn" if gap < -0.02 else "Both")
    print(f"  {mkey:<23} {n_L:>4} {bl:>6} {aa:>10.3f} {ma:>10.3f} {gap:>+8.3f} {drv:>10}")

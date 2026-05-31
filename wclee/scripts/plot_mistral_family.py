"""
Mistral Family Comparison: Layer Probe + Sublayer Probe
Visualizes how instruction tuning variant affects hallucination encoding depth.
"""

import sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

raw     = ROOT / "results" / "raw"
fig_dir = ROOT / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Ordered from most-base to most-tuned
MISTRAL_ORDER = [
    "mistral_base", "mistral_v02", "openhermes",
    "zephyr_7b", "openchat_35", "starling_7b",
    "nous_hermes_dpo", "mistral",
]
COLORS = {
    "mistral_base":   "#95a5a6",
    "mistral_v02":    "#f39c12",
    "openhermes":     "#8e44ad",
    "zephyr_7b":      "#27ae60",
    "openchat_35":    "#16a085",
    "starling_7b":    "#2471a3",
    "nous_hermes_dpo":"#d35400",
    "mistral":        "#e74c3c",
}
LABELS = {
    "mistral_base":   "Mistral-7B-v0.1 (BASE)",
    "mistral_v02":    "Mistral-7B-Instruct-v0.2 (SFT)",
    "openhermes":     "OpenHermes-2.5 (SFT only)",
    "zephyr_7b":      "Zephyr-7B-beta (SFT+DPO)",
    "openchat_35":    "OpenChat-3.5 (C-RLFT)",
    "starling_7b":    "Starling-LM-7B-alpha (C-RLFT+RLHF)",
    "nous_hermes_dpo":"NousHermes-2 (SFT+DPO) ← DPO!",
    "mistral":        "Mistral-7B-Instruct-v0.3 (ANOMALY)",
}

# ── Load Exp06 data ────────────────────────────────────────────
exp06 = {}
for m in MISTRAL_ORDER:
    files = sorted(raw.glob(f"06_layer_probe_{m}_*.json"), key=lambda x: x.stat().st_mtime)
    for pf in reversed(files):
        d = json.load(open(pf))
        if d.get("model", "") == m:
            exp06[m] = d
            break

# ── Load Exp12 data ────────────────────────────────────────────
exp12 = {}
for f in sorted(raw.glob("12_sublayer_*.json"), key=lambda x: x.stat().st_mtime):
    d = json.load(open(f))
    k = d["model"]
    if k in MISTRAL_ORDER:
        exp12[k] = d

present_06 = [m for m in MISTRAL_ORDER if m in exp06]
present_12 = [m for m in MISTRAL_ORDER if m in exp12]
print(f"Exp06 available: {present_06}")
print(f"Exp12 available: {present_12}")

# ── Figure 1: Exp06 Layer Probe AUROC ─────────────────────────
if present_06:
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in present_06:
        d = exp06[m]
        la = d["probe"]["layer_auroc"]
        n = len(la)  # use actual length (may include embedding layer)
        x = np.linspace(0, 1, n)
        bl = d["probe"]["best_layer"]
        ax.plot(x, la, color=COLORS[m], lw=2.5, label=LABELS[m])
        ax.scatter([bl / (n-1)], [la[bl]], color=COLORS[m], s=80, zorder=5)
        ax.annotate(f"L{bl}\n{la[bl]:.3f}", (bl/(n-1), la[bl]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7.5, color=COLORS[m], fontweight="bold")

    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.6)
    ax.axvspan(0, 0.15, alpha=0.08, color="#e74c3c", label="DPO early zone (0–15%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.35, 0.88)
    ax.set_xlabel("Normalized Layer Depth (0=input, 1=output)", fontsize=11)
    ax.set_ylabel("Layer Probe AUROC", fontsize=11)
    ax.set_title(
        "Mistral Architecture Family: Where Is Hallucination Encoded?\n"
        "Layer Probe (Exp06) — AUROC per layer across fine-tuning variants",
        fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = fig_dir / "MAIN_mistral_family_exp06.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ── Figure 2: Exp12 MLP-stream AUROC comparison ───────────────
if present_12:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_mlp, ax_gap = axes

    for m in present_12:
        d = exp12[m]
        n = d["n_layers"]
        x = np.linspace(0, 1, n)
        mlp_a  = np.array(d["mlp_stream_auroc"])
        attn_a = np.array(d["attn_stream_auroc"])
        gap    = mlp_a - attn_a
        bl_exp06 = d.get("exp06_best_layer")

        ax_mlp.plot(x, mlp_a, color=COLORS[m], lw=2.2, label=LABELS[m])
        if bl_exp06 is not None and bl_exp06 < n:
            bx = bl_exp06 / (n - 1)
            ax_mlp.scatter([bx], [mlp_a[bl_exp06]], color=COLORS[m], s=60, zorder=5)

        ax_gap.plot(x, gap, color=COLORS[m], lw=2.2, label=LABELS[m])
        if bl_exp06 is not None and bl_exp06 < n:
            bx = bl_exp06 / (n - 1)
            ax_gap.scatter([bx], [gap[bl_exp06]], color=COLORS[m], s=60, zorder=5)
            ax_gap.annotate(f"{gap[bl_exp06]:+.3f}", (bx, gap[bl_exp06]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=8, color=COLORS[m])

    ax_mlp.axhline(0.5, color="gray", ls="--", lw=1)
    ax_mlp.axvspan(0, 0.15, alpha=0.08, color="#e74c3c")
    ax_mlp.set_xlabel("Normalized Layer Depth"); ax_mlp.set_ylabel("MLP stream AUROC")
    ax_mlp.set_title("MLP Stream AUROC per Layer\n(bubble = Exp06 best layer)",
                     fontweight="bold")
    ax_mlp.set_ylim(0.35, 0.88)
    ax_mlp.legend(fontsize=8); ax_mlp.grid(True, alpha=0.3)

    ax_gap.axhline(0, color="gray", ls="--", lw=1)
    ax_gap.axvspan(0, 0.15, alpha=0.08, color="#e74c3c")
    ax_gap.set_xlabel("Normalized Layer Depth")
    ax_gap.set_ylabel("MLP AUROC − Attn AUROC")
    ax_gap.set_title("MLP vs Attention Dominance\n(+= MLP leads, −= Attention leads)",
                     fontweight="bold")
    ax_gap.legend(fontsize=8); ax_gap.grid(True, alpha=0.3)

    plt.suptitle(
        "Exp12: Sublayer Probe — Mistral Instruction-Tuning Variants\n"
        "Does early L4 MLP encoding in v0.3 appear in other variants?",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = fig_dir / "MAIN_mistral_family_exp12.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ── Summary table ─────────────────────────────────────────────
print(f"\n{'Model':<30} {'Exp06 Best':>10} {'Exp06 AUROC':>11} {'L4 AUROC':>9} {'Type':>10}")
print("-" * 78)
for m in MISTRAL_ORDER:
    exp06_d = exp06.get(m)
    exp12_d = exp12.get(m)
    if exp06_d:
        bl = exp06_d["probe"]["best_layer"]
        n  = len(exp06_d["probe"]["layer_auroc"])
        ba = exp06_d["probe"]["best_auroc"]
        l4 = exp06_d["probe"]["layer_auroc"][4] if len(exp06_d["probe"]["layer_auroc"]) > 4 else None
        depth = bl / (n-1) * 100
        if bl <= 5:
            t = "Type I (early MLP)"
        elif bl <= 16:
            t = "Intermediate"
        else:
            t = "Type II (late)"
        l4_str = f"{l4:.3f}" if l4 else "--"
        print(f"  {LABELS.get(m, m):<28} L{bl}/{n}={depth:4.0f}% {ba:>11.3f} {l4_str:>9} {t:>10}")
    else:
        print(f"  {LABELS.get(m, m):<28} {'--':>10}")

"""지금까지 수집된 Scaling + Layer 분석 결과 종합 시각화."""
import sys, json, numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

raw = ROOT / "results" / "raw"
fig_dir = ROOT / "results" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# ─── Load Layer Probe Results ────────────────────────────────────
layer_data = {}
for f in sorted(raw.glob("06_layer_probe_*.json")):
    d = json.load(open(f))
    key = d["model"]
    if key not in layer_data or d["n_samples"] > layer_data[key]["n_samples"]:
        layer_data[key] = d

# ─── Load Scaling Results ─────────────────────────────────────────
scaling_data = {}
for f in sorted(raw.glob("07_scaling_qwen_*.json"), key=lambda x: x.stat().st_size, reverse=True):
    for m in json.load(open(f)):
        k = m["model_key"]
        if k not in scaling_data:
            scaling_data[k] = m
qwen_scale = sorted(scaling_data.values(), key=lambda x: x["param_billions"])

# ─── Fig 1: Layer AUROC profiles — 모든 모델 overlay ─────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Layer AUROC curves
model_colors = {
    "exaone":        "#e74c3c",
    "mistral":       "#f39c12",
    "mistral_v02":   "#f39c12",
    "qwen_0_5b":     "#aed6f1",
    "qwen_1_5b":     "#5dade2",
    "qwen_3b":       "#3498db",
    "qwen_7b":       "#9b59b6",
    "qwen_14b":      "#2ecc71",
    "smollm2_1_7b":  "#1abc9c",
    "smollm2_360m":  "#a8d8a8",
    "opt_6_7b":      "#d35400",
}
model_labels = {
    "exaone":        "EXAONE-3.5 7.8B (32L)",
    "mistral":       "Mistral-7B v0.3 (32L)",
    "mistral_v02":   "Mistral-7B v0.2 (32L)",
    "qwen_0_5b":     "Qwen2.5-0.5B (24L)",
    "qwen_1_5b":     "Qwen2.5-1.5B (28L)",
    "qwen_3b":       "Qwen2.5-3B (36L)",
    "qwen_7b":       "Qwen2.5-7B (28L)",
    "qwen_14b":      "Qwen2.5-14B (48L)",
    "smollm2_1_7b":  "SmolLM2-1.7B (24L)",
    "smollm2_360m":  "SmolLM2-360M (32L)",
    "opt_6_7b":      "OPT-6.7B (32L)",
}

for key, d in layer_data.items():
    auroc = d["probe"]["layer_auroc"]
    n_layers = len(auroc)
    # 레이어 인덱스를 0~1 사이 깊이로 정규화
    x = [i / (n_layers - 1) for i in range(n_layers)]
    best = d["probe"]["best_layer"]
    best_x = best / (n_layers - 1)
    color = model_colors.get(key, "#95a5a6")
    label = model_labels.get(key, key)
    axes[0].plot(x, auroc, color=color, lw=2, label=label, alpha=0.85)
    axes[0].scatter([best_x], [auroc[best]], color=color, s=80, zorder=5)
    axes[0].annotate(f"{auroc[best]:.3f}@{best_x*100:.0f}%",
                     (best_x, auroc[best]), textcoords="offset points",
                     xytext=(5, 5), fontsize=7, color=color)

axes[0].axhline(0.5, color="gray", linestyle="--", lw=1, label="Random")
axes[0].set_xlabel("Normalized Layer Depth (0=input, 1=last layer)")
axes[0].set_ylabel("Probe AUROC")
axes[0].set_title("Layer-wise Hallucination Probe AUROC\n(per-layer hidden state → correctness prediction)", fontweight="bold")
axes[0].set_ylim(0.4, 1.0)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# ─── Fig 2: Qwen Scaling — 파라미터 vs 지표 ─────────────────────
params = [m["param_billions"] for m in qwen_scale]
accs = [m["accuracy"] for m in qwen_scale]
auroc_ent = [m["auroc_mean_entropy"] for m in qwen_scale]
auroc_nc = [m["auroc_n_clusters"] for m in qwen_scale]
gaps = [m["entropy_gap"] for m in qwen_scale]

ax2 = axes[1]
ax2_r = ax2.twinx()

ax2.plot(params, accs, "o-", color="#2ecc71", lw=2.5, ms=10, label="Accuracy")
ax2.plot(params, auroc_ent, "s--", color="#3498db", lw=2, ms=8, label="AUROC (Token Entropy)")
ax2.plot(params, auroc_nc, "^--", color="#9b59b6", lw=2, ms=8, label="AUROC (N Clusters)")
ax2_r.plot(params, gaps, "D:", color="#e74c3c", lw=2, ms=8, label="Entropy Gap", alpha=0.8)

for i, (p, a, ae, an, g) in enumerate(zip(params, accs, auroc_ent, auroc_nc, gaps)):
    ax2.annotate(f"{a:.2f}", (p, a), textcoords="offset points", xytext=(5, 3), fontsize=7, color="#2ecc71")
    ax2_r.annotate(f"{g:.3f}", (p, g), textcoords="offset points", xytext=(5, -10), fontsize=7, color="#e74c3c")

ax2.set_xscale("log")
ax2.set_xlabel("Parameters (Billions, log scale)")
ax2.set_ylabel("Accuracy / AUROC")
ax2_r.set_ylabel("Entropy Gap (wrong - correct)", color="#e74c3c")
ax2_r.tick_params(axis="y", labelcolor="#e74c3c")
ax2.set_title("Qwen2.5 Scaling (0.5B→14B)\nAccuracy UP vs Detection Harder Trade-off", fontweight="bold")
ax2.set_ylim(0.2, 1.0)
ax2.axhline(0.5, color="gray", linestyle=":", lw=1)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
ax2.grid(True, alpha=0.3)

# ─── Fig 3: Best Layer Depth vs Model Properties ─────────────────
model_info = []
for key, d in layer_data.items():
    n = d["n_layers"]
    best = d["probe"]["best_layer"]
    pct = best / n * 100
    params_b = {"exaone": 7.8, "mistral": 7.0, "mistral_v02": 7.0,
                "qwen_0_5b": 0.5, "qwen_1_5b": 1.5, "qwen_3b": 3.0,
                "qwen_7b": 7.0, "qwen_14b": 14.0,
                "smollm2_1_7b": 1.7, "smollm2_360m": 0.36, "opt_6_7b": 6.7}.get(key, 7.0)
    model_info.append({
        "key": key, "label": model_labels.get(key, key),
        "n_layers": n, "best_layer": best, "best_pct": pct,
        "best_auroc": d["probe"]["best_auroc"],
        "accuracy": d["accuracy"], "params": params_b,
        "color": model_colors.get(key, "#95a5a6")
    })

ax3 = axes[2]
for m in model_info:
    ax3.scatter(m["params"], m["best_pct"], s=m["best_auroc"]*500,
                color=m["color"], alpha=0.8, zorder=5,
                label=f"{m['label']}\n(Best@{m['best_pct']:.0f}%, AUROC={m['best_auroc']:.3f})")
    ax3.annotate(m["label"].split(" ")[0][:6],
                 (m["params"], m["best_pct"]),
                 textcoords="offset points", xytext=(8, 0), fontsize=8)

ax3.set_xlabel("Model Parameters (Billions)")
ax3.set_ylabel("Best Layer Depth (%)")
ax3.set_title("Best Probe Layer vs Model Scale\n(bubble size = Best AUROC)", fontweight="bold")
ax3.set_xlim(-0.5, 16)
ax3.set_ylim(5, 105)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=7, loc="lower right")
ax3.axhline(70, color="gray", linestyle=":", lw=1, alpha=0.5)
ax3.axhline(90, color="gray", linestyle=":", lw=1, alpha=0.5)
ax3.text(0.5, 70.5, "70% depth", fontsize=7, color="gray")
ax3.text(0.5, 90.5, "90% depth", fontsize=7, color="gray")

plt.suptitle("Layer Analysis & Scaling Laws for Hallucination Detection",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(fig_dir / "MAIN_scaling_layer_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: MAIN_scaling_layer_analysis.png")

# ─── Print summary ───────────────────────────────────────────────
print("\n[Layer Probe Summary]")
print(f"{'Model':<35} {'n_L':>4} {'Best_L':>7} {'Depth%':>7} {'AUROC':>7} {'Acc':>6}")
print("-"*65)
for m in sorted(model_info, key=lambda x: x["params"]):
    print(f"  {m['label'][:33]:<33} {m['n_layers']:>4} {m['best_layer']:>7} "
          f"{m['best_pct']:>6.0f}% {m['best_auroc']:>7.4f} {m['accuracy']:>6.3f}")

print("\n[Qwen Scaling Summary]")
print(f"{'Model':<35} {'Params':>6} {'Acc':>6} {'AUROC_ent':>10} {'AUROC_nc':>9} {'Gap':>7}")
print("-"*78)
for m in qwen_scale:
    print(f"  {m['model_name'][:33]:<33} {m['param_billions']:>5.1f}B "
          f"{m['accuracy']:>6.3f} {m['auroc_mean_entropy']:>10.4f} "
          f"{m['auroc_n_clusters']:>9.4f} {m['entropy_gap']:>+7.4f}")

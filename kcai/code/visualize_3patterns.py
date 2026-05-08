"""3 패턴 시각화 (이우창 결과와 비교용).

Pattern 1: Layer index vs AUROC (모델별 line)
Pattern 2: Params vs SEPs - SE gap (사이즈 ↑ → gap ↓)
Pattern 3: Params vs peak rel depth (사이즈 ↑ → 깊은 layer, 예외 표시)

데이터 소스: Phase 1 v1 (5 Instruct) + Sweep B (Pythia 7) + Sweep A (5 base).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 (matplotlib 등록 이름 기준)
plt.rcParams["font.family"] = ["NanumGothic", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

OUT_DIR = Path("/home/kcai/experiments/dl_team_v2/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "3patterns_visualization.png"


# ---- 데이터 로딩 ----
def load_phase1():
    """Phase 1 v1 (5 Instruct) — Llama 1B/3B + Qwen 1.5B/3B/7B"""
    base = Path("/home/kcai/experiments/dl_team_v2/01_se_seps/runs")
    models = [
        ("meta-llama__Llama-3.2-1B-Instruct", "Llama-3.2-1B-Inst", 1.24, "Instruct", "Llama"),
        ("meta-llama__Llama-3.2-3B-Instruct", "Llama-3.2-3B-Inst", 3.21, "Instruct", "Llama"),
        ("Qwen__Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Inst", 1.54, "Instruct", "Qwen"),
        ("Qwen__Qwen2.5-3B-Instruct", "Qwen2.5-3B-Inst", 3.09, "Instruct", "Qwen"),
        ("Qwen__Qwen2.5-7B-Instruct", "Qwen2.5-7B-Inst", 7.62, "Instruct", "Qwen"),
    ]
    return _collect(base, models)


def load_sweep_b():
    """Sweep B Pythia 7 사이즈"""
    base = Path("/home/kcai/experiments/dl_team_v2/04_sweep_b_pythia/runs")
    models = [
        ("EleutherAI__pythia-70m-deduped", "Pythia-70M", 0.07, "base", "Pythia"),
        ("EleutherAI__pythia-160m-deduped", "Pythia-160M", 0.16, "base", "Pythia"),
        ("EleutherAI__pythia-410m-deduped", "Pythia-410M", 0.41, "base", "Pythia"),
        ("EleutherAI__pythia-1b-deduped", "Pythia-1B", 1.0, "base", "Pythia"),
        ("EleutherAI__pythia-1.4b-deduped", "Pythia-1.4B", 1.4, "base", "Pythia"),
        ("EleutherAI__pythia-2.8b-deduped", "Pythia-2.8B", 2.8, "base", "Pythia"),
        ("EleutherAI__pythia-6.9b-deduped", "Pythia-6.9B", 6.9, "base", "Pythia"),
    ]
    return _collect(base, models)


def load_sweep_a():
    """Sweep A 5 base 패밀리"""
    base = Path("/home/kcai/experiments/dl_team_v2/06_sweep_a_family/runs")
    models = [
        ("EleutherAI__pythia-1.4b-deduped", "Pythia-1.4B", 1.4, "base", "Pythia"),
        ("meta-llama__Llama-3.2-1B", "Llama-3.2-1B-base", 1.24, "base", "Llama"),
        ("Qwen__Qwen2.5-1.5B", "Qwen2.5-1.5B-base", 1.54, "base", "Qwen"),
        ("facebook__opt-1.3b", "OPT-1.3B", 1.3, "base", "OPT"),
        ("EleutherAI__gpt-neo-1.3B", "GPT-Neo-1.3B", 1.3, "base", "GPT-Neo"),
    ]
    return _collect(base, models)


def _collect(base, models):
    out = []
    for slug, name, params, kind, family in models:
        for ds in ["triviaqa", "nq_open", "squad"]:
            d = base / slug / ds
            mp = d / "metrics.json"
            pp = d / "probes.json"
            if not mp.exists() or not pp.exists():
                continue
            try:
                m = json.loads(mp.read_text())
                p = json.loads(pp.read_text())
            except Exception:
                continue
            ga = m.get("greedy_acc", float("nan"))
            se = m.get("se_discrete", {}).get("auroc", float("nan"))
            seps = (
                p.get("best_logreg_halluc_auroc")
                or p.get("best_logreg_hallucination_auroc")
            )
            if seps is None or not isinstance(seps, (int, float)):
                continue
            # layer-wise AUROC
            layer_aurocs = []
            for l in p.get("layer_results", []):
                a = l.get("logreg_hallucination_auroc")
                if isinstance(a, (int, float)):
                    layer_aurocs.append(float(a))
            if not layer_aurocs:
                continue
            n_layers = len(layer_aurocs)
            best_idx = int(np.argmax(layer_aurocs))
            rel = best_idx / max(1, n_layers - 1)
            out.append({
                "model": name,
                "params_b": params,
                "kind": kind,
                "family": family,
                "ds": ds,
                "greedy_acc": ga,
                "se_auroc": float(se) if isinstance(se, (int, float)) else float("nan"),
                "seps_auroc": float(seps),
                "gap": float(seps) - float(se) if isinstance(se, (int, float)) else float("nan"),
                "layer_aurocs": layer_aurocs,
                "n_layers": n_layers,
                "peak_layer": best_idx,
                "peak_rel_depth": rel,
            })
    return out


# ---- 그래프 ----
def plot_3patterns():
    p1_data = load_phase1()
    sb_data = load_sweep_b()
    sa_data = load_sweep_a()
    all_data = p1_data + sb_data + sa_data
    print(f"Phase1: {len(p1_data)} cells, SweepB: {len(sb_data)}, SweepA: {len(sa_data)}")

    family_color = {
        "Llama": "tab:blue",
        "Qwen": "tab:orange",
        "Pythia": "tab:green",
        "OPT": "tab:red",
        "GPT-Neo": "tab:purple",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ─────────── Pattern 1: Layer index → AUROC ───────────
    ax = axes[0]
    # Phase 1 v1 (Instruct) 만 표시 (이우창 비교용 — Llama/Qwen 위주)
    seen_models = set()
    for r in p1_data:
        if r["model"] in seen_models:
            continue
        seen_models.add(r["model"])
        # 같은 모델 3 데이터셋 평균
        peers = [d for d in p1_data if d["model"] == r["model"]]
        max_n = max(len(d["layer_aurocs"]) for d in peers)
        avg = np.zeros(max_n)
        cnt = np.zeros(max_n)
        for d in peers:
            la = d["layer_aurocs"]
            for i, v in enumerate(la):
                avg[i] += v
                cnt[i] += 1
        avg = avg / np.maximum(cnt, 1)
        rel_x = np.linspace(0, 1, len(avg))
        col = family_color.get(r["family"], "gray")
        lw = 1.0 + 0.5 * np.log(r["params_b"] + 1)
        ax.plot(rel_x, avg, color=col, linewidth=lw, alpha=0.9, label=r["model"])

    ax.set_xlabel("상대 layer 깊이 (0=embed, 1=last)", fontsize=11)
    ax.set_ylabel("Hallucination AUROC", fontsize=11)
    ax.set_title("패턴 1 — layer ↑ ⇒ AUROC ↑\n(깊을수록 환각 신호 명확)", fontsize=12)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.45, 0.95)

    # ─────────── Pattern 2: Params → SEPs/SE gap ───────────
    ax = axes[1]
    # Sweep B (Pythia 7) + Phase 1 v1 (Instruct) + Sweep A (base 5)
    for label, data, marker in [
        ("Pythia base (Sweep B)", sb_data, "o"),
        ("Instruct (Phase 1 v1)", p1_data, "s"),
        ("Base 5 family (Sweep A)", sa_data, "^"),
    ]:
        for r in data:
            if not np.isfinite(r["gap"]):
                continue
            col = family_color.get(r["family"], "gray")
            ax.scatter(
                r["params_b"], r["gap"],
                color=col, marker=marker, s=60, alpha=0.65,
                edgecolors="black", linewidth=0.4,
            )
    # 경향선: 모든 데이터로
    xs = [r["params_b"] for r in all_data if np.isfinite(r["gap"])]
    ys = [r["gap"] for r in all_data if np.isfinite(r["gap"])]
    if xs:
        log_xs = np.log10(np.array(xs))
        z = np.polyfit(log_xs, ys, 1)
        x_fit = np.logspace(np.log10(min(xs)), np.log10(max(xs)), 50)
        y_fit = np.polyval(z, np.log10(x_fit))
        ax.plot(x_fit, y_fit, "--k", linewidth=1.2, alpha=0.5,
                label=f"trend (slope={z[0]:.3f}/log)")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("모델 파라미터 (B, log)", fontsize=11)
    ax.set_ylabel("SEPs AUROC − SE AUROC", fontsize=11)
    ax.set_title("패턴 2 — 파라미터 ↑ ⇒ gap ↓\n(큰 모델은 SE/SEPs 차이 줄어듦)", fontsize=12)
    # legend (마커별)
    handles = [
        plt.Line2D([], [], marker="o", color="gray", linestyle="None", markersize=8, label="Pythia base"),
        plt.Line2D([], [], marker="s", color="gray", linestyle="None", markersize=8, label="Instruct"),
        plt.Line2D([], [], marker="^", color="gray", linestyle="None", markersize=8, label="Base family"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # ─────────── Pattern 3: Params → peak rel depth ───────────
    ax = axes[2]
    for label, data, marker in [
        ("Pythia base", sb_data, "o"),
        ("Instruct", p1_data, "s"),
        ("Base family", sa_data, "^"),
    ]:
        for r in data:
            col = family_color.get(r["family"], "gray")
            ax.scatter(
                r["params_b"], r["peak_rel_depth"],
                color=col, marker=marker, s=60, alpha=0.65,
                edgecolors="black", linewidth=0.4,
            )

    # Llama 시리즈 예외 표시
    for r in p1_data:
        if r["family"] == "Llama":
            ax.annotate(
                "", xy=(r["params_b"], r["peak_rel_depth"]),
                xytext=(r["params_b"]*1.5, r["peak_rel_depth"]+0.1),
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            )
    ax.text(2, 0.95, "Llama 3B 예외\n(1B보다 얕음)",
            fontsize=8, color="red", ha="center")

    ax.axhline(0.68, color="green", linestyle=":", linewidth=1.2, alpha=0.7, label="원래 가설 0.68")
    ax.set_xscale("log")
    ax.set_xlabel("모델 파라미터 (B, log)", fontsize=11)
    ax.set_ylabel("peak layer / total layers", fontsize=11)
    ax.set_title("패턴 3 — 파라미터 ↑ ⇒ 깊은 layer\n(Llama 시리즈가 예외)", fontsize=12)
    # 패밀리 색깔 legend
    from matplotlib.patches import Patch
    family_handles = [Patch(color=c, label=f) for f, c in family_color.items()]
    ax.legend(handles=family_handles, loc="lower right", fontsize=8, ncol=1)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "3 패턴 시각화 — 이우창 결과 비교용\n"
        f"Phase 1 v1 ({len(p1_data)}) + Sweep B Pythia ({len(sb_data)}) + Sweep A base ({len(sa_data)}) = "
        f"{len(all_data)} cells",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {OUT_PNG}")


if __name__ == "__main__":
    plot_3patterns()

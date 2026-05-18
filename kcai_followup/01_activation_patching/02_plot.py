"""실험 1 plot: pair별 layer × patch effect bar.

입력: results/{src}__vs__{tgt}__{ds}/_summary.json
출력: plots/{src}__vs__{tgt}__{ds}_patch_effect.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "_shared"))

from paths import EXP1_PATCHING

RESULTS = EXP1_PATCHING / "results"
PLOTS = EXP1_PATCHING / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def plot_one(summary: dict, out_path: Path):
    base = summary["baseline_acc"]
    pr = summary.get("patch_results", {})
    cr = summary.get("control_results", {})
    layers_p = sorted([int(k.replace("L", "")) for k in pr.keys()])
    layers_c = sorted([int(k.replace("L", "")) for k in cr.keys()])
    deltas_p = [pr[f"L{l}"]["delta"] for l in layers_p]
    deltas_c = [cr[f"L{l}"]["delta"] for l in layers_c]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_p = np.array(layers_p)
    x_c = np.array(layers_c)
    ax.bar(x_p - 0.2, deltas_p, width=0.4, color="coral", edgecolor="black", label="patch layer (target L2-L4)")
    ax.bar(x_c + 0.2, deltas_c, width=0.4, color="steelblue", edgecolor="black", label="control layer")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(f"acc delta (baseline = {base:.3f})")
    ax.set_title(f"{summary['source'].split('/')[-1]} → {summary['target'].split('/')[-1]} / {summary['dataset']}\nMLP patching effect (n={summary['n_prompts']})", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    files = sorted(RESULTS.glob("*/_summary.json"))
    summaries = [json.loads(f.read_text()) for f in files]
    print(f"Found {len(summaries)} pairs")
    for s in summaries:
        out_path = PLOTS / f"{s['source'].replace('/', '__')}__vs__{s['target'].replace('/', '__')}__{s['dataset']}_patch.png"
        plot_one(s, out_path)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()

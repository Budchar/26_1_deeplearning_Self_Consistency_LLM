"""실험 1·2 결과의 paired t-test 통계 검정.

실험 1 (patching): 각 layer의 baseline vs patched per-prompt correct 비교
실험 2 (steering): 각 α vs baseline (α=0) per-prompt correct 비교

출력:
  /home/kcai/experiments/dl_team_followup/_docs/_paired_ttest_결과.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from paths import EXP1_PATCHING, EXP2_STEERING


def _read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def patching_ttest() -> list[dict]:
    """각 pair × dataset × layer에 대해 baseline vs patched per-prompt 비교."""
    out = []
    for pair_dir in sorted((EXP1_PATCHING / "results").glob("*/")):
        baseline_path = pair_dir / "baseline.jsonl"
        if not baseline_path.exists():
            continue
        baseline = {r["id"]: int(r["correct"]) for r in _read_jsonl(baseline_path)}

        for f in pair_dir.glob("patch_L*.jsonl"):
            li = int(f.stem.replace("patch_L", ""))
            patched = {r["id"]: int(r["correct"]) for r in _read_jsonl(f)}
            common = set(baseline) & set(patched)
            if len(common) < 5:
                continue
            b = np.array([baseline[k] for k in common])
            p = np.array([patched[k] for k in common])
            t, pval = stats.ttest_rel(p, b)
            d_mean = float((p - b).mean())
            d_std = float((p - b).std(ddof=1))
            out.append({
                "pair": pair_dir.name,
                "condition": "patch",
                "layer": li,
                "n": len(common),
                "baseline_acc": float(b.mean()),
                "patched_acc": float(p.mean()),
                "delta": d_mean,
                "delta_std": d_std,
                "t_stat": float(t),
                "p_value": float(pval),
                "significant_p05": bool(pval < 0.05),
            })
        for f in pair_dir.glob("control_L*.jsonl"):
            li = int(f.stem.replace("control_L", ""))
            patched = {r["id"]: int(r["correct"]) for r in _read_jsonl(f)}
            common = set(baseline) & set(patched)
            if len(common) < 5:
                continue
            b = np.array([baseline[k] for k in common])
            p = np.array([patched[k] for k in common])
            t, pval = stats.ttest_rel(p, b)
            d_mean = float((p - b).mean())
            d_std = float((p - b).std(ddof=1))
            out.append({
                "pair": pair_dir.name,
                "condition": "control",
                "layer": li,
                "n": len(common),
                "baseline_acc": float(b.mean()),
                "patched_acc": float(p.mean()),
                "delta": d_mean,
                "delta_std": d_std,
                "t_stat": float(t),
                "p_value": float(pval),
                "significant_p05": bool(pval < 0.05),
            })
    return out


def steering_ttest() -> list[dict]:
    """각 cell × α에 대해 α=0 baseline vs α!=0 per-prompt 비교."""
    out = []
    for cell_dir in sorted((EXP2_STEERING / "results").glob("*/")):
        baseline_f = None
        for f in cell_dir.glob("alpha_+0.00_L*.jsonl"):
            baseline_f = f
            break
        if baseline_f is None:
            continue
        baseline = {r["id"]: int(r["correct"]) for r in _read_jsonl(baseline_f)}

        for f in cell_dir.glob("alpha_*_L*.jsonl"):
            if "+0.00" in f.name:
                continue
            alpha = f.stem.split("_")[1]
            steered = {r["id"]: int(r["correct"]) for r in _read_jsonl(f)}
            common = set(baseline) & set(steered)
            if len(common) < 5:
                continue
            b = np.array([baseline[k] for k in common])
            s = np.array([steered[k] for k in common])
            t, pval = stats.ttest_rel(s, b)
            out.append({
                "cell": cell_dir.name,
                "alpha": alpha,
                "n": len(common),
                "baseline_acc": float(b.mean()),
                "steered_acc": float(s.mean()),
                "delta": float((s - b).mean()),
                "t_stat": float(t),
                "p_value": float(pval),
                "significant_p05": bool(pval < 0.05),
            })
    return out


def write_report(p_rows: list[dict], s_rows: list[dict], out_path: Path) -> None:
    lines = []
    lines.append("# Paired t-test 결과 — 실험 1 (Patching) · 실험 2 (Steering)\n")
    lines.append("> 작성: 2026-05-16 (자동 생성)\n\n---\n\n")
    lines.append("## 실험 1 — Patching\n\n")
    lines.append("| Pair | Cond | Layer | n | baseline | patched | Δ | t | p | sig (p<0.05) |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for r in p_rows:
        pair_short = r["pair"].replace("__vs__", "→").split("/")[-1][:50]
        lines.append(f"| {pair_short} | {r['condition']} | L{r['layer']} | {r['n']} | {r['baseline_acc']:.3f} | {r['patched_acc']:.3f} | {r['delta']:+.3f} | {r['t_stat']:+.2f} | {r['p_value']:.3f} | {'✅' if r['significant_p05'] else ''} |\n")

    sig_patch = sum(1 for r in p_rows if r["condition"] == "patch" and r["significant_p05"])
    sig_ctrl = sum(1 for r in p_rows if r["condition"] == "control" and r["significant_p05"])
    n_patch = sum(1 for r in p_rows if r["condition"] == "patch")
    n_ctrl = sum(1 for r in p_rows if r["condition"] == "control")
    lines.append(f"\n**요약**: patch 유의 {sig_patch}/{n_patch}, control 유의 {sig_ctrl}/{n_ctrl}\n\n---\n\n")

    lines.append("## 실험 2 — Steering\n\n")
    lines.append("| Cell | α | n | baseline | steered | Δ | t | p | sig (p<0.05) |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for r in s_rows:
        cell_short = r["cell"][:40]
        lines.append(f"| {cell_short} | {r['alpha']} | {r['n']} | {r['baseline_acc']:.3f} | {r['steered_acc']:.3f} | {r['delta']:+.3f} | {r['t_stat']:+.2f} | {r['p_value']:.3f} | {'✅' if r['significant_p05'] else ''} |\n")
    sig_s = sum(1 for r in s_rows if r["significant_p05"])
    lines.append(f"\n**요약**: steering 유의 {sig_s}/{len(s_rows)}\n")

    out_path.write_text("".join(lines))
    print(f"saved → {out_path}")


def main():
    p_rows = patching_ttest()
    s_rows = steering_ttest()
    out_path = Path("/home/kcai/experiments/dl_team_followup/_docs/_paired_ttest_결과.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(p_rows, s_rows, out_path)
    print(f"patch rows: {len(p_rows)}, steering rows: {len(s_rows)}")
    sig_patch = sum(1 for r in p_rows if r["condition"] == "patch" and r["significant_p05"])
    sig_ctrl = sum(1 for r in p_rows if r["condition"] == "control" and r["significant_p05"])
    sig_s = sum(1 for r in s_rows if r["significant_p05"])
    print(f"sig patch: {sig_patch}, sig control: {sig_ctrl}, sig steering: {sig_s}")


if __name__ == "__main__":
    main()

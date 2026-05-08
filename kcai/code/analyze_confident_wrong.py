"""Confident-but-Wrong analysis on Phase 1 data (이우창 제안 대응).

Find cases where the model's confidence is high but the output is wrong.
Operationalizes "overconfident hallucination" with 3 confidence metrics and
two correctness definitions, producing a 4-cell breakdown, ECE, Risk-Coverage,
metric disagreement, and per-model danger-zone examples.

Confidence metrics (higher = more confident):
  C_SE_disc = -se_discrete       (low semantic entropy → high confidence)
  C_SE_logp = -se_logprob        (logprob-weighted SE)
  C_logp    = mean(sample_logprobs)   (token-level confidence)

Correctness:
  EM = greedy_correct from se.jsonl  (already computed by se_compute.py)

Idempotent: writes per-(model, dataset) outputs and skips done ones.
CPU-only, GPU-safe to run while Phase 2 trains.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_se(records_path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(records_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def conf_metrics(rec: Dict) -> Dict[str, float]:
    se_d = rec.get("se_discrete")
    se_l = rec.get("se_logprob")
    lps = rec.get("sample_logprobs") or []
    return {
        "C_SE_disc": (-float(se_d)) if se_d is not None else float("nan"),
        "C_SE_logp": (-float(se_l)) if se_l is not None else float("nan"),
        "C_logp": (float(np.mean(lps)) if lps else float("nan")),
    }


def quartile_label(values: np.ndarray, x: float) -> str:
    q1, q2, q3 = np.nanpercentile(values, [25, 50, 75])
    if x <= q1:
        return "Q1"
    if x <= q2:
        return "Q2"
    if x <= q3:
        return "Q3"
    return "Q4"


def ece_score(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Standard equal-width ECE on confidence ∈ [min, max] mapped to [0,1]."""
    mask = np.isfinite(conf) & np.isfinite(correct)
    c, y = conf[mask], correct[mask].astype(float)
    if len(c) == 0:
        return float("nan")
    lo, hi = np.min(c), np.max(c)
    if hi - lo < 1e-9:
        return 0.0
    p = (c - lo) / (hi - lo)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if m.sum() == 0:
            continue
        avg_conf = p[m].mean()
        avg_acc = y[m].mean()
        e += (m.sum() / len(c)) * abs(avg_conf - avg_acc)
    return float(e)


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> List[Tuple[float, float, float]]:
    """Return [(coverage, accuracy_on_kept, threshold)]."""
    mask = np.isfinite(conf) & np.isfinite(correct)
    c, y = conf[mask], correct[mask].astype(float)
    if len(c) == 0:
        return []
    order = np.argsort(-c)  # high conf first
    c_sorted = c[order]
    y_sorted = y[order]
    n = len(c)
    out = []
    for cov_pct in range(10, 101, 10):
        k = max(1, int(round(n * cov_pct / 100)))
        out.append((cov_pct / 100.0, float(y_sorted[:k].mean()), float(c_sorted[k - 1])))
    return out


def four_cell(conf: np.ndarray, correct: np.ndarray) -> Dict[str, int]:
    mask = np.isfinite(conf) & np.isfinite(correct)
    c, y = conf[mask], correct[mask].astype(bool)
    if len(c) == 0:
        return {"high_correct": 0, "high_wrong": 0, "low_correct": 0, "low_wrong": 0}
    median = float(np.nanmedian(c))
    high = c >= median
    return {
        "high_correct": int((high & y).sum()),
        "high_wrong": int((high & ~y).sum()),
        "low_correct": int((~high & y).sum()),
        "low_wrong": int((~high & ~y).sum()),
        "median_threshold": median,
        "n": int(len(c)),
    }


def top_quartile_wrong(conf: np.ndarray, correct: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(conf) & np.isfinite(correct)
    c, y = conf[mask], correct[mask].astype(bool)
    if len(c) == 0:
        return {"n_top_q": 0, "wrong_in_top_q": 0, "rate": float("nan")}
    q3 = float(np.nanpercentile(c, 75))
    top = c >= q3
    n_top = int(top.sum())
    wrong = int((top & ~y).sum())
    return {"n_top_q": n_top, "wrong_in_top_q": wrong, "rate": (wrong / n_top) if n_top else float("nan")}


def disagreement(records: List[Dict]) -> Dict[str, int]:
    """Count cases where confidence metrics disagree (one says high, another low)."""
    metrics = ["C_SE_disc", "C_SE_logp", "C_logp"]
    arrays = {m: np.array([conf_metrics(r)[m] for r in records]) for m in metrics}
    medians = {m: float(np.nanmedian(arrays[m])) for m in metrics}
    high = {m: arrays[m] >= medians[m] for m in metrics}

    out: Dict[str, int] = {}
    pairs = [("C_SE_disc", "C_SE_logp"), ("C_SE_disc", "C_logp"), ("C_SE_logp", "C_logp")]
    for a, b in pairs:
        out[f"{a}_high__{b}_low"] = int((high[a] & ~high[b]).sum())
        out[f"{a}_low__{b}_high"] = int((~high[a] & high[b]).sum())
    return out


def per_pair_analysis(model_dir: Path, dataset: str) -> Dict:
    se_path = model_dir / dataset / "se.jsonl"
    if not se_path.exists():
        return {"skip_reason": f"missing {se_path}"}
    records = load_se(se_path)
    if not records:
        return {"skip_reason": "empty se.jsonl"}

    correct = np.array([1.0 if r.get("greedy_correct") else 0.0 for r in records])
    metrics = ["C_SE_disc", "C_SE_logp", "C_logp"]
    confs = {m: np.array([conf_metrics(r)[m] for r in records]) for m in metrics}

    summary: Dict = {
        "n_records": len(records),
        "greedy_acc": float(correct.mean()),
        "by_metric": {},
    }
    for m in metrics:
        c = confs[m]
        summary["by_metric"][m] = {
            "ece": ece_score(c, correct),
            "four_cell": four_cell(c, correct),
            "top_quartile_wrong": top_quartile_wrong(c, correct),
            "risk_coverage": [
                {"coverage": cov, "acc_on_kept": acc, "threshold": t}
                for cov, acc, t in risk_coverage(c, correct)
            ],
        }
    summary["disagreement_counts"] = disagreement(records)

    # danger-zone examples: top-quartile C_SE_disc AND wrong (max 50 per pair)
    c = confs["C_SE_disc"]
    q3 = float(np.nanpercentile(c, 75)) if np.isfinite(c).any() else float("nan")
    examples = []
    for r, ci in zip(records, c):
        if math.isfinite(ci) and ci >= q3 and not r.get("greedy_correct"):
            examples.append({
                "id": r.get("id"),
                "question": r.get("question"),
                "gold": r.get("answers"),
                "greedy": r.get("greedy"),
                "samples": r.get("samples"),
                "se_discrete": r.get("se_discrete"),
                "se_logprob": r.get("se_logprob"),
                "C_SE_disc": float(ci),
                "n_clusters": r.get("n_clusters"),
            })
            if len(examples) >= 50:
                break
    summary["danger_examples"] = examples

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/home/kcai/experiments/dl_team_v2/01_se_seps/runs")
    ap.add_argument("--out", default="/home/kcai/experiments/dl_team_v2/01_se_seps/results/confident_wrong")
    ap.add_argument("--datasets", nargs="+", default=["triviaqa", "nq_open", "squad"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    runs = Path(args.runs_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    big: Dict[str, Dict] = {}
    csv_rows = ["model,dataset,metric,n,greedy_acc,ece,top_q_wrong_rate,4cell_high_wrong,4cell_low_wrong"]

    for model_dir in sorted(runs.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("synthetic"):
            continue
        model_id = model_dir.name.replace("__", "/", 1)
        big[model_id] = {}
        for ds in args.datasets:
            done_path = out / f"{model_dir.name}__{ds}.json"
            if done_path.exists() and not args.force:
                summary = json.loads(done_path.read_text())
            else:
                summary = per_pair_analysis(model_dir, ds)
                done_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            big[model_id][ds] = summary

            if "by_metric" not in summary:
                continue
            for metric, m in summary["by_metric"].items():
                fc = m.get("four_cell", {})
                tqw = m.get("top_quartile_wrong", {})
                csv_rows.append(",".join([
                    model_id, ds, metric,
                    str(summary.get("n_records", 0)),
                    f"{summary.get('greedy_acc', 0):.4f}",
                    f"{m.get('ece', float('nan')):.4f}",
                    f"{tqw.get('rate', float('nan')):.4f}",
                    str(fc.get("high_wrong", 0)),
                    str(fc.get("low_wrong", 0)),
                ]))

    (out / "summary.json").write_text(json.dumps(big, ensure_ascii=False, indent=2))
    (out / "summary.csv").write_text("\n".join(csv_rows) + "\n")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()

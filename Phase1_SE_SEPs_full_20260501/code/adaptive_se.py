"""Cost-aware Adaptive SE (H4): dynamic sample budget based on early SE estimate.

Algorithm (per question):
    1. Use first n_initial=3 samples from the existing N=10 cache → SE_init.
    2. If SE_init < threshold_low      -> stop; effective_N = 3
    3. If SE_init > threshold_high     -> use all N=10
    4. else                            -> use first 5 (mid)
Final SE is the discrete entropy on the effective sample subset (recomputed via clustering).

This module operates on already-computed `se_jsonl` records (which contain cluster_ids
for the full N=10 set). To stay reproducible, it computes the *adaptive entropy* by
restricting the cluster_ids list to the first effective_N items (this is an approximation
to actually re-clustering the truncated sample set, which is valid under the assumption
that clusters are stable to sample order — which holds for greedy bidirectional clustering
on shuffled samples).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from metrics import detection_metrics


def discrete_entropy(cids: List[int]) -> float:
    counts = Counter(cids)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            H -= p * math.log(p)
    return H


def adaptive_pass(
    records: List[Dict],
    n_initial: int = 3,
    n_mid: int = 5,
    n_full: int = 10,
    th_low: float = 0.3,
    th_high: float = 0.9,
) -> List[Dict]:
    out: List[Dict] = []
    for r in records:
        cids = r["cluster_ids"]
        if len(cids) < n_full:
            n_full = len(cids)
        # First-pass SE estimate using n_initial
        cids_init = cids[:n_initial]
        se_init = discrete_entropy(cids_init)
        if se_init < th_low:
            eff = n_initial
        elif se_init > th_high:
            eff = n_full
        else:
            eff = min(n_mid, n_full)
        cids_eff = cids[:eff]
        se_eff = discrete_entropy(cids_eff)
        out.append({
            "id": r["id"],
            "se_init": se_init,
            "se_adaptive": se_eff,
            "se_full": r["se_discrete"],
            "n_used": eff,
            "n_full": n_full,
            "greedy_correct": r["greedy_correct"],
        })
    return out


def evaluate(records: List[Dict], adaptive: List[Dict]) -> Dict:
    correct = [r["greedy_correct"] for r in records]
    se_full = [r["se_discrete"] for r in records]
    se_ada = [a["se_adaptive"] for a in adaptive]
    n_used = [a["n_used"] for a in adaptive]
    n_full = adaptive[0]["n_full"] if adaptive else 10

    full_metrics = detection_metrics(se_full, correct)
    ada_metrics = detection_metrics(se_ada, correct)

    avg_n = float(np.mean(n_used)) if n_used else float(n_full)
    cost_save = 1.0 - (avg_n / n_full)
    return {
        "n": len(records),
        "n_full_per_q": n_full,
        "fixed": full_metrics,
        "adaptive": ada_metrics,
        "avg_n_adaptive": avg_n,
        "cost_save_frac": cost_save,
        "auroc_delta": ada_metrics["auroc"] - full_metrics["auroc"],
    }


def run(se_jsonl: Path, out_json: Path, **kwargs) -> Dict:
    records: List[Dict] = []
    with open(se_jsonl) as f:
        for line in f:
            records.append(json.loads(line))
    adaptive = adaptive_pass(records, **kwargs)
    summary = evaluate(records, adaptive)
    summary["params"] = kwargs
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_q": adaptive}, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--n-initial", type=int, default=3)
    ap.add_argument("--n-mid", type=int, default=5)
    ap.add_argument("--n-full", type=int, default=10)
    ap.add_argument("--th-low", type=float, default=0.3)
    ap.add_argument("--th-high", type=float, default=0.9)
    args = ap.parse_args()
    run(Path(args.in_path), Path(args.out_path),
        n_initial=args.n_initial, n_mid=args.n_mid, n_full=args.n_full,
        th_low=args.th_low, th_high=args.th_high)


if __name__ == "__main__":
    main()

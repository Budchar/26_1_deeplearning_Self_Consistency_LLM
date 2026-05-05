"""SEPs probe: predict semantic-entropy class from prompt hidden states.

For each layer ℓ:
  - Inputs:   X = stacked last-prompt-token hidden states (per question), shape (Q, H)
  - Targets:  y_se   = (SE > median(SE))            -> binary high-uncertainty
              y_corr = (greedy is incorrect)        -> binary hallucination
  - Train two probes per layer per target: LogisticRegression and 2-layer MLP.
  - 80/20 split (deterministic).
  - Output: layer × probe × target -> AUROC table (JSON).

This is the *Pre-hoc* signal in the H2 / H3 hypotheses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_se_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def stack_hidden(records: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    """Return (X[L+1, Q, H], records_used).

    Records without a valid hidden_path are skipped.
    """
    used: List[Dict] = []
    H_list: List[np.ndarray] = []
    for r in records:
        hp = r.get("hidden_path", "")
        if not hp or not Path(hp).exists():
            continue
        d = np.load(hp)
        H_list.append(d["greedy_h"])  # (L+1, H)
        used.append(r)
    if not H_list:
        raise RuntimeError(f"No hidden states found in {len(records)} records")
    X = np.stack(H_list, axis=0)  # (Q, L+1, H)
    X = np.transpose(X, (1, 0, 2))  # (L+1, Q, H)
    return X.astype(np.float32), used


def fit_probe_layer(X: np.ndarray, y: np.ndarray, kind: str, seed: int = 0) -> Tuple[float, float]:
    """Returns (val_auroc, train_auroc)."""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xva = sc.transform(Xva)
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=seed,
                            early_stopping=True, n_iter_no_change=10)
    else:
        raise ValueError(kind)
    clf.fit(Xtr, ytr)
    p_tr = clf.predict_proba(Xtr)[:, 1]
    p_va = clf.predict_proba(Xva)[:, 1]
    return float(roc_auc_score(yva, p_va)), float(roc_auc_score(ytr, p_tr))


def run_probes(se_jsonl: Path, out_json: Path, seed: int = 0) -> Dict:
    records = load_se_jsonl(se_jsonl)
    X, used = stack_hidden(records)  # (L+1, Q, H)
    se_vals = np.array([r["se_discrete"] for r in used], dtype=np.float32)
    correct = np.array([r["greedy_correct"] for r in used], dtype=np.int32)
    y_se = (se_vals > float(np.median(se_vals))).astype(np.int32)
    y_halluc = (1 - correct).astype(np.int32)

    L = X.shape[0]
    layer_results: List[Dict] = []
    for ell in tqdm(range(L), desc="probes"):
        Xl = X[ell]
        row = {"layer": ell}
        for kind in ("logreg", "mlp"):
            for tname, y in (("se_high", y_se), ("hallucination", y_halluc)):
                auc, auc_tr = fit_probe_layer(Xl, y, kind=kind, seed=seed)
                row[f"{kind}_{tname}_auroc"] = auc
                row[f"{kind}_{tname}_train_auroc"] = auc_tr
        layer_results.append(row)

    summary = {
        "n_used": len(used),
        "n_layers": L,
        "y_se_pos_rate": float(y_se.mean()),
        "y_halluc_pos_rate": float(y_halluc.mean()),
        "best_logreg_se_auroc": max((r["logreg_se_high_auroc"] for r in layer_results
                                     if not np.isnan(r["logreg_se_high_auroc"])), default=float("nan")),
        "best_logreg_halluc_auroc": max((r["logreg_hallucination_auroc"] for r in layer_results
                                         if not np.isnan(r["logreg_hallucination_auroc"])), default=float("nan")),
        "best_mlp_se_auroc": max((r["mlp_se_high_auroc"] for r in layer_results
                                  if not np.isnan(r["mlp_se_high_auroc"])), default=float("nan")),
        "best_mlp_halluc_auroc": max((r["mlp_hallucination_auroc"] for r in layer_results
                                      if not np.isnan(r["mlp_hallucination_auroc"])), default=float("nan")),
        "layer_results": layer_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "layer_results"}, indent=2), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_probes(Path(args.in_path), Path(args.out_path), seed=args.seed)


if __name__ == "__main__":
    main()

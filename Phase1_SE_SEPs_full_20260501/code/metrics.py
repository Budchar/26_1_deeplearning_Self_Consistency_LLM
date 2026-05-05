"""Evaluation metrics for Phase 1: AUROC, ECE, Brier, AURC, Wilcoxon, stratified accuracy."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, brier_score_loss


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC where higher score == higher predicted positive (e.g., higher SE -> hallucination=1)."""
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 15) -> float:
    """Expected Calibration Error. probs = predicted probability of correct=1 (i.e., 1 - hallucination)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(labels, dtype=np.int32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    n = len(p)
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        e += (mask.sum() / n) * abs(acc - conf)
    return float(e)


def brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(labels, dtype=np.int32)
    return float(brier_score_loss(y, p))


def aurc(uncertainty: Sequence[float], correct: Sequence[int]) -> float:
    """Area Under Risk-Coverage curve.

    Uncertainty: higher = more uncertain.
    correct: 1 if correct else 0.
    AURC sweeps coverage 1->0 by deferring most-uncertain first; risk = 1 - acc on covered.
    """
    u = np.asarray(uncertainty, dtype=np.float64)
    c = np.asarray(correct, dtype=np.int32)
    n = len(u)
    if n == 0:
        return float("nan")
    order = np.argsort(u)  # ascending (most certain first)
    c_sorted = c[order]
    coverage_risks: List[float] = []
    cumcorrect = 0
    for i in range(n):
        cumcorrect += int(c_sorted[i])
        cov = (i + 1) / n
        risk = 1.0 - (cumcorrect / (i + 1))
        coverage_risks.append(risk)
    return float(np.mean(coverage_risks))


def stratified_acc_by_quartile(
    se: Sequence[float], greedy_correct: Sequence[int], sc_correct: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    """For SE quartile Q1..Q4, return greedy/sc accuracy and improvement."""
    se_a = np.asarray(se, dtype=np.float64)
    g = np.asarray(greedy_correct, dtype=np.int32)
    s = np.asarray(sc_correct, dtype=np.int32)
    qs = np.quantile(se_a, [0.25, 0.5, 0.75])
    edges = [-np.inf, qs[0], qs[1], qs[2], np.inf]
    out: Dict[str, Dict[str, float]] = {}
    for i in range(4):
        mask = (se_a > edges[i]) & (se_a <= edges[i + 1])
        if mask.sum() == 0:
            continue
        out[f"Q{i+1}"] = {
            "n": int(mask.sum()),
            "greedy_acc": float(g[mask].mean()),
            "sc_acc": float(s[mask].mean()),
            "delta": float((s[mask] - g[mask]).mean()),
        }
    return out


def wilcoxon_paired(greedy_correct: Sequence[int], sc_correct: Sequence[int]) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test on SC - greedy correctness."""
    g = np.asarray(greedy_correct, dtype=np.int32)
    s = np.asarray(sc_correct, dtype=np.int32)
    diff = s - g
    if np.all(diff == 0):
        return {"stat": None, "p": 1.0, "n": int(len(diff)), "note": "all-zero diff"}
    try:
        stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        return {"stat": float(stat), "p": float(p), "n": int(len(diff))}
    except Exception as e:  # pragma: no cover
        return {"stat": None, "p": None, "n": int(len(diff)), "err": str(e)}


def detection_metrics(uncertainty: Sequence[float], correct: Sequence[int]) -> Dict[str, float]:
    """Wrap AUROC / ECE / Brier / AURC for hallucination-detection evaluation.

    Convention: hallucination = (correct == 0). Uncertainty is the *score* for hallucination.
    For ECE/Brier we convert to probability of correct via 1 - sigmoid(z-score(uncertainty)).
    """
    u = np.asarray(uncertainty, dtype=np.float64)
    c = np.asarray(correct, dtype=np.int32)
    halluc = 1 - c
    auc_halluc = auroc(u, halluc)
    # Calibrate: map u -> p_correct with simple sigmoid on z-scored u.
    if u.std() > 0:
        z = (u - u.mean()) / u.std()
        p_correct = 1.0 / (1.0 + np.exp(z))
    else:
        p_correct = np.full_like(u, 0.5, dtype=np.float64)
    return {
        "auroc": auc_halluc,
        "ece": ece(p_correct, c),
        "brier": brier(p_correct, c),
        "aurc": aurc(u, c),
    }

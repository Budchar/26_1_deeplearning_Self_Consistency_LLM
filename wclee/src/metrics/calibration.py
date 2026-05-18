"""
모델 Calibration 측정.

- ECE (Expected Calibration Error): 신뢰도와 정확도의 불일치 정량화
- Reliability Diagram: calibration 시각화
- Overconfidence Detection: 틀렸는데 high confidence인 케이스 분석
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
from typing import List, Tuple


def compute_ece(
    confidences: List[float],
    is_correct: List[bool],
    n_bins: int = 10,
) -> dict:
    """
    Expected Calibration Error 계산.
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Returns dict with ECE, MCE, bin stats.
    """
    confidences = np.array(confidences)
    is_correct = np.array(is_correct, dtype=float)
    n = len(confidences)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_acc[b] = is_correct[mask].mean()
            bin_conf[b] = confidences[mask].mean()
            bin_count[b] = mask.sum()

    ece = np.sum(bin_count / n * np.abs(bin_acc - bin_conf))
    mce = np.max(np.abs(bin_acc - bin_conf))  # Maximum Calibration Error

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_acc": bin_acc.tolist(),
        "bin_conf": bin_conf.tolist(),
        "bin_count": bin_count.tolist(),
        "bin_edges": bin_edges.tolist(),
        "n_bins": n_bins,
        "n_samples": n,
        "overall_accuracy": float(is_correct.mean()),
        "mean_confidence": float(confidences.mean()),
    }


def detect_overconfident(
    confidences: List[float],
    is_correct: List[bool],
    threshold: float = 0.8,
) -> dict:
    """
    핵심 분석: 높은 confidence인데 틀린 케이스 탐지.
    threshold 이상 confidence이면서 is_correct=False인 비율 반환.
    """
    confidences = np.array(confidences)
    is_correct = np.array(is_correct, dtype=bool)

    high_conf_mask = confidences >= threshold
    n_high_conf = high_conf_mask.sum()
    n_overconfident = (high_conf_mask & ~is_correct).sum()
    n_correct_high = (high_conf_mask & is_correct).sum()

    low_conf_mask = ~high_conf_mask
    n_low_conf = low_conf_mask.sum()
    n_underconf_correct = (low_conf_mask & is_correct).sum()

    return {
        "threshold": threshold,
        "n_total": len(confidences),
        "n_high_conf": int(n_high_conf),
        "n_overconfident": int(n_overconfident),
        "overconfident_rate": float(n_overconfident / n_high_conf) if n_high_conf > 0 else 0.0,
        "n_correct_high_conf": int(n_correct_high),
        "n_low_conf": int(n_low_conf),
        "n_underconfident_correct": int(n_underconf_correct),
        "underconfident_rate": float(n_underconf_correct / n_low_conf) if n_low_conf > 0 else 0.0,
    }


def plot_reliability_diagram(
    ece_result: dict,
    model_name: str = "",
    save_path: str = None,
    ax=None,
) -> plt.Axes:
    """Reliability Diagram (calibration curve) 시각화."""
    bin_acc = np.array(ece_result["bin_acc"])
    bin_conf = np.array(ece_result["bin_conf"])
    bin_count = np.array(ece_result["bin_count"])
    edges = np.array(ece_result["bin_edges"])
    n_bins = ece_result["n_bins"]
    ece = ece_result["ece"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    width = 1.0 / n_bins
    centers = (edges[:-1] + edges[1:]) / 2

    # gap (overconfidence = red, underconfidence = blue)
    for i in range(n_bins):
        if bin_count[i] == 0:
            continue
        gap = bin_conf[i] - bin_acc[i]
        color = "#e74c3c" if gap > 0 else "#3498db"
        ax.bar(centers[i], bin_acc[i], width=width * 0.9, color="#2ecc71", alpha=0.7, label="Accuracy" if i == 0 else "")
        ax.bar(centers[i], abs(gap), width=width * 0.9,
               bottom=bin_acc[i] if gap < 0 else bin_conf[i] - abs(gap),
               color=color, alpha=0.5, label=("Over/Underconf" if i == 0 else ""))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram — {model_name}\nECE = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    return ax


def plot_confidence_distribution(
    confidences: List[float],
    is_correct: List[bool],
    model_name: str = "",
    save_path: str = None,
):
    """정답/오답별 confidence 분포 히스토그램."""
    confidences = np.array(confidences)
    is_correct = np.array(is_correct, dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(confidences[is_correct], bins=bins, alpha=0.6, color="#2ecc71", label="Correct", density=True)
    ax.hist(confidences[~is_correct], bins=bins, alpha=0.6, color="#e74c3c", label="Wrong", density=True)
    ax.axvline(0.8, color="black", linestyle="--", linewidth=1, label="threshold=0.8")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"Confidence Distribution — {model_name}")
    ax.legend()

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    return fig


def normalize_log_prob_to_confidence(log_probs: List[float]) -> List[float]:
    """
    Sequence log probability → [0, 1] confidence로 정규화.
    min-max scaling after exponentiation (heuristic).
    """
    probs = np.exp(np.array(log_probs))
    p_min, p_max = probs.min(), probs.max()
    if p_max - p_min < 1e-8:
        return [0.5] * len(log_probs)
    return ((probs - p_min) / (p_max - p_min)).tolist()

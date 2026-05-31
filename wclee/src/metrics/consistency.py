"""
Self-Consistency 기반 Hallucination 탐지.

동일 질문을 여러 번 생성하고, 답변 간 일관성을 confidence proxy로 사용.
Wang et al. (2023) — https://arxiv.org/abs/2203.11171
"""

import numpy as np
from typing import List
from collections import Counter


def majority_vote(answers: List[str], normalize: bool = True) -> dict:
    """
    단순 문자열 majority vote.
    정규화된 답변으로 카운팅.
    """
    if normalize:
        normalized = [a.strip().lower() for a in answers]
    else:
        normalized = answers

    counter = Counter(normalized)
    top_answer, top_count = counter.most_common(1)[0]
    consistency = top_count / len(answers)

    return {
        "majority_answer": top_answer,
        "majority_count": top_count,
        "consistency_score": float(consistency),
        "n_unique": len(counter),
        "vote_distribution": dict(counter),
    }


def embedding_consistency(answers: List[str], embedding_model=None) -> dict:
    """
    임베딩 기반 pairwise 코사인 유사도 평균으로 일관성 측정.
    문자열이 달라도 의미가 같으면 높은 점수.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    if embedding_model is None:
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embs = embedding_model.encode(answers, convert_to_tensor=True, normalize_embeddings=True)
    sim_matrix = (embs @ embs.T).cpu().numpy()

    n = len(answers)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    mean_sim = float(upper_tri.mean()) if len(upper_tri) > 0 else 1.0

    return {
        "mean_pairwise_similarity": mean_sim,
        "min_pairwise_similarity": float(upper_tri.min()) if len(upper_tri) > 0 else 1.0,
        "consistency_score": mean_sim,
    }


def compute_consistency_score(
    answers: List[str],
    method: str = "majority",
    embedding_model=None,
) -> dict:
    """
    method: "majority" | "embedding"
    Returns dict with consistency_score in [0, 1].
    """
    if method == "majority":
        return majority_vote(answers)
    elif method == "embedding":
        return embedding_consistency(answers, embedding_model)
    else:
        raise ValueError(f"Unknown method: {method}")


def consistency_to_hallucination_pred(consistency_score: float, threshold: float = 0.5) -> int:
    """
    consistency가 낮으면 hallucination으로 예측.
    Returns 1 (hallucination) if consistency_score < threshold.
    """
    return 1 if consistency_score < threshold else 0


def analyze_consistency_vs_accuracy(
    consistency_scores: List[float],
    is_correct: List[bool],
    n_bins: int = 5,
) -> dict:
    """
    Consistency score 구간별 정확도 분석.
    높은 consistency = 높은 정확도 가설 검증.
    """
    scores = np.array(consistency_scores)
    correct = np.array(is_correct, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracy = []
    bin_count = []

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (scores >= lo) & (scores <= hi)
        bin_accuracy.append(float(correct[mask].mean()) if mask.sum() > 0 else 0.0)
        bin_count.append(int(mask.sum()))

    correlation = float(np.corrcoef(scores, correct)[0, 1]) if len(scores) > 1 else 0.0

    return {
        "correlation_consistency_accuracy": correlation,
        "bin_edges": bin_edges.tolist(),
        "bin_accuracy": bin_accuracy,
        "bin_count": bin_count,
        "overall_accuracy": float(correct.mean()),
        "mean_consistency": float(scores.mean()),
    }

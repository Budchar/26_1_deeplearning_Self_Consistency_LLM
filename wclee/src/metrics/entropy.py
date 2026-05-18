"""
Token Entropy & Semantic Entropy 계산.

- Token Entropy: 각 생성 스텝의 vocab distribution entropy
- Semantic Entropy: 의미적으로 동등한 답변 클러스터 기반 entropy
  (Kuhn et al., 2023 — https://arxiv.org/abs/2302.09664)
"""

import numpy as np
from typing import List, Tuple
from collections import defaultdict


# ─── Token-level Entropy ────────────────────────────────────────────────────

def compute_token_entropy_scores(token_entropies: List[float]) -> dict:
    """GenerationResult.token_entropies 로부터 집계 통계 반환."""
    if not token_entropies:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "last": 0.0}
    arr = np.array(token_entropies)
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "std": float(arr.std()),
        "last": float(arr[-1]),
        "first_10_mean": float(arr[:10].mean()) if len(arr) >= 10 else float(arr.mean()),
    }


def entropy_threshold_predict(token_entropies: List[float], threshold: float) -> bool:
    """
    mean entropy가 threshold를 초과하면 hallucination으로 예측.
    Returns True if predicted hallucination.
    """
    if not token_entropies:
        return False
    return float(np.mean(token_entropies)) > threshold


def find_best_entropy_threshold(
    entropy_scores: List[float],
    labels: List[int],  # 1=hallucination, 0=correct
) -> Tuple[float, float]:
    """
    ROC curve 없이 간단히 F1 최대화 threshold 탐색.
    Returns (best_threshold, best_f1)
    """
    from sklearn.metrics import f1_score
    best_f1, best_thresh = 0.0, 0.0
    for thresh in np.linspace(min(entropy_scores), max(entropy_scores), 100):
        preds = [1 if s > thresh else 0 for s in entropy_scores]
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return best_thresh, best_f1


# ─── Semantic Entropy ────────────────────────────────────────────────────────

def cluster_answers_by_nli(
    answers: List[str],
    nli_model=None,
    threshold: float = 0.8,
) -> List[int]:
    """
    NLI (entailment) 기반으로 답변을 의미적 클러스터로 묶는다.
    같은 의미 = 같은 cluster_id.

    nli_model: sentence_transformers CrossEncoder or transformers pipeline.
    threshold: entailment score >= threshold 이면 같은 클러스터.

    실제 NLI 모델 없을 때는 임베딩 코사인 유사도로 fallback.
    """
    if nli_model is None:
        return _cluster_by_embedding(answers, threshold)
    return _cluster_by_nli(answers, nli_model, threshold)


_EMBED_MODEL_CACHE = {}

def _get_embed_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if model_name not in _EMBED_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBED_MODEL_CACHE[model_name]


def _cluster_by_embedding(answers: List[str], threshold: float = 0.8) -> List[int]:
    """임베딩 코사인 유사도 기반 greedy clustering."""
    import torch
    model = _get_embed_model()
    embeddings = model.encode(answers, convert_to_tensor=True, normalize_embeddings=True)
    sim = (embeddings @ embeddings.T).cpu().numpy()

    cluster_ids = [-1] * len(answers)
    next_id = 0
    for i in range(len(answers)):
        if cluster_ids[i] == -1:
            cluster_ids[i] = next_id
            for j in range(i + 1, len(answers)):
                if cluster_ids[j] == -1 and sim[i, j] >= threshold:
                    cluster_ids[j] = next_id
            next_id += 1
    return cluster_ids


def _cluster_by_nli(answers: List[str], nli_model, threshold: float) -> List[int]:
    """양방향 NLI entailment 기반 clustering."""
    cluster_ids = [-1] * len(answers)
    next_id = 0
    for i in range(len(answers)):
        if cluster_ids[i] == -1:
            cluster_ids[i] = next_id
            for j in range(i + 1, len(answers)):
                if cluster_ids[j] == -1:
                    # 양방향 entailment 확인
                    score_ij = nli_model.predict([(answers[i], answers[j])])
                    score_ji = nli_model.predict([(answers[j], answers[i])])
                    if score_ij >= threshold and score_ji >= threshold:
                        cluster_ids[j] = next_id
            next_id += 1
    return cluster_ids


def compute_semantic_entropy(
    answers: List[str],
    log_probs: List[float],  # 각 답변의 sequence log probability
    cluster_ids: List[int],
) -> float:
    """
    Semantic Entropy = -sum_c [ P(c) * log P(c) ]
    P(c) = sum of exp(log_prob) for answers in cluster c, normalized.

    Kuhn et al. (2023) Eq. (1)
    """
    # cluster별 log probability logsumexp
    cluster_log_probs = defaultdict(list)
    for answer, lp, cid in zip(answers, log_probs, cluster_ids):
        cluster_log_probs[cid].append(lp)

    from scipy.special import logsumexp
    cluster_lse = {}
    all_lp = []
    for cid, lps in cluster_log_probs.items():
        cluster_lse[cid] = logsumexp(lps)
        all_lp.append(cluster_lse[cid])

    # normalize to get P(c)
    total = logsumexp(all_lp)
    entropy = 0.0
    for cid, lse in cluster_lse.items():
        p_c = np.exp(lse - total)
        if p_c > 1e-10:
            entropy -= p_c * np.log(p_c)

    return float(entropy)


def compute_semantic_entropy_from_results(
    generation_results,  # List[GenerationResult]
    nli_model=None,
    threshold: float = 0.8,
) -> dict:
    """편의 함수: GenerationResult 리스트에서 바로 semantic entropy 계산."""
    answers = [r.generated_text for r in generation_results]
    log_probs = [r.sequence_log_prob for r in generation_results]
    cluster_ids = cluster_answers_by_nli(answers, nli_model, threshold)
    se = compute_semantic_entropy(answers, log_probs, cluster_ids)

    n_clusters = len(set(cluster_ids))
    max_se = np.log(n_clusters) if n_clusters > 1 else 0.0

    return {
        "semantic_entropy": se,
        "n_answers": len(answers),
        "n_clusters": n_clusters,
        "normalized_se": se / max_se if max_se > 0 else 0.0,
        "cluster_ids": cluster_ids,
        "unique_answers": list(set(answers)),
    }

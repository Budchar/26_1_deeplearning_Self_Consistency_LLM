"""Semantic Entropy + SC majority vote.

For each question:
  - Cluster N samples using bidirectional NLI entailment (Kuhn et al. 2023).
  - Discrete semantic entropy = entropy of cluster size distribution.
  - SE-logprob = entropy weighted by sample average log-prob (length-normalized).
  - Self-consistency (SC) majority vote over clusters.
  - Correctness label per sample using normalized exact match against gold answers.
  - Greedy correctness using same metric.

Entailment model: microsoft/deberta-v2-xlarge-mnli (HF pipeline `text-classification`).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


import os
_NLI_MODEL_ID = os.environ.get("PHASE1_NLI_MODEL", "microsoft/deberta-v2-xlarge-mnli")


# ----------------- Normalization / EM ---------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.I)
_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize_text(s: str) -> str:
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def first_sentence(s: str) -> str:
    """Trim model output to a short answer-like form: stop at newline / final punctuation."""
    s = s.strip()
    for sep in ["\n", "\r"]:
        if sep in s:
            s = s.split(sep, 1)[0]
    # Trim trailing 'Question:' echoes
    for stop in ["Question:", "question:", "Q:"]:
        if stop in s:
            s = s.split(stop, 1)[0]
    return s.strip()


def is_correct(pred: str, golds: List[str]) -> int:
    p = normalize_text(first_sentence(pred))
    if not p:
        return 0
    for g in golds:
        gn = normalize_text(g)
        if not gn:
            continue
        if p == gn or gn in p:
            return 1
    return 0


# ----------------- Entailment-based clustering -------------------------

class NLIEntailer:
    def __init__(self, model_id: str = _NLI_MODEL_ID, device: Optional[str] = None):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch.float16,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()
        # deberta-v2-xlarge-mnli labels: 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT
        # Fall back to id2label if present.
        id2label = self.model.config.id2label
        self.label_to_id = {v.upper(): k for k, v in id2label.items()}
        self.entail_id = self.label_to_id.get("ENTAILMENT", 2)

    @torch.no_grad()
    def entails_batch(self, premises: List[str], hypotheses: List[str], batch_size: int = 16) -> np.ndarray:
        out: List[float] = []
        for i in range(0, len(premises), batch_size):
            p = premises[i:i + batch_size]
            h = hypotheses[i:i + batch_size]
            enc = self.tok(p, h, padding=True, truncation=True, return_tensors="pt", max_length=256).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            out.extend(probs[:, self.entail_id].tolist())
        return np.array(out, dtype=np.float32)


def cluster_samples(
    question: str,
    samples: List[str],
    entailer: NLIEntailer,
    threshold: float = 0.5,
) -> List[int]:
    """Greedy bidirectional-entailment clustering (Kuhn et al. 2023, Algorithm 1).

    Returns: list of cluster ids, length = len(samples).
    """
    cluster_ids: List[int] = [-1] * len(samples)
    # Unique-ish reps to speed: cluster by first-sentence trimmed text.
    trimmed = [first_sentence(s) or "(empty)" for s in samples]
    # Build all pair queries lazily using a representative-per-cluster loop.
    representatives: List[int] = []  # indices of samples that are representatives
    pending: List[Tuple[int, int]] = []  # (sample_i, rep_i) queries to score

    for i, txt in enumerate(trimmed):
        # Try to attach to existing cluster
        attached = False
        for rid in representatives:
            rep_txt = trimmed[rid]
            # Score both directions on prompt-conditioned strings.
            ctx = f"Question: {question}\nAnswer: "
            p1 = ctx + rep_txt
            h1 = ctx + txt
            p2 = ctx + txt
            h2 = ctx + rep_txt
            e_fwd = entailer.entails_batch([p1], [h1])[0]
            e_bwd = entailer.entails_batch([p2], [h2])[0]
            if e_fwd >= threshold and e_bwd >= threshold:
                cluster_ids[i] = cluster_ids[rid]
                attached = True
                break
        if not attached:
            cluster_ids[i] = (max(cluster_ids) + 1) if max(cluster_ids) >= 0 else 0
            representatives.append(i)
    return cluster_ids


# ----------------- SE / SC --------------------------------------------

def discrete_entropy(cluster_ids: List[int]) -> float:
    counts = Counter(cluster_ids)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            H -= p * math.log(p)
    return H


def logprob_weighted_entropy(cluster_ids: List[int], logprobs: List[float]) -> float:
    """Approximation to Eq.(4) of Kuhn et al.: sum over clusters of -p_c log p_c
    where p_c = sum_{i in c} exp(logprob_i) / Z."""
    weights = np.exp(np.asarray(logprobs, dtype=np.float64))
    if weights.sum() <= 0 or not np.isfinite(weights).all():
        return discrete_entropy(cluster_ids)
    cluster_weight: Dict[int, float] = defaultdict(float)
    for cid, w in zip(cluster_ids, weights):
        cluster_weight[cid] += float(w)
    Z = sum(cluster_weight.values())
    H = 0.0
    for w in cluster_weight.values():
        p = w / Z
        if p > 0:
            H -= p * math.log(p)
    return H


def sc_majority(cluster_ids: List[int], samples: List[str]) -> str:
    """Pick the cluster with most members; within it, return the modal sample text."""
    counts = Counter(cluster_ids)
    if not counts:
        return ""
    top_cid, _ = counts.most_common(1)[0]
    members = [samples[i] for i, cid in enumerate(cluster_ids) if cid == top_cid]
    text_counts = Counter([first_sentence(m) for m in members])
    return text_counts.most_common(1)[0][0]


# ----------------- Pipeline over a generations.jsonl ------------------

def process_jsonl(
    in_path: Path,
    out_path: Path,
    entailer: Optional[NLIEntailer] = None,
    threshold: float = 0.5,
    limit: Optional[int] = None,
) -> Dict:
    if entailer is None:
        entailer = NLIEntailer()
    in_records: List[Dict] = []
    with open(in_path) as f:
        for line in f:
            in_records.append(json.loads(line))
    if limit is not None:
        in_records = in_records[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_correct_greedy = 0
    n_correct_sc = 0
    se_vals: List[float] = []
    se_lp_vals: List[float] = []
    n_clusters_vals: List[int] = []
    with open(out_path, "w") as fout:
        for rec in tqdm(in_records, desc=f"se {in_path.parent.name}"):
            samples = rec["samples"]
            logprobs = rec.get("sample_logprobs", [0.0] * len(samples))
            cids = cluster_samples(rec["question"], samples, entailer, threshold=threshold)
            se = discrete_entropy(cids)
            se_lp = logprob_weighted_entropy(cids, logprobs)
            sc_pred = sc_majority(cids, samples)
            greedy_correct = is_correct(rec["greedy"], rec["answers"])
            sc_correct = is_correct(sc_pred, rec["answers"])
            sample_correct = [is_correct(s, rec["answers"]) for s in samples]
            out = {
                "id": rec["id"],
                "dataset": rec["dataset"],
                "question": rec["question"],
                "answers": rec["answers"],
                "greedy": rec["greedy"],
                "greedy_correct": greedy_correct,
                "samples": samples,
                "sample_correct": sample_correct,
                "sample_logprobs": logprobs,
                "cluster_ids": cids,
                "n_clusters": len(set(cids)),
                "se_discrete": se,
                "se_logprob": se_lp,
                "sc_pred": sc_pred,
                "sc_correct": sc_correct,
                "hidden_path": rec.get("hidden_path", ""),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_correct_greedy += greedy_correct
            n_correct_sc += sc_correct
            se_vals.append(se)
            se_lp_vals.append(se_lp)
            n_clusters_vals.append(len(set(cids)))
    n = len(in_records)
    summary = {
        "n": n,
        "greedy_acc": n_correct_greedy / max(n, 1),
        "sc_acc": n_correct_sc / max(n, 1),
        "se_discrete_mean": float(np.mean(se_vals)) if se_vals else 0.0,
        "se_logprob_mean": float(np.mean(se_lp_vals)) if se_lp_vals else 0.0,
        "n_clusters_mean": float(np.mean(n_clusters_vals)) if n_clusters_vals else 0.0,
        "out": str(out_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    process_jsonl(Path(args.in_path), Path(args.out_path),
                  threshold=args.threshold, limit=args.limit)


if __name__ == "__main__":
    main()

"""Dataset loaders for Phase 1: Semantic Entropy + SEPs.

Loads validation splits from TriviaQA (rc.nocontext), Natural Questions (nq_open),
and SQuAD (no context). Each loader returns a list of dicts with stable schema:
    {
        "id": str,
        "question": str,
        "answers": list[str],   # set of acceptable gold answers
        "dataset": str,
    }

Datasets are cached to ~/experiments/dl_team_v2/_data/cache/ as JSON so that
subsequent runs are fast and deterministic.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Iterable

DATA_CACHE = Path(os.path.expanduser("~/experiments/dl_team_v2/_data/cache"))
DATA_CACHE.mkdir(parents=True, exist_ok=True)


def _stable_subsample(records: List[Dict], n: int, seed: int) -> List[Dict]:
    """Deterministic subsample without replacement."""
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    return [records[i] for i in idx[:n]]


def _save_cache(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, ensure_ascii=False)


def _load_cache(path: Path) -> List[Dict] | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_triviaqa(n: int = 1000, seed: int = 42) -> List[Dict]:
    cache = DATA_CACHE / f"triviaqa_n{n}_s{seed}.json"
    cached = _load_cache(cache)
    if cached is not None:
        return cached
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    records: List[Dict] = []
    for ex in ds:
        ans = ex.get("answer", {})
        gold: List[str] = []
        if isinstance(ans, dict):
            for k in ("aliases", "normalized_aliases"):
                v = ans.get(k) or []
                gold.extend([a for a in v if isinstance(a, str)])
            for k in ("value", "normalized_value"):
                v = ans.get(k)
                if isinstance(v, str):
                    gold.append(v)
        gold = list({g.strip() for g in gold if g and g.strip()})
        if not gold:
            continue
        records.append({
            "id": ex["question_id"],
            "question": ex["question"],
            "answers": gold,
            "dataset": "triviaqa",
        })
    out = _stable_subsample(records, n, seed)
    _save_cache(cache, out)
    return out


def load_nq_open(n: int = 1000, seed: int = 42) -> List[Dict]:
    cache = DATA_CACHE / f"nq_open_n{n}_s{seed}.json"
    cached = _load_cache(cache)
    if cached is not None:
        return cached
    from datasets import load_dataset
    # Try canonical first, fall back to mirror.
    last_err = None
    ds = None
    for name in ("google-research-datasets/nq_open", "nq_open"):
        try:
            ds = load_dataset(name, split="validation")
            break
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(f"Failed to load nq_open: {last_err}")
    records: List[Dict] = []
    for i, ex in enumerate(ds):
        gold = ex.get("answer") or []
        if isinstance(gold, str):
            gold = [gold]
        gold = list({g.strip() for g in gold if g and g.strip()})
        if not gold:
            continue
        records.append({
            "id": f"nq_{i}",
            "question": ex["question"],
            "answers": gold,
            "dataset": "nq_open",
        })
    out = _stable_subsample(records, n, seed)
    _save_cache(cache, out)
    return out


def load_squad(n: int = 1000, seed: int = 42) -> List[Dict]:
    cache = DATA_CACHE / f"squad_n{n}_s{seed}.json"
    cached = _load_cache(cache)
    if cached is not None:
        return cached
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")
    records: List[Dict] = []
    for ex in ds:
        gold = ex["answers"].get("text", []) if isinstance(ex.get("answers"), dict) else []
        gold = list({g.strip() for g in gold if g and g.strip()})
        if not gold:
            continue
        # NOTE: SQuAD context is intentionally dropped to make it a closed-book QA task.
        records.append({
            "id": ex["id"],
            "question": ex["question"],
            "answers": gold,
            "dataset": "squad",
        })
    out = _stable_subsample(records, n, seed)
    _save_cache(cache, out)
    return out


LOADERS = {
    "triviaqa": load_triviaqa,
    "nq_open": load_nq_open,
    "squad": load_squad,
}


def load_dataset_by_name(name: str, n: int = 1000, seed: int = 42) -> List[Dict]:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(LOADERS)}")
    return LOADERS[name](n=n, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(LOADERS.keys()))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for name in args.datasets:
        recs = load_dataset_by_name(name, n=args.n, seed=args.seed)
        print(f"[data] {name}: {len(recs)} examples cached", flush=True)
        if recs:
            print(f"  example: {recs[0]['question'][:80]!r} -> {recs[0]['answers'][:2]}", flush=True)


if __name__ == "__main__":
    main()

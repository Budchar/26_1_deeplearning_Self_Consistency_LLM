"""Pile / C4 / OpenWebText 1B token shards 준비 (Option 3 from-scratch 학습용).

같은 GPT-2 BPE 토크나이저로 각 dataset에서 1B 토큰을 streaming + tokenize → uint16 .bin 저장.
OpenWebText는 Phase 2 _data/owt_shards/train.bin 재사용 (이미 있음).

출력:
  /home/kcai/experiments/dl_team_v2/_data/pile_shards/train.bin (2GB)
  /home/kcai/experiments/dl_team_v2/_data/c4_shards/train.bin (2GB)
  + val.bin (10M tokens) 각각

소요: 각 dataset ~1-2h (streaming + tokenization 병목).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/home/kcai/experiments/dl_team_v2/_data")
N_TRAIN_TOKENS = 1_000_000_000  # 1B
N_VAL_TOKENS = 10_000_000  # 10M


def build_shard(dataset_name: str, dataset_config: str | None,
                split_used: str, out_path: Path, n_tokens: int):
    if out_path.exists() and out_path.stat().st_size >= n_tokens * 2:
        print(f"[skip] {out_path} already exists ({out_path.stat().st_size/1e9:.2f}GB)")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    print(f"[build] {dataset_name} ({dataset_config or 'default'}) → {out_path} ({n_tokens/1e6:.0f}M tokens)")
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=split_used, streaming=True, trust_remote_code=True)
    else:
        ds = load_dataset(dataset_name, split=split_used, streaming=True, trust_remote_code=True)

    buf = np.empty(n_tokens, dtype=np.uint16)
    pos = 0
    n_docs = 0
    t0 = time.time()
    for ex in ds:
        text = ex.get("text") or ex.get("content") or ex.get("article") or ""
        if not text:
            continue
        ids = enc.encode_ordinary(text)
        ids.append(eot)
        a = np.asarray(ids, dtype=np.uint32).astype(np.uint16)
        room = n_tokens - pos
        if room <= 0:
            break
        n = min(len(a), room)
        buf[pos: pos + n] = a[:n]
        pos += n
        n_docs += 1
        if n_docs % 10000 == 0:
            print(f"  {n_docs} docs, {pos/1e6:.1f}M / {n_tokens/1e6:.0f}M  ({time.time()-t0:.0f}s)", flush=True)

    if pos < n_tokens:
        buf[pos:] = eot
    buf.tofile(out_path)
    print(f"[done] {out_path} ({pos/1e6:.1f}M usable tokens, {time.time()-t0:.0f}s)")


def main():
    # HF token
    tok_file = Path.home() / "hf_token.txt"
    if tok_file.exists():
        os.environ["HF_TOKEN"] = tok_file.read_text().strip()

    # 1. Pile (deduplicated — Pythia 학습 데이터와 같은 출처)
    print("\n=== Pile shard ===")
    build_shard(
        dataset_name="EleutherAI/the_pile_deduplicated",
        dataset_config=None,
        split_used="train",
        out_path=DATA_ROOT / "pile_shards" / "train.bin",
        n_tokens=N_TRAIN_TOKENS,
    )
    build_shard(
        dataset_name="EleutherAI/the_pile_deduplicated",
        dataset_config=None,
        split_used="train",
        out_path=DATA_ROOT / "pile_shards" / "val.bin",
        n_tokens=N_VAL_TOKENS,
    )

    # 2. C4 (en)
    print("\n=== C4 shard ===")
    try:
        build_shard(
            dataset_name="allenai/c4",
            dataset_config="en",
            split_used="train",
            out_path=DATA_ROOT / "c4_shards" / "train.bin",
            n_tokens=N_TRAIN_TOKENS,
        )
        build_shard(
            dataset_name="allenai/c4",
            dataset_config="en",
            split_used="train",
            out_path=DATA_ROOT / "c4_shards" / "val.bin",
            n_tokens=N_VAL_TOKENS,
        )
    except Exception as e:
        print(f"C4 failed: {e}")

    print("\n=== Final state ===")
    for d in ["owt_shards", "pile_shards", "c4_shards"]:
        for f in ["train.bin", "val.bin"]:
            p = DATA_ROOT / d / f
            if p.exists():
                print(f"  {p}: {p.stat().st_size/1e9:.2f}GB")
            else:
                print(f"  {p}: MISSING")


if __name__ == "__main__":
    main()

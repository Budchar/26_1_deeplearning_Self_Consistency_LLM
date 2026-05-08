"""HF 캐시에 모델·토크나이저 미리 다운로드. CPU/네트워크만 사용 — Phase 2 GPU 영향 0.

대상:
  - Pythia 7개 (Sweep B용): 70m/160m/410m/1b/1.4b/2.8b/6.9b
  - Sweep A base 5개: Pythia-1.4b(중복), Llama-3.2-1B(base), Qwen2.5-1.5B(base), OPT-1.3B, GPT-Neo-1.3B
  - NLI 모델 (이미 캐시 있음 — 검증만)

병렬 3개 동시 다운로드 (네트워크 병목 회피). 이미 캐시된 모델은 즉시 skip.
HF_TOKEN 자동 로드 (~/hf_token.txt).
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load HF token if file exists
TOK_FILE = Path.home() / "hf_token.txt"
if TOK_FILE.exists():
    os.environ["HF_TOKEN"] = TOK_FILE.read_text().strip()

from huggingface_hub import snapshot_download  # noqa: E402


PYTHIA_SIZES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
PYTHIA_MODELS = [f"EleutherAI/pythia-{sz}-deduped" for sz in PYTHIA_SIZES]

SWEEP_A_BASE = [
    # pythia-1.4b는 PYTHIA_MODELS에 이미 있음
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B",
    "facebook/opt-1.3b",
    "EleutherAI/gpt-neo-1.3B",
]

NLI_MODEL = "microsoft/deberta-v2-xlarge-mnli"

ALL_MODELS = list(dict.fromkeys(PYTHIA_MODELS + SWEEP_A_BASE + [NLI_MODEL]))


def is_cached(model_id: str) -> bool:
    """Heuristic: refs/main file present in HF cache."""
    cache = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    safe = model_id.replace("/", "--")
    refs = cache / "hub" / f"models--{safe}" / "refs" / "main"
    if not refs.exists():
        return False
    sha = refs.read_text().strip()
    snap = cache / "hub" / f"models--{safe}" / "snapshots" / sha
    if not snap.exists():
        return False
    # Check at least the config + a weight file are present (not symlink-only stub)
    has_config = (snap / "config.json").exists()
    has_weights = any(snap.glob("*.safetensors")) or any(snap.glob("*.bin"))
    return has_config and has_weights


def download_one(model_id: str) -> tuple[str, str, float]:
    t0 = time.time()
    if is_cached(model_id):
        return model_id, "cached_skip", time.time() - t0
    try:
        snapshot_download(
            repo_id=model_id,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.bin",
                "tokenizer*",
                "*.txt",
                "*.model",
                "*.tiktoken",
            ],
            ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*flax*"],
        )
        return model_id, "downloaded", time.time() - t0
    except Exception as e:
        return model_id, f"FAIL: {e}", time.time() - t0


def main():
    print(f"[prefetch] {len(ALL_MODELS)} models, max 3 concurrent")
    print(f"[prefetch] HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'not set'}")
    print()
    for m in ALL_MODELS:
        print(f"  - {m}  ({'cached' if is_cached(m) else 'pending'})")
    print()

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(download_one, m): m for m in ALL_MODELS}
        for fut in as_completed(futs):
            m, status, dt = fut.result()
            results.append((m, status, dt))
            print(f"[{time.strftime('%H:%M:%S')}] {m}  →  {status}  ({dt:.1f}s)", flush=True)

    print()
    print(f"[prefetch] total wall: {(time.time()-t0)/60:.1f} min")
    print(f"[prefetch] {sum(1 for _,s,_ in results if 'FAIL' in s)} failures, "
          f"{sum(1 for _,s,_ in results if s == 'downloaded')} downloaded, "
          f"{sum(1 for _,s,_ in results if s == 'cached_skip')} cached_skip")


if __name__ == "__main__":
    main()

"""Mistral 8변형 중 핵심 3개 다운로드. Resumable (HF cache).

실행:
  python download_mistral.py 2>&1 | tee /tmp/mistral_download.log
"""
import sys
import time
from huggingface_hub import snapshot_download

REPOS = [
    "mistralai/Mistral-7B-v0.1",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
]


def main():
    for i, repo in enumerate(REPOS, 1):
        print(f"\n[{i}/{len(REPOS)}] {repo}", flush=True)
        t0 = time.time()
        try:
            path = snapshot_download(
                repo_id=repo,
                repo_type="model",
                allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt"],
                max_workers=4,
            )
            print(f"  ✅ {time.time() - t0:.0f}s → {path}", flush=True)
        except Exception as e:
            print(f"  ❌ {e}", flush=True)
            sys.exit(1)
    print("\n=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()

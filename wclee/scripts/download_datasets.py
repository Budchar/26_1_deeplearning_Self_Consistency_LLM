"""
실험에 필요한 데이터셋을 HuggingFace에서 캐시 다운로드.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --datasets triviaqa truthfulqa
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))


def download_triviaqa():
    print("[download] TriviaQA (rc.nocontext)...")
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", cache_dir=str(DATA_DIR))
    print(f"  validation: {len(ds['validation'])} / train: {len(ds['train'])}")
    return True


def download_truthfulqa():
    print("[download] TruthfulQA (generation)...")
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation", cache_dir=str(DATA_DIR))
    print(f"  validation: {len(ds['validation'])}")
    return True


def download_halueval():
    print("[download] HaluEval (qa / summarization / dialogue)...")
    from datasets import load_dataset
    for task in ["qa_samples", "summarization_samples", "dialogue_samples"]:
        try:
            ds = load_dataset("pminervini/HaluEval", task, cache_dir=str(DATA_DIR))
            split = list(ds.keys())[0]
            print(f"  {task}: {len(ds[split])}")
        except Exception as e:
            print(f"  {task} 실패: {e}")
    return True


DATASET_FNS = {
    "triviaqa": download_triviaqa,
    "truthfulqa": download_truthfulqa,
    "halueval": download_halueval,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASET_FNS.keys()))
    args = p.parse_args()

    print(f"[download] 저장 위치: {DATA_DIR}\n")
    for name in args.datasets:
        if name not in DATASET_FNS:
            print(f"[download] 알 수 없는 데이터셋: {name}")
            continue
        try:
            DATASET_FNS[name]()
            print(f"  -> {name} 완료\n")
        except Exception as e:
            print(f"  -> {name} 실패: {e}\n")

    print("[download] 모든 다운로드 완료!")


if __name__ == "__main__":
    main()

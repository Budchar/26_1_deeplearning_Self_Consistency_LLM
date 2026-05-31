"""
Experiment 08: Cross-model Comparison
동급 크기(7-8B)의 다른 아키텍처 모델들 비교.
EXAONE / Llama / Qwen / Mistral

Usage:
    python experiments/08_cross_model/run.py --n_samples 200
    python experiments/08_cross_model/run.py --models exaone qwen_7b llama_8b --n_samples 200
"""

import sys
import json
import argparse
import datetime
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Exp07와 동일한 run_single_model 재사용
sys.path.insert(0, str(ROOT / "experiments" / "07_model_scaling"))
from run import run_single_model
from src.datasets.loader import load_dataset_by_name

DEFAULT_MODELS = ["exaone", "qwen_7b", "llama_8b", "mistral"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--dataset", default="triviaqa")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--n_gen_se", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=str(ROOT / "results" / "raw"))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n[Exp08] Cross-model | models={args.models} | n={args.n_samples}")

    samples = load_dataset_by_name(args.dataset, n_samples=args.n_samples, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    all_summaries = []
    for model_key in args.models:
        try:
            summary = run_single_model(model_key, samples, n_gen_se=args.n_gen_se)
            all_summaries.append(summary)
        except Exception as e:
            print(f"  [{model_key}] 실패: {e}")
            import traceback; traceback.print_exc()

    out_path = output_dir / f"08_cross_model_{args.dataset}_{ts}.json"
    save_data = [{k: v for k, v in s.items() if k != "records"} for s in all_summaries]
    with open(out_path, "w") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n[Exp08] Saved: {out_path}")

    # Quick summary
    print(f"\n{'Model':<35} {'Acc':>6} {'EntAUROC':>9} {'SE_AUROC':>9} {'NC_AUROC':>9}")
    print("-" * 75)
    for s in sorted(all_summaries, key=lambda x: -x["auroc_n_clusters"]):
        print(f"  {s['model_name']:<33} {s['accuracy']:>6.3f} "
              f"{s['auroc_mean_entropy']:>9.4f} {s['auroc_semantic_entropy']:>9.4f} "
              f"{s['auroc_n_clusters']:>9.4f}")

    return out_path


if __name__ == "__main__":
    main()

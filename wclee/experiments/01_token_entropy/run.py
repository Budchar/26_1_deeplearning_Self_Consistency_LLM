"""
Experiment 01: Token Entropy
각 토큰 생성 시점의 entropy로 hallucination을 사전 탐지.

Usage:
    python experiments/01_token_entropy/run.py --model exaone --dataset triviaqa --n_samples 200
"""

import sys
import json
import argparse
import datetime
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.model_loader import load_model
from src.generation import generate_with_scores
from src.datasets.loader import load_dataset_by_name, check_correctness
from src.metrics.entropy import compute_token_entropy_scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="exaone", help="모델 키 (exaone/llama/qwen/mistral)")
    p.add_argument("--dataset", default="triviaqa", help="데이터셋 (triviaqa/truthfulqa)")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=str(ROOT / "results" / "raw"))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n[Exp01] Token Entropy | model={args.model} | dataset={args.dataset} | n={args.n_samples}")

    wrapper = load_model(args.model)
    samples = load_dataset_by_name(args.dataset, n_samples=args.n_samples, seed=args.seed)
    print(f"[Exp01] Loaded {len(samples)} samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"01_token_entropy_{args.model}_{args.dataset}_{ts}.jsonl"

    results = []
    with open(out_path, "w") as f:
        for sample in tqdm(samples, desc="Generating"):
            gen = generate_with_scores(
                wrapper,
                sample["question"],
                temperature=args.temperature,
            )
            is_correct = check_correctness(gen.generated_text, sample["answers"])
            entropy_stats = compute_token_entropy_scores(gen.token_entropies)

            record = {
                "experiment": "token_entropy",
                "model": args.model,
                "dataset": args.dataset,
                "question": sample["question"],
                "gold_answers": sample["answers"],
                "generated_text": gen.generated_text,
                "is_correct": is_correct,
                # 핵심 지표
                "mean_entropy": entropy_stats["mean"],
                "max_entropy": entropy_stats["max"],
                "min_entropy": entropy_stats["min"],
                "std_entropy": entropy_stats["std"],
                "last_token_entropy": entropy_stats["last"],
                "first_10_mean_entropy": entropy_stats["first_10_mean"],
                "sequence_log_prob": gen.sequence_log_prob,
                # 상세 (분석용)
                "token_entropies": gen.token_entropies,
                "token_log_probs": gen.token_log_probs,
                "token_texts": gen.token_texts,
                "n_tokens": len(gen.token_ids),
            }
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 간단한 통계 출력
    n_correct = sum(r["is_correct"] for r in results)
    correct_entropy = [r["mean_entropy"] for r in results if r["is_correct"]]
    wrong_entropy = [r["mean_entropy"] for r in results if not r["is_correct"]]

    print(f"\n[Exp01] Results saved to {out_path}")
    print(f"  Accuracy       : {n_correct}/{len(results)} = {n_correct/len(results):.3f}")
    if correct_entropy:
        import numpy as np
        print(f"  Entropy (correct): {np.mean(correct_entropy):.4f}")
    if wrong_entropy:
        import numpy as np
        print(f"  Entropy (wrong)  : {np.mean(wrong_entropy):.4f}")

    wrapper.unload()
    return out_path


if __name__ == "__main__":
    main()

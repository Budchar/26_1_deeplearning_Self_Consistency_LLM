"""
Experiment 03: Self-Consistency
다수결 일관성 점수를 hallucination confidence proxy로 사용.
Wang et al. (2023).

Usage:
    python experiments/03_self_consistency/run.py --model exaone --dataset triviaqa --n_samples 200 --n_gen 10
"""

import sys
import json
import argparse
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.model_loader import load_model
from src.generation import generate_multiple
from src.datasets.loader import load_dataset_by_name, check_correctness
from src.metrics.consistency import compute_consistency_score, analyze_consistency_vs_accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="exaone")
    p.add_argument("--dataset", default="triviaqa")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--n_gen", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=str(ROOT / "results" / "raw"))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n[Exp03] Self-Consistency | model={args.model} | n_gen={args.n_gen}")

    wrapper = load_model(args.model)
    samples = load_dataset_by_name(args.dataset, n_samples=args.n_samples, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"03_self_consistency_{args.model}_{args.dataset}_{ts}.jsonl"

    results = []
    with open(out_path, "w") as f:
        for sample in tqdm(samples, desc="Generating"):
            gens = generate_multiple(
                wrapper, sample["question"],
                n=args.n_gen, temperature=args.temperature,
            )
            answers = [g.generated_text for g in gens]

            majority_result = compute_consistency_score(answers, method="majority")
            embedding_result = compute_consistency_score(answers, method="embedding")

            majority_answer = majority_result["majority_answer"]
            is_correct = check_correctness(majority_answer, sample["answers"])
            individual_correct = [check_correctness(a, sample["answers"]) for a in answers]

            record = {
                "experiment": "self_consistency",
                "model": args.model,
                "dataset": args.dataset,
                "question": sample["question"],
                "gold_answers": sample["answers"],
                "majority_answer": majority_answer,
                "is_correct": is_correct,
                "any_correct": any(individual_correct),
                "individual_correct_rate": float(np.mean(individual_correct)),
                # 핵심 지표
                "majority_consistency": majority_result["consistency_score"],
                "n_unique_answers": majority_result["n_unique"],
                "embedding_consistency": embedding_result["consistency_score"],
                "mean_pairwise_sim": embedding_result["mean_pairwise_similarity"],
                # 보조
                "all_answers": answers,
                "vote_distribution": majority_result["vote_distribution"],
            }
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 간단 통계
    is_correct_list = [r["is_correct"] for r in results]
    maj_scores = [r["majority_consistency"] for r in results]
    emb_scores = [r["embedding_consistency"] for r in results]

    analysis = analyze_consistency_vs_accuracy(maj_scores, is_correct_list)

    print(f"\n[Exp03] Saved to {out_path}")
    print(f"  Accuracy: {np.mean(is_correct_list):.3f}")
    print(f"  Correlation (majority_consistency vs accuracy): {analysis['correlation_consistency_accuracy']:.4f}")
    print(f"  Mean majority consistency (correct): {np.mean([s for s,c in zip(maj_scores, is_correct_list) if c]):.3f}")
    print(f"  Mean majority consistency (wrong):   {np.mean([s for s,c in zip(maj_scores, is_correct_list) if not c]):.3f}")

    wrapper.unload()
    return out_path


if __name__ == "__main__":
    main()

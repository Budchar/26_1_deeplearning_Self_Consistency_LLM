"""
Experiment 02: Semantic Entropy
같은 질문을 여러 번 생성 → 의미적 클러스터 entropy로 hallucination 탐지.
Kuhn et al. (2023) 방법론 구현.

Usage:
    python experiments/02_semantic_entropy/run.py --model exaone --dataset triviaqa --n_samples 100 --n_gen 10
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
from src.metrics.entropy import compute_semantic_entropy_from_results, compute_token_entropy_scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="exaone")
    p.add_argument("--dataset", default="triviaqa")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--n_gen", type=int, default=10, help="질문당 생성 횟수")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--similarity_threshold", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=str(ROOT / "results" / "raw"))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n[Exp02] Semantic Entropy | model={args.model} | n_gen={args.n_gen}")

    wrapper = load_model(args.model)
    samples = load_dataset_by_name(args.dataset, n_samples=args.n_samples, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"02_semantic_entropy_{args.model}_{args.dataset}_{ts}.jsonl"

    results = []
    with open(out_path, "w") as f:
        for sample in tqdm(samples, desc="Generating (x{args.n_gen})"):
            # n_gen번 생성
            gens = generate_multiple(
                wrapper,
                sample["question"],
                n=args.n_gen,
                temperature=args.temperature,
            )

            # Semantic Entropy 계산
            se_result = compute_semantic_entropy_from_results(
                gens,
                threshold=args.similarity_threshold,
            )

            # 다수결 답변으로 정답 판단
            from collections import Counter
            answer_counts = Counter(g.generated_text.strip().lower() for g in gens)
            majority_answer = answer_counts.most_common(1)[0][0]
            # 원본 텍스트로 복원
            majority_text = next(g.generated_text for g in gens
                                 if g.generated_text.strip().lower() == majority_answer)
            is_correct = check_correctness(majority_text, sample["answers"])

            # 개별 생성들의 정답률도 기록
            individual_correct = [check_correctness(g.generated_text, sample["answers"]) for g in gens]
            any_correct = any(individual_correct)

            # 평균 token entropy도 함께 기록
            mean_token_entropy = np.mean([g.mean_entropy for g in gens])

            record = {
                "experiment": "semantic_entropy",
                "model": args.model,
                "dataset": args.dataset,
                "question": sample["question"],
                "gold_answers": sample["answers"],
                "majority_answer": majority_text,
                "is_correct": is_correct,
                "any_correct": any_correct,
                "individual_correct_rate": float(np.mean(individual_correct)),
                # 핵심 지표
                "semantic_entropy": se_result["semantic_entropy"],
                "normalized_se": se_result["normalized_se"],
                "n_clusters": se_result["n_clusters"],
                "n_unique_answers": len(se_result["unique_answers"]),
                # 보조 지표
                "mean_token_entropy": float(mean_token_entropy),
                "mean_seq_log_prob": float(np.mean([g.sequence_log_prob for g in gens])),
                # 모든 생성 텍스트
                "all_generated": [g.generated_text for g in gens],
                "unique_answers": se_result["unique_answers"],
                "cluster_ids": se_result["cluster_ids"],
            }
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_correct = sum(r["is_correct"] for r in results)
    se_correct = [r["semantic_entropy"] for r in results if r["is_correct"]]
    se_wrong = [r["semantic_entropy"] for r in results if not r["is_correct"]]

    print(f"\n[Exp02] Saved to {out_path}")
    print(f"  Accuracy: {n_correct}/{len(results)} = {n_correct/len(results):.3f}")
    if se_correct:
        print(f"  SE (correct): {np.mean(se_correct):.4f}")
    if se_wrong:
        print(f"  SE (wrong):   {np.mean(se_wrong):.4f}")

    wrapper.unload()
    return out_path


if __name__ == "__main__":
    main()

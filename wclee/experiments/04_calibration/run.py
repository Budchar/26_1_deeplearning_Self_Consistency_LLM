"""
Experiment 04: Calibration (ECE + Reliability Diagram)
sequence log probability → confidence로 변환 후 ECE 측정.
Overconfident wrong answers 탐지.

Usage:
    python experiments/04_calibration/run.py --model exaone --dataset triviaqa --n_samples 500
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
from src.generation import greedy_generate
from src.datasets.loader import load_dataset_by_name, check_correctness
from src.metrics.calibration import (
    compute_ece, detect_overconfident,
    plot_reliability_diagram, plot_confidence_distribution,
    normalize_log_prob_to_confidence,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="exaone")
    p.add_argument("--dataset", default="triviaqa")
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--overconf_threshold", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=str(ROOT / "results" / "raw"))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n[Exp04] Calibration | model={args.model} | dataset={args.dataset}")

    wrapper = load_model(args.model)
    samples = load_dataset_by_name(args.dataset, n_samples=args.n_samples, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"04_calibration_{args.model}_{args.dataset}_{ts}.jsonl"

    raw_log_probs = []
    is_correct_list = []
    results = []

    with open(out_path, "w") as f:
        for sample in tqdm(samples, desc="Generating (greedy)"):
            gen = greedy_generate(wrapper, sample["question"])
            is_correct = check_correctness(gen.generated_text, sample["answers"])

            raw_log_probs.append(gen.sequence_log_prob)
            is_correct_list.append(is_correct)

            record = {
                "experiment": "calibration",
                "model": args.model,
                "dataset": args.dataset,
                "question": sample["question"],
                "gold_answers": sample["answers"],
                "generated_text": gen.generated_text,
                "is_correct": is_correct,
                "sequence_log_prob": gen.sequence_log_prob,
                "mean_token_prob": float(np.exp(np.mean(gen.token_log_probs))) if gen.token_log_probs else 0.0,
                "mean_entropy": gen.mean_entropy,
                "n_tokens": len(gen.token_ids),
            }
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # confidence 정규화
    confidences = normalize_log_prob_to_confidence(raw_log_probs)

    # ECE 계산
    ece_result = compute_ece(confidences, is_correct_list, n_bins=args.n_bins)
    overconf_result = detect_overconfident(confidences, is_correct_list, threshold=args.overconf_threshold)

    print(f"\n[Exp04] Results saved to {out_path}")
    print(f"  Accuracy       : {ece_result['overall_accuracy']:.3f}")
    print(f"  Mean Confidence: {ece_result['mean_confidence']:.3f}")
    print(f"  ECE            : {ece_result['ece']:.4f}")
    print(f"  MCE            : {ece_result['mce']:.4f}")
    print(f"  Overconfident (conf>={args.overconf_threshold} & wrong): "
          f"{overconf_result['n_overconfident']}/{overconf_result['n_high_conf']} "
          f"({overconf_result['overconfident_rate']:.1%})")

    # Reliability Diagram
    model_name = f"{args.model} / {args.dataset}"
    plot_reliability_diagram(
        ece_result,
        model_name=model_name,
        save_path=str(fig_dir / f"04_reliability_{args.model}_{args.dataset}_{ts}.png"),
    )
    plot_confidence_distribution(
        confidences, is_correct_list,
        model_name=model_name,
        save_path=str(fig_dir / f"04_conf_dist_{args.model}_{args.dataset}_{ts}.png"),
    )

    # ECE 결과를 별도 json으로도 저장
    summary_path = output_dir / f"04_calibration_summary_{args.model}_{args.dataset}_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "dataset": args.dataset,
            "n_samples": len(results),
            "ece": ece_result,
            "overconfidence": overconf_result,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Summary saved: {summary_path}")

    wrapper.unload()
    return out_path


if __name__ == "__main__":
    main()

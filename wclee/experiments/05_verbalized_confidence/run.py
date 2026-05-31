"""
Experiment 05: Verbalized Confidence
모델에게 직접 "몇 % 확신하냐"고 물어보고 실제 정확도와 비교.
Kadavath et al. (2022): "Language Models (Mostly) Know What They Know"

Usage:
    python experiments/05_verbalized_confidence/run.py --model exaone --dataset triviaqa --n_samples 300
"""

import sys
import json
import re
import argparse
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.model_loader import load_model
from src.generation import generate_with_scores, greedy_generate
from src.datasets.loader import load_dataset_by_name, check_correctness
from src.metrics.calibration import compute_ece, plot_reliability_diagram


CONFIDENCE_PROMPT_TEMPLATE = """\
Question: {question}

My answer: {answer}

How confident are you that the above answer is correct? \
Please respond with ONLY a number between 0 and 100 representing your confidence percentage. \
Do not include any explanation."""


def extract_confidence(text: str) -> float | None:
    """텍스트에서 0-100 숫자 추출."""
    text = text.strip()
    patterns = [
        r"^(\d{1,3})%?$",
        r"(\d{1,3})%",
        r"confidence[:\s]+(\d{1,3})",
        r"(\d{1,3})\s*(?:percent|%|out of 100)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val / 100.0
    # 마지막: 첫 번째 숫자 추출
    nums = re.findall(r"\d+", text)
    if nums:
        val = int(nums[0])
        if 0 <= val <= 100:
            return val / 100.0
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="exaone")
    p.add_argument("--dataset", default="triviaqa")
    p.add_argument("--n_samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=str(ROOT / "results" / "raw"))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n[Exp05] Verbalized Confidence | model={args.model} | dataset={args.dataset}")

    wrapper = load_model(args.model)
    samples = load_dataset_by_name(args.dataset, n_samples=args.n_samples, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"05_verbalized_confidence_{args.model}_{args.dataset}_{ts}.jsonl"

    results = []
    n_parse_fail = 0

    with open(out_path, "w") as f:
        for sample in tqdm(samples, desc="Step1: Answering"):
            # Step 1: 답변 생성
            answer_gen = greedy_generate(wrapper, sample["question"])
            answer = answer_gen.generated_text
            is_correct = check_correctness(answer, sample["answers"])

            # Step 2: 모델에게 confidence 질문
            conf_question = CONFIDENCE_PROMPT_TEMPLATE.format(
                question=sample["question"],
                answer=answer,
            )
            conf_gen = generate_with_scores(
                wrapper, conf_question,
                system="You are a precise confidence estimator. Respond with only a number 0-100.",
                temperature=0.1,  # confidence 추출은 deterministic하게
                do_sample=False,
            )
            verbalized_conf = extract_confidence(conf_gen.generated_text)
            if verbalized_conf is None:
                n_parse_fail += 1
                verbalized_conf = 0.5  # fallback

            record = {
                "experiment": "verbalized_confidence",
                "model": args.model,
                "dataset": args.dataset,
                "question": sample["question"],
                "gold_answers": sample["answers"],
                "answer": answer,
                "is_correct": is_correct,
                # 핵심 지표
                "verbalized_confidence": verbalized_conf,
                "verbalized_raw": conf_gen.generated_text,
                "parse_success": extract_confidence(conf_gen.generated_text) is not None,
                # 보조: sequence log prob 기반 confidence (비교용)
                "seq_log_prob": answer_gen.sequence_log_prob,
                "mean_entropy": answer_gen.mean_entropy,
            }
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 분석
    is_correct_list = [r["is_correct"] for r in results]
    verb_confs = [r["verbalized_confidence"] for r in results]
    parse_ok = [r["parse_success"] for r in results]

    ece_result = compute_ece(verb_confs, is_correct_list)

    print(f"\n[Exp05] Saved to {out_path}")
    print(f"  Accuracy          : {np.mean(is_correct_list):.3f}")
    print(f"  Mean conf (correct): {np.mean([c for c,ok in zip(verb_confs, is_correct_list) if ok]):.3f}")
    print(f"  Mean conf (wrong):  {np.mean([c for c,ok in zip(verb_confs, is_correct_list) if not ok]):.3f}")
    print(f"  ECE (verbalized)  : {ece_result['ece']:.4f}")
    print(f"  Parse fail rate   : {n_parse_fail}/{len(results)} = {n_parse_fail/len(results):.1%}")

    plot_reliability_diagram(
        ece_result,
        model_name=f"{args.model} / {args.dataset} (verbalized)",
        save_path=str(fig_dir / f"05_verbalized_reliability_{args.model}_{args.dataset}_{ts}.png"),
    )

    wrapper.unload()
    return out_path


if __name__ == "__main__":
    main()

"""
Experiment 07: Parameter Scaling Study
같은 아키텍처(Qwen2.5), 다른 크기의 모델들에서 hallucination 지표 비교.
0.5B → 1.5B → 3B → 7B → 14B

Usage:
    python experiments/07_model_scaling/run.py --n_samples 200
    python experiments/07_model_scaling/run.py --family qwen --n_samples 200
    python experiments/07_model_scaling/run.py --family llama --n_samples 200
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

from src.model_loader import load_model, load_config
from src.generation import generate_with_scores, generate_multiple
from src.datasets.loader import load_dataset_by_name, check_correctness
from src.metrics.entropy import compute_token_entropy_scores, compute_semantic_entropy_from_results


QWEN_MODELS = ["qwen_0_5b", "qwen_1_5b", "qwen_3b", "qwen_7b", "qwen_14b"]
LLAMA_MODELS = ["llama_1b", "llama_3b", "llama_8b"]


def run_single_model(model_key, samples, n_gen_se=5):
    """단일 모델에 대해 token entropy + semantic entropy 측정."""
    cfg = load_config()
    model_cfg = cfg["models"][model_key]
    print(f"\n  [{model_key}] {model_cfg['name']} ({model_cfg['param_billions']}B)")

    wrapper = load_model(model_key)
    records = []

    for sample in tqdm(samples, desc=f"  {model_key}", leave=False):
        # Token entropy (greedy)
        gen = generate_with_scores(wrapper, sample["question"], do_sample=False, temperature=1.0, top_p=1.0)
        is_correct = check_correctness(gen.generated_text, sample["answers"])
        ent = compute_token_entropy_scores(gen.token_entropies)

        # Semantic entropy (n_gen_se samples)
        gens = generate_multiple(wrapper, sample["question"], n=n_gen_se, temperature=1.0)
        se_result = compute_semantic_entropy_from_results(gens)

        records.append({
            "question": sample["question"],
            "gold_answers": sample["answers"],
            "generated_text": gen.generated_text,
            "is_correct": is_correct,
            "mean_entropy": ent["mean"],
            "max_entropy": ent["max"],
            "sequence_log_prob": gen.sequence_log_prob,
            "semantic_entropy": se_result["semantic_entropy"],
            "n_clusters": se_result["n_clusters"],
            "n_unique_answers": len(se_result["unique_answers"]),
        })

    wrapper.unload()

    # 집계
    is_c = [r["is_correct"] for r in records]
    me_c = [r["mean_entropy"] for r in records if r["is_correct"]]
    me_w = [r["mean_entropy"] for r in records if not r["is_correct"]]
    se_c = [r["semantic_entropy"] for r in records if r["is_correct"]]
    se_w = [r["semantic_entropy"] for r in records if not r["is_correct"]]
    nc_c = [r["n_clusters"] for r in records if r["is_correct"]]
    nc_w = [r["n_clusters"] for r in records if not r["is_correct"]]

    from sklearn.metrics import roc_auc_score
    labels = [0 if c else 1 for c in is_c]
    me_all = [r["mean_entropy"] for r in records]
    se_all = [r["semantic_entropy"] for r in records]
    nc_all = [r["n_clusters"] for r in records]

    summary = {
        "model_key": model_key,
        "model_name": model_cfg["name"],
        "param_billions": model_cfg["param_billions"],
        "family": model_cfg.get("family", "unknown"),
        "n_samples": len(records),
        "accuracy": float(np.mean(is_c)),
        "mean_entropy_correct": float(np.mean(me_c)) if me_c else 0,
        "mean_entropy_wrong": float(np.mean(me_w)) if me_w else 0,
        "entropy_gap": float(np.mean(me_w) - np.mean(me_c)) if me_c and me_w else 0,
        "semantic_entropy_correct": float(np.mean(se_c)) if se_c else 0,
        "semantic_entropy_wrong": float(np.mean(se_w)) if se_w else 0,
        "n_clusters_correct": float(np.mean(nc_c)) if nc_c else 0,
        "n_clusters_wrong": float(np.mean(nc_w)) if nc_w else 0,
        "auroc_mean_entropy": float(roc_auc_score(labels, me_all)) if len(set(labels)) > 1 else 0.5,
        "auroc_semantic_entropy": float(roc_auc_score(labels, se_all)) if len(set(labels)) > 1 else 0.5,
        "auroc_n_clusters": float(roc_auc_score(labels, nc_all)) if len(set(labels)) > 1 else 0.5,
        "records": records,
    }

    print(f"    acc={summary['accuracy']:.3f} | entropy_gap={summary['entropy_gap']:+.4f} | "
          f"AUROC_ent={summary['auroc_mean_entropy']:.4f} | AUROC_SE={summary['auroc_semantic_entropy']:.4f}")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="qwen", choices=["qwen", "llama", "both"])
    p.add_argument("--dataset", default="triviaqa")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--n_gen_se", type=int, default=5, help="Semantic entropy 생성 횟수")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=str(ROOT / "results" / "raw"))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n[Exp07] Model Scaling | family={args.family} | dataset={args.dataset} | n={args.n_samples}")

    if args.family == "qwen":
        models = QWEN_MODELS
    elif args.family == "llama":
        models = LLAMA_MODELS
    else:
        models = QWEN_MODELS + LLAMA_MODELS

    samples = load_dataset_by_name(args.dataset, n_samples=args.n_samples, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    all_summaries = []
    for model_key in models:
        try:
            summary = run_single_model(model_key, samples, n_gen_se=args.n_gen_se)
            all_summaries.append(summary)

            # 중간 저장
            mid_path = output_dir / f"07_scaling_{args.family}_{args.dataset}_{ts}.json"
            save_data = [
                {k: v for k, v in s.items() if k != "records"}
                for s in all_summaries
            ]
            with open(mid_path, "w") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"  [{model_key}] 실패: {e}")
            import traceback; traceback.print_exc()

    out_path = output_dir / f"07_scaling_{args.family}_{args.dataset}_{ts}.json"
    save_data = [{k: v for k, v in s.items() if k != "records"} for s in all_summaries]
    with open(out_path, "w") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n[Exp07] Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    main()

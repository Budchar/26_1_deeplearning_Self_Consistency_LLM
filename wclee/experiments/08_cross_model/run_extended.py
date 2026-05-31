"""
Exp08-Extended: Cross-Architecture Comparison (확장판)
12개 모델 전체에 대해 TriviaQA hallucination 탐지 지표 비교.
기존 08_cross_model/run.py를 기반으로 모델 목록만 확장.

Usage:
  python experiments/08_cross_model/run_extended.py --n_samples 200
  python experiments/08_cross_model/run_extended.py --models smollm2_360m smollm2_1_7b falcon_7b opt_6_7b
"""

import sys, json, argparse, gc, torch
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.model_loader import load_model, load_config
from src.datasets.loader import load_triviaqa, check_correctness
from src.generation import generate_with_scores, generate_multiple
from src.metrics.entropy import compute_token_entropy_scores, compute_semantic_entropy_from_results

ALL_MODELS_ORDERED = [
    "smollm2_360m",   # 0.36B
    "qwen_0_5b",      # 0.5B
    "smollm2_1_7b",   # 1.7B
    "qwen_1_5b",      # 1.5B
    "qwen_3b",        # 3.0B
    "qwen_7b",        # 7.0B
    "mistral_v02",    # 7.0B
    "mistral",        # 7.0B
    "opt_6_7b",       # 6.7B
    "falcon_7b",      # 7.0B
    "exaone",         # 7.8B
    "qwen_14b",       # 14.0B
]


def run_one_model(model_key, samples, n_samples):
    wrapper = load_model(model_key)
    cfg = load_config()["models"][model_key]
    results = []

    for i, item in enumerate(samples[:n_samples]):
        if i % 25 == 0:
            print(f"  [{model_key}] {i}/{n_samples}", flush=True)
        question = item["question"]
        gold = item["answers"]
        try:
            gen = generate_with_scores(wrapper, question, do_sample=False)
            ent = compute_token_entropy_scores(gen.token_entropies)
            gens = generate_multiple(wrapper, question, n=5)
            se = compute_semantic_entropy_from_results(gens)
            is_correct = check_correctness(gen.generated_text, gold)
            results.append({
                "is_correct": is_correct,
                "mean_entropy": ent["mean"],
                "max_entropy": ent["max"],
                "sequence_log_prob": gen.sequence_log_prob,
                "semantic_entropy": se["semantic_entropy"],
                "n_clusters": se["n_clusters"],
            })
        except Exception as e:
            print(f"  [WARN] {i}: {e}")
            continue

    wrapper.unload()
    gc.collect()
    torch.cuda.empty_cache()

    if not results:
        return None

    correct = [r for r in results if r["is_correct"]]
    wrong   = [r for r in results if not r["is_correct"]]
    labels  = [1 if r["is_correct"] else 0 for r in results]

    def auroc(scores):
        try:
            return float(roc_auc_score(labels, [-s for s in scores])) if len(set(labels)) > 1 else 0.5
        except Exception:
            return 0.5

    return {
        "model_key": model_key,
        "model_name": cfg["name"],
        "param_billions": cfg["param_billions"],
        "family": cfg["family"],
        "n_samples": len(results),
        "accuracy": round(len(correct) / len(results), 4),
        "auroc_entropy": round(auroc([r["mean_entropy"] for r in results]), 4),
        "auroc_se": round(auroc([r["semantic_entropy"] for r in results]), 4),
        "auroc_nc": round(auroc([r["n_clusters"] for r in results]), 4),
        "entropy_gap": round(
            float(np.mean([r["mean_entropy"] for r in wrong]) -
                  np.mean([r["mean_entropy"] for r in correct])), 4
        ) if correct and wrong else 0.0,
        "entropy_correct": round(float(np.mean([r["mean_entropy"] for r in correct])), 4) if correct else None,
        "entropy_wrong":   round(float(np.mean([r["mean_entropy"] for r in wrong])), 4)   if wrong   else None,
        "clusters_correct": round(float(np.mean([r["n_clusters"] for r in correct])), 4) if correct else None,
        "clusters_wrong":   round(float(np.mean([r["n_clusters"] for r in wrong])), 4)   if wrong   else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--models", nargs="+", default=None,
                        help="실행할 모델 목록. 미지정시 ALL_MODELS_ORDERED 전체.")
    parser.add_argument("--dataset", default="triviaqa")
    args = parser.parse_args()

    target_models = args.models or ALL_MODELS_ORDERED
    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"08_cross_extended_{ts}.json"

    print(f"=== Exp08-Extended: {len(target_models)} models, n={args.n_samples} ===")
    from src.datasets.loader import load_dataset_by_name
    samples = load_dataset_by_name(args.dataset, n_samples=args.n_samples * 2)
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    all_results = []
    for model_key in target_models:
        print(f"\n--- {model_key} ---")
        try:
            res = run_one_model(model_key, samples, args.n_samples)
            if res:
                all_results.append(res)
                print(f"  acc={res['accuracy']:.3f} auroc_ent={res['auroc_entropy']:.4f} "
                      f"auroc_nc={res['auroc_nc']:.4f} gap={res['entropy_gap']:+.4f}")
                json.dump(all_results, open(out_path, "w"), indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [ERROR] {model_key}: {e}")
            continue

    print(f"\nAll done. Saved: {out_path}")
    print("\n[Final Summary]")
    print(f"{'Model':<28} {'Params':>6} {'Acc':>6} {'AUROC_ent':>10} {'AUROC_nc':>9} {'Gap':>8}")
    print("-" * 74)
    for r in sorted(all_results, key=lambda x: x["param_billions"]):
        print(f"  {r['model_name'][:26]:<26} {r['param_billions']:>5.1f}B "
              f"{r['accuracy']:>6.3f} {r['auroc_entropy']:>10.4f} "
              f"{r['auroc_nc']:>9.4f} {r['entropy_gap']:>+8.4f}")


if __name__ == "__main__":
    main()

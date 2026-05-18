"""
Exp09: Multi-Dataset Comparison
동일 모델에 대해 TriviaQA / MMLU / NaturalQuestions 세 데이터셋에서
hallucination 탐지 지표가 얼마나 일관되는지 비교.

Usage:
  python experiments/09_multi_dataset/run.py --model exaone --n_samples 200
  python experiments/09_multi_dataset/run.py --model qwen_7b --n_samples 200
"""

import sys, json, argparse, gc, torch
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.model_loader import load_model
from src.datasets.loader import load_dataset_by_name, check_correctness
from src.generation import generate_with_scores, generate_multiple
from src.metrics.entropy import compute_token_entropy_scores, compute_semantic_entropy_from_results


def run_on_dataset(wrapper, samples, dataset_name, n_samples):
    results = []
    for i, item in enumerate(samples[:n_samples]):
        if i % 20 == 0:
            print(f"  [{dataset_name}] {i}/{min(n_samples, len(samples))}", flush=True)
        question = item["question"]
        gold = item["answers"]

        try:
            gen = generate_with_scores(wrapper, question, do_sample=False)
            ent = compute_token_entropy_scores(gen.token_entropies)
            is_correct = check_correctness(gen.generated_text, gold)

            gens = generate_multiple(wrapper, question, n=5)
            se = compute_semantic_entropy_from_results(gens)

            results.append({
                "question": question,
                "generated": gen.generated_text,
                "is_correct": is_correct,
                "mean_entropy": ent["mean"],
                "max_entropy": ent["max"],
                "sequence_log_prob": gen.sequence_log_prob,
                "semantic_entropy": se["semantic_entropy"],
                "n_clusters": se["n_clusters"],
                "dataset": dataset_name,
                "model": wrapper.model_key,
            })
        except Exception as e:
            print(f"  [WARN] {i}: {e}")
            continue

    return results


def compute_auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    y = np.array(labels)
    if len(set(y)) < 2:
        return 0.5
    # uncertainty → hallucination: higher score = more likely wrong
    try:
        return roc_auc_score(y, [-s for s in scores])
    except Exception:
        return 0.5


def summarize(results, dataset_name, model_key):
    import numpy as np
    correct = [r for r in results if r["is_correct"]]
    wrong   = [r for r in results if not r["is_correct"]]
    acc = len(correct) / len(results) if results else 0

    labels = [1 if r["is_correct"] else 0 for r in results]
    auroc_ent  = compute_auroc([r["mean_entropy"] for r in results], labels)
    auroc_nc   = compute_auroc([r["n_clusters"] for r in results], labels)
    auroc_se   = compute_auroc([r["semantic_entropy"] for r in results], labels)
    entropy_gap = (
        np.mean([r["mean_entropy"] for r in wrong]) -
        np.mean([r["mean_entropy"] for r in correct])
    ) if correct and wrong else 0.0

    return {
        "model": model_key,
        "dataset": dataset_name,
        "n_samples": len(results),
        "accuracy": round(acc, 4),
        "auroc_entropy": round(auroc_ent, 4),
        "auroc_n_clusters": round(auroc_nc, 4),
        "auroc_semantic_entropy": round(auroc_se, 4),
        "entropy_gap": round(float(entropy_gap), 4),
        "entropy_correct": round(float(np.mean([r["mean_entropy"] for r in correct])), 4) if correct else None,
        "entropy_wrong":   round(float(np.mean([r["mean_entropy"] for r in wrong])), 4)   if wrong   else None,
        "n_correct": len(correct),
        "n_wrong": len(wrong),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="exaone")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--datasets", nargs="+", default=["triviaqa", "mmlu", "naturalquestions"])
    args = parser.parse_args()

    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"09_multi_dataset_{args.model}_{ts}.json"

    print(f"=== Exp09: Multi-Dataset | model={args.model} n={args.n_samples} ===")
    wrapper = load_model(args.model)

    all_summaries = []
    for ds_name in args.datasets:
        print(f"\n--- Loading {ds_name} ---")
        samples = load_dataset_by_name(ds_name, n_samples=args.n_samples * 2)
        if not samples:
            print(f"  [SKIP] {ds_name} 로드 실패")
            continue
        print(f"  Loaded {len(samples)} samples, running {args.n_samples}...")
        results = run_on_dataset(wrapper, samples, ds_name, args.n_samples)
        if not results:
            continue
        summary = summarize(results, ds_name, args.model)
        all_summaries.append(summary)
        print(f"  {ds_name}: acc={summary['accuracy']:.3f} "
              f"auroc_ent={summary['auroc_entropy']:.4f} "
              f"auroc_nc={summary['auroc_n_clusters']:.4f}")

    wrapper.unload()
    gc.collect()
    torch.cuda.empty_cache()

    json.dump(all_summaries, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    print("\n[Summary]")
    print(f"{'Dataset':<20} {'Acc':>6} {'AUROC_ent':>10} {'AUROC_nc':>9} {'Gap':>8}")
    print("-" * 58)
    for s in all_summaries:
        print(f"  {s['dataset']:<18} {s['accuracy']:>6.3f} "
              f"{s['auroc_entropy']:>10.4f} {s['auroc_n_clusters']:>9.4f} "
              f"{s['entropy_gap']:>+8.4f}")


if __name__ == "__main__":
    main()

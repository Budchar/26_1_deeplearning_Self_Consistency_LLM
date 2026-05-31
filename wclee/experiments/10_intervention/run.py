"""
Exp10: Layer Probe Intervention (C)
사전 학습된 layer probe를 이용해 hallucination 위험을 탐지하고,
위험 시 재샘플링(regeneration)으로 응답 품질을 개선.

Pipeline:
  1. 질문 → 1차 답변 생성 + hidden state 추출
  2. Best layer probe로 hallucination 확률 추정
  3. P(hallucination) > threshold → 재샘플링 (최대 max_retries회)
  4. 재샘플링된 답변 중 가장 낮은 hallucination 확률의 것 선택
  5. 개입 전후 accuracy, AUROC 비교

Usage:
  python experiments/10_intervention/run.py --model exaone --n_samples 150 --threshold 0.6
  python experiments/10_intervention/run.py --model qwen_7b --n_samples 150 --threshold 0.6
"""

import sys, json, argparse, gc, torch
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.model_loader import load_model
from src.datasets.loader import load_triviaqa, check_correctness
from src.hidden_states import extract_hidden_states, train_layer_probes
from src.metrics.entropy import compute_semantic_entropy


def train_probe_from_layer_result(layer_probe_path):
    """기존 layer probe JSON에서 best_layer 정보 로드."""
    import json
    data = json.load(open(layer_probe_path))
    return data["probe"]["best_layer"], data["probe"]["best_auroc"]


def build_probe(wrapper, train_samples, best_layer_hint=None):
    """
    소규모 학습 데이터로 probe를 직접 학습.
    Returns: (scaler, clf, best_layer)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print(f"  Building probe on {len(train_samples)} samples...")
    hs_list, labels = [], []
    for i, item in enumerate(train_samples):
        if i % 20 == 0:
            print(f"    probe training {i}/{len(train_samples)}", flush=True)
        try:
            result = extract_hidden_states(wrapper, item["question"])
            result.is_correct = check_correctness(result.generated_text, item["answers"])
            hs_list.append(result.hidden_states_last_token)
            labels.append(1 if result.is_correct else 0)
        except Exception as e:
            print(f"    [WARN] {e}")
            continue

    if len(hs_list) < 10:
        raise RuntimeError("Probe 학습에 충분한 샘플이 없습니다.")

    probe_result = train_layer_probes(hs_list, labels)
    best_layer = probe_result["best_layer"]
    print(f"  Best probe layer: {best_layer} (AUROC={probe_result['best_auroc']:.4f})")

    # best layer에서 scaler + clf 재학습
    X = np.stack(hs_list)[:, best_layer, :]
    y = np.array(labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(X_scaled, y)

    return scaler, clf, best_layer, probe_result


def predict_hallucination_prob(wrapper, question, scaler, clf, best_layer):
    """단일 질문에 대해 hallucination 확률 추정 (1 - P(correct))."""
    result = extract_hidden_states(wrapper, question)
    h = result.hidden_states_last_token[best_layer].reshape(1, -1)
    h_scaled = scaler.transform(h)
    p_correct = clf.predict_proba(h_scaled)[0, 1]
    return result, 1.0 - p_correct  # hallucination probability


def run_intervention(wrapper, test_samples, scaler, clf, best_layer, threshold, max_retries):
    """개입 실험 실행."""
    from src.generation import generate_with_scores
    from src.metrics.entropy import compute_token_entropy_scores

    baseline_results = []
    intervention_results = []

    for i, item in enumerate(test_samples):
        if i % 20 == 0:
            print(f"  [{i}/{len(test_samples)}]", flush=True)
        question = item["question"]
        gold = item["answers"]

        try:
            # 1차 생성
            result, hall_prob = predict_hallucination_prob(wrapper, question, scaler, clf, best_layer)
            first_answer = result.generated_text
            first_correct = check_correctness(first_answer, gold)

            baseline_results.append({
                "question": question,
                "generated": first_answer,
                "is_correct": first_correct,
                "hall_prob": hall_prob,
                "intervened": False,
            })

            # 개입 결정
            final_answer = first_answer
            final_correct = first_correct
            intervened = False
            n_retries = 0

            if hall_prob > threshold:
                intervened = True
                candidates = [(first_answer, hall_prob, first_correct)]
                for retry in range(max_retries):
                    r2, hp2 = predict_hallucination_prob(wrapper, question, scaler, clf, best_layer)
                    ans2 = r2.generated_text
                    ok2 = check_correctness(ans2, gold)
                    candidates.append((ans2, hp2, ok2))
                    n_retries += 1
                    if hp2 < threshold:
                        break  # 충분히 낮으면 조기 중단

                # 가장 낮은 hallucination 확률의 후보 선택
                best_cand = min(candidates, key=lambda x: x[1])
                final_answer, final_hall_prob, final_correct = best_cand
            else:
                final_hall_prob = hall_prob

            intervention_results.append({
                "question": question,
                "generated": final_answer,
                "is_correct": final_correct,
                "hall_prob": final_hall_prob,
                "hall_prob_initial": hall_prob,
                "intervened": intervened,
                "n_retries": n_retries,
            })

        except Exception as e:
            print(f"  [WARN] {i}: {e}")
            continue

    return baseline_results, intervention_results


def compute_metrics(results, label=""):
    from sklearn.metrics import roc_auc_score
    labels = [1 if r["is_correct"] else 0 for r in results]
    acc = np.mean(labels)
    hall_probs = [r["hall_prob"] for r in results]
    try:
        auroc = roc_auc_score(labels, [-p for p in hall_probs]) if len(set(labels)) > 1 else 0.5
    except Exception:
        auroc = 0.5
    intervened = [r for r in results if r.get("intervened", False)]
    print(f"  {label}: acc={acc:.4f} auroc={auroc:.4f} "
          f"n_intervened={len(intervened)}/{len(results)}")
    return {"accuracy": float(acc), "auroc": float(auroc),
            "n_samples": len(results), "n_intervened": len(intervened)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="exaone")
    parser.add_argument("--n_samples", type=int, default=150)
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Hallucination probability threshold for intervention")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"10_intervention_{args.model}_{ts}.json"

    print(f"=== Exp10: Layer Probe Intervention | model={args.model} ===")
    print(f"  threshold={args.threshold} max_retries={args.max_retries} n={args.n_samples}")

    samples = load_triviaqa(n_samples=args.n_samples * 2)
    n_train = int(args.n_samples * args.train_ratio)
    train_samples = samples[:n_train]
    test_samples  = samples[n_train: n_train + args.n_samples - n_train]

    print(f"  Train probe: {len(train_samples)} | Test: {len(test_samples)}")

    wrapper = load_model(args.model)

    # Probe 학습
    scaler, clf, best_layer, probe_info = build_probe(wrapper, train_samples)

    # 개입 실험
    print(f"\n--- Running intervention (threshold={args.threshold}) ---")
    baseline_results, intervention_results = run_intervention(
        wrapper, test_samples, scaler, clf, best_layer,
        args.threshold, args.max_retries
    )

    wrapper.unload()
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== Results ===")
    baseline_metrics    = compute_metrics(baseline_results, "Baseline (no intervention)")
    intervention_metrics = compute_metrics(intervention_results, "With intervention")

    acc_gain  = intervention_metrics["accuracy"] - baseline_metrics["accuracy"]
    auroc_gain = intervention_metrics["auroc"] - baseline_metrics["auroc"]
    print(f"\n  Accuracy gain: {acc_gain:+.4f}")
    print(f"  AUROC gain:    {auroc_gain:+.4f}")
    n_intervened = intervention_metrics["n_intervened"]
    n_test = intervention_metrics["n_samples"]
    print(f"  Intervened on {n_intervened}/{n_test} samples ({n_intervened/n_test*100:.1f}%)")

    output = {
        "model": args.model,
        "threshold": args.threshold,
        "max_retries": args.max_retries,
        "best_layer": best_layer,
        "probe_auroc": probe_info["best_auroc"],
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "baseline": baseline_metrics,
        "intervention": intervention_metrics,
        "acc_gain": round(acc_gain, 4),
        "auroc_gain": round(auroc_gain, 4),
        "n_intervened": n_intervened,
        "intervention_rate": round(n_intervened / n_test, 4) if n_test else 0,
        "sample_results": intervention_results,
    }
    json.dump(output, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

"""
Experiment 06: Layer-wise Hidden State Probe
각 Transformer 레이어의 hidden state에 로지스틱 회귀 probe를 달아
"어느 레이어가 hallucination 정보를 가장 잘 인코딩하는가" 분석.

Usage:
    python experiments/06_layer_probe/run.py --model exaone --n_samples 300
    python experiments/06_layer_probe/run.py --model qwen_7b --n_samples 300
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
from src.hidden_states import extract_hidden_states, train_layer_probes, compute_layer_entropy_profile
from src.datasets.loader import load_dataset_by_name, check_correctness


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
    print(f"\n[Exp06] Layer Probe | model={args.model} | n={args.n_samples}")

    wrapper = load_model(args.model)
    samples = load_dataset_by_name(args.dataset, n_samples=args.n_samples, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    all_hidden = []
    all_labels = []
    meta_records = []

    for sample in tqdm(samples, desc="Extracting hidden states"):
        result = extract_hidden_states(wrapper, sample["question"])
        is_correct = check_correctness(result.generated_text, sample["answers"])
        result.is_correct = is_correct

        all_hidden.append(result.hidden_states_last_token)
        all_labels.append(1 if is_correct else 0)

        meta_records.append({
            "question": sample["question"],
            "gold_answers": sample["answers"],
            "generated_text": result.generated_text,
            "is_correct": is_correct,
            "mean_entropy": result.mean_entropy,
            "sequence_log_prob": result.sequence_log_prob,
            "n_layers": result.n_layers,
            "hidden_dim": result.hidden_dim,
        })

    n_correct = sum(all_labels)
    print(f"\n  Accuracy: {n_correct}/{len(all_labels)} = {n_correct/len(all_labels):.3f}")
    print(f"  n_layers: {meta_records[0]['n_layers']}, hidden_dim: {meta_records[0]['hidden_dim']}")

    # ── Layer Probe ────────────────────────────────────────────────
    print("\n  Training layer probes...")
    probe_result = train_layer_probes(all_hidden, all_labels)

    print(f"  Best layer: {probe_result['best_layer']} / {probe_result['n_layers']}")
    print(f"  Best AUROC: {probe_result['best_auroc']:.4f}")

    # ── Representation Analysis ────────────────────────────────────
    rep_metrics = compute_layer_entropy_profile(all_hidden, all_labels)

    # ── Save ───────────────────────────────────────────────────────
    result_data = {
        "experiment": "layer_probe",
        "model": args.model,
        "dataset": args.dataset,
        "n_samples": len(all_labels),
        "accuracy": n_correct / len(all_labels),
        "n_layers": meta_records[0]["n_layers"],
        "hidden_dim": meta_records[0]["hidden_dim"],
        "probe": probe_result,
        "representation": rep_metrics,
        "sample_meta": meta_records,
    }

    out_path = output_dir / f"06_layer_probe_{args.model}_{args.dataset}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")

    wrapper.unload()
    return out_path


if __name__ == "__main__":
    main()

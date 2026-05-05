"""Synthetic smoke test: validate the post-generation pipeline without an LLM.

Produces a fake generations.jsonl + hidden npz files, then runs:
  se_compute -> seps_probe -> adaptive_se -> metrics aggregation.

This isolates network-bound model downloads from the rest of the code.
The NLI model is still needed (deberta-v2-xlarge-mnli) but that download
is the single dependency for SE computation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from se_compute import process_jsonl, NLIEntailer  # noqa: E402
from seps_probe import run_probes  # noqa: E402
from adaptive_se import run as adaptive_run  # noqa: E402
from metrics import detection_metrics, wilcoxon_paired, stratified_acc_by_quartile  # noqa: E402


def make_synthetic(out_dir: Path, n_q: int = 30, n_samples: int = 10,
                   n_layers: int = 13, hidden_size: int = 256, seed: int = 0) -> Path:
    """Construct fake QA records: half "easy" (consistent samples), half "hard" (varied)."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir = out_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "generations.jsonl"

    easy_pairs = [
        ("What is the capital of France?", ["Paris"]),
        ("Who wrote Hamlet?", ["William Shakespeare", "Shakespeare"]),
        ("What is 2 + 2?", ["4", "four"]),
        ("Largest planet in our solar system?", ["Jupiter"]),
        ("Chemical symbol for water?", ["H2O"]),
    ]
    hard_pairs = [
        ("Who invented the telephone?", ["Alexander Graham Bell"]),
        ("Capital of Australia?", ["Canberra"]),
        ("What year did WW2 end?", ["1945"]),
        ("Author of 1984?", ["George Orwell"]),
        ("Speed of light in m/s?", ["299792458", "3 x 10^8"]),
    ]

    # Build fake hidden state with a useful linear signal:
    #   layer 0..3: pure noise
    #   layer 4..8: weak signal correlated with "is_hard"
    #   layer 9..L: strong signal correlated with "is_hard" (hidden state norm differs)
    records = []
    with open(jsonl, "w") as f:
        for i in range(n_q):
            is_hard = (i % 2 == 1)
            base_pairs = hard_pairs if is_hard else easy_pairs
            q, golds = base_pairs[i % len(base_pairs)]
            if is_hard:
                # samples: mix of variants -> high SE
                samples = [
                    f"{golds[0]}",
                    "I don't know",
                    f"Maybe {golds[0]}",
                    "Not sure",
                    f"Possibly something else",
                    "Unknown",
                    f"{golds[0][:3] if golds[0] else 'xx'}",
                    "Multiple answers",
                    "Could be many things",
                    "Unsure",
                ][:n_samples]
                greedy = "I don't know"
            else:
                samples = [golds[0]] * (n_samples - 1) + [f"{golds[0]} (the answer)"]
                greedy = golds[0]
            lps = list(rng.normal(-0.5 if not is_hard else -1.5, 0.2, size=n_samples))

            # Hidden state with discriminative signal in deep layers
            h = rng.standard_normal((n_layers, hidden_size), dtype=np.float32)
            signal_strength = np.linspace(0.0, 3.0 if is_hard else -3.0, n_layers)
            h[:, 0] += signal_strength  # first dim becomes a "uncertainty axis"
            samples_h = np.broadcast_to(h[None], (n_samples, n_layers, hidden_size)).copy()
            hp = hidden_dir / f"q{i:03d}.npz"
            np.savez_compressed(hp, greedy_h=h.astype(np.float16), samples_h=samples_h.astype(np.float16))

            records.append({
                "id": f"q{i:03d}",
                "dataset": "synthetic",
                "question": q,
                "answers": golds,
                "greedy": greedy,
                "samples": samples,
                "sample_logprobs": [float(x) for x in lps],
                "hidden_path": str(hp),
            })
            f.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
    print(f"[synth] wrote {n_q} records to {jsonl}", flush=True)
    return jsonl


def run(out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    gen_dir = out_root / "synthetic_run"
    jsonl = make_synthetic(gen_dir, n_q=30, n_samples=10)

    # SE step (this loads the NLI model)
    se_path = gen_dir / "se.jsonl"
    print("[synth] loading NLI entailer...", flush=True)
    entailer = NLIEntailer()
    print("[synth] computing SE...", flush=True)
    se_summary = process_jsonl(jsonl, se_path, entailer=entailer)

    # Probes
    probes_path = gen_dir / "probes.json"
    probe_summary = run_probes(se_path, probes_path)

    # Adaptive
    adaptive_path = gen_dir / "adaptive.json"
    adaptive_summary = adaptive_run(se_path, adaptive_path)

    # Aggregate metrics
    se_records = [json.loads(l) for l in open(se_path)]
    correct = [r["greedy_correct"] for r in se_records]
    sc_correct = [r["sc_correct"] for r in se_records]
    se = [r["se_discrete"] for r in se_records]
    full_metrics = {
        "n": len(se_records),
        "greedy_acc": float(sum(correct) / max(len(correct), 1)),
        "sc_acc": float(sum(sc_correct) / max(len(sc_correct), 1)),
        "se_discrete": detection_metrics(se, correct),
        "wilcoxon_sc_vs_greedy": wilcoxon_paired(correct, sc_correct),
        "stratified_acc": stratified_acc_by_quartile(se, correct, sc_correct),
        "probe": {
            "best_logreg_se_auroc": probe_summary["best_logreg_se_auroc"],
            "best_logreg_halluc_auroc": probe_summary["best_logreg_halluc_auroc"],
            "best_mlp_se_auroc": probe_summary["best_mlp_se_auroc"],
            "best_mlp_halluc_auroc": probe_summary["best_mlp_halluc_auroc"],
        },
        "adaptive": {
            "auroc_delta": adaptive_summary["auroc_delta"],
            "cost_save_frac": adaptive_summary["cost_save_frac"],
            "avg_n": adaptive_summary["avg_n_adaptive"],
        },
    }
    print(json.dumps(full_metrics, indent=2), flush=True)
    return full_metrics


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "~/experiments/dl_team_v2/01_se_seps/runs/synthetic"
    ).expanduser()
    summary = run(out)
    summary_path = Path(
        "~/experiments/dl_team_v2/01_se_seps/results/smoke_synthetic_summary.json"
    ).expanduser()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[synth] summary -> {summary_path}", flush=True)

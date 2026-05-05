"""Main Phase-1 entry point.

For each (model, dataset) pair:
    1. Generate N samples + greedy with hidden states (sample_generator).
    2. Compute semantic entropy + SC + correctness (se_compute).
    3. Train SEPs probes per layer (seps_probe).
    4. Run cost-aware adaptive SE (adaptive_se).
    5. Aggregate detection metrics + Wilcoxon (metrics).

All steps are checkpointed (skip if output exists). Outputs land under:
    01_se_seps/runs/<model_safe>/<dataset>/
        generations.jsonl
        se.jsonl
        probes.json
        adaptive.json
        metrics.json
        hidden/<id>.npz

`results/sweep_summary.json` aggregates all runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from data_loader import load_dataset_by_name  # noqa: E402
from sample_generator import run as gen_run  # noqa: E402
from se_compute import process_jsonl, NLIEntailer  # noqa: E402
from seps_probe import run_probes  # noqa: E402
from adaptive_se import run as adaptive_run  # noqa: E402
from metrics import detection_metrics, wilcoxon_paired, stratified_acc_by_quartile  # noqa: E402

ROOT = Path(os.path.expanduser("~/experiments/dl_team_v2/01_se_seps"))
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def aggregate_metrics(se_jsonl: Path) -> Dict:
    records: List[Dict] = []
    with open(se_jsonl) as f:
        for line in f:
            records.append(json.loads(line))
    correct = [r["greedy_correct"] for r in records]
    sc_correct = [r["sc_correct"] for r in records]
    se = [r["se_discrete"] for r in records]
    se_lp = [r["se_logprob"] for r in records]

    out = {
        "n": len(records),
        "greedy_acc": float(sum(correct) / max(len(correct), 1)),
        "sc_acc": float(sum(sc_correct) / max(len(sc_correct), 1)),
        "se_discrete": detection_metrics(se, correct),
        "se_logprob": detection_metrics(se_lp, correct),
        "wilcoxon_sc_vs_greedy": wilcoxon_paired(correct, sc_correct),
        "stratified_acc": stratified_acc_by_quartile(se, correct, sc_correct),
    }
    return out


def run_one(
    model_id: str,
    dataset: str,
    n: int = 1000,
    n_samples: int = 10,
    four_bit: bool = False,
    dtype: str = "fp16",
    limit: int | None = None,
    save_hidden: bool = True,
    skip_probes: bool = False,
    skip_adaptive: bool = False,
    entailer: NLIEntailer | None = None,
) -> Dict:
    out_dir = RUNS / safe_name(model_id) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "generations.jsonl"
    se_path = out_dir / "se.jsonl"
    probes_path = out_dir / "probes.json"
    adaptive_path = out_dir / "adaptive.json"
    metrics_path = out_dir / "metrics.json"

    t0 = time.time()
    # Step 1: generate
    recs = load_dataset_by_name(dataset, n=n)
    if limit is not None:
        recs = recs[:limit]
    gen_run(
        model_id=model_id, dataset_records=recs, out_dir=out_dir,
        n_samples=n_samples, max_new_tokens=64, temperature=1.0, top_p=0.95,
        four_bit=four_bit, dtype=dtype, save_hidden=save_hidden,
    )
    # Free GPU mem before NLI
    torch.cuda.empty_cache()

    # Step 2: SE
    if not se_path.exists() or se_path.stat().st_size == 0:
        if entailer is None:
            entailer = NLIEntailer()
        process_jsonl(gen_path, se_path, entailer=entailer)
    else:
        print(f"[run] reuse {se_path}", flush=True)

    # Step 3: probes
    if not skip_probes and save_hidden:
        if not probes_path.exists():
            run_probes(se_path, probes_path)
        else:
            print(f"[run] reuse {probes_path}", flush=True)

    # Step 4: adaptive
    if not skip_adaptive:
        if not adaptive_path.exists():
            adaptive_run(se_path, adaptive_path)
        else:
            print(f"[run] reuse {adaptive_path}", flush=True)

    # Step 5: aggregate metrics
    metrics = aggregate_metrics(se_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    metrics["elapsed_s"] = time.time() - t0
    metrics["model"] = model_id
    metrics["dataset"] = dataset
    print(f"[run] {model_id} / {dataset} done in {metrics['elapsed_s']:.1f}s", flush=True)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["triviaqa", "nq_open", "squad"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--four-bit", nargs="*", default=[],
                    help="model ids to load in 4-bit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-hidden", action="store_true")
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--skip-adaptive", action="store_true")
    ap.add_argument("--summary-name", default="sweep_summary.json")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary: List[Dict] = []
    entailer: NLIEntailer | None = None
    for model in args.models:
        four_bit = model in args.four_bit
        for dataset in args.datasets:
            try:
                m = run_one(
                    model_id=model, dataset=dataset,
                    n=args.n, n_samples=args.n_samples,
                    four_bit=four_bit, limit=args.limit,
                    save_hidden=not args.no_hidden,
                    skip_probes=args.skip_probes,
                    skip_adaptive=args.skip_adaptive,
                    entailer=entailer,
                )
                summary.append(m)
            except Exception as e:
                print(f"[run] FAILED {model}/{dataset}: {e}", flush=True)
                summary.append({"model": model, "dataset": dataset, "error": str(e)})
    out = RESULTS / args.summary_name
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] summary -> {out}", flush=True)


if __name__ == "__main__":
    main()

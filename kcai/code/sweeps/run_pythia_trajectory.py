"""Pythia 1.4B 5 체크포인트 SEPs gap 측정 (Option 2).

같은 모델 (Pythia 1.4B) 의 학습 step별 체크포인트:
  step1000, step10000, step50000, step100000, step143000

각 체크포인트 × {triviaqa, nq_open, squad} = 15 cells.
같은 모델 + 같은 데이터, capability만 변경 → confound 없음.

run_phase1.py에 revision 인자가 없으므로 별도 wrapper로 사용.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add Phase 1 code path
PHASE1_CODE = "/home/kcai/experiments/dl_team_v2/01_se_seps/code"
sys.path.insert(0, PHASE1_CODE)

# HF token
TOK_FILE = Path.home() / "hf_token.txt"
if TOK_FILE.exists():
    os.environ["HF_TOKEN"] = TOK_FILE.read_text().strip()

from data_loader import load_dataset_by_name
from sample_generator import sample_one, build_prompt, existing_ids
from se_compute import process_jsonl, NLIEntailer
from seps_probe import run_probes
from metrics import auroc, ece, brier, aurc

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ROOT = Path("/home/kcai/experiments/dl_team_v2/09_option2_pythia_trajectory")
MODEL_BASE = "EleutherAI/pythia-1.4b-deduped"
CHECKPOINTS = ["step1000", "step10000", "step50000", "step100000", "step143000"]
DATASETS = ["triviaqa", "nq_open", "squad"]


def load_model_at_revision(revision: str):
    print(f"[load] {MODEL_BASE} @ {revision}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_BASE, revision=revision, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE, revision=revision, trust_remote_code=True,
        torch_dtype=torch.float16, device_map={"": 0},
    )
    model.eval()
    print(f"[load] OK params={sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)
    return tok, model, next(model.parameters()).device


def generate_for_step(revision: str, tok, model, device, n_samples=10, max_new=64):
    step_dir = ROOT / "runs" / revision
    step_dir.mkdir(parents=True, exist_ok=True)
    for ds_name in DATASETS:
        ds_dir = step_dir / ds_name
        ds_dir.mkdir(exist_ok=True)
        hidden_dir = ds_dir / "hidden"
        hidden_dir.mkdir(exist_ok=True)
        jsonl_path = ds_dir / "generations.jsonl"
        done = existing_ids(jsonl_path)
        recs = load_dataset_by_name(ds_name, n=1000)
        if len(done) >= len(recs):
            print(f"  [skip-gen] {revision}/{ds_name}", flush=True)
            continue
        print(f"  [gen] {revision}/{ds_name} {len(recs)} 질문", flush=True)
        t0 = time.time()
        with open(jsonl_path, "a") as fout:
            for rec in tqdm(recs, desc=f"{revision}/{ds_name}"):
                if rec["id"] in done:
                    continue
                prompt = build_prompt(tok, rec["question"])
                try:
                    greedy, samples, lps, h_prompt, samples_h = sample_one(
                        model, tok, prompt, n_samples, max_new, 1.0, 0.95, device,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    print(f"  OOM: {e}", flush=True)
                    continue
                hp = hidden_dir / f"{rec['id']}.npz"
                np.savez_compressed(hp, greedy_h=h_prompt, samples_h=samples_h)
                out = {
                    "id": rec["id"],
                    "dataset": rec["dataset"],
                    "question": rec["question"],
                    "answers": rec["answers"],
                    "greedy": greedy,
                    "samples": samples,
                    "sample_logprobs": lps,
                    "hidden_path": str(hp),
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
        print(f"  [gen] {revision}/{ds_name} done in {time.time()-t0:.1f}s", flush=True)


def compute_se_seps_for_step(revision: str, entailer):
    step_dir = ROOT / "runs" / revision
    for ds_name in DATASETS:
        ds_dir = step_dir / ds_name
        gen = ds_dir / "generations.jsonl"
        se_path = ds_dir / "se.jsonl"
        metrics_path = ds_dir / "metrics.json"
        probes_path = ds_dir / "probes.json"
        if not gen.exists():
            continue
        if not se_path.exists():
            print(f"  [se] {revision}/{ds_name}", flush=True)
            process_jsonl(gen, se_path, entailer=entailer)
        if not metrics_path.exists():
            # compute basic metrics
            recs = []
            with open(se_path) as f:
                for line in f:
                    recs.append(json.loads(line))
            greedy_correct = [r["greedy_correct"] for r in recs]
            se_disc = [r["se_discrete"] for r in recs]
            se_lp = [r["se_logprob"] for r in recs]
            sc_correct = [r["sc_correct"] for r in recs]
            wrong = [1 - c for c in greedy_correct]
            metrics = {
                "n": len(recs),
                "greedy_acc": float(sum(greedy_correct)) / len(recs),
                "sc_acc": float(sum(sc_correct)) / len(recs),
                "se_discrete": {
                    "auroc": auroc(se_disc, wrong),
                    "ece": ece(se_disc, wrong),
                    "brier": brier(se_disc, wrong),
                    "aurc": aurc(se_disc, greedy_correct),
                },
                "se_logprob": {
                    "auroc": auroc(se_lp, wrong),
                    "ece": ece(se_lp, wrong),
                    "brier": brier(se_lp, wrong),
                    "aurc": aurc(se_lp, greedy_correct),
                },
            }
            metrics_path.write_text(json.dumps(metrics, indent=2))
            print(f"  [metrics] {revision}/{ds_name} greedy_acc={metrics['greedy_acc']:.3f} se_auroc={metrics['se_discrete']['auroc']:.3f}", flush=True)
        if not probes_path.exists():
            print(f"  [probes] {revision}/{ds_name}", flush=True)
            run_probes(se_path, probes_path)


def main():
    print(f"=== Option 2 start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    # 1단계: 5 체크포인트 generation (각각 모델 로드)
    for revision in CHECKPOINTS:
        tok, model, device = load_model_at_revision(revision)
        generate_for_step(revision, tok, model, device)
        del model
        torch.cuda.empty_cache()
    # 2단계: SE + SEPs (NLI entailer 한 번 로드해서 모든 cell 처리)
    print(f"\n=== {time.strftime('%H:%M:%S')} all generations done, computing SE+SEPs ===", flush=True)
    entailer = NLIEntailer()
    for revision in CHECKPOINTS:
        compute_se_seps_for_step(revision, entailer)
    print(f"=== Option 2 done {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)


if __name__ == "__main__":
    main()

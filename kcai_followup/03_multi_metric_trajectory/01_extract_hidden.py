"""실험 3 Step 1: Phase 1 prompt를 다시 forward해서 layer별 hidden state 추출.

입력:
  Phase 1 generations.jsonl (id, question, answers, greedy)

출력 (cell별):
  hidden_states/{model}/{dataset}/
    hidden.npz: layer_00 ... layer_NN, ids
    prompts.jsonl: id, question, answers, greedy, correct
    meta.json
    done.marker

Resumable: 모든 cell 단위로 done.marker. 부분 처리 시 prompts.jsonl에서 처리된 id skip.
실행:
    python 01_extract_hidden.py --models all --datasets all
    python 01_extract_hidden.py --models Qwen/Qwen2.5-1.5B-Instruct --datasets triviaqa  # 단일 cell
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "_shared"))

from paths import (
    HIDDEN_CACHE,
    PHASE1,
    PHASE1_MODELS,
    DATASETS,
    hidden_cell_dir,
)
from model_loader import load_model, unload, model_n_layers, model_hidden_dim
from hidden_state import extract_for_cell
from prompt_format import format_record
from eval_utils import is_correct, extract_first_line
from resumable import is_done, append_jsonl, processed_ids


def phase1_cell_dir(model: str, dataset: str) -> Path:
    """Phase 1 결과 디렉토리. 모델명 형식: meta-llama__Llama-3.2-1B-Instruct"""
    return PHASE1 / model.replace("/", "__") / dataset


def load_phase1_prompts(model: str, dataset: str) -> list[dict]:
    """Phase 1 generations.jsonl → [{id, question, answers, greedy, correct}, ...]."""
    gen_path = phase1_cell_dir(model, dataset) / "generations.jsonl"
    if not gen_path.exists():
        raise FileNotFoundError(gen_path)

    records = []
    with open(gen_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            greedy = rec.get("greedy", "")
            answers = rec.get("answers", [])
            greedy_short = extract_first_line(greedy)
            correct = is_correct(greedy_short, answers)
            records.append({
                "id": str(rec["id"]),
                "question": rec["question"],
                "answers": answers,
                "dataset": rec.get("dataset", dataset),
                "greedy": greedy,
                "greedy_short": greedy_short,
                "correct": correct,
            })
    return records


def run_one_cell(model_name: str, dataset: str) -> dict:
    cell = hidden_cell_dir(model_name, dataset)
    if is_done(cell):
        print(f"[skip] {model_name} / {dataset} — already done", flush=True)
        return {"cell": str(cell), "status": "skipped"}

    print(f"\n[load] Phase 1 prompts {model_name} / {dataset}", flush=True)
    prompts = load_phase1_prompts(model_name, dataset)
    print(f"  n_prompts: {len(prompts)}, correct: {sum(p['correct'] for p in prompts)} ({sum(p['correct'] for p in prompts) / len(prompts) * 100:.1f}%)", flush=True)

    print(f"[load_model] {model_name}", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(model_name, dtype="fp16")
    print(f"  loaded in {time.time() - t0:.1f}s, n_layers: {model_n_layers(model)}, hidden_dim: {model_hidden_dim(model)}", flush=True)

    print(f"[extract] {model_name} / {dataset} ({len(prompts)} prompts)", flush=True)
    t0 = time.time()
    result = extract_for_cell(
        model, tokenizer, prompts, cell,
        prompt_format_fn=lambda r: format_record(tokenizer, r),
        batch_save_every=100,
    )
    print(f"  extracted in {time.time() - t0:.1f}s, status: {result['status']}", flush=True)

    # 정답 라벨도 prompts.jsonl 옆에 별도 저장 (분석 단계에서 사용)
    labels_path = cell / "labels.jsonl"
    if not labels_path.exists():
        for p in prompts:
            append_jsonl(labels_path, {
                "id": p["id"],
                "correct": p["correct"],
                "greedy": p["greedy_short"],
                "n_gold_answers": len(p["answers"]),
            })

    unload(model)
    return {"cell": str(cell), "status": "ok", "n_prompts": len(prompts), "n_correct": sum(p["correct"] for p in prompts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"], help="모델 이름 또는 'all'")
    ap.add_argument("--datasets", nargs="+", default=["all"], help="데이터셋 또는 'all'")
    args = ap.parse_args()

    models = PHASE1_MODELS if args.models == ["all"] else args.models
    datasets = DATASETS if args.datasets == ["all"] else args.datasets

    HIDDEN_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {HIDDEN_CACHE}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")

    summary = []
    for model in models:
        for dataset in datasets:
            try:
                r = run_one_cell(model, dataset)
                summary.append(r)
            except Exception as e:
                print(f"[error] {model} / {dataset}: {type(e).__name__}: {e}", flush=True)
                summary.append({"model": model, "dataset": dataset, "error": str(e)})

    print("\n=== SUMMARY ===")
    for s in summary:
        print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()

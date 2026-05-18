"""실험 3 Step 4: layer별 logit entropy · attention entropy · answer-token logit/prob 추출.

One forward per prompt (output_hidden_states + output_attentions) →
  - Logit Lens: hidden_states[ℓ][:, -1, :] @ lm_head.T → softmax → entropy (마지막 토큰 위치)
  - Attention entropy: attentions[ℓ][:, :, -1, :] (heads) → mean over heads → entropy
  - Answer-token logit/prob: final-layer logits[:, -1, :]에서 정답 첫 토큰 위치

저장 (cell별):
  _data/logit_attn/{model}/{dataset}/
    extra_metrics.npz : logit_entropy, attn_entropy, answer_logit, answer_prob, answer_token_id, ids
    meta.json
    done.marker
    progress.jsonl (resumable checkpoint)

Resumable: cell done.marker로 skip. cell 진행 중에도 progress.jsonl로 처리된 id skip.

실행:
    python 04_extract_logits_attn.py --models all --datasets all
    python 04_extract_logits_attn.py --models Qwen/Qwen2.5-1.5B-Instruct --datasets triviaqa
    python 04_extract_logits_attn.py --force  # 강제 재실행
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "_shared"))

from paths import DATA, PHASE1, PHASE1_MODELS, DATASETS
from model_loader import load_model, unload, model_n_layers
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_format import format_record
from eval_utils import extract_first_line
from resumable import (
    append_jsonl,
    atomic_write_json,
    cell_log,
    is_done,
    mark_done,
    processed_ids,
    read_jsonl,
)


# 출력 위치: _data/logit_attn/{model}/{dataset}/
LOGIT_ATTN_CACHE = DATA / "logit_attn"

# Llama·Qwen 모두 1B-7B 범위는 4 모델만 사용 (7B는 OOM 우려로 제외)
TARGET_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

# Phase 1 prompt 최대 길이. attention output (heads, seq, seq) 메모리 부담 큼.
MAX_PROMPT_LEN = 512


def logit_attn_cell_dir(model: str, dataset: str) -> Path:
    return LOGIT_ATTN_CACHE / model.replace("/", "__") / dataset


def _load_model_eager(name: str, trust_remote_code: bool = True, dtype: str = "bf16"):
    """attn_implementation=eager로 로딩 (output_attentions 지원 위해).

    NOTE: eager attention + fp16은 Qwen·일부 모델에서 softmax 분모 overflow로 NaN을
    만든다 (Qwen 1.5B 재현 확인). bf16은 exponent 범위가 넓어 안전. 4 target 모델
    (1B-3B)은 bf16으로 단일 12GB GPU에 fit. RTX 5070(sm_120) bf16 지원 확인됨.
    """
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        name,
        trust_remote_code=trust_remote_code,
        dtype=torch_dtype,
        attn_implementation="eager",
        device_map={"": "cuda"},  # 전체를 GPU에. offload 없음.
    )
    model.eval()
    return model, tokenizer


def phase1_cell_dir(model: str, dataset: str) -> Path:
    return PHASE1 / model.replace("/", "__") / dataset


def load_phase1_prompts(model: str, dataset: str) -> list[dict]:
    """Phase 1 generations.jsonl → [{id, question, answers, greedy, greedy_short}, ...]."""
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
            records.append({
                "id": str(rec["id"]),
                "question": rec["question"],
                "answers": answers,
                "dataset": rec.get("dataset", dataset),
                "greedy": greedy,
                "greedy_short": extract_first_line(greedy),
            })
    return records


def _get_lm_head_weight(model) -> torch.Tensor:
    """lm_head weight tensor. 일부 모델은 lm_head 속성 없이 get_output_embeddings 사용."""
    head = getattr(model, "lm_head", None)
    if head is None:
        head = model.get_output_embeddings()
    if head is None:
        raise RuntimeError("cannot locate lm_head / output embeddings")
    return head.weight  # (vocab, hidden)


def _entropy_from_logits(logits: torch.Tensor) -> float:
    """logits (vocab,) → softmax entropy in nats. fp32 stable."""
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    return float(-(p * logp).sum().item())


def _entropy_from_probs(probs: torch.Tensor) -> float:
    """이미 정규화된 attention weights (seq,) → entropy in nats."""
    p = probs.float()
    s = p.sum()
    if float(s) <= 0:
        return float("nan")
    p = p / s
    # 0 확률 mask
    nz = p > 0
    return float(-(p[nz] * p[nz].log()).sum().item())


def _resolve_answer_token_id(tokenizer, greedy_short: str, answers: list[str]) -> int | None:
    """정답 첫 토큰 ID 추출.

    우선순위:
      1. greedy_short 첫 단어 토큰
      2. answers[0] 첫 단어 토큰
    leading space 처리: BPE 모델 대부분 첫 단어 앞 공백을 별도 토큰화. " word" 형태로 인코딩 시도.
    """
    def first_token_id(text: str) -> int | None:
        text = (text or "").strip()
        if not text:
            return None
        first_word = text.split()[0]
        # leading space 포함 시도 (Llama/Qwen 모두 " word" 형태가 보통)
        for candidate in (f" {first_word}", first_word):
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if ids:
                return int(ids[0])
        return None

    tid = first_token_id(greedy_short)
    if tid is not None:
        return tid
    if answers:
        tid = first_token_id(answers[0])
        if tid is not None:
            return tid
    return None


@torch.no_grad()
def extract_extras(model, tokenizer, prompts: list[dict], cell_out: Path, device: str = "cuda") -> dict:
    """1 forward로 logit entropy + attention entropy + answer-token logit/prob 동시 추출.

    Resumable: cell_out/progress.jsonl에 처리된 id 누적. 재시작 시 skip.
    최종적으로 cell_out/extra_metrics.npz, meta.json, done.marker 저장.
    """
    cell_out.mkdir(parents=True, exist_ok=True)
    progress_path = cell_out / "progress.jsonl"
    npz_path = cell_out / "extra_metrics.npz"
    meta_path = cell_out / "meta.json"

    already = processed_ids(progress_path, id_key="id")

    n_layers = model_n_layers(model)  # transformer block 수 (embedding 제외)
    lm_head_w = _get_lm_head_weight(model)  # (vocab, hidden)

    # 누적 buffer (재시작 시 progress.jsonl에서 복원)
    rows: list[dict] = []
    if progress_path.exists():
        for rec in read_jsonl(progress_path):
            rows.append(rec)

    n_processed_run = 0
    t0 = time.time()

    for i, rec in enumerate(prompts):
        pid = str(rec["id"])
        if pid in already:
            continue

        prompt = format_record(tokenizer, rec)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_LEN,
        ).to(device)

        try:
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        # hidden_states: tuple len = n_layers + 1 (embedding 포함). 각 (1, seq, hidden)
        # attentions: tuple len = n_layers. 각 (1, heads, seq, seq)
        hs = outputs.hidden_states
        atts = outputs.attentions

        # Logit Lens entropy per layer (embedding 포함 → n_layers + 1)
        logit_ent = []
        for h in hs:
            last_h = h[0, -1, :]  # (hidden,)
            # vocab projection: (vocab,)
            logits = last_h.to(lm_head_w.dtype) @ lm_head_w.t()
            logit_ent.append(_entropy_from_logits(logits))
        logit_ent = np.array(logit_ent, dtype=np.float32)  # (n_layers + 1,)

        # Attention entropy per layer (heads 평균 후 entropy)
        attn_ent = []
        for a in atts:
            # a: (1, heads, seq, seq). 마지막 query 위치의 attention.
            last_attn = a[0, :, -1, :]  # (heads, seq) — softmax 정규화 이미 적용됨
            mean_attn = last_attn.mean(dim=0)  # (seq,)
            attn_ent.append(_entropy_from_probs(mean_attn))
        attn_ent = np.array(attn_ent, dtype=np.float32)  # (n_layers,)

        # Answer-token logit·probability (final-layer)
        final_logits = outputs.logits[0, -1, :]  # (vocab,)
        ans_tid = _resolve_answer_token_id(tokenizer, rec.get("greedy_short", ""), rec.get("answers", []))
        if ans_tid is None or ans_tid >= final_logits.shape[-1]:
            ans_logit = float("nan")
            ans_prob = float("nan")
            ans_tid_save = -1
        else:
            ans_logit = float(final_logits[ans_tid].item())
            probs = F.softmax(final_logits.float(), dim=-1)
            ans_prob = float(probs[ans_tid].item())
            ans_tid_save = int(ans_tid)

        row = {
            "id": pid,
            "logit_entropy": logit_ent.tolist(),
            "attn_entropy": attn_ent.tolist(),
            "answer_logit": ans_logit,
            "answer_prob": ans_prob,
            "answer_token_id": ans_tid_save,
        }
        rows.append(row)
        append_jsonl(progress_path, row)

        # 메모리 정리 (큰 attention tensor)
        del outputs, hs, atts, final_logits

        n_processed_run += 1
        if n_processed_run % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(prompts)}] processed (run={n_processed_run}, elapsed={elapsed:.0f}s)", flush=True)

    # 최종 npz 저장
    if not rows:
        raise RuntimeError(f"no rows extracted for {cell_out}")

    ids_arr = np.array([r["id"] for r in rows], dtype=object)
    logit_ent_arr = np.array([r["logit_entropy"] for r in rows], dtype=np.float32)
    attn_ent_arr = np.array([r["attn_entropy"] for r in rows], dtype=np.float32)
    answer_logit_arr = np.array([r["answer_logit"] for r in rows], dtype=np.float32)
    answer_prob_arr = np.array([r["answer_prob"] for r in rows], dtype=np.float32)
    answer_tid_arr = np.array([r["answer_token_id"] for r in rows], dtype=np.int64)

    tmp = npz_path.parent / (npz_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        logit_entropy=logit_ent_arr,
        attn_entropy=attn_ent_arr,
        answer_logit=answer_logit_arr,
        answer_prob=answer_prob_arr,
        answer_token_id=answer_tid_arr,
        ids=ids_arr,
    )
    import os
    os.replace(tmp, npz_path)

    meta = {
        "n_prompts": len(rows),
        "n_layers": n_layers,
        "n_layers_p1": logit_ent_arr.shape[1],  # embedding 포함
        "shapes": {
            "logit_entropy": list(logit_ent_arr.shape),
            "attn_entropy": list(attn_ent_arr.shape),
            "answer_logit": list(answer_logit_arr.shape),
            "answer_prob": list(answer_prob_arr.shape),
        },
        "max_prompt_len": MAX_PROMPT_LEN,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(meta_path, meta)
    return {"n_done": len(rows), "n_layers": n_layers, "n_layers_p1": logit_ent_arr.shape[1]}


def run_one_cell(model_name: str, dataset: str, force: bool = False) -> dict:
    cell = logit_attn_cell_dir(model_name, dataset)
    if is_done(cell) and not force:
        print(f"[skip] {model_name} / {dataset} — already done", flush=True)
        return {"cell": str(cell), "status": "skipped"}

    print(f"\n[load] Phase 1 prompts {model_name} / {dataset}", flush=True)
    prompts = load_phase1_prompts(model_name, dataset)
    print(f"  n_prompts: {len(prompts)}", flush=True)

    print(f"[load_model] {model_name} (attn_implementation=eager, bf16)", flush=True)
    t0 = time.time()
    # SDPA backend는 output_attentions 미지원 → 처음부터 eager attention으로 로드.
    # bf16 사용: fp16 eager softmax overflow로 NaN 발생함.
    model, tokenizer = _load_model_eager(model_name, dtype="bf16")
    print(f"  loaded in {time.time() - t0:.1f}s, n_layers: {model_n_layers(model)}", flush=True)

    print(f"[extract_extras] {model_name} / {dataset}", flush=True)
    t0 = time.time()
    with cell_log(cell, "extract_logits_attn"):
        result = extract_extras(model, tokenizer, prompts, cell)
    elapsed = time.time() - t0
    print(f"  extracted in {elapsed:.1f}s, n_done: {result['n_done']}", flush=True)

    mark_done(cell, {
        "n_prompts": result["n_done"],
        "n_layers": result["n_layers"],
        "n_layers_p1": result["n_layers_p1"],
        "elapsed_sec": elapsed,
    })

    unload(model)
    return {"cell": str(cell), "status": "ok", **result, "elapsed_sec": elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"], help="모델 이름 또는 'all' (Qwen 7B 제외 4 모델)")
    ap.add_argument("--datasets", nargs="+", default=["all"], help="데이터셋 또는 'all'")
    ap.add_argument("--force", action="store_true", help="done.marker 무시하고 재실행")
    args = ap.parse_args()

    models = TARGET_MODELS if args.models == ["all"] else args.models
    datasets = DATASETS if args.datasets == ["all"] else args.datasets

    LOGIT_ATTN_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {LOGIT_ATTN_CACHE}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")

    summary = []
    for model_name in models:
        for dataset in datasets:
            try:
                r = run_one_cell(model_name, dataset, force=args.force)
                summary.append({"model": model_name, "dataset": dataset, **r})
            except Exception as e:
                import traceback
                traceback.print_exc()
                summary.append({"model": model_name, "dataset": dataset, "error": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY ===")
    for s in summary:
        print(json.dumps(s, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

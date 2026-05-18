"""Hidden state 추출. Layer별 last-token hidden 저장. Resumable.

저장 형식 (cell별):
    hidden.npz: {layer_0, layer_1, ..., layer_L} 각 shape=(N, hidden_dim) fp16
    meta.json: {model, dataset, n_prompts, n_layers, hidden_dim, ids}
    prompts.jsonl: prompt별 raw (id, question, gold_answers, generated, correct)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from resumable import append_jsonl, atomic_write_json, cell_log, is_done, mark_done, processed_ids


@torch.no_grad()
def extract_last_token_hidden(model, tokenizer, prompt: str, device: str = "cuda") -> np.ndarray:
    """1 forward → 모든 layer의 마지막 토큰 hidden state.

    Returns: array shape (n_layers+1, hidden_dim), fp16
        (n_layers+1 = embedding layer 포함)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    # hidden_states: tuple of (batch, seq, hidden) for each layer (n_layers+1 total)
    hs = outputs.hidden_states
    last = [h[0, -1].detach().cpu().to(torch.float16).numpy() for h in hs]
    return np.stack(last, axis=0)  # (n_layers+1, hidden_dim)


def extract_for_cell(
    model,
    tokenizer,
    prompts: list[dict],  # [{id, question, ...}, ...]
    cell: Path,
    prompt_format_fn,  # (record) -> str
    batch_save_every: int = 50,
    device: str = "cuda",
) -> dict:
    """한 cell (모델 x 데이터셋) 전체 처리. Resumable.

    prompts[i]는 다음 키 보유 가정: id (str), question (str), answers (list)
    prompt_format_fn(record) → 모델에 넣을 최종 prompt 문자열.

    Returns: {n_done, n_layers, hidden_dim, hidden_path, prompts_path}
    """
    if is_done(cell):
        return {"status": "skipped_already_done", **_load_meta(cell)}

    cell.mkdir(parents=True, exist_ok=True)
    prompts_path = cell / "prompts.jsonl"
    hidden_path = cell / "hidden.npz"
    meta_path = cell / "meta.json"

    already = processed_ids(prompts_path, id_key="id")

    with cell_log(cell, "extract_hidden"):
        all_layers_hidden: list[np.ndarray] = []
        all_ids: list[str] = []

        # 기존 데이터 로드 (재시작 시)
        if hidden_path.exists():
            data = np.load(hidden_path)
            all_layers_hidden = [data[k] for k in data.files]
            all_ids = [rec["id"] for rec in _iter_jsonl(prompts_path)]

        n_layers_p1 = None
        hidden_dim = None
        n_processed_this_run = 0

        for i, rec in enumerate(prompts):
            pid = str(rec["id"])
            if pid in already:
                continue

            prompt = prompt_format_fn(rec)
            try:
                h = extract_last_token_hidden(model, tokenizer, prompt, device=device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                h = extract_last_token_hidden(model, tokenizer, prompt, device=device)

            if n_layers_p1 is None:
                n_layers_p1, hidden_dim = h.shape

            all_layers_hidden.append(h)
            all_ids.append(pid)

            append_jsonl(prompts_path, {
                "id": pid,
                "question": rec.get("question", "")[:500],
                "answers": rec.get("answers", []),
                "dataset": rec.get("dataset", ""),
            })

            n_processed_this_run += 1
            if n_processed_this_run % batch_save_every == 0:
                _save_hidden(hidden_path, all_layers_hidden, all_ids)
                _save_meta(meta_path, all_ids, n_layers_p1, hidden_dim)
                print(f"  [{i + 1}/{len(prompts)}] saved checkpoint", flush=True)

        _save_hidden(hidden_path, all_layers_hidden, all_ids)
        _save_meta(meta_path, all_ids, n_layers_p1 or 0, hidden_dim or 0)

    mark_done(cell, {"n_prompts": len(all_ids), "n_layers": n_layers_p1, "hidden_dim": hidden_dim})
    return {"status": "ok", "n_done": len(all_ids), "n_layers": n_layers_p1, "hidden_dim": hidden_dim}


def _save_hidden(path: Path, hiddens: list[np.ndarray], ids: list[str]) -> None:
    """N개 (n_layers+1, hidden_dim) → (N, n_layers+1, hidden_dim) 저장.
    Atomic write (tmp → rename)."""
    if not hiddens:
        return
    arr = np.stack(hiddens, axis=0)  # (N, n_layers+1, hidden_dim)
    # np.savez_compressed는 path가 .npz 안 끝나면 자동 추가. 명시적 .npz로.
    tmp = path.parent / (path.stem + ".tmp.npz")
    # layer별로 저장 (메모리 효율, 부분 로드 가능)
    by_layer = {f"layer_{i:02d}": arr[:, i, :] for i in range(arr.shape[1])}
    by_layer["ids"] = np.array(ids, dtype=object)
    np.savez_compressed(tmp, **by_layer)
    import os
    os.replace(tmp, path)


def _save_meta(path: Path, ids: list[str], n_layers_p1: int, hidden_dim: int) -> None:
    atomic_write_json(path, {
        "n_prompts": len(ids),
        "n_layers_p1": n_layers_p1,
        "hidden_dim": hidden_dim,
        "ids": ids,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


def _load_meta(cell: Path) -> dict:
    p = cell / "meta.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_hidden_cell(cell: Path) -> tuple[dict, dict, list[dict]]:
    """저장된 cell 로드. (hidden_by_layer, meta, prompts).
    hidden_by_layer[k] shape = (N, hidden_dim) fp16
    """
    hidden_path = cell / "hidden.npz"
    if not hidden_path.exists():
        raise FileNotFoundError(hidden_path)
    data = np.load(hidden_path, allow_pickle=True)
    by_layer = {k: data[k] for k in data.files if k.startswith("layer_")}
    ids = data["ids"].tolist() if "ids" in data.files else []

    meta = _load_meta(cell)
    prompts = list(_iter_jsonl(cell / "prompts.jsonl"))
    return by_layer, {**meta, "ids": ids}, prompts

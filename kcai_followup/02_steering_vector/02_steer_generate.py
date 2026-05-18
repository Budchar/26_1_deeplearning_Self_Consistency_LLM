"""실험 2 Step 2: direction을 hook으로 hidden state에 더해 generation 수행.

각 (model, dataset, layer, α)에 대해:
  hook: residual stream에서 h_l → h_l + α * d_l
  generation: Phase 1 prompt 200개 (subsample) greedy 생성
  정답률 평가 (eval_utils.is_correct)

기본:
  - target_layer: direction norm이 최대인 layer (또는 모든 mid-late layer)
  - alphas: [-2, -1, -0.5, 0, 0.5, 1, 2]
  - n_prompts: 200 (시간 절약)
  - models: Qwen2.5-1.5B, 3B-Instruct부터 (작은 것)

입력: directions/{model}__{dataset}_directions.npz, Phase 1 generations.jsonl
출력: results/{model}__{dataset}__L{layer}__alpha{α}.jsonl (id, generated, correct)
       results/_aggregate.json (α별 정답률 표)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "_shared"))

from paths import PHASE1, hidden_cell_dir, EXP2_STEERING
from model_loader import load_model, unload
from prompt_format import format_record
from eval_utils import is_correct, extract_first_line
from resumable import append_jsonl, atomic_write_json, processed_ids


DIRECTIONS = EXP2_STEERING / "directions"
RESULTS = EXP2_STEERING / "results"


def load_phase1_prompts(model: str, dataset: str, n: int = 200) -> list[dict]:
    """Phase 1 generations에서 첫 n개 prompt."""
    gen_path = PHASE1 / model.replace("/", "__") / dataset / "generations.jsonl"
    recs = []
    with open(gen_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            recs.append({
                "id": str(rec["id"]),
                "question": rec["question"],
                "answers": rec.get("answers", []),
                "dataset": dataset,
            })
            if len(recs) >= n:
                break
    return recs


def load_directions(model: str, dataset: str) -> dict[str, np.ndarray]:
    path = DIRECTIONS / f"{model.replace('/', '__')}__{dataset}_directions.npz"
    data = np.load(path)
    return {k: data[k] for k in data.files}


class SteeringHook:
    """residual stream에 d 벡터 더하는 forward pre-hook."""

    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction  # (hidden_dim,)
        self.alpha = alpha
        self.handle = None

    def install(self, module: torch.nn.Module) -> None:
        def fwd_hook(mod, inputs, outputs):
            # transformer block output: (hidden, ...) or hidden
            if isinstance(outputs, tuple):
                hidden = outputs[0]
                rest = outputs[1:]
            else:
                hidden = outputs
                rest = None
            # hidden shape: (batch, seq, hidden_dim) — 모든 토큰에 동일 direction 추가
            hidden = hidden + self.alpha * self.direction.to(hidden.dtype).to(hidden.device)
            return (hidden,) + rest if rest else hidden

        self.handle = module.register_forward_hook(fwd_hook)

    def remove(self) -> None:
        if self.handle:
            self.handle.remove()
            self.handle = None


def get_layer_module(model, layer_idx: int):
    """transformer block (layer) module 가져오기. Llama/Qwen/Mistral 호환."""
    # 흔한 path: model.model.layers[i]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    # GPT-NeoX style: model.gpt_neox.layers[i]
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers[layer_idx]
    raise AttributeError(f"cannot locate layer modules for {type(model).__name__}")


@torch.no_grad()
def generate_with_steering(model, tokenizer, prompt: str, direction: torch.Tensor, alpha: float, layer_idx: int, max_new_tokens: int = 80) -> str:
    """direction을 hook으로 layer_idx에 주입한 채 generation."""
    hook = SteeringHook(direction, alpha)
    layer_mod = get_layer_module(model, layer_idx)
    hook.install(layer_mod)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text
    finally:
        hook.remove()


def run_one_cell(model_name: str, dataset: str, alphas: list[float], n_prompts: int, target_layer: int | None = None) -> dict:
    print(f"\n[load_dirs] {model_name}/{dataset}", flush=True)
    directions = load_directions(model_name, dataset)
    layer_keys = sorted([k for k in directions.keys() if k.startswith("layer_")])
    # target layer: 명시 안 되면 direction norm 최대인 mid-late layer (rel_d 0.4-0.9)
    n_layers = len(layer_keys)
    if target_layer is None:
        mid_late = [(i, np.linalg.norm(directions[k])) for i, k in enumerate(layer_keys) if 0.4 <= i / (n_layers - 1) <= 0.9]
        target_layer = max(mid_late, key=lambda x: x[1])[0]
    target_key = layer_keys[target_layer]
    d = torch.from_numpy(directions[target_key].astype(np.float32))
    print(f"  target_layer={target_layer} (rel_d={target_layer / (n_layers - 1):.2f}), ||d||={d.norm().item():.2f}", flush=True)

    prompts = load_phase1_prompts(model_name, dataset, n=n_prompts)
    print(f"  n_prompts={len(prompts)}", flush=True)

    print(f"[load_model] {model_name}", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(model_name, dtype="fp16")
    # transformer block 위치 확인 (실패 시 빠르게 fail)
    get_layer_module(model, target_layer)

    cell_out = RESULTS / f"{model_name.replace('/', '__')}__{dataset}"
    cell_out.mkdir(parents=True, exist_ok=True)
    aggregate = {}

    for alpha in alphas:
        result_path = cell_out / f"alpha_{alpha:+.2f}_L{target_layer}.jsonl"
        already = processed_ids(result_path, id_key="id")
        n_done = 0
        n_correct = 0
        ta = time.time()
        for p in prompts:
            if p["id"] in already:
                continue
            prompt_text = format_record(tokenizer, p)
            try:
                gen = generate_with_steering(model, tokenizer, prompt_text, d, alpha, target_layer)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gen = generate_with_steering(model, tokenizer, prompt_text, d, alpha, target_layer)
            short = extract_first_line(gen)
            correct = is_correct(short, p["answers"])
            append_jsonl(result_path, {"id": p["id"], "generated": gen, "short": short, "correct": correct})
            n_done += 1
            n_correct += int(correct)

        # 누적 정답률 (지금까지 처리된 모든 prompt)
        all_correct = sum(1 for r in _read_jsonl(result_path) if r.get("correct"))
        all_n = sum(1 for _ in _read_jsonl(result_path))
        acc = all_correct / max(1, all_n)
        aggregate[f"alpha_{alpha:+.2f}"] = {"acc": acc, "n": all_n, "this_run_n": n_done, "this_run_acc": n_correct / max(1, n_done), "time_sec": time.time() - ta}
        print(f"  alpha={alpha:+.2f}: acc={acc:.3f} ({all_correct}/{all_n}), this_run={n_done} prompts in {time.time() - ta:.0f}s", flush=True)

    summary_path = cell_out / "_summary.json"
    atomic_write_json(summary_path, {
        "model": model_name,
        "dataset": dataset,
        "target_layer": target_layer,
        "target_rel_depth": target_layer / (n_layers - 1),
        "n_prompts": len(prompts),
        "alpha_results": aggregate,
        "compute_time_sec": time.time() - t0,
    })

    unload(model)
    return {"status": "ok", "summary": str(summary_path), "alpha_results": aggregate}


def _read_jsonl(path: Path):
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"])
    ap.add_argument("--datasets", nargs="+", default=["triviaqa"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--target_layer", type=int, default=None, help="None이면 자동 (direction norm max layer in rel_d 0.4-0.9)")
    args = ap.parse_args()

    summary = []
    for model in args.models:
        for dataset in args.datasets:
            try:
                r = run_one_cell(model, dataset, args.alphas, args.n_prompts, args.target_layer)
                summary.append({"model": model, "dataset": dataset, **r})
            except Exception as e:
                import traceback
                traceback.print_exc()
                summary.append({"model": model, "dataset": dataset, "error": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY ===")
    for s in summary:
        print(json.dumps(s, ensure_ascii=False, default=str)[:500])


if __name__ == "__main__":
    main()

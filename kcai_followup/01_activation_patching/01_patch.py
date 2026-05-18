"""실험 1 Activation patching.

두 모델 (source, target) 같은 base 다른 fine-tuning 비교:
  - source에서 prompt forward, layer ℓ의 MLP output 캐싱
  - target에서 같은 prompt forward, 같은 layer ℓ MLP output을 source 값으로 교체 (hook)
  - target 출력 (greedy generation) 측정

비교:
  - target unmodified accuracy vs target+patch accuracy
  - control: patch layer ≠ critical layer (예: L0) → 효과 없음 확인

먼저 PoC: Qwen2.5-1.5B (base) vs Qwen2.5-1.5B-Instruct (instruct), 같은 architecture
Mistral 다운로드 후: OpenHermes vs NousHermes-DPO 등 동일 base 변형

자원 제약 (5070 12GB):
  - 두 모델 동시 로딩 불가 (7B). 1.5B는 6GB 가능
  - sequential: source 미리 cache → target만 로딩 + 주입

출력:
  results/{src}__vs__{tgt}__{dataset}/
    cache_src/{prompt_id}.npz    (layer별 MLP output)
    patched/L{layer}.jsonl       (id, generated, correct)
    baseline/no_patch.jsonl      (id, generated, correct)
    _summary.json                (layer별 acc delta)
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

from paths import PHASE1, EXP1_PATCHING
from model_loader import load_model, unload
from prompt_format import format_record
from eval_utils import is_correct, extract_first_line
from resumable import append_jsonl, atomic_write_json, processed_ids


RESULTS = EXP1_PATCHING / "results"


def load_phase1_prompts(model: str, dataset: str, n: int = 100) -> list[dict]:
    """Phase 1 generations.jsonl 우선. 없으면 데이터셋 직접 로드 (dl_team_v2 data_loader)."""
    gen_path = PHASE1 / model.replace("/", "__") / dataset / "generations.jsonl"
    if gen_path.exists():
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

    # Fallback: dl_team_v2 data_loader 직접 호출
    from data_loader import load_dataset_by_name
    recs = load_dataset_by_name(dataset, n=n, seed=42)
    return recs[:n]


def get_mlp_module(model, layer_idx: int):
    """transformer block 내 MLP (FFN) sub-module 가져오기."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[layer_idx]
        # Llama / Qwen / Mistral 공통: layer.mlp
        if hasattr(layer, "mlp"):
            return layer.mlp
    raise AttributeError(f"cannot locate MLP for {type(model).__name__} layer {layer_idx}")


@torch.no_grad()
def cache_mlp_outputs(model, tokenizer, prompt: str, layers: list[int], device: str = "cuda") -> dict[int, torch.Tensor]:
    """forward 1회로 지정 layer들의 MLP output 캐싱.
    Returns: {layer_idx: tensor (seq, hidden), cpu}
    """
    cache = {}
    handles = []

    def make_hook(li):
        def hook(mod, inputs, output):
            # MLP output: tensor (batch, seq, hidden)
            cache[li] = output[0].detach().cpu()
        return hook

    for li in layers:
        h = get_mlp_module(model, li).register_forward_hook(make_hook(li))
        handles.append(h)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    model(**inputs)

    for h in handles:
        h.remove()
    return cache


class PatchHook:
    """target 모델 layer의 MLP output을 source 값으로 교체."""

    def __init__(self, source_mlp_out: torch.Tensor, debug: bool = False):
        self.src = source_mlp_out  # cpu tensor (seq, hidden)
        self.handle = None
        self.debug = debug
        self.n_called = 0
        self.n_patched = 0

    def install(self, mod):
        def hook(m, inputs, output):
            self.n_called += 1
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None
            # hidden: (batch, seq, hidden); src: (seq, hidden)
            if self.debug and self.n_called == 1:
                print(f"    [PatchHook] hidden shape={tuple(hidden.shape)} src shape={tuple(self.src.shape)}", flush=True)
            if hidden.shape[1] == self.src.shape[0]:
                hidden = self.src.unsqueeze(0).to(hidden.dtype).to(hidden.device)
                self.n_patched += 1
            return (hidden,) + rest if rest else hidden
        self.handle = mod.register_forward_hook(hook)

    def remove(self):
        if self.handle:
            self.handle.remove()


@torch.no_grad()
def generate_with_patch(model, tokenizer, prompt: str, mlp_cache_per_layer: dict[int, torch.Tensor], max_new_tokens: int = 80, debug: bool = False) -> tuple[str, dict]:
    handles = []
    for li, src in mlp_cache_per_layer.items():
        ph = PatchHook(src, debug=debug)
        ph.install(get_mlp_module(model, li))
        handles.append(ph)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        stats = {f"L{li}": {"called": h.n_called, "patched": h.n_patched} for li, h in zip(mlp_cache_per_layer.keys(), handles)}
        return text, stats
    finally:
        for h in handles:
            h.remove()


def run_pair(source_model_name: str, target_model_name: str, dataset: str, patch_layers: list[int], control_layers: list[int], n_prompts: int = 100, dtype: str = "fp16") -> dict:
    """source MLP → target patching. n_prompts 처리. patch_layers·control_layers 각각 실험."""
    pair_dir = RESULTS / f"{source_model_name.replace('/', '__')}__vs__{target_model_name.replace('/', '__')}__{dataset}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = pair_dir / "cache_src"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1에 source 데이터 없을 수 있음 (base 모델 등). target 기준으로 로드.
    prompts = load_phase1_prompts(target_model_name, dataset, n=n_prompts)
    print(f"\n[{source_model_name} → {target_model_name}, {dataset}] n={len(prompts)} prompts, patch_layers={patch_layers}, control={control_layers}", flush=True)

    # Step 1: source 모델에서 prompt별 MLP output 캐싱
    print(f"[step1] cache source MLP outputs", flush=True)
    all_target_layers = set(patch_layers + control_layers)
    t0 = time.time()
    src_model, src_tok = load_model(source_model_name, dtype=dtype)
    for p in prompts:
        cache_file = cache_dir / f"{p['id']}.npz"
        if cache_file.exists():
            continue
        prompt_text = format_record(src_tok, p, force_plain=True)
        try:
            cache = cache_mlp_outputs(src_model, src_tok, prompt_text, sorted(all_target_layers))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            cache = cache_mlp_outputs(src_model, src_tok, prompt_text, sorted(all_target_layers))
        np.savez_compressed(cache_file, **{f"L{li}": cache[li].to(torch.float16).numpy() for li in cache})
    print(f"  cached in {time.time() - t0:.0f}s", flush=True)
    unload(src_model)
    torch.cuda.empty_cache()

    # Step 2: target 모델 로딩, baseline + patch + control 각각 generation
    print(f"[step2] load target + generate", flush=True)
    t0 = time.time()
    tgt_model, tgt_tok = load_model(target_model_name, dtype=dtype)

    # 2a. baseline (no patch) — patched와 같은 prompt format 사용 (force_plain)
    baseline_path = pair_dir / "baseline.jsonl"
    already = processed_ids(baseline_path, id_key="id")
    for p in prompts:
        if p["id"] in already:
            continue
        prompt_text = format_record(tgt_tok, p, force_plain=True)
        inputs = tgt_tok(prompt_text, return_tensors="pt").to(tgt_model.device)
        out = tgt_model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tgt_tok.eos_token_id)
        text = tgt_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        short = extract_first_line(text)
        correct = is_correct(short, p["answers"])
        append_jsonl(baseline_path, {"id": p["id"], "generated": text, "short": short, "correct": correct})
    base_acc = _acc(baseline_path)
    print(f"  baseline acc={base_acc:.3f}, {time.time() - t0:.0f}s", flush=True)

    # 2b. patch experiments
    results = {"baseline_acc": base_acc, "patch_results": {}, "control_results": {}}
    for cond_name, layers in [("patch", patch_layers), ("control", control_layers)]:
        for li in layers:
            out_path = pair_dir / f"{cond_name}_L{li}.jsonl"
            already = processed_ids(out_path, id_key="id")
            ta = time.time()
            first_in_layer = True
            for p in prompts:
                if p["id"] in already:
                    continue
                # src cache 로딩
                cache_file = cache_dir / f"{p['id']}.npz"
                if not cache_file.exists():
                    continue
                data = np.load(cache_file)
                if f"L{li}" not in data.files:
                    continue
                src_mlp = torch.from_numpy(data[f"L{li}"].astype(np.float32))
                prompt_text = format_record(tgt_tok, p, force_plain=True)
                _debug = first_in_layer
                try:
                    gen, pstats = generate_with_patch(tgt_model, tgt_tok, prompt_text, {li: src_mlp}, debug=_debug)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gen, pstats = generate_with_patch(tgt_model, tgt_tok, prompt_text, {li: src_mlp})
                if first_in_layer:
                    print(f"    [debug L{li}] hook stats={pstats}", flush=True)
                    first_in_layer = False
                short = extract_first_line(gen)
                correct = is_correct(short, p["answers"])
                append_jsonl(out_path, {"id": p["id"], "generated": gen, "short": short, "correct": correct})
            acc = _acc(out_path)
            (results["patch_results"] if cond_name == "patch" else results["control_results"])[f"L{li}"] = {"acc": acc, "delta": acc - base_acc, "time_sec": time.time() - ta}
            print(f"  {cond_name} L{li}: acc={acc:.3f} (Δ={acc - base_acc:+.3f}), {time.time() - ta:.0f}s", flush=True)

    unload(tgt_model)

    summary_path = pair_dir / "_summary.json"
    atomic_write_json(summary_path, {
        "source": source_model_name,
        "target": target_model_name,
        "dataset": dataset,
        "n_prompts": len(prompts),
        "patch_layers": patch_layers,
        "control_layers": control_layers,
        **results,
    })
    return results


def _acc(path: Path) -> float:
    n = 0
    nc = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            nc += int(r.get("correct", False))
    return nc / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--datasets", nargs="+", default=["triviaqa"])
    ap.add_argument("--patch_layers", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--control_layers", nargs="+", type=int, default=[0, 20])
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "8bit", "4bit"])
    args = ap.parse_args()

    summary = []
    for dataset in args.datasets:
        try:
            r = run_pair(args.source, args.target, dataset, args.patch_layers, args.control_layers, args.n_prompts, dtype=args.dtype)
            summary.append({"dataset": dataset, **r})
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary.append({"dataset": dataset, "error": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY ===")
    for s in summary:
        print(json.dumps(s, ensure_ascii=False, default=str)[:500])


if __name__ == "__main__":
    main()

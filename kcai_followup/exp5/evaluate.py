"""실험 5 Step 3 — B와 C 모델 평가.

각 adapter (B, C)에 대해:
  1. base에 adapter 합쳐서 모델 로드
  2. TriviaQA·NQ-Open·SQuAD 1000 prompt씩 forward
  3. 각 layer last-token hidden state 추출
  4. Layer probe (5-fold CV logistic regression) → AUROC per layer
  5. peak layer 추출
  6. B vs C 비교 plot

실행 예:
    python evaluate.py \
        --base mistralai/Mistral-7B-v0.1 \
        --adapters B_adapter C_adapter \
        --datasets triviaqa nq_open squad \
        --n_prompts 1000 \
        --output ./results_exp5

자원: A100 40GB 7B fp16+adapter ~ 약 15GB · 모델당 약 1시간 추출 + probe 30분
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# Prompt format (실험 3·4와 동일)
# ============================================================

SYSTEM_QA = "You are a helpful assistant that answers questions concisely."


def format_qa_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_QA},
        {"role": "user", "content": f"Question: {question}\nGive a short factual answer."},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{SYSTEM_QA}\n\nQuestion: {question}\nGive a short factual answer."


# ============================================================
# Eval utils (실험 3·4와 동일)
# ============================================================

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = _ARTICLE_RE.sub(" ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_correct(prediction: str, gold_answers: list[str]) -> bool:
    if not prediction or not gold_answers:
        return False
    pred = normalize_answer(prediction)
    for ga in gold_answers:
        gold = normalize_answer(ga)
        if gold and (pred == gold or gold in pred):
            return True
    return False


def extract_first_line(text: str) -> str:
    text = text.strip()
    line = text.split("\n", 1)[0].strip()
    for stop in [". ", "? ", "! ", "."]:
        if stop in line:
            line = line.split(stop)[0].strip()
            break
    return line


# ============================================================
# Dataset loading (TriviaQA·NQ-Open·SQuAD)
# ============================================================

def load_dataset_prompts(name: str, n: int = 1000, seed: int = 42) -> list[dict]:
    if name == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        recs = []
        for ex in ds.shuffle(seed=seed).select(range(min(n, len(ds)))):
            recs.append({
                "id": ex["question_id"],
                "question": ex["question"],
                "answers": ex["answer"]["aliases"] + [ex["answer"]["value"]],
            })
        return recs
    elif name == "nq_open":
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
        recs = []
        for i, ex in enumerate(ds.shuffle(seed=seed).select(range(min(n, len(ds))))):
            recs.append({
                "id": f"nq_{i}",
                "question": ex["question"],
                "answers": ex["answer"],
            })
        return recs
    elif name == "squad":
        ds = load_dataset("rajpurkar/squad", split="validation")
        recs = []
        for ex in ds.shuffle(seed=seed).select(range(min(n, len(ds)))):
            recs.append({
                "id": ex["id"],
                "question": ex["question"],
                "answers": ex["answers"]["text"],
            })
        return recs
    else:
        raise ValueError(f"unknown dataset: {name}")


# ============================================================
# Hidden state extraction
# ============================================================

@torch.no_grad()
def extract_last_token_hidden(model, tokenizer, prompt: str, device: str = "cuda") -> np.ndarray:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hs = outputs.hidden_states
    last = [h[0, -1].detach().cpu().to(torch.float16).numpy() for h in hs]
    return np.stack(last, axis=0)


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ============================================================
# Layer probe (실험 4와 동일 절차)
# ============================================================

def probe_layer(H: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 42) -> dict:
    H = H.astype(np.float32)
    if y.sum() < n_folds or (~y).sum() < n_folds:
        return {"auroc_mean": float("nan"), "auroc_std": float("nan"), "error": "imbalance"}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aurocs = []
    for tr, te in skf.split(H, y):
        scaler = StandardScaler()
        Htr = scaler.fit_transform(H[tr])
        Hte = scaler.transform(H[te])
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(Htr, y[tr])
        proba = clf.predict_proba(Hte)[:, 1]
        try:
            aurocs.append(roc_auc_score(y[te], proba))
        except ValueError:
            aurocs.append(float("nan"))
    arr = np.array(aurocs)
    return {
        "auroc_mean": float(np.nanmean(arr)),
        "auroc_std": float(np.nanstd(arr)),
        "n_folds": n_folds,
    }


# ============================================================
# Per-adapter pipeline
# ============================================================

def evaluate_adapter(base_name: str, adapter_path: Path, datasets: list[str], n_prompts: int, out_dir: Path) -> dict:
    label = adapter_path.name
    print(f"\n=== Evaluating {label} ({adapter_path}) ===", flush=True)
    cache_dir = out_dir / f"{label}_hidden_states"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load base + adapter (fp16 inference)
    print(f"[load] base={base_name} + adapter={adapter_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name, dtype=torch.float16, device_map={"": 0}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    all_results = {}
    for dataset in datasets:
        print(f"\n[{label}/{dataset}] loading prompts...", flush=True)
        prompts = load_dataset_prompts(dataset, n=n_prompts)
        print(f"  n={len(prompts)} prompts")

        # 1) generate greedy + correctness
        print(f"[{label}/{dataset}] generating + extracting hidden states...", flush=True)
        t0 = time.time()
        labels_list = []
        hidden_list = []
        ids = []
        for i, p in enumerate(prompts):
            prompt_text = format_qa_prompt(tokenizer, p["question"])
            try:
                gen = greedy_generate(model, tokenizer, prompt_text, max_new_tokens=80)
                h = extract_last_token_hidden(model, tokenizer, prompt_text)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gen = greedy_generate(model, tokenizer, prompt_text, max_new_tokens=80)
                h = extract_last_token_hidden(model, tokenizer, prompt_text)
            short = extract_first_line(gen)
            correct = is_correct(short, p["answers"])
            labels_list.append(int(correct))
            hidden_list.append(h)
            ids.append(p["id"])
            if (i + 1) % 100 == 0:
                print(f"    [{i + 1}/{len(prompts)}] ({time.time() - t0:.0f}s) acc so far {np.mean(labels_list):.3f}", flush=True)
        elapsed = time.time() - t0
        acc = float(np.mean(labels_list))
        print(f"  done in {elapsed:.0f}s, acc={acc:.3f} ({sum(labels_list)}/{len(labels_list)})")

        # Save cache
        H = np.stack(hidden_list, axis=0)  # (N, n_layers+1, hidden_dim)
        cache_file = cache_dir / f"{dataset}.npz"
        np.savez_compressed(
            cache_file,
            hidden=H,
            labels=np.array(labels_list, dtype=bool),
            ids=np.array(ids, dtype=object),
        )

        # 2) layer probe
        print(f"[{label}/{dataset}] running layer probes...", flush=True)
        y = np.array(labels_list, dtype=bool)
        n_layers_p1 = H.shape[1]
        layer_results = []
        t0 = time.time()
        for layer in range(n_layers_p1):
            r = probe_layer(H[:, layer, :], y)
            r["layer"] = layer
            r["rel_depth"] = layer / max(1, n_layers_p1 - 1)
            layer_results.append(r)

        aurocs = np.array([r["auroc_mean"] for r in layer_results])
        aurocs_safe = np.where(np.isnan(aurocs), -np.inf, aurocs)
        peak_layer = int(aurocs_safe.argmax())
        peak_auroc = float(aurocs[peak_layer])
        peak_rel_depth = peak_layer / max(1, n_layers_p1 - 1)

        print(f"  peak layer L{peak_layer} (rel_d={peak_rel_depth:.2f}), AUROC={peak_auroc:.3f} ({time.time() - t0:.0f}s)")

        all_results[dataset] = {
            "n_prompts": len(prompts),
            "n_correct": int(sum(labels_list)),
            "accuracy": acc,
            "n_layers": n_layers_p1,
            "peak_layer": peak_layer,
            "peak_rel_depth": peak_rel_depth,
            "peak_auroc": peak_auroc,
            "layer_results": layer_results,
        }

    # Unload model
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Save per-adapter summary
    out_json = out_dir / f"{label}_probe_results.json"
    out_json.write_text(json.dumps({"adapter": label, "results": all_results}, indent=2, ensure_ascii=False))
    print(f"\n[{label}] saved → {out_json}")
    return all_results


# ============================================================
# Compare + plot B vs C
# ============================================================

def make_comparison_plot(all_results: dict, out_path: Path):
    """B와 C의 layer 별 AUROC trajectory + peak 비교."""
    datasets = list(next(iter(all_results.values())).keys())
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(n_datasets * 5, 4), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        for adapter_label, ds_results in all_results.items():
            if ds not in ds_results:
                continue
            lr = ds_results[ds]["layer_results"]
            rel_d = [r["rel_depth"] for r in lr]
            auroc = [r["auroc_mean"] for r in lr]
            ax.plot(rel_d, auroc, "-o", label=adapter_label, markersize=4, lw=1.5)
            peak_rel = ds_results[ds]["peak_rel_depth"]
            peak_auroc = ds_results[ds]["peak_auroc"]
            ax.scatter([peak_rel], [peak_auroc], s=100, marker="*", zorder=5)
        ax.axvspan(0.55, 0.81, color="green", alpha=0.10, label="H3 band")
        ax.axhline(0.5, color="gray", linestyle=":", lw=1)
        ax.set_xlabel("relative depth")
        ax.set_title(ds)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Layer Probe AUROC (5-fold CV)")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("B (SFT) vs C (SFT + DPO) — Layer Probe Trajectory", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--adapters", nargs="+", required=True, help="평가할 adapter 폴더 (예: B_adapter C_adapter)")
    ap.add_argument("--datasets", nargs="+", default=["triviaqa", "nq_open", "squad"])
    ap.add_argument("--n_prompts", type=int, default=1000)
    ap.add_argument("--output", default="./results_exp5")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for adapter_path_str in args.adapters:
        adapter_path = Path(adapter_path_str)
        if not adapter_path.exists():
            print(f"[error] adapter not found: {adapter_path}", file=sys.stderr)
            continue
        results = evaluate_adapter(args.base, adapter_path, args.datasets, args.n_prompts, out_dir)
        all_results[adapter_path.name] = results

    if len(all_results) >= 2:
        plot_path = out_dir / "comparison_plot.png"
        make_comparison_plot(all_results, plot_path)
        print(f"\ncomparison plot → {plot_path}")

    # Final summary
    summary = {
        "base": args.base,
        "datasets": args.datasets,
        "n_prompts": args.n_prompts,
        "adapters": list(all_results.keys()),
        "results": {
            label: {ds: {k: v for k, v in res.items() if k != "layer_results"} for ds, res in r.items()}
            for label, r in all_results.items()
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nsummary → {summary_path}")
    print(f"\n=== ALL DONE ===")
    print(json.dumps(summary["results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

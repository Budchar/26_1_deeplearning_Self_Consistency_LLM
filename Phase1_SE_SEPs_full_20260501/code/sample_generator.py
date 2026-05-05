"""Sample generator: load LLM, run N sampled generations + 1 greedy per question,
saving generations and last-token hidden states (per-layer).

Output JSONL schema, one record per question:
    {
      "id": str,
      "dataset": str,
      "question": str,
      "answers": list[str],
      "greedy": str,
      "samples": list[str],            # length N
      "sample_logprobs": list[float],  # length N (avg token logprob, for SE-logprob baseline)
      "hidden_path": str,              # path to .npz file with greedy & sample hidden states
    }

Hidden states .npz layout:
    greedy_h:   shape (L+1, H)   last-token hidden state per layer (incl. embedding)
    samples_h:  shape (N, L+1, H)

Hidden states are taken from the last *prompt* token (matches Kossen et al. SEPs setup
where probes are trained from the prompt representation, before any generation).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROMPT_TEMPLATE = (
    "Answer the following question with a short factual answer.\n"
    "Question: {q}\n"
    "Answer:"
)


def build_prompt(tokenizer, question: str) -> str:
    """Use chat template if available, else fall back to plain prompt."""
    msg = [
        {"role": "system", "content": "You are a helpful assistant that answers questions concisely."},
        {"role": "user", "content": f"Question: {question}\nGive a short factual answer."},
    ]
    try:
        return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    except Exception:
        return PROMPT_TEMPLATE.format(q=question)


def load_model(model_id: str, dtype: str = "fp16", four_bit: bool = False):
    """Load tokenizer + model. Returns (tokenizer, model, device)."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kwargs: Dict = {"trust_remote_code": True, "output_hidden_states": False}
    if four_bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        kwargs["quantization_config"] = bnb
        kwargs["device_map"] = "auto"
        # 7B+ 모델 5070 12GB OOM 방지: 한도 명시 + 잔여는 CPU
        kwargs["max_memory"] = {0: "10GiB", "cpu": "60GiB"}
    else:
        td = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        kwargs["torch_dtype"] = td
        kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    device = next(model.parameters()).device
    return tok, model, device


@torch.no_grad()
def get_prompt_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """Forward only on prompt; return last-token hidden states for every layer.

    Output shape: (num_layers + 1, hidden_size), float16 numpy.
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    # tuple of (L+1) tensors of shape (1, T, H); take last position
    last_idx = attention_mask[0].sum().item() - 1
    layers = []
    for h in out.hidden_states:
        layers.append(h[0, last_idx].detach().to(torch.float16).cpu().numpy())
    return np.stack(layers, axis=0)  # (L+1, H)


@torch.no_grad()
def sample_one(
    model, tokenizer, prompt: str, n_samples: int, max_new_tokens: int,
    temperature: float, top_p: float, device,
) -> Tuple[str, List[str], List[float], np.ndarray, np.ndarray]:
    """Returns greedy_text, sample_texts, sample_avg_logprobs, greedy_h, samples_h."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    attn = enc.attention_mask
    prompt_len = input_ids.shape[1]

    # Hidden state at last prompt token (computed once, shared across greedy/samples).
    h_prompt = get_prompt_hidden(model, input_ids, attn)  # (L+1, H)

    # Greedy generation
    g_out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    greedy_text = tokenizer.decode(g_out.sequences[0, prompt_len:], skip_special_tokens=True).strip()

    # Sampling N times in one batched call
    s_out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n_samples,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    samples_text: List[str] = []
    samples_logprob: List[float] = []
    seqs = s_out.sequences  # (N, prompt_len + new)
    # Compute average token logprob per sequence using output_scores
    # scores: tuple of length new_tokens; each (N, V)
    for i in range(n_samples):
        gen_ids = seqs[i, prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        samples_text.append(text)
        lps = []
        for t, score in enumerate(s_out.scores):
            if t >= gen_ids.shape[0]:
                break
            tok_id = gen_ids[t].item()
            if tok_id == tokenizer.pad_token_id:
                continue
            logp = torch.log_softmax(score[i].float(), dim=-1)[tok_id].item()
            lps.append(logp)
        samples_logprob.append(float(np.mean(lps)) if lps else 0.0)

    # samples_h: we duplicate prompt hidden state for all N (shared prompt).
    samples_h = np.broadcast_to(h_prompt[None], (n_samples,) + h_prompt.shape).copy()
    return greedy_text, samples_text, samples_logprob, h_prompt, samples_h


def existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids: set = set()
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                ids.add(rec["id"])
            except Exception:
                continue
    return ids


def run(
    model_id: str,
    dataset_records: List[Dict],
    out_dir: Path,
    n_samples: int = 10,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
    four_bit: bool = False,
    dtype: str = "fp16",
    save_hidden: bool = True,
    limit: Optional[int] = None,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir = out_dir / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "generations.jsonl"
    done = existing_ids(jsonl_path)

    print(f"[gen] loading model {model_id} (4bit={four_bit}, dtype={dtype})", flush=True)
    tok, model, device = load_model(model_id, dtype=dtype, four_bit=four_bit)
    print(f"[gen] model on {device}; {len(done)} already done; total {len(dataset_records)}", flush=True)

    if limit is not None:
        dataset_records = dataset_records[:limit]

    t0 = time.time()
    n_processed = 0
    with open(jsonl_path, "a") as fout:
        for rec in tqdm(dataset_records, desc=f"gen {Path(model_id).name}"):
            if rec["id"] in done:
                continue
            prompt = build_prompt(tok, rec["question"])
            try:
                greedy, samples, lps, h_prompt, samples_h = sample_one(
                    model, tok, prompt, n_samples, max_new_tokens,
                    temperature, top_p, device,
                )
            except torch.cuda.OutOfMemoryError as e:  # pragma: no cover
                torch.cuda.empty_cache()
                print(f"[gen] OOM on id={rec['id']}: {e}", flush=True)
                continue

            hidden_path = ""
            if save_hidden:
                hp = hidden_dir / f"{rec['id']}.npz"
                np.savez_compressed(hp, greedy_h=h_prompt, samples_h=samples_h)
                hidden_path = str(hp)

            out_rec = {
                "id": rec["id"],
                "dataset": rec["dataset"],
                "question": rec["question"],
                "answers": rec["answers"],
                "greedy": greedy,
                "samples": samples,
                "sample_logprobs": lps,
                "hidden_path": hidden_path,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_processed += 1
    dt = time.time() - t0
    print(f"[gen] {n_processed} processed in {dt:.1f}s ({dt / max(n_processed, 1):.2f}s/q)", flush=True)
    return {"processed": n_processed, "elapsed_s": dt, "out": str(jsonl_path)}


def main() -> None:
    from data_loader import load_dataset_by_name
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["triviaqa", "nq_open", "squad"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-hidden", action="store_true")
    args = ap.parse_args()

    recs = load_dataset_by_name(args.dataset, n=args.n)
    summary = run(
        model_id=args.model,
        dataset_records=recs,
        out_dir=Path(args.out_dir),
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        four_bit=args.four_bit,
        dtype=args.dtype,
        save_hidden=not args.no_hidden,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

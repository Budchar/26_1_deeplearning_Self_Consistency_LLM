"""실험 5 Step 2 — B (SFT adapter) 위에 QLoRA DPO 학습 (C 모델 학습).

Argilla DPO-mix-7k preference dataset으로 1 epoch DPO.
B와 C의 유일한 차이는 이 DPO 한 단계.

실행 예:
    nohup python train_dpo.py \
        --base mistralai/Mistral-7B-v0.1 \
        --sft_adapter ./B_adapter \
        --data argilla/dpo-mix-7k \
        --output ./C_adapter \
        > train_dpo.log 2>&1 &

자원: A100 40GB QLoRA + reference model 4-bit ~ 약 15GB · 1 epoch 약 12-15시간
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig


def format_dpo(example: dict) -> dict | None:
    """Argilla DPO-mix-7k 형식 → trl DPOTrainer 표준 형식.

    Argilla 데이터 schema에 따라 chosen·rejected는 list-of-dict 또는 string.
    """
    chosen = example.get("chosen")
    rejected = example.get("rejected")
    prompt = example.get("prompt")

    # case 1: chosen/rejected가 list of message dicts (conversation 형식)
    if isinstance(chosen, list):
        # 마지막 assistant 응답만 추출
        def last_assistant(msgs):
            for m in reversed(msgs):
                if m.get("role") in ("assistant", "gpt"):
                    return m.get("content") or m.get("value", "")
            return ""
        chosen_text = last_assistant(chosen)
        rejected_text = last_assistant(rejected)
        # prompt가 따로 없으면 user message로 구성
        if not prompt:
            user_msgs = [m for m in chosen if m.get("role") in ("user", "human")]
            prompt = user_msgs[-1].get("content") or user_msgs[-1].get("value", "") if user_msgs else ""
    else:
        chosen_text = str(chosen or "")
        rejected_text = str(rejected or "")
        prompt = str(prompt or "")

    if not prompt or not chosen_text or not rejected_text:
        return None
    return {"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--sft_adapter", required=True, help="train_sft.py로 만든 B adapter 경로")
    ap.add_argument("--data", default="argilla/dpo-mix-7k")
    ap.add_argument("--output", default="./C_adapter")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== train_dpo.py ===")
    print(f"base: {args.base}")
    print(f"sft_adapter: {args.sft_adapter}")
    print(f"data: {args.data}")
    print(f"output: {out_dir}")
    print(f"DPO: beta={args.beta} lr={args.lr}\n", flush=True)

    # 1) Tokenizer
    print("[1/4] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2) Base 모델 4-bit + B adapter load
    print("[2/4] Loading base in 4-bit + applying SFT adapter...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
    )
    # Load B adapter
    model = PeftModel.from_pretrained(base_model, args.sft_adapter, is_trainable=True)
    model.print_trainable_parameters()

    # 3) Dataset
    print(f"[3/4] Loading dataset {args.data}...", flush=True)
    ds = load_dataset(args.data, split="train", trust_remote_code=True)
    ds = ds.map(format_dpo, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x is not None and x.get("prompt") and x.get("chosen") and x.get("rejected"))
    print(f"  dataset size: {len(ds)}")

    # 4) DPOTrainer
    print("[4/4] Starting DPO training...", flush=True)
    dpo_config = DPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT 사용 시 ref는 adapter disable 상태로 자동 처리
        args=dpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 3600

    # Save adapter (B + DPO merged adapter)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "base": args.base,
        "sft_adapter": str(args.sft_adapter),
        "data": args.data,
        "n_samples": len(ds),
        "elapsed_hours": elapsed,
        "beta": args.beta,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"\n=== DONE === {elapsed:.1f} hours")
    print(f"C adapter saved to: {out_dir}")


if __name__ == "__main__":
    main()

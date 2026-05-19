"""실험 5 Step 1 — Mistral-7B-v0.1 QLoRA SFT (B 모델 학습).

OpenHermes-2.5 dataset 일부(기본 100k samples)로 1 epoch SFT.

실행 예:
    nohup python train_sft.py \
        --base mistralai/Mistral-7B-v0.1 \
        --data teknium/OpenHermes-2.5 \
        --n_samples 100000 \
        --output ./B_adapter \
        > train_sft.log 2>&1 &

자원: A100 40GB QLoRA 약 12-15GB · 1 epoch 약 15-20시간
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
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig


SYSTEM_QA = "You are a helpful assistant that answers questions concisely."


def format_openhermes(example: dict) -> dict:
    """OpenHermes-2.5는 conversation 형식. SFT용 plain text로 변환."""
    msgs = example.get("conversations", [])
    parts = []
    for m in msgs:
        role = m.get("from", "")
        value = m.get("value", "")
        if role in ("human", "user"):
            parts.append(f"### Instruction:\n{value}")
        elif role in ("gpt", "assistant"):
            parts.append(f"### Response:\n{value}")
        elif role == "system":
            parts.append(f"### System:\n{value}")
    return {"text": "\n\n".join(parts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--data", default="teknium/OpenHermes-2.5")
    ap.add_argument("--n_samples", type=int, default=100_000)
    ap.add_argument("--output", default="./B_adapter")
    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== train_sft.py ===")
    print(f"base: {args.base}")
    print(f"data: {args.data} (first {args.n_samples} samples)")
    print(f"output: {out_dir}")
    print(f"QLoRA: r={args.lora_r} alpha={args.lora_alpha} lr={args.lr}")
    print(f"batch={args.batch_size} grad_accum={args.grad_accum} max_seq={args.max_seq_length}\n", flush=True)

    # 1) Tokenizer
    print("[1/4] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2) Model (4-bit QLoRA)
    print("[2/4] Loading model in 4-bit...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA adapter
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 3) Dataset
    print(f"[3/4] Loading dataset {args.data}...", flush=True)
    ds = load_dataset(args.data, split="train", trust_remote_code=True)
    if args.n_samples and len(ds) > args.n_samples:
        ds = ds.shuffle(seed=42).select(range(args.n_samples))
    print(f"  dataset size: {len(ds)}")
    ds = ds.map(format_openhermes, remove_columns=ds.column_names, num_proc=4)

    # 4) SFTTrainer
    print("[4/4] Starting SFT training...", flush=True)
    sft_config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=sft_config,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 3600

    # Save adapter only
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "base": args.base,
        "data": args.data,
        "n_samples": min(args.n_samples, len(ds)),
        "elapsed_hours": elapsed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_length": args.max_seq_length,
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"\n=== DONE === {elapsed:.1f} hours")
    print(f"adapter saved to: {out_dir}")


if __name__ == "__main__":
    main()
